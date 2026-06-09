import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import END, StateGraph

from src.llm.llm_switch import get_llm
from src.llm.prompts import (
    build_answer_format_retry_prompt,
    build_context_decision_prompt,
    build_question_output_retry_prompt,
    build_query_refine_prompt,
    get_pass_master_prompt,
    get_question_only_prompt,
)
from src.retrieval.embedder import TEIEmbedder
from src.retrieval.graph import GraphRetriever


load_dotenv()


class RAGGraphState(TypedDict, total=False):
    # LangGraph 노드들이 공유하는 상태입니다.
    # 기존 run_stream()의 지역 변수를 그래프 전체에서 이어받을 수 있게 명시적으로 분리했습니다.
    user_query: str
    session_id: str
    history: InMemoryChatMessageHistory
    recent_history: list[Any]
    # 더 보여줘/전체 보여줘 요청은 직전 검색 결과와 offset이 있어야 동작합니다.
    page_state: dict | None
    is_contextual_request: bool
    is_more_question_request: bool
    is_all_question_request: bool
    context: str | None
    refined_query: str
    query_vector: list[float]
    raw_results: list[dict]
    filtered_results: list[dict]
    image_paths: dict[str, list[str]]
    remaining_question_count: int
    expected_question_ids: list[str]
    missing_question_ids: list[str]
    pending_page_state: dict | None
    question_output_retry_count: int
    question_output_valid: bool
    answer_format_errors: list[str]
    answer_format_retry_count: int
    answer_format_valid: bool
    # 일반 답변과 기출 더보기 답변은 프롬프트 규칙이 다르므로 모드만 상태로 넘깁니다.
    answer_prompt_mode: Literal["default", "question_only"]
    response: str
    status: str
    # 더 보여줄 문제가 없거나 page_state가 없으면 LLM 호출 없이 즉시 답변합니다.
    should_generate: bool


class PassMasterGraphChain:
    """기존 PassMasterChain.run_stream 흐름을 LangGraph 노드로 옮긴 버전입니다.

    기존 동작을 유지하되, 기출 문제 출력처럼 누락되면 사용자 경험이 크게 흔들리는 단계에는
    LangGraph 검증 노드를 붙입니다.
    """

    def __init__(self):
        self.embedder = TEIEmbedder()
        self.retriever = GraphRetriever(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
        self.llm = get_llm()
        self.history_store = {}
        self.question_page_store = {}
        self.prompt = get_pass_master_prompt(os.getenv("LLM_MODEL"))
        self.question_only_prompt = get_question_only_prompt()
        self.graph = self._build_graph()

    def _get_session_history(self, session_id: str):
        """세션별 대화 기록 반환"""
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def _build_graph(self):
        # 기존 run_stream()의 큰 if/else 블록을 LangGraph 노드와 조건부 edge로 옮긴 구조입니다.
        # 이 단계에서는 재시도/자가검토를 넣지 않고, 기존 동작을 그대로 복제하는 것을 우선합니다.
        graph = StateGraph(RAGGraphState)
        graph.add_node("analyze_request", self._analyze_request)
        graph.add_node("handle_question_page", self._handle_question_page)
        graph.add_node("handle_missing_page_state", self._handle_missing_page_state)
        graph.add_node("decide_context_reuse", self._decide_context_reuse)
        graph.add_node("mark_new_search", self._mark_new_search)
        graph.add_node("refine_query", self._refine_query)
        graph.add_node("embed_query", self._embed_query)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_question_output", self._validate_question_output)
        graph.add_node("retry_question_output", self._retry_question_output)
        graph.add_node("validate_answer_format", self._validate_answer_format)
        graph.add_node("retry_answer_format", self._retry_answer_format)
        graph.add_node("finalize_answer", self._finalize_answer)
        graph.add_node("save_history", self._save_history)

        graph.set_entry_point("analyze_request")
        # 첫 분기: 사용자 요청이 더보기인지, 이전 맥락 요청인지, 신규 검색인지 결정합니다.
        graph.add_conditional_edges(
            "analyze_request",
            self._route_after_analyze,
            {
                "question_page": "handle_question_page",
                "missing_page_state": "handle_missing_page_state",
                "contextual": "decide_context_reuse",
                "new_search": "mark_new_search",
            },
        )
        # 더보기 경로는 context가 준비되면 바로 답변 생성으로 가고,
        # 더 보여줄 문제가 없으면 LLM을 호출하지 않고 history 저장으로 끝냅니다.
        graph.add_conditional_edges(
            "handle_question_page",
            self._route_after_context_ready,
            {
                "generate": "generate_answer",
                "save": "save_history",
            },
        )
        graph.add_edge("handle_missing_page_state", "save_history")
        # 이전 맥락 키워드가 있는 경우만 LLM에게 "검색 없이 답변 가능한지"를 묻습니다.
        # 검색이 필요하다고 판단되면 일반 검색 경로로 다시 합류합니다.
        graph.add_conditional_edges(
            "decide_context_reuse",
            self._route_after_context_ready,
            {
                "generate": "generate_answer",
                "search": "refine_query",
                "save": "save_history",
            },
        )
        graph.add_edge("mark_new_search", "refine_query")
        graph.add_edge("refine_query", "embed_query")
        graph.add_edge("embed_query", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_answer")
        graph.add_edge("generate_answer", "validate_question_output")
        graph.add_conditional_edges(
            "validate_question_output",
            self._route_after_question_validation,
            {
                "retry": "retry_question_output",
                "validate_format": "validate_answer_format",
                "finalize": "finalize_answer",
            },
        )
        graph.add_edge("retry_question_output", "validate_question_output")
        graph.add_conditional_edges(
            "validate_answer_format",
            self._route_after_answer_format_validation,
            {
                "retry": "retry_answer_format",
                "finalize": "finalize_answer",
            },
        )
        graph.add_edge("retry_answer_format", "validate_question_output")
        graph.add_edge("finalize_answer", "save_history")
        graph.add_edge("save_history", END)
        return graph.compile()

    def _analyze_request(self, state: RAGGraphState) -> dict:
        """사용자 입력을 라우팅하기 위한 최소 상태를 준비합니다."""
        user_query = state["user_query"]
        session_id = state["session_id"]
        history = self._get_session_history(session_id)
        context_keywords = ["방금", "그거", "이거", "앞서", "다시", "정답만", "해설만"]

        return {
            "history": history,
            "recent_history": history.messages[-2:],
            "page_state": self.question_page_store.get(session_id),
            "is_contextual_request": any(word in user_query for word in context_keywords),
            "is_more_question_request": self._is_more_question_request(user_query),
            "is_all_question_request": self._is_all_question_request(user_query),
            "image_paths": {},
            "remaining_question_count": 0,
            "expected_question_ids": [],
            "missing_question_ids": [],
            "pending_page_state": None,
            "question_output_retry_count": 0,
            "question_output_valid": True,
            "answer_format_errors": [],
            "answer_format_retry_count": 0,
            "answer_format_valid": True,
            "answer_prompt_mode": "default",
            "context": None,
            "should_generate": True,
            "status": "질문 맥락 분석 중...",
        }

    def _route_after_analyze(self, state: RAGGraphState) -> str:
        """기존 run_stream()의 최상단 if/elif 라우팅을 LangGraph edge로 표현합니다."""
        wants_more = state.get("is_more_question_request") or state.get("is_all_question_request")
        if wants_more and state.get("page_state"):
            return "question_page"
        if wants_more and not state.get("page_state"):
            return "missing_page_state"
        if state.get("is_contextual_request"):
            return "contextual"
        return "new_search"

    def _handle_question_page(self, state: RAGGraphState) -> dict:
        """이전 검색 결과에서 다음 기출 페이지를 구성합니다.

        이 경로에서는 새 검색을 하지 않고, 저장된 검색 결과와 offset만 사용합니다.
        """
        page_state = state["page_state"] or {}
        filtered_results = page_state["results"]
        question_offset = page_state["offset"]
        question_limit = None if state.get("is_all_question_request") else page_state["limit"]
        total_questions = self.retriever.count_related_questions(filtered_results)

        if question_offset >= total_questions:
            return {
                "response": "이전 검색 결과에서 더 보여줄 관련 기출 문제가 없습니다.",
                "image_paths": {},
                "should_generate": False,
                "status": "이전 검색 결과에서 더 보여줄 기출 확인 완료",
            }

        context = self.retriever.format_context_for_llm(
            filtered_results,
            question_offset=question_offset,
            question_limit=question_limit,
        )
        image_paths = self.retriever.collect_image_paths(
            filtered_results,
            question_offset=question_offset,
            question_limit=question_limit,
        )

        expected_question_ids = self._get_page_question_ids(
            filtered_results,
            question_offset=question_offset,
            question_limit=question_limit,
        )

        # offset은 LLM 답변 검증이 끝난 뒤에만 커밋합니다.
        # 생성 실패 상태에서 먼저 이동시키면 LLM이 빼먹은 문제가 다음 페이지에서 사라질 수 있습니다.
        shown_count = total_questions - question_offset if state.get("is_all_question_request") else page_state["limit"]
        pending_page_state = {
            **page_state,
            "offset": min(question_offset + shown_count, total_questions),
        }

        status = (
            "이전 검색 결과에서 남은 기출 전체 LLM 검수 준비 중..."
            if state.get("is_all_question_request")
            else "이전 검색 결과에서 다음 기출 3문제 LLM 검수 준비 중..."
        )
        return {
            "context": context,
            "image_paths": image_paths,
            "answer_prompt_mode": "question_only",
            "expected_question_ids": expected_question_ids,
            "pending_page_state": pending_page_state,
            "remaining_question_count": max(total_questions - pending_page_state["offset"], 0),
            "question_output_retry_count": 0,
            "answer_format_retry_count": 0,
            "should_generate": True,
            "status": status,
        }

    def _handle_missing_page_state(self, state: RAGGraphState) -> dict:
        """더보기 요청이지만 이전 검색 결과가 없는 경우입니다."""
        return {
            "response": "이전 검색 결과가 없어 이어서 보여줄 관련 기출 문제가 없습니다. 먼저 궁금한 개념을 질문해 주세요.",
            "image_paths": {},
            "should_generate": False,
            "status": "이전 검색 결과 없음",
        }

    def _decide_context_reuse(self, state: RAGGraphState) -> dict:
        """방금/그거/다시 같은 맥락 요청에서 검색을 생략할 수 있는지 판단합니다."""
        decision = self.llm.invoke(
            build_context_decision_prompt(state.get("recent_history", []), state["user_query"])
        ).content.strip().upper()

        if "YES" in decision:
            print("💡 [Decision] 명시적 키워드 기반 맥락 활용 (History Reuse)")
            return {
                "context": "이전 대화 내용을 바탕으로 답변하세요.",
                "status": "이전 대화 맥락 재사용",
            }

        print(f"🔎 [Decision] 키워드는 있으나 검색 필요 판단: {state['user_query']}")
        return {
            "context": None,
            "status": "새로운 지식 그래프 검색 필요",
        }

    def _mark_new_search(self, state: RAGGraphState) -> dict:
        """신규 질문은 별도 판단 없이 검색 경로로 보냅니다."""
        print(f"🔎 [Decision] 신규 검색 실행: {state['user_query']}")
        return {"status": "신규 검색 경로 선택"}

    def _refine_query(self, state: RAGGraphState) -> dict:
        """사용자 질문을 검색용 키워드로 정제합니다."""
        refined_query = self.llm.invoke(
            build_query_refine_prompt(state["user_query"])
        ).content.strip()
        print(f"🧹 [Refine] 검색어 정제: {state['user_query']} -> {refined_query}")
        return {
            "refined_query": refined_query,
            "status": f"검색어 정제 완료: {refined_query}",
        }

    def _embed_query(self, state: RAGGraphState) -> dict:
        """정제된 검색어를 TEI 임베딩 벡터로 변환합니다."""
        return {
            "query_vector": self.embedder.get_embedding(state["refined_query"]),
            "status": "TEI 임베딩 생성 중...",
        }

    def _retrieve_context(self, state: RAGGraphState) -> dict:
        """hybrid retriever로 top1 Concept와 연결 기출 context를 구성합니다."""
        raw_results = self.retriever.search_concepts_with_questions(
            query_vector=state["query_vector"],
            query_text=state["refined_query"],
            top_k=1,
        )
        threshold = 0.7
        filtered_results = [result for result in raw_results if result.get("score", 0) > threshold]

        if not filtered_results:
            # 검색 결과가 없으면 이전 더보기 상태도 의미가 없으므로 제거합니다.
            self.question_page_store.pop(state["session_id"], None)
            return {
                "raw_results": raw_results,
                "filtered_results": [],
                "context": "지식 베이스에서 관련 내용을 찾을 수 없습니다.",
                "image_paths": {},
                "remaining_question_count": 0,
                "status": "검색 결과 필터링 및 컨텍스트 구성 중...",
            }

        question_limit = 3
        question_offset = 0
        total_questions = self.retriever.count_related_questions(filtered_results)
        context = self.retriever.format_context_for_llm(
            filtered_results,
            question_offset=question_offset,
            question_limit=question_limit,
        )
        image_paths = self.retriever.collect_image_paths(
            filtered_results,
            question_offset=question_offset,
            question_limit=question_limit,
        )
        pending_page_state = {
            "results": filtered_results,
            "offset": min(question_limit, total_questions),
            "limit": question_limit,
        }
        return {
            "raw_results": raw_results,
            "filtered_results": filtered_results,
            "context": context,
            "image_paths": image_paths,
            "expected_question_ids": self._get_page_question_ids(
                filtered_results,
                question_offset=question_offset,
                question_limit=question_limit,
            ),
            "pending_page_state": pending_page_state,
            "question_output_retry_count": 0,
            "answer_format_retry_count": 0,
            "remaining_question_count": max(
                total_questions - pending_page_state["offset"],
                0,
            ),
            "status": "검색 결과 필터링 및 컨텍스트 구성 중...",
        }

    def _generate_answer(self, state: RAGGraphState) -> dict:
        """준비된 context와 모드에 맞는 프롬프트로 최종 답변을 생성합니다."""
        answer_prompt = self.question_only_prompt if state.get("answer_prompt_mode") == "question_only" else self.prompt
        prompt_value = answer_prompt.invoke({
            "history": state.get("recent_history", []),
            "context": state.get("context"),
            "question": state["user_query"],
        })
        response = self._extract_llm_content(self.llm.invoke(prompt_value))
        return {
            "response": response,
            "status": "LLM 답변 생성 중...",
        }

    def _validate_question_output(self, state: RAGGraphState) -> dict:
        """LLM 답변에 이번 페이지의 기출 문제 ID가 모두 포함됐는지 검사합니다."""
        expected_question_ids = state.get("expected_question_ids", [])
        if not expected_question_ids:
            return {
                "missing_question_ids": [],
                "question_output_valid": True,
            }

        response = state.get("response", "")
        missing_question_ids = [
            question_id
            for question_id in expected_question_ids
            if not self._contains_question_id(response, question_id)
        ]
        is_valid = not missing_question_ids
        if missing_question_ids:
            self._log_validation_failure(
                stage="question_output",
                state=state,
                errors=[f"기출 문제 ID 누락: {question_id}" for question_id in missing_question_ids],
                extra={
                    "missing_question_ids": missing_question_ids,
                    "will_retry": state.get("question_output_retry_count", 0) < 1,
                },
            )

        return {
            "missing_question_ids": missing_question_ids,
            "question_output_valid": is_valid,
            "status": (
                "기출 문제 출력 검증 완료"
                if is_valid
                else f"기출 문제 출력 누락 감지: {', '.join(missing_question_ids)}"
            ),
        }

    def _retry_question_output(self, state: RAGGraphState) -> dict:
        """누락된 기출 ID가 있으면 같은 context로 답변 생성을 한 번 더 시도합니다."""
        retry_count = state.get("question_output_retry_count", 0) + 1
        retry_prompt = build_question_output_retry_prompt(
            context=state.get("context", ""),
            question=state["user_query"],
            previous_response=state.get("response", ""),
            missing_question_ids=state.get("missing_question_ids", []),
        )
        response = self._extract_llm_content(self.llm.invoke(retry_prompt))
        return {
            "response": response,
            "question_output_retry_count": retry_count,
            "status": "누락된 기출 문제 재생성 중...",
        }

    def _validate_answer_format(self, state: RAGGraphState) -> dict:
        """필수 답변 형식과 정답 마스킹이 지켜졌는지 검사합니다."""
        response = state.get("response", "")
        expected_question_ids = state.get("expected_question_ids", [])
        expected_image_question_ids = sorted((state.get("image_paths") or {}).keys())
        is_question_only = state.get("answer_prompt_mode") == "question_only"
        errors = []

        if is_question_only:
            if not self._has_heading(response, "실제 기출 문제"):
                errors.append("[실제 기출 문제] 섹션 누락")
        else:
            required_sections = ["단원 정보", "요약 정보", "실제 기출 문제"]
            for section in required_sections:
                if not self._has_heading(response, section):
                    errors.append(f"[{section}] 섹션 누락")

            # ID와 출제 횟수는 독립 섹션이 아니라 [단원 정보] 안의 라벨로 출력하는 형식입니다.
            for label in ["ID", "출제 횟수", "연결된 기출 문제", "중요도"]:
                if not self._has_label(response, label):
                    errors.append(f"[단원 정보]의 {label} 항목 누락")

        # 실제 기출을 출력하는 답변이면 정답 마스킹 HTML이 적어도 한 번은 있어야 합니다.
        if expected_question_ids and not self._has_answer_mask(response):
            errors.append("정답 마스킹 HTML 누락")

        # 이미지가 연결된 문제는 답변 안에 해당 위치를 표시하는 토큰이 반드시 있어야 합니다.
        for question_id in expected_image_question_ids:
            if not self._has_image_token(response, question_id):
                errors.append(f"이미지 토큰 누락: [[IMAGE:{question_id}]]")

        if errors:
            self._log_validation_failure(
                stage="answer_format",
                state=state,
                errors=errors,
                extra={
                    "will_retry": state.get("answer_format_retry_count", 0) < 1,
                    "expected_image_question_ids": expected_image_question_ids,
                },
            )

        return {
            "answer_format_errors": errors,
            "answer_format_valid": not errors,
            "status": (
                "답변 형식 검증 완료"
                if not errors
                else f"답변 형식 누락 감지: {', '.join(errors)}"
            ),
        }

    def _retry_answer_format(self, state: RAGGraphState) -> dict:
        """내용은 유지하되 필수 섹션과 정답 마스킹을 맞추도록 한 번 더 생성합니다."""
        retry_count = state.get("answer_format_retry_count", 0) + 1
        retry_prompt = build_answer_format_retry_prompt(
            context=state.get("context", ""),
            question=state["user_query"],
            previous_response=state.get("response", ""),
            validation_errors=state.get("answer_format_errors", []),
            question_only=state.get("answer_prompt_mode") == "question_only",
        )
        response = self._extract_llm_content(self.llm.invoke(retry_prompt))
        return {
            "response": response,
            "answer_format_retry_count": retry_count,
            "status": "답변 형식 재생성 중...",
        }

    def _finalize_answer(self, state: RAGGraphState) -> dict:
        """검증 결과에 따라 더보기 offset을 커밋하고 최종 안내 문구를 붙입니다."""
        response = state.get("response", "")

        if state.get("question_output_valid", True):
            pending_page_state = state.get("pending_page_state")
            if pending_page_state is not None:
                self.question_page_store[state["session_id"]] = pending_page_state
            response = self._append_remaining_question_notice(
                response,
                state.get("remaining_question_count", 0),
            )
        else:
            # 재시도 후에도 누락되면 offset을 움직이지 않습니다.
            # 사용자가 다시 "더 보여줘"를 입력했을 때 같은 묶음을 다시 시도할 수 있게 하기 위함입니다.
            missing_ids = ", ".join(state.get("missing_question_ids", []))
            response = (
                f"{response}\n\n"
                f"시스템 검증 결과, 다음 기출 문제가 답변에서 누락되었습니다: {missing_ids}. "
                "누락 방지를 위해 다음 기출 페이지로는 아직 넘어가지 않았습니다."
            )

        if not state.get("answer_format_valid", True):
            errors = ", ".join(state.get("answer_format_errors", []))
            response = (
                f"{response}\n\n"
                f"시스템 검증 결과, 답변 형식 문제가 남아 있습니다: {errors}."
            )

        return {
            "response": response,
            "status": "답변 검증 및 페이지 상태 반영 완료",
        }

    def _save_history(self, state: RAGGraphState) -> dict:
        """LangGraph 실행이 끝나기 전에 기존 체인과 동일하게 세션 history를 갱신합니다."""
        history = state["history"]
        history.add_user_message(state["user_query"])
        history.add_ai_message(state.get("response", ""))
        return {"status": "대화 기록 저장 완료"}

    def _route_after_context_ready(self, state: RAGGraphState) -> str:
        """context 준비 상태에 따라 검색, 답변 생성, 즉시 종료 경로를 고릅니다."""
        if not state.get("should_generate", True):
            return "save"
        if state.get("context") is None:
            return "search"
        return "generate"

    def _route_after_question_validation(self, state: RAGGraphState) -> str:
        """기출 ID 누락이 있으면 한 번만 재생성하고, 이후에는 최종 응답으로 확정합니다."""
        if state.get("question_output_valid", True):
            return "validate_format"
        if state.get("question_output_retry_count", 0) < 1:
            return "retry"
        return "finalize"

    def _route_after_answer_format_validation(self, state: RAGGraphState) -> str:
        """필수 답변 형식이 빠졌으면 한 번만 재생성합니다."""
        if state.get("answer_format_valid", True):
            return "finalize"
        if state.get("answer_format_retry_count", 0) < 1:
            return "retry"
        return "finalize"

    def run(self, user_query: str, session_id: str = "default_user"):
        """비스트리밍 호출용 실행 함수입니다. 기존 PassMasterChain.run()과 같은 외부 인터페이스를 맞춥니다."""
        try:
            result = self.graph.invoke({
                "user_query": user_query,
                "session_id": session_id,
            })
            return result.get("response", "")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ 파이프라인 실행 중 오류 발생: {e}"

    def run_stream(self, user_query: str, session_id: str = "default_user"):
        """Streamlit이 기대하는 status/answer 이벤트 형태로 LangGraph 실행 결과를 변환합니다."""
        try:
            final_state = {}
            for event in self.graph.stream(
                {"user_query": user_query, "session_id": session_id},
                stream_mode="updates",
            ):
                for node_update in event.values():
                    if not isinstance(node_update, dict):
                        continue
                    final_state.update(node_update)
                    # 각 노드가 반환한 status를 기존 st.status UI에 그대로 흘려보냅니다.
                    # 저장 완료 메시지는 사용자에게 의미가 약해서 화면에는 노출하지 않습니다.
                    status = node_update.get("status")
                    if status and status != "대화 기록 저장 완료":
                        yield {"type": "status", "message": status}

            yield {
                "type": "answer",
                "content": final_state.get("response", ""),
                "images": final_state.get("image_paths", {}),
            }
        except Exception as e:
            yield {"type": "answer", "content": f"❌ 오류 발생: {e}", "images": {}}

    def _is_more_question_request(self, user_query: str) -> bool:
        """이전 검색 결과의 다음 기출 3개를 요청하는 표현을 감지합니다."""
        more_keywords = ["더 보여", "더보기", "더 보기", "다음", "이어서", "계속"]
        return any(keyword in user_query for keyword in more_keywords)

    def _is_all_question_request(self, user_query: str) -> bool:
        """이전 검색 결과의 남은 기출 전체를 요청하는 표현을 감지합니다."""
        all_keywords = ["전체 보여", "전부 보여", "다 보여", "전체 기출", "나머지 전부", "남은 거 전부"]
        return any(keyword in user_query for keyword in all_keywords)

    def _append_remaining_question_notice(self, response: str, remaining_count: int) -> str:
        """LLM 답변의 중복 안내 문구를 제거하고 코드가 계산한 남은 기출 수로 안내를 붙입니다."""
        if remaining_count <= 0:
            return response

        notice_pattern = (
            r"\n*(?:현재 그래프DB에 연결된 )?관련 기출이 \d+개 더 있습니다\.\s*"
            r"다음 대화에서 [\"']더 보여줘[\"']라고 입력하면 이어서 3개씩 보여드릴게요\."
        )
        response = re.sub(notice_pattern, "", response).rstrip()
        notice = (
            f"현재 그래프DB에 연결된 관련 기출이 {remaining_count}개 더 있습니다. "
            "다음 대화에서 \"더 보여줘\"라고 입력하면 이어서 3개씩 보여드릴게요."
        )
        return f"{response}\n\n{notice}"

    def _extract_llm_content(self, response):
        """LangChain 모델별 응답 객체 차이를 문자열로 통일합니다."""
        return response.content if hasattr(response, "content") else str(response)

    def _get_page_question_ids(
        self,
        search_results: list,
        question_offset: int,
        question_limit: int | None,
    ) -> list[str]:
        """현재 LLM에 넘긴 페이지 범위의 문제 ID만 뽑아 검증 기준으로 사용합니다."""
        questions = self.retriever.get_sorted_related_questions(search_results, primary_only=True)
        if question_limit is None:
            page_questions = questions[question_offset:]
        else:
            page_questions = questions[question_offset:question_offset + question_limit]
        return [str(question.get("id")) for question in page_questions if question.get("id")]

    def _contains_question_id(self, response: str, question_id: str) -> bool:
        """문제 ID가 답변에 실제로 등장했는지 느슨하게 검사합니다."""
        escaped_id = re.escape(question_id)
        return re.search(rf"(?<![\w]){escaped_id}(?![\w])", response) is not None

    def _has_answer_mask(self, response: str) -> bool:
        """정답 마스킹 HTML이 답변에 포함되어 있는지 확인합니다."""
        return re.search(r"<span\s+class=[\"']answer-mask[\"']>", response) is not None

    def _has_heading(self, response: str, heading: str) -> bool:
        """대괄호/마크다운/콜론 유무와 관계없이 제목 텍스트가 있는지 확인합니다."""
        normalized_response = re.sub(r"\s+", "", response)
        normalized_heading = re.sub(r"\s+", "", heading)
        return normalized_heading in normalized_response

    def _has_label(self, response: str, label: str) -> bool:
        """단원 정보 안의 `ID:`, `출제 횟수:` 같은 라벨 표기를 검사합니다."""
        normalized_response = re.sub(r"\s+", "", response)
        normalized_label = re.sub(r"\s+", "", label)
        return f"{normalized_label}:" in normalized_response

    def _has_image_token(self, response: str, question_id: str) -> bool:
        """이미지가 있는 문제의 `[[IMAGE:문제ID]]` 토큰이 답변에 남아 있는지 확인합니다."""
        return f"[[IMAGE:{question_id}]]" in response

    def _log_validation_failure(
        self,
        stage: str,
        state: RAGGraphState,
        errors: list[str],
        extra: dict | None = None,
    ) -> None:
        """검증 실패 사례를 JSONL로 남겨 나중에 재시도 품질을 평가할 수 있게 합니다."""
        log_path = Path(
            os.getenv("RAG_VALIDATION_LOG_PATH", "logs/rag_validation_failures.jsonl")
        )
        response = state.get("response", "")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "errors": errors,
            "session_id": state.get("session_id"),
            "user_query": state.get("user_query"),
            "refined_query": state.get("refined_query"),
            "answer_prompt_mode": state.get("answer_prompt_mode"),
            "question_output_retry_count": state.get("question_output_retry_count", 0),
            "answer_format_retry_count": state.get("answer_format_retry_count", 0),
            "expected_question_ids": state.get("expected_question_ids", []),
            "image_question_ids": sorted((state.get("image_paths") or {}).keys()),
            "remaining_question_count": state.get("remaining_question_count", 0),
            "response_length": len(response),
            "response_excerpt": response[:2000],
        }
        if extra:
            record.update(extra)

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # 로그 저장 실패가 사용자 답변 생성까지 막으면 안 되므로 콘솔에만 남깁니다.
            print(f"[ValidationLog] 검증 실패 로그 저장 실패: {exc}")
