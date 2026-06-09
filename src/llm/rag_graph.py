import os
import re
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import END, StateGraph

from src.llm.llm_switch import get_llm
from src.llm.prompts import (
    build_context_decision_prompt,
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
    # 일반 답변과 기출 더보기 답변은 프롬프트 규칙이 다르므로 모드만 상태로 넘깁니다.
    answer_prompt_mode: Literal["default", "question_only"]
    response: str
    status: str
    # 더 보여줄 문제가 없거나 page_state가 없으면 LLM 호출 없이 즉시 답변합니다.
    should_generate: bool


class PassMasterGraphChain:
    """기존 PassMasterChain.run_stream 흐름을 LangGraph 노드로 옮긴 버전입니다.

    첫 적용 단계에서는 검토/재시도 로직을 추가하지 않고, 기존 동작과 같은 결과를 내는 것을 목표로 합니다.
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
        graph.add_edge("generate_answer", "save_history")
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

        # 전체보기는 남은 문제를 모두 소비하고, 더보기는 기존 limit만큼만 offset을 이동합니다.
        shown_count = total_questions - question_offset if state.get("is_all_question_request") else page_state["limit"]
        page_state["offset"] = min(question_offset + shown_count, total_questions)
        self.question_page_store[state["session_id"]] = page_state

        status = (
            "이전 검색 결과에서 남은 기출 전체 LLM 검수 준비 중..."
            if state.get("is_all_question_request")
            else "이전 검색 결과에서 다음 기출 3문제 LLM 검수 준비 중..."
        )
        return {
            "context": context,
            "image_paths": image_paths,
            "answer_prompt_mode": "question_only",
            "remaining_question_count": max(total_questions - page_state["offset"], 0),
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
        self.question_page_store[state["session_id"]] = {
            "results": filtered_results,
            "offset": min(question_limit, total_questions),
            "limit": question_limit,
        }
        return {
            "raw_results": raw_results,
            "filtered_results": filtered_results,
            "context": context,
            "image_paths": image_paths,
            "remaining_question_count": max(
                total_questions - self.question_page_store[state["session_id"]]["offset"],
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
        response = self._append_remaining_question_notice(
            response,
            state.get("remaining_question_count", 0),
        )
        return {
            "response": response,
            "status": "LLM 답변 생성 중...",
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
