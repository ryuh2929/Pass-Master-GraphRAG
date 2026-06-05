import os
import re
from dotenv import load_dotenv

# LangChain 관련 컴포넌트
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.retrieval.embedder import TEIEmbedder
from src.retrieval.graph import GraphRetriever

from src.llm.llm_switch import get_llm
from src.llm.prompts import (
    build_context_decision_prompt,
    build_query_refine_prompt,
    get_condense_question_prompt,
    get_pass_master_prompt,
)

load_dotenv()

class PassMasterChain:
    def __init__(self):
        # 1. 원본 컴포넌트 로드
        self.embedder = TEIEmbedder()
        self.retriever = GraphRetriever(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        # 2. LangChain용 LLM 설정 (ChatOllama 혹은 ChatOpenAI 사용)
        self.llm = get_llm() 
        
        # 3. 대화 기록 저장소 (세션별 관리)
        self.history_store = {} 
        self.question_page_store = {}

        # 4. 프롬프트 템플릿 구성
        self.condense_question_prompt = get_condense_question_prompt()
        self.prompt = get_pass_master_prompt(os.getenv("LLM_MODEL"))

        # 메인 RAG 체인
        base_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(self._get_refined_context)
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        self.chain_with_history = RunnableWithMessageHistory(
            base_chain,
            get_session_history=self._get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def _get_session_history(self, session_id: str):
        """세션별 대화 기록 반환"""
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]
        
    def _execute_vector_search(self, query, history=None):
        """
        질문 재구성 후 Vector DB 검색 실행
        
        불용어를 제거하고 핵심 키워드 위주로 검색을 수행하여 
        '알려줘' 노이즈 현상을 방지함
        """
        # 1. LLM을 사용하여 검색용 '핵심 명사'만 추출 (Condense & Clean)
        refined_query = self.llm.invoke(build_query_refine_prompt(query)).content.strip()
        print(f"🧹 [Refine] 검색어 정제: {query} -> {refined_query}")
        
        # 2. 정제된 키워드로 임베딩 및 Neo4j 검색 수행
        query_vector = self.embedder.get_embedding(refined_query)
        raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=3)
        
        # 3. 유사도가 너무 낮은 결과는 무시하는 로직 추가 가능
        threshold = 0.7
        filtered_results = [r for r in raw_results if r.get('score', 0) > threshold]

        # 결과가 하나도 없으면 LLM이 검색 결과 없음을 인지하도록 빈 값 처리
        if not filtered_results:
            return "지식 베이스에서 관련 내용을 찾을 수 없습니다."
        
        return self.retriever.format_context_for_llm(filtered_results)

    def _extract_llm_content(self, response):
        return response.content if hasattr(response, "content") else str(response)

    def _is_more_question_request(self, user_query: str) -> bool:
        more_keywords = ["더 보여", "더보기", "더 보기", "다음", "이어서", "계속"]
        return any(keyword in user_query for keyword in more_keywords)

    def _is_all_question_request(self, user_query: str) -> bool:
        all_keywords = ["전체 보여", "전부 보여", "다 보여", "전체 기출", "나머지 전부", "남은 거 전부"]
        return any(keyword in user_query for keyword in all_keywords)

    def _append_remaining_question_notice(self, response: str, remaining_count: int) -> str:
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

    def _get_refined_context(self, x):
        """
        맥락이 포함된 재구성된 질문으로 검색 수행
            - LLM에게 검색 필요성 판단을 맡기는 대신, 명시적 키워드가 없을 경우 무조건 검색을 수행하는 하드 라우팅 로직
        """
        # 이 시점에서 history가 주입된 상태가 아니므로, 별도로 condense 과정이 필요할 수 있음
        # 간단한 구현을 위해 여기서는 현재 질문을 사용하되, 검색 퀄리티를 위해 로그 확인
        # 1. 여기서 history를 직접 제어합니다. (최근 2개 메시지만 참조)
        # x['history']는 RunnableWithMessageHistory가 주입해줍니다.
        user_query = x["question"]
        recent_history = x.get("history", [])[-2:] # 직전 문답만 참고
        
        # '이전 맥락'을 써야만 하는 명시적 키워드 리스트 (Hard Rule)
        # 이 단어들이 포함되지 않았다면 LLM 판단 없이 바로 검색으로 보냄
        context_keywords = ["방금", "그거", "이거", "앞서", "다시", "정답만", "해설만"]
        is_contextual_request = any(word in user_query for word in context_keywords)

        if not is_contextual_request:
            print(f"🔎 [Decision] 신규 검색 실행: {user_query}")
            return self._execute_vector_search(user_query, recent_history)

        # 2. 키워드가 있는 경우, LLM에게 검색 필요성 판단
        decision = self.llm.invoke(
            build_context_decision_prompt(recent_history, user_query)
        ).content.strip().upper()

        if "YES" in decision:
            print("💡 [Decision] 명시적 키워드 기반 맥락 활용 (History Reuse)")
            return "이전 대화 내용을 바탕으로 답변하세요."
        
        print(f"🔎 [Decision] 키워드는 있으나 검색 필요 판단: {user_query}")
        return self._execute_vector_search(user_query, recent_history)

    # def _get_session_history(self, session_id: str):
    #     # RunnableWithMessageHistory와 쓰려면 객체 구조를 맞춰야 함
    #     if session_id not in self.history_store:
    #         self.history_store[session_id] = ConversationTokenBufferMemory(
    #             llm=self.llm, 
    #             max_token_limit=self.max_token_limit,
    #             return_messages=True
    #         )
    #     return self.history_store[session_id]

    # def _condense_question(self, query, history):
    #     """
    #     TokenBufferMemory를 쓰면 history 자체가 이미 '최적화된 최근 대화'입니다.
    #     따라서 여기서 별도로 history[-3:] 처럼 슬라이싱 할 필요가 없어집니다.
    #     """
    #     condense_prompt = f"""
    #     [최적화된 이전 대화 기록]
    #     {history}
        
    #     [사용자 현재 질문]
    #     {query}
        
    #     위 내용을 바탕으로 검색을 위한 한국어 핵심 키워드 한 문장만 생성해줘.
    #     """
    #     response = self.llm.invoke(condense_prompt)
    #     return response.content if hasattr(response, 'content') else str(response)
    
    def run(self, user_query: str, session_id: str = "default_user"):
        try:
            return self.chain_with_history.invoke(
                {"question": user_query},
                config={"configurable": {"session_id": session_id}}
            )
        except Exception as e:
            import traceback
            traceback.print_exc() # 상세 에러 로그 확인용
            return f"❌ 파이프라인 실행 중 오류 발생: {e}"
        
    def run_stream(self, user_query: str, session_id: str = "default_user"):
        try:
            history = self._get_session_history(session_id)
            recent_history = history.messages[-2:]
            image_paths = {}
            remaining_question_count = 0
            page_state = self.question_page_store.get(session_id)

            yield {"type": "status", "message": "질문 맥락 분석 중..."}
            context_keywords = ["방금", "그거", "이거", "앞서", "다시", "정답만", "해설만"]
            is_contextual_request = any(word in user_query for word in context_keywords)
            is_more_question_request = self._is_more_question_request(user_query)
            is_all_question_request = self._is_all_question_request(user_query)

            if (is_more_question_request or is_all_question_request) and page_state:
                if is_all_question_request:
                    yield {"type": "status", "message": "이전 검색 결과에서 남은 기출 전체 LLM 검수 준비 중..."}
                else:
                    yield {"type": "status", "message": "이전 검색 결과에서 다음 기출 3문제 LLM 검수 준비 중..."}

                filtered_results = page_state["results"]
                question_offset = page_state["offset"]
                question_limit = None if is_all_question_request else page_state["limit"]
                total_questions = self.retriever.count_related_questions(filtered_results)

                if question_offset >= total_questions:
                    response = "이전 검색 결과에서 더 보여줄 관련 기출 문제가 없습니다."
                    history.add_user_message(user_query)
                    history.add_ai_message(response)
                    yield {"type": "answer", "content": response, "images": {}}
                    return

                context = self.retriever.format_context_for_llm(
                    filtered_results,
                    question_offset=question_offset,
                    question_limit=question_limit
                )
                context += (
                    "\n\n### 응답 모드\n"
                    "- 이 요청은 이전 검색 결과의 기출 더보기입니다.\n"
                    "- [단원 정보], [요약 정보], [보충 설명], [합격 포인트]는 반복하지 마십시오.\n"
                    "- [실제 기출 문제] 섹션만 출력하십시오.\n"
                    "- 제공된 문제 원문과 코드는 LLM이 읽기 좋게 줄바꿈과 코드블럭을 정리하되, 문제 내용과 정답은 바꾸지 마십시오.\n"
                )
                image_paths = self.retriever.collect_image_paths(
                    filtered_results,
                    question_offset=question_offset,
                    question_limit=question_limit
                )

                shown_count = total_questions - question_offset if is_all_question_request else page_state["limit"]
                page_state["offset"] = min(question_offset + shown_count, total_questions)
                self.question_page_store[session_id] = page_state
                remaining_question_count = max(
                    total_questions - page_state["offset"],
                    0
                )
            elif (is_more_question_request or is_all_question_request) and not page_state:
                response = "이전 검색 결과가 없어 이어서 보여줄 관련 기출 문제가 없습니다. 먼저 궁금한 개념을 질문해 주세요."
                history.add_user_message(user_query)
                history.add_ai_message(response)
                yield {"type": "answer", "content": response, "images": {}}
                return
            elif is_contextual_request:
                yield {"type": "status", "message": "이전 대화 재사용 여부 판단 중..."}
                decision = self.llm.invoke(
                    build_context_decision_prompt(recent_history, user_query)
                ).content.strip().upper()

                if "YES" in decision:
                    print("💡 [Decision] 명시적 키워드 기반 맥락 활용 (History Reuse)")
                    yield {"type": "status", "message": "이전 대화 맥락 재사용"}
                    context = "이전 대화 내용을 바탕으로 답변하세요."
                else:
                    print(f"🔎 [Decision] 키워드는 있으나 검색 필요 판단: {user_query}")
                    yield {"type": "status", "message": "새로운 지식 그래프 검색 필요"}
                    context = None
            else:
                print(f"🔎 [Decision] 신규 검색 실행: {user_query}")
                yield {"type": "status", "message": "신규 검색 경로 선택"}
                context = None

            if context is None:
                yield {"type": "status", "message": "검색어 정제 중..."}
                refined_query = self.llm.invoke(
                    build_query_refine_prompt(user_query)
                ).content.strip()
                print(f"🧹 [Refine] 검색어 정제: {user_query} -> {refined_query}")

                yield {"type": "status", "message": f"검색어 정제 완료: {refined_query}"}
                yield {"type": "status", "message": "TEI 임베딩 생성 중..."}
                query_vector = self.embedder.get_embedding(refined_query)

                yield {"type": "status", "message": "Neo4j 지식 그래프 검색 중..."}
                raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=3)

                yield {"type": "status", "message": "검색 결과 필터링 및 컨텍스트 구성 중..."}
                threshold = 0.7
                filtered_results = [r for r in raw_results if r.get('score', 0) > threshold]

                if not filtered_results:
                    context = "지식 베이스에서 관련 내용을 찾을 수 없습니다."
                    self.question_page_store.pop(session_id, None)
                else:
                    question_limit = 3
                    question_offset = 0
                    total_questions = self.retriever.count_related_questions(filtered_results)
                    context = self.retriever.format_context_for_llm(
                        filtered_results,
                        question_offset=question_offset,
                        question_limit=question_limit
                    )
                    image_paths = self.retriever.collect_image_paths(
                        filtered_results,
                        question_offset=question_offset,
                        question_limit=question_limit
                    )
                    self.question_page_store[session_id] = {
                        "results": filtered_results,
                        "offset": min(question_limit, total_questions),
                        "limit": question_limit,
                    }
                    remaining_question_count = max(
                        total_questions - self.question_page_store[session_id]["offset"],
                        0
                    )

            yield {"type": "status", "message": "LLM 답변 생성 중..."}
            prompt_value = self.prompt.invoke({
                "history": recent_history,
                "context": context,
                "question": user_query,
            })
            response = self._extract_llm_content(self.llm.invoke(prompt_value))
            response = self._append_remaining_question_notice(
                response,
                remaining_question_count
            )

            history.add_user_message(user_query)
            history.add_ai_message(response)

            yield {"type": "answer", "content": response, "images": image_paths}
            
        except Exception as e:
            yield {"type": "answer", "content": f"❌ 오류 발생: {e}", "images": {}}
        
if __name__ == "__main__":
    chain = PassMasterChain()
    test_question = "블랙박스 테스트 설명"
    
    print("\n--- [Pass-Master LangChain RAG 실행] ---")
    answer = chain.run(test_question)
    print("\n[Pass-Master 답변]:")
    print(answer)

    # 두 번째 질문 (맥락 확인)
    print("\n--- [질문 2 (맥락 확인)] ---")
    print(chain.run("방금 말한 기출문제 정답만 다시 알려줘."))
