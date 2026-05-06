import os
from dotenv import load_dotenv

# LangChain 관련 컴포넌트
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.retrieval.embedder import TEIEmbedder
from src.retrieval.graph import GraphRetriever

from src.llm.llm_switch import get_llm

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

        # 4. 프롬프트 템플릿 구성
        self.condense_question_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
            ("system", "위 대화를 바탕으로, 지식 베이스에서 검색하기 위한 독립적인 한 문장의 한국어 질문으로 다시 작성해줘. 핵심 키워드를 포함해야 해.")
        ])

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 한국어만 사용하는 국가기술자격증 전문 튜터 'Pass-Master'입니다.
            
            [반드시 지켜야 할 규칙]
            1. 모든 답변은 반드시 한국어로 작성하십시오. 영어 사용을 금지합니다.
            2. 정답과 해설은 반드시 다음 HTML 형식을 사용하십시오: 
            <span style="color: black; background-color: black;">정답: [내용]</span>
            3. [학습 지식]에 없는 내용은 절대 지어내지 마십시오. (특히 프로토콜 3요소: 구문, 의미, 타이밍 확인)
            4. 사용자가 '정답만' 요구하면 다른 설명 없이 문제와 마스킹된 정답만 출력하십시오."""),
                        MessagesPlaceholder(variable_name="history"), 
                        ("human", """[학습 지식]
            {context}

            [사용자 질문]
            {question}

            ---
            구조화된 한국어 답변 (질문에 해당하는 내용만):
            1. 💡 핵심 개념 요약
            2. 📝 관련 실전 문제 (정답/해설 마스킹 필수)
            3. 🚀 합격 가이드 (Tip - 반드시 한국어로 작성할 것)""")
        ])

        # 검색용 질문을 재구성하는 체인
        # self.condense_chain = self.condense_question_prompt | self.llm | StrOutputParser()

        # self.chain = self.prompt | self.llm | self.output_parser

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
        
        # self.prompt = ChatPromptTemplate.from_messages([
        #     ("system", """너는 국가기술자격증 합격을 가이드하는 전문 AI 튜터 'Pass-Master'야.
        #     제공된 [학습 지식]과 이전 대화 맥락을 바탕으로 한국어로 답변해줘."""),
        # MessagesPlaceholder(variable_name="history"), # 대화 기록이 들어갈 자리
        #     ("human", """[학습 지식]
        #     {context}
             
        #      [사용자 질문]
        #     {question}

        #     ---
        #     반드시 아래 구조로 답변할 것:
        #     1. 💡 핵심 개념 요약 (중요도/기출날짜 포함)
        #     2. 📝 관련 실전 문제 (정답/해설은 반드시 검은색 마스킹 HTML 사용)
        #     3. 🚀 합격 가이드 (Tip)""")
        # ])

        # # 5. 체인 조립 (LCEL 방식)
        # base_chain = (
        #     RunnablePassthrough.assign(
        #         context=RunnableLambda(lambda x: self._get_graph_context(x["question"]))
        #     )
        #     | self.prompt
        #     | self.llm
        #     | StrOutputParser()
        # )

        # # 6. 메모리가 통합된 최종 체인
        # self.chain_with_history = RunnableWithMessageHistory(
        #     base_chain,
        #     get_session_history=self._get_session_history,
        #     input_messages_key="question",
        #     history_messages_key="history"
        # )

    # def _get_graph_context(self, user_query: str) -> str:
    #     """기존 검색 로직을 LangChain 파이프라인에 이식"""
    #     """질문을 벡터화하여 Neo4j에서 관련 지식 추출"""
    #     query_vector = self.embedder.get_embedding(user_query)
    #     raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=3)
    #     return self.retriever.format_context_for_llm(raw_results)

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
        clean_query_prompt = f"""
        사용자 질문: "{query}"
        위 질문에서 '알려줘', '설명해줘', '알려주세요' 같은 조기 서술어를 제외하고,
        지식 베이스 검색에 필요한 핵심 전문 용어(명사)만 한 단어로 추출해줘.
        추출된 단어: """

        refined_query = self.llm.invoke(clean_query_prompt).content.strip()
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
        decision_prompt = f"""
        [이전 대화] {recent_history}
        [현재 질문] {user_query}
        
        질문이 이전 대화에 나온 내용의 '정답'이나 '부연 설명'만을 요구합니까?
        그렇다면 'YES', 새로운 정보 검색이 필요해 보인다면 'NO'라고만 답하세요.
        답변: """
        
        decision = self.llm.invoke(decision_prompt).content.strip().upper()

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
            # 1. 초기 로딩 (VRAM 로딩 느낌)
            yield "⚡ GPU VRAM 캐시 최적화 및 엔진 로딩 중..."
            
            # 2. 질문 분석 (기존 _get_refined_context 로직 내부에서 yield를 쓰기 위해 분리 가능)
            yield "🧼 검색 노이즈 제거 및 핵심 키워드 추출 중..."
            # (실제 로직 수행 - 예시)
            # refined_query = self.llm.invoke(...) 

            yield "🔍 지식 그래프(Neo4j) 탐색 및 관련 지식 추출 중..."
            # context = self._execute_vector_search(...)

            yield "✍️ Gemma 4가 정밀 해설을 작성하고 있습니다..."
            # 최종 결과 생성
            response = self.chain_with_history.invoke(
                {"question": user_query},
                config={"configurable": {"session_id": session_id}}
            )
            
            # 마지막에 최종 답변 객체를 yield
            yield response
            
        except Exception as e:
            yield f"❌ 오류 발생: {e}"
        
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