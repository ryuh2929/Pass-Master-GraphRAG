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
        
        # 2. LangChain용 LLM 설정 (ChatOllama 사용)
        self.llm = ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3:latest"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=0.1 # 분석가용 사실 중심 답변 설정
        )

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
        self.condense_chain = self.condense_question_prompt | self.llm | StrOutputParser()

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

    def _get_refined_context(self, x):
        """맥락이 포함된 재구성된 질문으로 검색 수행"""
        # 이 시점에서 history가 주입된 상태가 아니므로, 별도로 condense 과정이 필요할 수 있음
        # 간단한 구현을 위해 여기서는 현재 질문을 사용하되, 검색 퀄리티를 위해 로그 확인
        # 1. 여기서 history를 직접 제어합니다. (최근 2개 메시지만 참조)
        # x['history']는 RunnableWithMessageHistory가 주입해줍니다.
        user_query = x["question"]
        recent_history = x.get("history", [])[-2:] # 직전 문답만 참고

        # 이전 대화 기록이 없으면 바로 검색
        if not recent_history:
            return self._execute_vector_search(user_query)

        # 2. LLM에게 검색 필요성 판단
        decision_prompt = f"""
        당신은 데이터 검색 전문가입니다.
        
        [최근 대화 맥락]
        {recent_history}
        
        [새로운 질문]
        {user_query}
        
        [판단 기준]
        1. 새로운 질문이 이전 대화에서 설명된 '구체적인 개념'의 정답이나 요약만 요구합니까? -> 'USE_HISTORY'
        2. 새로운 질문에 이전 대화에 없던 '새로운 용어'나 '개념'이 등장합니까? -> 'SEARCH'
        3. 이전 대화가 불충분하여 더 자세한 '기출 데이터'가 필요합니까? -> 'SEARCH'
        
        답변은 반드시 'USE_HISTORY' 또는 'SEARCH' 둘 중 하나만 출력하십시오.
        """
        decision = self.llm.invoke(decision_prompt).content.strip()
        
        # 3. 결정에 따른 분기 처리 (데이터 노이즈 차단)
        if "USE_HISTORY" in decision:
            print("💡 [Decision] 기존 맥락 활용 (History Reuse)")
            # 이전 대화에서 사용되었던 핵심 지식은 이미 LLM의 History에 있으므로 빈 컨텍스트 전달
            return "이전 대화의 지식 베이스를 바탕으로 답변하세요."
        
        # 4. 검색이 필요한 경우에만 DB 쿼리 실행
        print(f"🔎 [Decision] 지식 베이스 신규 검색 실행")
        return self._execute_vector_search(user_query, recent_history)
    
    def _execute_vector_search(self, query, history=None):
        """실제 DB 검색을 수행하는 유틸리티 함수"""
        refined_query = query
        if history:
            # 맥락을 섞어 검색어 정제
            condense_prompt = f"대화: {history}\n질문: {query}\n위 맥락을 포함한 한국어 검색 키워드 한 문장:"
            refined_query = self.llm.invoke(condense_prompt).content

        query_vector = self.embedder.get_embedding(refined_query)
        raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=2)
        return self.retriever.format_context_for_llm(raw_results)

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