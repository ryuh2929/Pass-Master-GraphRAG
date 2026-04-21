import os
from dotenv import load_dotenv

# LangChain 관련 컴포넌트
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
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
            ("system", """너는 국가기술자격증 전문 AI 튜터 'Pass-Master'야. 모든 답변은 반드시 한국어(Korean)로 작성해.
제공된 [학습 지식]에 근거하여 답변하고, 모르는 내용은 지어내지 마."""),
            MessagesPlaceholder(variable_name="history"), 
            ("human", """[학습 지식]
{context}

[사용자 질문]
{question}

---
구조화된 한국어 답변:
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
    
    # def _get_session_history(self, session_id: str):
    #     """세션별 대화 기록 반환"""
    #     if session_id not in self.history_store:
    #         self.history_store[session_id] = InMemoryChatMessageHistory()
    #     return self.history_store[session_id]

    def _get_refined_context(self, x):
        """맥락이 포함된 재구성된 질문으로 검색 수행"""
        # 이 시점에서 history가 주입된 상태가 아니므로, 별도로 condense 과정이 필요할 수 있음
        # 간단한 구현을 위해 여기서는 현재 질문을 사용하되, 검색 퀄리티를 위해 로그 확인
        query = x["question"]
        query_vector = self.embedder.get_embedding(query)
        raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=3)
        return self.retriever.format_context_for_llm(raw_results)

    def _get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def run(self, user_query: str, session_id: str = "default_user"):
        print(f"🔎 [{session_id}] 질문 분석 중: {user_query}")
        try:
            # invoke 시 config에 session_id를 전달해야 함
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
    test_question = "프로토콜의 기본 요소 3가지는?"
    
    print("\n--- [Pass-Master LangChain RAG 실행] ---")
    answer = chain.run(test_question)
    print("\n[Pass-Master 답변]:")
    print(answer)

    # 두 번째 질문 (맥락 확인)
    print("\n--- [질문 2 (맥락 확인)] ---")
    print(chain.run("방금 말한 기출문제 정답만 다시 알려줘."))