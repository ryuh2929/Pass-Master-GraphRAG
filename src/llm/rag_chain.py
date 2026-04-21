import os
from dotenv import load_dotenv

# LangChain 관련 컴포넌트
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 기존 분석가님이 작성하신 로직
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

        # 3. 프롬프트 템플릿 구성
        self.prompt = ChatPromptTemplate.from_template("""
        ### [System Role]
        너는 국가기술자격증 합격을 가이드하는 전문 AI 튜터 'Pass-Master'야. 
        분석가적인 관점에서 수험생에게 정확한 개념과 실전 문제 풀이 전략을 한국어로 제공해야 해.

        ### [Provided Context]
        아래는 검색 엔진을 통해 추출된 데이터베이스 내 지식이야. 이 내용에만 기반해서 답변해.
        {context}

        ### [Output Instructions]
        반드시 다음 구조를 지켜서 한국어로 답변할 것:

        1. **💡 핵심 개념 요약**: 
        - 제공된 'document' 내용을 바탕으로 정의와 특징을 일목요연하게 정리해줘.
        - 중요도(importance)나 기출 날짜가 있다면 언급하며 강조해줘.

        2. **📝 관련 실전 문제**:
        - 제공된 문제 데이터(no, question, answer)가 있다면 해당 문제를 소개하고 해설해줘.
        - **정답과 해설 부분은 반드시 아래와 같이 HTML 태그로 감싸서 작성해줘:**
            <span style="color: black; background-color: black;">
            정답: [내용] <br>
            해설: [내용]</span>
        - 드래그했을 때만 보이게 <br> 태그로 줄바꿈을 명확히 해줘.

        3. **🚀 합격 가이드 (Tip)**:
        - 해당 개념이 시험에 어떻게 나오는지 분석가적인 팁을 줘.

        ---
        ### [User Question]
        {question}

        ### [Final Answer]
        """)

        # 4. 체인 조립 (LCEL 방식)
        self.chain = (
            {
                "context": RunnableLambda(self._get_graph_context),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_graph_context(self, user_query: str) -> str:
        """기존 분석가님의 검색 로직을 LangChain 파이프라인에 이식"""
        query_vector = self.embedder.get_embedding(user_query)
        raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=3)
        return self.retriever.format_context_for_llm(raw_results)

    def run(self, user_query: str):
        print(f"🔎 질문 분석 중 (LangChain): {user_query}")
        try:
            return self.chain.invoke(user_query)
        except Exception as e:
            return f"❌ 파이프라인 실행 중 오류 발생: {e}"

if __name__ == "__main__":
    chain = PassMasterChain()
    test_question = "프로토콜의 기본 요소 3가지는?"
    
    print("\n--- [Pass-Master LangChain RAG 실행] ---")
    answer = chain.run(test_question)
    print("\n[Pass-Master 답변]:")
    print(answer)