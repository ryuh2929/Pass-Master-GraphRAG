import os
import requests
from dotenv import load_dotenv

# 내부 모듈 import 시 절대 경로 사용
from src.retrieval.embedder import TEIEmbedder
from src.retrieval.graph import GraphRetriever

load_dotenv()

class PassMasterChain:
    def __init__(self):
        # 1. 컴포넌트 초기화
        self.embedder = TEIEmbedder()
        
        # 2. GraphRetriever 초기화 (환경 변수 직접 참조)
        self.retriever = GraphRetriever(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        # 3. LLM 설정 (Ollama)
        self.llm_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        self.model_name = os.getenv("LLM_MODEL", "llama3:latest") # 사용하는 모델명 확인

    def run(self, user_query: str):
        print(f"🔎 질문 분석 중: {user_query}")
        
        try:
            # STEP 1: 질문 벡터 생성
            query_vector = self.embedder.get_embedding(user_query)
            
            # STEP 2: Neo4j 검색
            raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=3)
            context = self.retriever.format_context_for_llm(raw_results)
            
            # STEP 3: 프롬프트 구성
            prompt = self._build_prompt(user_query, context)
            
            # STEP 4: LLM 호출
            return self._call_llm(prompt)
        except Exception as e:
            return f"❌ 파이프라인 실행 중 오류 발생: {e}"

    def _build_prompt(self, query, context):
        return f"""
        ### [System Role]
        너는 국가기술자격증(정보처리기사, ADsP 등) 합격을 가이드하는 전문 AI 튜터 'Pass-Master'야. 
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
        - 문제에서 요구하는 정답을 명확히 명시해.

        3. **🚀 합격 가이드 (Tip)**:
        - 해당 개념이 시험에 어떻게 나오는지(다중도 표기법 등) 분석가적인 팁을 줘.

        ---
        ### [User Question]
        {query}

        ### [Final Answer]
        """

    def _call_llm(self, prompt):
        try:
            response = requests.post(
                self.llm_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False # 결과를 한 번에 받기 위해 False
                },
                timeout=30 # LLM 추론 시간을 고려한 타임아웃
            )
            response.raise_for_status()
            return response.json().get("response", "답변 생성 실패")
        except Exception as e:
            return f"❌ LLM 서버(Ollama) 통신 실패: {e}"

if __name__ == "__main__":
    # 테스트 실행
    chain = PassMasterChain()
    test_question = "프로토콜의 기본 요소 3가지는?"
    
    print("\n--- [Pass-Master RAG 실행] ---")
    answer = chain.run(test_question)
    print("\n[Pass-Master 답변]:")
    print(answer)