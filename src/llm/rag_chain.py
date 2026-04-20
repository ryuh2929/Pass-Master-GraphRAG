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
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # 3. LLM 설정 (Ollama)
        self.llm_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        self.model_name = os.getenv("LLM_MODEL", "llama3") # 사용하는 모델명 확인

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
        return f"""너는 데이터 분석 자격증 전문가 'Pass-Master'야. 
아래의 [학습 지식]을 참고해서 질문에 답해줘.

[학습 지식]
{context}

[사용자 질문]
{query}

답변 (전문가답게 구조적으로 작성해줘):"""

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
    test_question = "발생주의와 관련된 기출문제가 있어?"
    
    print("\n--- [Pass-Master RAG 실행] ---")
    answer = chain.run(test_question)
    print("\n[Pass-Master 답변]:")
    print(answer)