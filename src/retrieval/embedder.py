import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

class TEIEmbedder:
    def __init__(self, endpoint: str = None):
        """
        TEI(Text Embeddings Inference) 서버와 통신하는 클래스입니다.
        """
        self.endpoint = endpoint or os.getenv("TEI_ENDPOINT", "http://localhost:8080/embed")

    def get_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트를 벡터로 변환합니다.
        """
        try:
            response = requests.post(
                self.endpoint,
                json={"inputs": text},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # TEI는 결과로 [ [vector] ] 형태를 반환합니다 (배치 처리 대응 때문)
            result = response.json()
            return result[0]
            
        except requests.exceptions.RequestException as e:
            print(f"❌ TEI 서버 통신 에러: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 한꺼번에 벡터로 변환합니다. (배치 처리)
        """
        try:
            response = requests.post(
                self.endpoint,
                json={"inputs": texts},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ TEI 서버 배치 통신 에러: {e}")
            raise

if __name__ == "__main__":
    # 간단한 동작 테스트
    embedder = TEIEmbedder()
    test_text = "회계 원칙 중 발생주의란 무엇인가?"
    
    print(f"--- [테스트] TEI 임베딩 시작 ---")
    try:
        vector = embedder.get_embedding(test_text)
        print(f"✅ 벡터 생성 성공! (차원 수: {len(vector)})")
        print(f"샘플 (앞 5자리): {vector[:5]}")
    except Exception as e:
        print(f"❌ 테스트 실패: TEI 서버가 {embedder.endpoint}에서 실행 중인지 확인하세요.")