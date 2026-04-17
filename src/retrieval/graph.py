from langchain_neo4j import Neo4jGraph

class GraphRetriever:
    def __init__(self, url, username, password):
        """
        데이터 조회 전용 클래스. 
        ingestion/loader.py에서 생성한 인덱스와 관계를 활용합니다.
        """
        self.graph = Neo4jGraph(url=url, username=username, password=password)

    def search_concepts_with_questions(self, query_vector: list, top_k: int = 3):
        """
        사용자 질문 벡터를 기반으로 관련 개념(Concept)과 
        그에 연결된 기출문제(Question)를 한꺼번에 추출합니다.
        """
        
        # Cypher: Concept을 먼저 찾고, 연결된 Question들을 1:N으로 묶어서 가져옴
        query = """
        CALL db.index.vector.queryNodes('concept_index', $top_k, $vector) 
        YIELD node AS c, score
        
        // 해당 개념과 VERIFIED_MENTIONS로 연결된 기출문제들을 수집
        OPTIONAL MATCH (q:Question)-[:VERIFIED_MENTIONS]->(c)
        WITH c, score, q
        ORDER BY q.id  // 문제 번호순 정렬(선택 사항)
        
        WITH c, score, collect({
            id: q.id, 
            question: q.question, 
            answer: q.answer
        })[..3] AS related_questions
        
        RETURN 
            c.title AS title, 
            c.document AS content, 
            score, 
            related_questions
        """
        
        params = {
            "vector": query_vector,
            "top_k": top_k
        }
        
        return self.graph.query(query, params)

    def format_context_for_llm(self, search_results: list) -> str:
        """
        추출된 리스트 데이터를 LLM 프롬프트에 주입할 텍스트 형식으로 변환합니다.
        """
        if not search_results:
            return "관련된 지식 베이스 내용을 찾을 수 없습니다."

        context_parts = []
        for i, res in enumerate(search_results):
            # 유사도 점수를 함께 표기하여 분석 시 참고 가능하게 구성
            part = f"### 관련 지식 {i+1} (유사도: {res['score']:.4f})\n"
            part += f"- 주요 개념: {res['title']}\n"
            part += f"- 상세 설명: {res['content']}\n"
            
            if res['related_questions'] and res['related_questions'][0]['id'] is not None:
                part += "- 실제 기출 사례:\n"
                for q in res['related_questions']:
                    part += f"  * [문제 {q['id']}] {q['question']} (정답: {q['answer']})\n"
            
            context_parts.append(part)
            
        return "\n\n".join(context_parts)
    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # 1. 환경 변수 로드 (DB 접속 정보)
    load_dotenv()
    
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    print(f"✅ 환경 변수 로드 완료: {NEO4J_URI}, {NEO4J_USER}, {'*'*len(NEO4J_PASSWORD)}")

    # 2. 리트리버 초기화
    retriever = GraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# 3. 테스트용 유효 벡터 생성 
    # (실제 환경에서는 TEI API를 통해 질문을 벡터화한 값을 넣어야 합니다)
    # 여기서는 차원수만 맞춘 랜덤 벡터 혹은 기존에 존재하는 더미 데이터를 가정합니다.
    # 예: [0.12, -0.05, 0.34, ...] (bge-m3의 경우 1024차원)
    # 모든 요소가 0이면 L2-norm이 0이 되어 에러가 발생합니다.
    # 최소한 하나 이상의 요소에 유효한 값을 넣어줍니다.
    import random
    dummy_vector = [random.uniform(-0.1, 0.1) for _ in range(1024)]

    print("--- [테스트] 그래프 검색 시작 ---")
    try:
        # 개념 및 관련 기출문제 조회
        raw_results = retriever.search_concepts_with_questions(dummy_vector, top_k=2)
        
        # LLM 주입용 텍스트 변환
        formatted_context = retriever.format_context_for_llm(raw_results)
        
        print("\n[검색 결과 요약]")
        print(formatted_context)
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
    finally:
        # langchain_neo4j의 Neo4jGraph는 내부적으로 드라이버를 관리하므로 
        # 명시적인 close가 없어도 세션 종료 시 정리됩니다.
        print("\n--- 테스트 종료 ---")