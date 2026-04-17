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