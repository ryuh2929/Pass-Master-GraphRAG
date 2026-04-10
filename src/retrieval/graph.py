import os
import json
from dotenv import load_dotenv
# 변경된 임포트 경로
from langchain_neo4j import Neo4jGraph

load_dotenv()

class GraphDataManager:
    def __init__(self):
        # 최신 패키지인 langchain-neo4j의 Neo4jGraph를 사용합니다.
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD")
        )

    def load_exam_data(self, json_path: str):
        if not os.path.exists(json_path):
            print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        query = """
        MERGE (e:Exam {year: $year, round: $round})
        WITH e
        UNWIND $problems AS prob
        MERGE (q:Question {id: $year + "_" + $round + "_" + prob.no})
        SET q.no = prob.no, 
            q.question = prob.question, 
            q.answer = prob.answer
        MERGE (e)-[:HAS_QUESTION]->(q)
        """
        
        try:
            self.graph.query(query, {
                "year": data['year'],
                "round": data['round'],
                "problems": data['problems']
            })
            print(f"✅ {data['year']}년 {data['round']}회 기출 적재 완료")
        except Exception as err:
            print(f"❌ Neo4j 적재 중 오류 발생: {err}")

    def check_stats(self):
        """저장된 데이터의 통계를 출력합니다."""
        query = """
        MATCH (e:Exam)-[:HAS_QUESTION]->(q:Question)
        RETURN e.year AS year, e.round AS round, count(q) AS count
        """
        results = self.graph.query(query)
        print("\n--- DB 적재 통계 ---")
        for res in results:
            print(f"📅 {res['year']}년 {res['round']}회: {res['count']}문제 저장됨")

if __name__ == "__main__":
    manager = GraphDataManager()
    # 실제 데이터 경로로 테스트 (예: 2024년 1회)
    manager.load_exam_data("data/processed/exam_2024_1.json")
    manager.check_stats()