import os
import json
import glob
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

class GraphDataManager:
    def __init__(self):
        try:
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USER"),
                password=os.getenv("NEO4J_PASSWORD")
            )
            print("✅ Neo4j 연결 성공")
        except Exception as e:
            print(f"❌ Neo4j 연결 실패: {e}")
            raise e

    def load_exam_data(self, json_path: str):
        """기출 JSON 구조에 맞춘 적재 로직"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        query = """
        MERGE (e:Exam {year: $year, round: $round})
        SET e.practical_dates = $practical_dates, e.url = $url
        WITH e
        UNWIND $problems AS prob
        MERGE (q:Question {id: $year + "_" + $round + "_" + prob.no})
        SET q.no = prob.no, 
            q.question = prob.question, 
            q.answer = prob.answer
        MERGE (e)-[:HAS_QUESTION]->(q)
        """
        self.graph.query(query, {
            "year": data['year'],
            "round": data['round'],
            "practical_dates": data.get('practical_dates'),
            "url": data.get('url'),
            "problems": data['problems']
        })

    def load_summary_chunks(self, json_path: str):
        """요약본(processed_chunks.json) 구조에 맞춘 적재 로직"""
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        query = """
        UNWIND $chunks AS chunk
        MERGE (c:Concept {id: chunk.metadata.id})
        SET c.title = chunk.metadata.title,
            c.document = chunk.document,
            c.chapter = chunk.metadata.chapter,
            c.importance = chunk.metadata.importance,
            c.exam_dates = chunk.metadata.exam_dates
        
        // 해당 개념이 출제된 시험(Exam)과 자동으로 관계 맺기
        WITH c, chunk
        UNWIND chunk.metadata.exam_dates AS date
        MATCH (e:Exam {practical_dates: date})
        MERGE (c)-[:APPEARED_IN]->(e)
        """
        try:
            self.graph.query(query, {"chunks": chunks})
            print(f"✅ 요약본 데이터({len(chunks)}개 청크) 적재 및 관계 형성 완료")
        except Exception as e:
            print(f"❌ 요약본 적재 실패: {e}")

    def load_all_exams(self, target_dir: str):
        json_files = glob.glob(os.path.join(target_dir, "exam_*.json"))
        for file_path in sorted(json_files):
            self.load_exam_data(file_path)
            print(f"✅ 기출 적재 완료: {os.path.basename(file_path)}")

if __name__ == "__main__":
    manager = GraphDataManager()
    
    # 1. 기출문제 먼저 다 넣기 (Exam 노드가 있어야 요약본과 연결됨)
    manager.load_all_exams("data/processed")
    
    # 2. 요약본 넣기
    manager.load_summary_chunks("data/processed/processed_chunks.json")