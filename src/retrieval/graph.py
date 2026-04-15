import os
import json
import glob
import requests
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
            self.tei_url = "http://localhost:8080/embed"
            print("✅ Neo4j & TEI 연동 준비 완료")
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            raise e
        
    def load_all_exams(self, target_dir: str):
        json_files = glob.glob(os.path.join(target_dir, "exam_*.json"))
        for file_path in sorted(json_files):
            self.load_exam_data(file_path)
            print(f"✅ 기출 적재 완료: {os.path.basename(file_path)}")
        
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
        // 1. Chapter 노드 생성 (이미 있으면 가져옴)
        MERGE (ch:Chapter {name: chunk.metadata.chapter})
        
        // 2. Concept 노드 생성
        MERGE (c:Concept {id: chunk.metadata.id})
        SET c.title = chunk.metadata.title,
            c.document = chunk.document,
            c.importance = chunk.metadata.importance,
            c.exam_dates = chunk.metadata.exam_dates,
            c.practical_dates = chunk.metadata.practical_dates
        
        // 3. Chapter와 Concept 연결 (소속 관계)
        MERGE (c)-[:BELONGS_TO]->(ch)
        
        // 4. 기존 Exam과의 관계 형성 (실기 날짜 기반)
        WITH c, chunk
        UNWIND chunk.metadata.practical_dates AS p_date  // 필기가 아닌 '실기' 날짜 리스트를 풉니다.
        MATCH (e:Exam) WHERE trim(e.practical_dates) = trim(p_date)        // Exam 노드의 날짜와 정확히 매칭
        MERGE (c)-[:APPEARED_IN_PRACTICAL]->(e)         // 관계명도 구체화하면 분석이 더 쉬워집니다.
        """
        try:
            self.graph.query(query, {"chunks": chunks})
            print(f"✅ 요약본 데이터({len(chunks)}개 청크) 적재 및 관계 형성 완료")
        except Exception as e:
            print(f"❌ 요약본 적재 실패: {e}")

    def get_embedding(self, text: str):
        """TEI 컨테이너를 호출하여 bge-m3 임베딩을 가져옵니다."""
        try:
            response = requests.post(self.tei_url, json={"inputs": text}, timeout=10)
            return response.json()[0]
        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

    def embed_nodes(self):
        """Question과 Concept 노드에 임베딩을 추가합니다."""
        # Concept 노드 임베딩
        concepts = self.graph.query("MATCH (c:Concept) WHERE c.embedding IS NULL RETURN c.id as id, c.document as text")
        for rec in concepts:
            vector = self.get_embedding(rec['text'])
            if vector:
                self.graph.query("MATCH (c:Concept {id: $id}) CALL db.create.setNodeVectorProperty(c, 'embedding', $vector)", 
                                 {"id": rec['id'], "vector": vector})
        
        # Question 노드: 본문과 정답을 결합하여 정보량 증대
        questions = self.graph.query("""
            MATCH (q:Question) 
            WHERE q.embedding IS NULL 
            RETURN q.id as id, q.question as question, q.answer as answer
        """)
        
        for rec in questions:
            # 문제와 정답을 합쳐서 문맥을 풍부하게 만듦
            combined_text = f"문제: {rec['question']} / 정답: {rec['answer']}"
            vector = self.get_embedding(combined_text)
            
            if vector:
                self.graph.query("""
                    MATCH (q:Question {id: $id}) 
                    CALL db.create.setNodeVectorProperty(q, 'embedding', $vector)
                """, {"id": rec['id'], "vector": vector})
        print("✅ 모든 노드 벡터화 완료")

    def create_vector_index(self):
        """벡터 조회를 위한 인덱스 생성 (bge-m3: 1024차원)"""
        self.graph.query("""
        CREATE VECTOR INDEX concept_index IF NOT EXISTS
        FOR (c:Concept) ON (c.embedding)
        OPTIONS {indexConfig: { `vector.dimensions`: 1024, `vector.similarity_function`: 'cosine' }}
        """)

    def link_with_semantic_verification(self, threshold=0.8):
        """기존 검증 관계를 초기화하고 엄격한 로직으로 재연결"""
        # 기존에 생성된 모든 VERIFIED_MENTIONS 관계 삭제 (노드는 유지, 선만 삭제)
        print("🧹 기존 검증 관계 초기화 중...")
        self.graph.query("MATCH (:Question)-[r:VERIFIED_MENTIONS]->(:Concept) DELETE r")

        """의미 유사도 확인 후 날짜 데이터로 검증하여 연결"""
        # 1. 벡터 유사도로 후보 탐색
        # 2. 문제(Question)가 속한 시험(Exam)의 날짜가 개념(Concept)의 출제 날짜에 있는지 검증
        query = """
        MATCH (q:Question)-[:HAS_QUESTION]-(e:Exam)
        CALL db.index.vector.queryNodes('concept_index', 20, q.embedding) 
        YIELD node AS c, score
        WHERE score >= $threshold
          AND any(d IN c.practical_dates WHERE trim(d) = trim(e.practical_dates))  // 날짜 검증 로직
        
        // 문제(q)별로 가장 점수가 높은 개념(c) 하나만 남기기
        WITH q, c, score
        ORDER BY q.id, score DESC
        WITH q, collect({concept: c, score: score})[0] AS best_match

        // 최종적으로 검증된 연결만 생성
        MATCH (target:Concept {id: best_match.concept.id})
        MERGE (q)-[r:VERIFIED_MENTIONS]->(target)
        SET r.similarity_score = best_match.score
        RETURN count(r) as link_count
        """
        result = self.graph.query(query, {"threshold": threshold})
        print(f"✅ 검증된 의미적 연결 {result[0]['link_count']}개 생성 완료")

if __name__ == "__main__":
    manager = GraphDataManager()
    
    # 1. 기출문제 먼저 다 넣기 (Exam 노드가 있어야 요약본과 연결됨)
    manager.load_all_exams("data/processed")
    
    # 2. 요약본 넣기
    manager.load_summary_chunks("data/processed/processed_chunks.json")

    # 3. 임베딩 및 인덱스 생성
    manager.embed_nodes()
    manager.create_vector_index()
    
    # 4. 딥러닝(임베딩) + 날짜 검증 기반 연결
    manager.link_with_semantic_verification(threshold=0.75)