import os
import json
import glob
import requests
import warnings
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from langchain_neo4j import Neo4jGraph
from src.ingestion.hybrid_linker import (
    BM25ConceptRanker,
    RerankerClient,
    build_hybrid_candidates,
    rerank_candidates,
)

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
        MERGE (q:Question {problem_id: $year + "_" + $round + "_" + prob.no})
        SET q.no = prob.no, 
            q.question = prob.question, 
            q.answer = prob.answer,
            q.images = coalesce(prob.images, [])
        MERGE (e)-[:HAS_QUESTION]->(q)
        """
        self.graph.query(query, {
            "year": data['year'],
            "round": data['round'],
            "practical_dates": data.get('practical_dates'),
            "url": data.get('url'),
            "problems": data['problems']
        })

    def update_question_images(self, target_dir: str):
        """기존 Question 노드에 exam_*.json의 images 속성만 갱신합니다."""
        json_files = glob.glob(os.path.join(target_dir, "exam_*.json"))

        query = """
        UNWIND $problems AS prob
        MATCH (q:Question {problem_id: $year + "_" + $round + "_" + prob.no})
        SET q.images = coalesce(prob.images, [])
        RETURN count(q) AS updated_count
        """

        total_updated = 0
        for file_path in sorted(json_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result = self.graph.query(query, {
                "year": data['year'],
                "round": data['round'],
                "problems": data['problems']
            })
            total_updated += result[0]["updated_count"] if result else 0

        print(f"✅ Question 이미지 경로 {total_updated}개 노드 갱신 완료")

    def load_summary_chunks(self, json_path: str):
        """요약본(processed_chunks.json) 구조에 맞춘 적재 로직"""
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        query = """
        UNWIND $chunks AS chunk
        // 1. Chapter 노드 생성 (이미 있으면 가져옴)
        MERGE (ch:Chapter {name: chunk.metadata.chapter})
        
        // 2. Concept 노드 생성
        MERGE (c:Concept {section_id: chunk.metadata.id})
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
        concepts = self.graph.query("MATCH (c:Concept) WHERE c.embedding IS NULL RETURN c.section_id as id, c.document as text")
        for rec in concepts:
            vector = self.get_embedding(rec['text'])
            if vector:
                self.graph.query("MATCH (c:Concept {section_id: $id}) CALL db.create.setNodeVectorProperty(c, 'embedding', $vector)", 
                                 {"id": rec['id'], "vector": vector})
        
        # Question 노드: 본문과 정답을 결합하여 정보량 증대
        questions = self.graph.query("""
            MATCH (q:Question) 
            WHERE q.embedding IS NULL 
            RETURN q.problem_id as id, q.question as question, q.answer as answer
        """)
        
        for rec in questions:
            # 문제와 정답을 합쳐서 문맥을 풍부하게 만듦
            combined_text = f"문제: {rec['question']} / 정답: {rec['answer']}"
            vector = self.get_embedding(combined_text)
            
            if vector:
                self.graph.query("""
                    MATCH (q:Question {problem_id: $id}) 
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
        """Link each question to the best date-valid Concept using vector and BM25 scores."""
        concepts = self._fetch_concepts_for_bm25()
        ranker = BM25ConceptRanker(concepts)
        reranker = self._create_reranker()
        questions = self._fetch_questions_for_linking()
        reranker_top_k = int(os.getenv("RERANKER_TOP_K", "3"))
        vector_top_k = int(os.getenv("VECTOR_TOP_K", "500"))
        links = []

        for index, question in enumerate(questions, start=1):
            vector_scores = self._fetch_vector_scores(question["embedding"], top_k=vector_top_k)
            bm25_text = f"{question.get('question') or ''} {question.get('answer') or ''}"
            bm25_scores = ranker.get_scores(bm25_text)
            candidates = build_hybrid_candidates(
                question_text=question.get("question") or "",
                practical_date=question.get("practical_date") or "",
                vector_scores=vector_scores,
                bm25_scores=bm25_scores,
                ranker=ranker,
            )
            candidates = rerank_candidates(
                question_text=question.get("question") or "",
                candidates=candidates,
                reranker=reranker,
                top_k=reranker_top_k,
            )

            if not candidates:
                continue

            best = candidates[0]
            links.append({
                "problem_id": question["problem_id"],
                "section_id": best.concept.section_id,
                "final_score": best.final_score,
                "vector_score": best.vector_score,
                "bm25_score": best.bm25_score,
                "reranker_score": best.reranker_score,
            })

            if index % 50 == 0:
                print(f"Prepared hybrid links: {index}/{len(questions)}")

        # Keep the old relationship layer until all new links are ready.
        print("Resetting VERIFIED_MENTIONS relationships...")
        self.graph.query("MATCH (:Question)-[r:VERIFIED_MENTIONS]->(:Concept) DELETE r")

        result = self.graph.query("""
        UNWIND $links AS link
        MATCH (q:Question {problem_id: link.problem_id})
        MATCH (c:Concept {section_id: link.section_id})
        MERGE (q)-[r:VERIFIED_MENTIONS]->(c)
        SET r.similarity_score = link.final_score,
            r.vector_score = link.vector_score,
            r.bm25_score = link.bm25_score,
            r.reranker_score = link.reranker_score
        RETURN count(r) AS link_count
        """, {"links": links})

        link_count = result[0]["link_count"] if result else 0
        print(f"Hybrid verified links created: {link_count}")

    def _create_reranker(self):
        if os.getenv("USE_RERANKER", "false").lower() not in {"1", "true", "yes"}:
            print("Reranker disabled. Set USE_RERANKER=true to enable.")
            return None

        endpoint = os.getenv("RERANKER_ENDPOINT", "http://localhost:8081/rerank")
        reranker = RerankerClient(endpoint=endpoint)
        if reranker.available:
            print(f"Reranker enabled: {endpoint}")
            return reranker

        print(f"Reranker unavailable, falling back to hybrid scores: {endpoint}")
        return None

    def _fetch_concepts_for_bm25(self):
        return self.graph.query("""
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(ch:Chapter)
        RETURN
            c.section_id AS section_id,
            c.title AS title,
            c.document AS document,
            c.practical_dates AS practical_dates,
            ch.name AS chapter
        """)

    def _fetch_questions_for_linking(self):
        return self.graph.query("""
        MATCH (q:Question)-[:HAS_QUESTION]-(e:Exam)
        WHERE q.embedding IS NOT NULL
        RETURN
            q.problem_id AS problem_id,
            q.question AS question,
            q.answer AS answer,
            q.embedding AS embedding,
            e.practical_dates AS practical_date
        ORDER BY q.problem_id
        """)

    def _fetch_vector_scores(self, embedding, top_k=30):
        rows = self.graph.query("""
        CALL db.index.vector.queryNodes('concept_index', $top_k, $vector)
        YIELD node AS c, score
        RETURN c.section_id AS section_id, score
        """, {"top_k": top_k, "vector": embedding})

        return {
            str(row["section_id"]): float(row["score"])
            for row in rows
            if row["score"] >= 0
        }

if __name__ == "__main__":
    import sys

    manager = GraphDataManager()

    if len(sys.argv) > 1 and sys.argv[1] == "update-images":
        manager.update_question_images("data/processed")
        raise SystemExit(0)
    
    # 1. 기출문제 먼저 다 넣기 (Exam 노드가 있어야 요약본과 연결됨)
    manager.load_all_exams("data/processed")
    
    # 2. 요약본 넣기
    manager.load_summary_chunks("data/processed/processed_chunks.json")

    # 3. 임베딩 및 인덱스 생성
    manager.embed_nodes()
    manager.create_vector_index()
    
    # 4. 딥러닝(임베딩) + 날짜 검증 기반 연결
    manager.link_with_semantic_verification(threshold=0.75)
