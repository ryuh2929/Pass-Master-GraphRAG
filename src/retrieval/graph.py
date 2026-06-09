from langchain_neo4j import Neo4jGraph

from src.ingestion.hybrid_linker import BM25ConceptRanker

class GraphRetriever:
    def __init__(self, url, username, password):
        """
        데이터 조회 전용 클래스. 
        ingestion/loader.py에서 생성한 인덱스와 관계를 활용합니다.
        """
        self.graph = Neo4jGraph(url=url, username=username, password=password)
        self._bm25_ranker = None

    def search_concepts_with_questions(
        self,
        query_vector: list,
        query_text: str | None = None,
        top_k: int = 1,
        candidate_top_k: int = 100,
        vector_weight: float = 0.8,
        bm25_weight: float = 0.2,
    ):
        """
        사용자 질문으로 관련 개념(Concept)과 연결 기출문제(Question)를 조회합니다.

        query_text가 있으면 vector 후보를 넓게 가져온 뒤 BM25 점수를 20% 섞어 재정렬합니다.
        query_text가 없으면 기존처럼 vector-only 검색으로 동작합니다.
        """
        if query_text:
            return self.search_concepts_with_questions_hybrid(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                candidate_top_k=candidate_top_k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )

        return self._fetch_concepts_with_questions_by_vector(
            query_vector=query_vector,
            top_k=top_k,
        )

    def search_concepts_with_questions_hybrid(
        self,
        query_vector: list,
        query_text: str,
        top_k: int = 1,
        candidate_top_k: int = 100,
        vector_weight: float = 0.8,
        bm25_weight: float = 0.2,
    ):
        """
        runtime RAG 검색용 hybrid retriever입니다.

        평가 결과가 가장 좋았던 80:20 비율을 기본값으로 사용합니다.
        실제 LLM 주입은 top1 중심으로 하되, 내부 후보는 넓게 가져와 BM25가 재정렬할 여지를 둡니다.
        """
        vector_scores = self._fetch_vector_scores(query_vector, top_k=candidate_top_k)
        bm25_scores = self._get_bm25_ranker().get_scores(query_text)
        candidate_ids = set(vector_scores) | set(bm25_scores)

        # 각 점수군은 자기 점수 분포 안에서 먼저 정규화합니다.
        # 후보 union에 없는 점수를 0으로 채워 넣고 정규화하면 평가 스크립트와 순위가 달라질 수 있습니다.
        normalized_vector = self._normalize_scores(vector_scores)
        normalized_bm25 = self._normalize_scores(bm25_scores)

        ranked_candidates = []
        for section_id in candidate_ids:
            final_score = (
                vector_weight * normalized_vector.get(section_id, 0.0)
                + bm25_weight * normalized_bm25.get(section_id, 0.0)
            )
            ranked_candidates.append({
                "section_id": section_id,
                "score": final_score,
                "vector_score": vector_scores.get(section_id, 0.0),
                "bm25_score": bm25_scores.get(section_id, 0.0),
            })

        ranked_candidates.sort(key=lambda item: item["score"], reverse=True)
        return self._fetch_concepts_with_questions_by_ids(ranked_candidates[:top_k])

    def _fetch_concepts_with_questions_by_vector(self, query_vector: list, top_k: int):
        # Cypher: Concept을 먼저 찾고, 연결된 Question들을 1:N으로 묶어서 가져옴
        query = """
        CALL db.index.vector.queryNodes('concept_index', $top_k, $vector) 
        YIELD node AS c, score
        
        // 해당 개념과 VERIFIED_MENTIONS로 연결된 기출문제들을 수집
        OPTIONAL MATCH (q:Question)-[:VERIFIED_MENTIONS]->(c)
        WITH c, score, q
        ORDER BY q.problem_id  // 문제 번호순 정렬(선택 사항)
        
        WITH c, score, collect({
            id: q.problem_id, 
            question: q.question, 
            answer: q.answer,
            images: coalesce(q.images, [])
        }) AS related_questions
        
        RETURN 
            c.section_id AS section_id,
            c.title AS title, 
            c.document AS content, 
            c.importance AS importance,
            c.practical_dates AS practical_dates,
            score, 
            related_questions
        """
        
        params = {
            "vector": query_vector,
            "top_k": top_k
        }
        
        return self.graph.query(query, params)

    def _fetch_vector_scores(self, query_vector: list, top_k: int) -> dict[str, float]:
        rows = self.graph.query(
            """
            CALL db.index.vector.queryNodes('concept_index', $top_k, $vector)
            YIELD node AS c, score
            RETURN c.section_id AS section_id, score
            """,
            {"top_k": top_k, "vector": query_vector},
        )
        return {
            str(row["section_id"]): float(row["score"])
            for row in rows
            if row["section_id"] is not None and row["score"] >= 0
        }

    def _fetch_concepts_with_questions_by_ids(self, ranked_candidates: list[dict]):
        if not ranked_candidates:
            return []

        query = """
        UNWIND $candidates AS candidate
        MATCH (c:Concept {section_id: candidate.section_id})
        OPTIONAL MATCH (q:Question)-[:VERIFIED_MENTIONS]->(c)
        WITH candidate, c, q
        ORDER BY q.problem_id

        WITH candidate, c, collect({
            id: q.problem_id,
            question: q.question,
            answer: q.answer,
            images: coalesce(q.images, [])
        }) AS related_questions

        RETURN
            c.section_id AS section_id,
            c.title AS title,
            c.document AS content,
            c.importance AS importance,
            c.practical_dates AS practical_dates,
            candidate.score AS score,
            candidate.vector_score AS vector_score,
            candidate.bm25_score AS bm25_score,
            related_questions
        ORDER BY candidate.score DESC
        """
        return self.graph.query(query, {"candidates": ranked_candidates})

    def _get_bm25_ranker(self):
        # Concept 문서는 앱 실행 중 자주 바뀌지 않으므로 최초 검색 시 한 번만 BM25 인덱스를 만듭니다.
        if self._bm25_ranker is None:
            self._bm25_ranker = BM25ConceptRanker(self._fetch_concepts_for_bm25())
        return self._bm25_ranker

    def _fetch_concepts_for_bm25(self):
        return self.graph.query(
            """
            MATCH (c:Concept)
            OPTIONAL MATCH (c)-[:BELONGS_TO]->(ch:Chapter)
            RETURN
                c.section_id AS section_id,
                c.title AS title,
                c.document AS document,
                c.practical_dates AS practical_dates,
                ch.name AS chapter
            """
        )

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        min_score = min(scores.values())
        max_score = max(scores.values())
        if max_score == min_score:
            return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}

        return {
            key: (value - min_score) / (max_score - min_score)
            for key, value in scores.items()
        }

    def format_context_for_llm(
        self,
        search_results: list,
        question_offset: int = 0,
        question_limit: int | None = 3
    ) -> str:
        """
        추출된 리스트 데이터를 LLM 프롬프트에 주입할 텍스트 형식으로 변환합니다.
        """
        if not search_results:
            return "관련된 지식 베이스 내용을 찾을 수 없습니다."

        context_parts = []
        for i, res in enumerate(search_results):
            # 유사도 점수를 함께 표기하여 분석 시 참고 가능하게 구성
            part = f"### 관련 지식 {i+1} (유사도: {res['score']:.4f})\n"
            part += f"- ID: {res.get('section_id')}\n"
            part += f"- 주요 개념: {res['title']}\n"
            part += f"- 중요도: {res.get('importance')}\n"
            practical_dates = res.get('practical_dates') or []
            linked_question_count = len(
                self.get_sorted_related_questions([res], primary_only=True)
            )
            part += f"- 실기 출제 횟수: {len(practical_dates)}회\n"
            part += f"- 그래프DB 연결 기출 수: {linked_question_count}문제\n"
            part += f"- 실기 출제 날짜: {', '.join(practical_dates) if practical_dates else '없음'}\n"
            part += f"- 상세 설명: {res['content']}\n"
            
            context_parts.append(part)

        question_parts = self._format_question_page(
            search_results,
            question_offset=question_offset,
            question_limit=question_limit
        )
        if question_parts:
            context_parts.append(question_parts)

        return "\n\n".join(context_parts)

    def _format_question_page(
        self,
        search_results: list,
        question_offset: int = 0,
        question_limit: int | None = 3
    ) -> str:
        questions = self.get_sorted_related_questions(search_results, primary_only=True)
        if not questions:
            return ""

        if question_limit is None:
            page_questions = questions[question_offset:]
        else:
            page_questions = questions[question_offset:question_offset + question_limit]
        remaining_count = max(len(questions) - (question_offset + len(page_questions)), 0)
        start_no = question_offset + 1
        end_no = question_offset + len(page_questions)

        part = "### 실제 기출 문제\n"
        part += f"- 관련 기출 문제 수: {len(questions)}개\n"
        part += f"- 현재 표시 범위: {start_no}~{end_no}번째 최신 기출\n"
        part += f"- 남은 기출 문제 수: {remaining_count}개\n"
        part += f"- 기본 표시 후보: 아래 {len(page_questions)}개 문제를 답변에 모두 출력해야 함\n"

        for index, q in enumerate(page_questions, start=start_no):
            part += f"\n#### {index}. [문제 {q['id']}]\n"
            part += "[문제 원문]\n"
            part += f"{q['question']}\n"
            if q.get("images"):
                part += f"[이미지 토큰]\n[[IMAGE:{q['id']}]]\n"
            part += "[정답 원문]\n"
            part += f"{q['answer']}\n"

        return part

    def get_sorted_related_questions(
        self,
        search_results: list,
        primary_only: bool = False
    ) -> list[dict]:
        questions_by_id = {}
        question_sources = search_results[:1] if primary_only else search_results

        for res in question_sources:
            for question in res.get("related_questions", []):
                question_id = question.get("id")
                if question_id and question_id not in questions_by_id:
                    questions_by_id[question_id] = question

        return sorted(
            questions_by_id.values(),
            key=self._question_sort_key,
            reverse=True
        )

    def count_related_questions(self, search_results: list) -> int:
        return len(self.get_sorted_related_questions(search_results, primary_only=True))

    def _question_sort_key(self, question: dict):
        try:
            year, round_no, question_no = str(question["id"]).split("_")
            return int(year), int(round_no), int(question_no)
        except (KeyError, ValueError):
            return 0, 0, 0

    def collect_image_paths(
        self,
        search_results: list,
        question_offset: int = 0,
        question_limit: int | None = 3
    ) -> dict[str, list[str]]:
        """검색 결과에 포함된 관련 기출 이미지 경로를 문제 ID별로 수집합니다."""
        images_by_question = {}
        questions = self.get_sorted_related_questions(search_results, primary_only=True)
        if question_limit is None:
            page_questions = questions[question_offset:]
        else:
            page_questions = questions[question_offset:question_offset + question_limit]

        for question in page_questions:
            question_id = question.get("id")
            if not question_id:
                continue

            for image_path in question.get("images") or []:
                normalized_path = image_path.replace("\\", "/")
                images_by_question.setdefault(question_id, [])
                if normalized_path not in images_by_question[question_id]:
                    images_by_question[question_id].append(normalized_path)

        return images_by_question
    
# 테스트 코드: 임의의 임베딩 벡터를 만들어서 Neo4j DB에 저장된 데이터 중 가장 유사도가 높은 노드 2개와 그에 연결된 기출문제 조회
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.retrieval.embedder import TEIEmbedder

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
    # import random
    # test_vector = [random.uniform(-0.1, 0.1) for _ in range(1024)]

    # 테스트용이 아닌 실제 임베딩 데이터 사용
    embedder = TEIEmbedder()
    query = "블랙박스 테스트에 대해 알려줘"
    test_vector = embedder.get_embedding(query)

    print("--- [테스트] 그래프 검색 시작 ---")
    try:
        # 개념 및 관련 기출문제 조회
        # raw_results = retriever.search_concepts_with_questions(test_vector, top_k=2)

        # 실제 벡터로 검색
        raw_results = retriever.search_concepts_with_questions(test_vector, top_k=2)
        
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
