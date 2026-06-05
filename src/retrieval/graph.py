import html

from langchain_neo4j import Neo4jGraph

ANSWER_MASK_CLASS = "answer-mask"

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

    def format_context_for_llm(
        self,
        search_results: list,
        question_offset: int = 0,
        question_limit: int = 3
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
        question_limit: int = 3
    ) -> str:
        questions = self.get_sorted_related_questions(search_results, primary_only=True)
        if not questions:
            return ""

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

    def format_questions_for_answer(
        self,
        search_results: list,
        question_offset: int = 0,
        question_limit: int | None = 3
    ) -> str:
        """더보기 응답에서 LLM을 거치지 않고 기출 문제 원문만 출력합니다."""
        questions = self.get_sorted_related_questions(search_results, primary_only=True)
        if not questions:
            return "이전 검색 결과에서 더 보여줄 관련 기출 문제가 없습니다."

        if question_limit is None:
            page_questions = questions[question_offset:]
        else:
            page_questions = questions[question_offset:question_offset + question_limit]

        if not page_questions:
            return "이전 검색 결과에서 더 보여줄 관련 기출 문제가 없습니다."

        lines = ["### 실제 기출 문제"]
        for index, question in enumerate(page_questions, start=question_offset + 1):
            question_id = question.get("id", "unknown")
            question_text = self._format_question_text_for_answer(
                str(question.get("question") or "").rstrip()
            )
            answer_text = str(question.get("answer") or "").rstrip()

            lines.append(f"#### {index}. [문제 {question_id}]")
            lines.append("[문제]")
            lines.append(question_text)

            if question.get("images"):
                lines.append(f"[[IMAGE:{question_id}]]")

            if answer_text:
                masked_answer = (
                    f'<span class="{ANSWER_MASK_CLASS}">'
                    f'정답: {html.escape(answer_text)}'
                    "</span>"
                )
                lines.append("[정답]")
                lines.append(masked_answer)

        return "\n\n".join(lines)

    def _format_question_text_for_answer(self, question_text: str) -> str:
        """문제 원문은 유지하고 [Source Code] 영역만 Markdown 코드블럭으로 감쌉니다."""
        source_marker = "[Source Code]"
        if source_marker not in question_text:
            return question_text

        before_source, source_code = question_text.split(source_marker, 1)
        source_code = source_code.strip()
        if not source_code:
            return question_text

        # 코드 내용은 복원하거나 재배열하지 않고, 표시 형식만 코드블럭으로 바꾼다.
        safe_source_code = source_code.replace("```", "`\u200b``")
        return (
            f"{before_source.rstrip()}\n\n"
            f"{source_marker}\n\n"
            "```text\n"
            f"{safe_source_code}\n"
            "```"
        ).strip()

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
