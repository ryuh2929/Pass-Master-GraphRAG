# Pass-Master Troubleshooting 

각 항목은 다음 기준으로 작성한다.

```text
문제 상황
원인
해결 방법
개선 효과
정리
```

---

## 1. Neo4j 컨테이너 hostname 해석 실패

### 문제 상황

`docker compose up` 또는 `docker compose up -d neo4j` 실행 후 `passmaster-neo4j`가 바로 종료된다.

로그에는 다음과 유사한 메시지가 반복된다.

```text
Could not determine local host name
java.net.UnknownHostException: 75e4f4e26d8f
Neo4j Server shutdown initiated by request
```

### 원인

Neo4j 자체 인증 실패가 아니다.

Neo4j/Java가 컨테이너 내부 hostname을 확인하는 과정에서 Docker가 부여한 임의 hostname을 해석하지 못해 Log4j 초기화 단계에서 실패한다.

로그의 다음 메시지는 인증 실패가 아니라 초기 비밀번호 설정 안내다.

```text
Changed password for user 'neo4j'
```

### 해결 방법

`neo4j` 서비스에 고정 hostname을 설정한다.

수정 파일: `docker-compose.yml`

```yaml
neo4j:
  image: neo4j:5.18
  container_name: passmaster-neo4j
  hostname: passmaster-neo4j
```

Neo4j 5.x 기준 pagecache 설정은 deprecated 된 이름 대신 새 이름을 사용한다.

수정 파일: `docker-compose.yml`

```yaml
environment:
  - NEO4J_server_memory_pagecache_size=1G
```

확인 명령:

```powershell
docker compose logs neo4j --tail 60
docker compose ps neo4j
```

정상 로그:

```text
Bolt enabled on 0.0.0.0:7687.
HTTP enabled on 0.0.0.0:7474.
Started.
```

### 개선 효과

Docker Desktop 재시작, 네트워크 캐시, 컨테이너 재생성 이후에도 Neo4j가 안정적으로 기동된다.

### 정리

Neo4j가 시작 직후 죽고 `UnknownHostException`이 보이면 비밀번호보다 hostname 설정을 먼저 확인한다.

---

## 2. OpenAI API 모드에서 TEI 임베딩 서버가 실행되지 않음

### 문제 상황

`LLM_MODEL=openai`로 바꾼 뒤 앱은 OpenAI API를 사용하지만, 질문 처리 중 임베딩 또는 검색 단계에서 실패한다.

예상되는 에러 흐름:

```text
TEI 서버 통신 에러
Connection refused
http://localhost:8080/embed 연결 실패
```

### 원인

OpenAI API 전환은 최종 답변 생성 LLM만 바꾼다.

검색 임베딩은 여전히 로컬 TEI의 `BAAI/bge-m3`를 사용한다.

Neo4j에는 BGE-M3로 생성한 벡터가 저장되어 있으므로, 질의 임베딩도 같은 모델로 생성해야 vector index 검색이 정상 동작한다.

### 해결 방법

`tei`는 기본 `docker compose up` 대상에 포함한다.

수정 파일: `docker-compose.yml`

```yaml
tei:
  image: ghcr.io/huggingface/text-embeddings-inference:latest
  container_name: passmaster-tei
  ports:
    - "8080:80"
  command: ["--model-id", "BAAI/bge-m3", "--port", "80"]
```

`tei`에는 `profiles: ["local"]`을 두지 않는다.

실행 명령:

```powershell
docker compose up -d
```

기본 실행 서비스 확인:

```powershell
docker compose config --services
```

기대값:

```text
neo4j
tei
```

### 개선 효과

OpenAI API 모드에서도 GraphRAG 검색 단계가 동일한 임베딩 모델로 동작한다.

### 정리

`LLM_MODEL=openai`는 LLM만 OpenAI로 바꾸는 설정이다. Neo4j와 TEI는 검색 백엔드이므로 계속 필요하다.

---

## 3. Ollama가 받은 모델과 앱이 호출하는 모델이 다름

### 문제 상황

Ollama 컨테이너는 특정 모델을 pull했는데 앱은 다른 모델을 호출한다.

예시:

```text
docker-compose.yml: ollama pull gemma4:e4b
.env: LLM_MODEL=llama3:latest
```

이 경우 Ollama 안에 `llama3:latest`가 없으면 모델 없음 오류가 발생한다.

### 원인

앱이 실제 호출하는 모델은 `docker-compose.yml`의 `ollama pull` 값이 아니라 `.env`의 `LLM_MODEL`이다.

`docker-compose.yml`은 컨테이너 시작 시 모델을 미리 받아두는 역할만 한다.

### 해결 방법

로컬 기본 모델을 하나로 정하고 `.env`, `.env.example`, `docker-compose.yml`, LLM fallback 값을 맞춘다.

수정 파일: `.env`

```env
LLM_MODEL=gemma4:e4b
OLLAMA_HOST=http://localhost:11434
```

수정 파일: `docker-compose.yml`

```yaml
entrypoint: ["/bin/sh", "-c", "ollama serve & sleep 10 && ollama pull gemma4:e4b && wait"]
```

수정 파일: `src/llm/llm_switch.py`

```python
model_name = os.getenv("LLM_MODEL", "gemma4:e4b")
```

### 개선 효과

로컬 LLM 실행 시 컨테이너가 준비한 모델과 앱이 호출하는 모델이 일치한다.

### 정리

Ollama 모델 불일치 문제는 `.env`의 `LLM_MODEL`을 기준으로 확인한다. `docker-compose.yml`의 pull 명령은 fallback이 아니다.

---

## 4. 관련 기출문제가 LLM 컨텍스트에 들어가지 않음

### 문제 상황

Concept 검색은 되는 것 같은데 답변의 관련 기출 문제가 비어 있거나 출력되지 않는다.

GraphDB에는 `Question`과 `Concept`의 `VERIFIED_MENTIONS` 관계가 있는데도 LLM 컨텍스트의 기출 사례가 누락된다.

### 원인

`Question` 노드의 식별 속성은 `id`가 아니라 `problem_id`다.

적재 로직은 다음처럼 `problem_id`를 만든다.

수정 파일: `src/ingestion/loader.py`

```cypher
MERGE (q:Question {problem_id: $year + "_" + $round + "_" + prob.no})
```

조회 쿼리에서 `q.id`를 읽으면 `NULL`이 반환된다.

그 결과 포맷터가 관련 문제가 없다고 판단할 수 있다.

### 해결 방법

검색 쿼리에서 `q.problem_id`를 사용한다.

수정 파일: `src/retrieval/graph.py`

```cypher
ORDER BY q.problem_id

WITH c, score, collect({
    id: q.problem_id,
    question: q.question,
    answer: q.answer
}) AS related_questions
```

DB 확인 명령:

```powershell
docker exec passmaster-neo4j cypher-shell -u neo4j -p <password> "MATCH (q:Question)-[:VERIFIED_MENTIONS]->(c:Concept) RETURN q.problem_id AS problem_id, q.id AS id, c.title AS title LIMIT 5"
```

정상 상태에서는 `problem_id`는 값이 있고 `id`는 `NULL`이다.

### 개선 효과

`VERIFIED_MENTIONS` 관계로 연결된 기출문제가 LLM 컨텍스트의 `실제 기출 사례` 섹션에 포함된다.

### 정리

기출문제가 안 나오면 먼저 `Question` 식별 필드가 `problem_id`인지 확인한다.

---

## 5. 크롤링 중 정답 박스와 문제 본문이 중첩되어 데이터가 누락됨

### 문제 상황

기출 JSON 생성 후 다음 문제가 발생할 수 있다.

```text
문항 수가 20개가 아님
정답이 비어 있음
문제 본문 끝에 정답이 섞임
소스코드 또는 표 일부가 누락됨
```

### 원인

Tistory 페이지의 문제 본문, 정답 moreLess 박스, 코드 블록, 표, 이미지가 중첩된 HTML 구조로 들어오는 경우가 있다.

단순 visited 처리로 부모 태그를 방문 처리하면 자식 태그의 실제 데이터까지 누락될 수 있다.

반대로 방문 처리를 너무 느슨하게 하면 정답 박스 내용이 문제 본문에 섞일 수 있다.

### 해결 방법

크롤러는 다음 기준을 함께 적용한다.

수정 파일: `src/ingestion/crawling.py`

```python
last_prob_no = 0
match = re.search(r'^(\d+)[\.\)]', text)
```

문제 번호는 `last_prob_no + 1` 순서인지 확인한다.

정답 박스는 `data-ke-type="moreLess"` 또는 `.moreless-content`에서 별도로 추출한다.

코드 블록은 `colorscripter-code` 또는 `colorscripter-code-table`을 별도 처리한다.

데이터 검증 스크립트를 반드시 실행한다.

```powershell
$env:PYTHONIOENCODING='utf-8'
python src\test_exam_integrity.py
```

정상 결과:

```text
최종 결과: 19/19 통과
```

### 개선 효과

문항 수, 정답 누락, 본문-정답 혼입 문제를 조기에 발견할 수 있다.

### 정리

크롤링 로직을 수정한 뒤에는 반드시 `test_exam_integrity.py`로 20문항/정답/본문 무결성을 확인한다.

---

## 6. PDF 청킹 과정에서 섹션이 누락되거나 합쳐짐

### 문제 상황

`processed_chunks.json` 생성 후 특정 개념 섹션이 누락되거나 두 섹션이 하나로 합쳐진다.

예시:

```text
총 탐지된 섹션 수가 301개가 아님
특정 ID가 누락됨
298번과 299번처럼 인접 섹션이 섞임
```

### 원인

PDF 텍스트 추출 결과가 페이지 단 구성, 줄바꿈, 날짜 표기, 제목 패턴에 따라 불안정하게 나온다.

현재 청커는 정규식으로 섹션 헤더를 찾으므로, PDF 텍스트가 예상 패턴에서 벗어나면 누락 또는 병합이 발생할 수 있다.

### 해결 방법

섹션 패턴과 예외 패치를 함께 관리한다.

수정 파일: `src/ingestion/chunker.py`

```python
section_pattern = re.compile(
    r'(\d{6})\s+'
    r'([\d\.,\s]*?)'
    r'(?:필기\s+([\d\.,\s]+?)\s+)?'
    r'[\s…,]*'
    r'(\d{3})\s+'
    r'([^\n]+?)\s+'
    r'([A-C])(?:\s|\n|$)'
)
```

이미 알려진 섹션 병합은 수동 패치한다.

수정 파일: `src/ingestion/chunker.py`

```python
if chunk["metadata"]["id"] == "298" and "299 C" in chunk["document"]:
    ...
```

청킹 후 ID 검사를 실행한다.

```powershell
python src\test_check_ids.py
```

정상 결과:

```text
총 탐지된 섹션: 301개
누락된 번호가 없습니다
```

### 개선 효과

요약 Concept 노드가 누락되지 않고 1~301번 범위를 안정적으로 유지한다.

### 정리

PDF 청킹을 다시 돌린 뒤에는 `test_check_ids.py`로 섹션 ID 완전성을 먼저 확인한다.

---

## 7. 의미적으로 비슷하지만 날짜가 맞지 않는 기출-개념 연결이 생성됨

### 문제 상황

임베딩 유사도는 높지만 실제 해당 회차에 출제된 개념이 아닌 `Question`과 `Concept`이 연결된다.

예시:

```text
Java 코드 문제가 Python 단원과 연결됨
비슷한 용어가 반복되는 다른 단원이 더 높은 후보로 잡힘
```

### 원인

벡터 유사도는 의미적으로 가까운 후보를 찾는 데 강하지만, 시험 회차와 출제 날짜의 정합성을 보장하지 않는다.

특히 프로그래밍 언어, SQL, 테스트 기법처럼 유사 문맥이 많은 단원은 의미 검색만으로 오연결이 생길 수 있다.

### 해결 방법

`VERIFIED_MENTIONS` 관계 생성 시 임베딩 유사도와 실기 출제 날짜를 함께 검증한다.

수정 파일: `src/ingestion/loader.py`

```cypher
MATCH (q:Question)-[:HAS_QUESTION]-(e:Exam)
CALL db.index.vector.queryNodes('concept_index', 20, q.embedding)
YIELD node AS c, score
WHERE score >= $threshold
  AND any(d IN c.practical_dates WHERE trim(d) = trim(e.practical_dates))
```

문제 하나는 대표 Concept 하나에만 연결한다.

수정 파일: `src/ingestion/loader.py`

```cypher
WITH q, collect({concept: c, score: score})[0] AS best_match
MERGE (q)-[r:VERIFIED_MENTIONS]->(target)
```

### 개선 효과

단순 의미 유사도만 높은 후보가 아니라, 실제 실기 날짜와 맞는 후보만 검증 관계로 남는다.

### 정리

기출-개념 연결은 벡터 점수만 보지 않는다. `score threshold + practical_dates 검증 + 대표 Concept 1개`가 현재 프로젝트의 의도된 설계다.

---

## 8. 검색 결과가 불용어 또는 요청 표현에 끌려감

### 문제 상황

사용자가 다음처럼 질문했을 때 핵심 개념이 아니라 요청 표현에 가까운 엉뚱한 결과가 검색된다.

```text
블랙박스 테스트 알려줘
프로토콜 설명해줘
정답만 다시 알려줘
```

### 원인

질문 전체를 그대로 임베딩하면 `알려줘`, `설명해줘`, `다시` 같은 요청 표현이 검색 벡터에 섞인다.

짧은 질의일수록 이런 표현이 의미 벡터에 미치는 영향이 커질 수 있다.

### 해결 방법

검색 전에 LLM으로 핵심 검색어를 추출한다.

수정 파일: `src/llm/rag_chain.py`

```python
refined_query = self.llm.invoke(build_query_refine_prompt(query)).content.strip()
query_vector = self.embedder.get_embedding(refined_query)
```

프롬프트에서는 요청 표현을 제거하고 핵심 전문 용어만 반환하도록 제한한다.

수정 파일: `src/llm/prompts.py`

```python
def build_query_refine_prompt(query: str) -> str:
    ...
```

### 개선 효과

벡터 검색이 질문의 말투가 아니라 실제 개념 키워드 중심으로 수행된다.

### 정리

검색 품질이 흔들릴 때는 먼저 정제된 검색어가 무엇으로 나왔는지 로그를 확인한다.

```text
[Refine] 검색어 정제: 원문 -> 정제어
```

---

## 9. GraphRAG 진행 상태가 실제 처리 단계와 맞지 않음

### 문제 상황

Streamlit UI에는 다음과 같은 상태 문구가 순서대로 표시된다.

```text
검색어 정제 중...
Neo4j 지식 그래프 검색 중...
LLM 답변 생성 중...
```

하지만 실제로는 각 단계가 그 시점에 실행되지 않고, 마지막에 전체 chain이 한 번에 실행된다.

이 경우 사용자는 현재 어떤 단계에서 시간이 걸리는지 알 수 없고, 디버깅할 때도 TEI, Neo4j, LLM 중 어느 지점이 병목인지 구분하기 어렵다.

### 원인

기존 `run_stream()`이 실제 GraphRAG 단계를 나누어 실행하지 않고, 안내 문구만 먼저 `yield`한 뒤 마지막에 `chain_with_history.invoke()`를 호출했다.

수정 파일: `src/llm/rag_chain.py`

```python
yield "검색 노이즈 제거 및 핵심 키워드 추출 중..."
yield "지식 그래프(Neo4j) 탐색 및 관련 지식 추출 중..."

response = self.chain_with_history.invoke(...)
yield response
```

이 구조에서는 상태 문구가 실제 처리 상태를 나타내지 않는다.

### 해결 방법

토큰 스트리밍이 아니라 GraphRAG 내부 단계 이벤트를 스트리밍한다.

`rag_chain.py`는 Streamlit UI를 직접 만지지 않고, 현재 처리 단계를 event로 반환한다.

수정 파일: `src/llm/rag_chain.py`

```python
yield {"type": "status", "message": "검색어 정제 중..."}
refined_query = self.llm.invoke(build_query_refine_prompt(user_query)).content.strip()

yield {"type": "status", "message": "TEI 임베딩 생성 중..."}
query_vector = self.embedder.get_embedding(refined_query)

yield {"type": "status", "message": "Neo4j 지식 그래프 검색 중..."}
raw_results = self.retriever.search_concepts_with_questions(query_vector, top_k=3)

yield {"type": "status", "message": "LLM 답변 생성 중..."}
response = self._extract_llm_content(self.llm.invoke(prompt_value))

yield {"type": "answer", "content": response}
```

`app.py`는 event를 받아 Streamlit 상태 UI만 갱신한다.

수정 파일: `app.py`

```python
for event in stream:
    if isinstance(event, dict) and event.get("type") == "status":
        status.update(label=event["message"])
    elif isinstance(event, dict) and event.get("type") == "answer":
        response = event["content"]
```

상태 박스가 답변 생성 후 계속 남지 않게 하려면 `st.empty()` placeholder로 감싼 뒤 완료 후 비운다.

수정 파일: `app.py`

```python
status_placeholder = st.empty()

with status_placeholder.container():
    with st.status("분석 준비 중...", expanded=True) as status:
        ...

status_placeholder.empty()
st.markdown(response, unsafe_allow_html=True)
```

### 개선 효과

UI 안내 문구와 실제 GraphRAG 실행 단계가 일치한다.

TEI 임베딩, Neo4j 검색, 컨텍스트 구성, LLM 생성 중 어느 단계에서 병목이나 오류가 발생하는지 파악하기 쉬워진다.

토큰 스트리밍 없이도 사용자에게 실제 처리 흐름을 보여줄 수 있다.

### 정리

상태 UI는 단순 안내 문구를 미리 출력하는 방식이 아니라, GraphRAG 내부 단계가 실행되기 직전에 event를 반환하는 방식으로 구현한다.

토큰 스트리밍은 필요하지 않다. 이 프로젝트에서 필요한 것은 LLM 내부 토큰 출력이 아니라 GraphRAG 파이프라인 단계 표시다.

---

## 10. 기출 이미지가 문제와 1:1로 매칭되지 않음

### 문제 상황

기출 문제에 이미지가 포함되어 있는데 답변에서는 이미지가 문제 바로 아래에 나오지 않거나, 답변 맨 마지막에 몰려서 출력된다.

이미지가 너무 크게 표시되어 답변 가독성이 떨어지고, 어떤 문제에 연결된 이미지인지 확인하기 어렵다.

### 원인

기출 JSON의 `images` 항목을 GraphDB의 `Question` 노드에 저장하지 않거나, LLM 답변과 별도로 이미지 목록만 후처리하면 문제와 이미지의 위치 관계가 깨진다.

또한 `st.image(width=...)`처럼 width 기준으로만 크기를 고정하면 이미지 원본 비율에 따라 세로 길이가 크게 달라진다.

### 해결 방법

기출 적재 시 `Question.images` 속성을 함께 저장한다.

수정 파일: `src/ingestion/loader.py`

```cypher
SET q.no = prob.no,
    q.question = prob.question,
    q.answer = prob.answer,
    q.images = coalesce(prob.images, [])
```

검색 결과에서 현재 표시 범위의 문제 이미지 경로만 문제 ID별로 수집한다.

수정 파일: `src/retrieval/graph.py`

```python
def collect_image_paths(
    self,
    search_results: list,
    question_offset: int = 0,
    question_limit: int | None = 3
) -> dict[str, list[str]]:
    ...
```

LLM 컨텍스트에는 이미지가 필요한 문제 바로 아래에 `[[IMAGE:problem_id]]` 토큰을 넣고, Streamlit에서는 이 토큰 위치에 이미지를 삽입한다.

수정 파일: `app.py`

```python
token_pattern = re.compile(r"\[\[IMAGE:([^\]]+)\]\]")

for match in token_pattern.finditer(content):
    ...
    question_id = match.group(1).strip()
    for image_path in images_by_question.get(question_id, []):
        render_local_image(image_path)
```

이미지 크기는 width 고정 대신 HTML 이미지 태그와 CSS `max-height`로 제어한다.

수정 파일: `app.py`

```python
.question-image {
    display: block;
    max-height: 420px;
    max-width: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
}
```

### 개선 효과

각 이미지는 해당 기출 문제의 `[[IMAGE:problem_id]]` 위치에 출력된다.

답변 맨 아래에 관련 없는 이미지가 몰리는 문제를 방지하고, 이미지 크기가 답변 영역을 과도하게 차지하지 않는다.

### 정리

기출 이미지는 답변 후처리 목록으로 붙이지 않는다. 문제 ID 기반 이미지 토큰을 사용해 LLM 답변 위치와 Streamlit 렌더링 위치를 연결한다.

---

## 11. 기출 더보기 요청에서 답변 범위가 흐려짐

### 문제 상황

관련 기출 문제가 여러 개 있을 때 첫 답변에 모든 문제를 출력하면 답변이 길어진다.

반대로 일부만 보여주면 사용자가 다음 문제를 이어서 보고 싶어 할 수 있는데, `더 보여줘` 같은 요청을 일반 RAG 질문처럼 처리하면 단원 요약, 보충 설명, 합격 포인트가 반복될 수 있다.

또한 랜덤하게 기출을 보여주면 최신 기출부터 학습하기 어렵고, 같은 문제만 반복될 수 있다.

### 원인

일반 개념 질문과 후속 기출 더보기 요청을 같은 흐름으로 처리하면 사용자 의도를 구분하기 어렵다.

일반 질문은 개념 설명과 기출 예시가 필요하지만, `더 보여줘`는 이전 검색 결과의 다음 기출 문제만 이어서 보고 싶은 요청이다.

### 해결 방법

사용자 입력을 키워드 기반으로 먼저 라우팅한다.

수정 파일: `src/llm/rag_chain.py`

```python
def _is_more_question_request(self, user_query: str) -> bool:
    more_keywords = ["더 보여", "더보기", "더 보기", "다음", "이어서", "계속"]
    return any(keyword in user_query for keyword in more_keywords)

def _is_all_question_request(self, user_query: str) -> bool:
    all_keywords = ["전체 보여", "전부 보여", "다 보여", "전체 기출", "나머지 전부", "남은 거 전부"]
    return any(keyword in user_query for keyword in all_keywords)
```

일반 질문이 들어오면 검색 결과와 현재 offset을 세션에 저장한다.

수정 파일: `src/llm/rag_chain.py`

```python
self.question_page_store[session_id] = {
    "results": filtered_results,
    "offset": min(question_limit, total_questions),
    "limit": question_limit,
}
```

후속 요청은 저장된 검색 결과와 offset을 사용한다.

```text
일반 질문
-> 최신순 3문제 context 구성
-> offset = 3 저장

더 보여줘
-> 저장된 검색 결과에서 offset부터 다음 3문제 context 구성
-> offset 갱신

전체 보여줘
-> 저장된 검색 결과에서 현재 offset 이후 남은 전체 context 구성
-> offset 끝까지 갱신
```

현재 구조에서는 질문 라우팅과 카운팅은 코드가 담당하고, 문제 본문과 코드블럭 정리는 LLM이 담당한다.

수정 파일: `src/llm/rag_chain.py`

```python
context = self.retriever.format_context_for_llm(
    filtered_results,
    question_offset=question_offset,
    question_limit=question_limit
)

context += """
### 응답 모드
- 이 요청은 이전 검색 결과의 기출 더보기입니다.
- [단원 정보], [요약 정보], [보충 설명], [합격 포인트]는 반복하지 마십시오.
- [실제 기출 문제] 섹션만 출력하십시오.
"""
```

### 개선 효과

첫 답변은 개념 설명과 최신순 기출 3문제를 함께 제공한다.

`더 보여줘`는 이전 검색 결과에서 다음 3문제만 이어서 보여주고, `전체 보여줘`는 남은 전체 문제를 보여준다.

사용자는 긴 답변에 압도되지 않고 관련 기출을 단계적으로 확인할 수 있다.

### 정리

기출 더보기는 LLM에게 전부 맡기지 않는다. 요청 라우팅, offset, 표시 범위는 코드가 관리하고, 현재 표시 범위의 문제 출력 품질은 LLM이 검수한다.

---

## 12. Question-Concept 자동 연결에서 코드형 문제가 엉뚱한 단원으로 연결됨

### 문제 상황

기출 문제를 `Concept` 노드에 자동 연결할 때, 단순 벡터 유사도만 사용하면 코드형 문제가 엉뚱한 프로그래밍 언어 단원으로 연결되는 사례가 있었다.

예를 들어 Java 코드 문제가 C언어 단원으로 연결되거나, SQL 문제 안의 일부 키워드만 보고 인접 SQL 단원으로 잘못 연결되는 식이다.

전체 연결 커버리지는 높아 보여도, 코드형 문제의 언어 대분류나 세부 단원 연결이 틀리면 이후 RAG 답변에서 잘못된 기출 문제가 함께 출력된다.

### 원인

임베딩 모델은 문장 전체의 의미 유사도에는 강하지만, 코드형 문제에서 다음 신호를 안정적으로 구분하기 어렵다.

- `SQL`, `C언어`, `Java`, `Python` 같은 명시 언어명
- `printf`, `System.out.print`, `range`, `slice` 같은 코드 키워드
- 문제 출제 날짜와 단원 출제 날짜의 정합성
- 같은 언어 내부의 세부 단원 차이

또한 문제 본문에 코드가 길게 들어가면 임베딩이 코드 토큰에 끌려, 실제 평가 의도와 다른 단원으로 이동할 수 있다.

### 해결 방법

기출-개념 자동 연결 로직을 vector-only에서 날짜 검증과 키워드 기반 보정을 포함한 hybrid 방식으로 확장했다.

수정 파일: `src/ingestion/hybrid_linker.py`

```python
def build_hybrid_candidates(
    *,
    question_text: str,
    practical_date: str,
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    ranker: BM25ConceptRanker,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    explicit_language_vector_weight: float = 1.0,
    explicit_language_bm25_weight: float = 0.0,
    code_vector_weight: float = 0.3,
    code_bm25_weight: float = 0.7,
) -> list[HybridCandidate]:
```

핵심 보정은 다음과 같다.

- Question의 실기 출제 날짜와 Concept의 `practical_dates`가 맞지 않으면 후보에서 제거
- 문제에 언어명이 명시되면 해당 언어 Concept 범위 안에서만 후보 선택
- 언어명이 없지만 코드/SQL 힌트가 있으면 BM25 가중치를 높여 표면 키워드 반영
- `System.out.printf()`가 C의 `printf()`로 오인되지 않도록 정규식 보정
- Concept 계열 판별은 본문 전체가 아니라 제목/챕터 중심으로 수행

평가셋도 별도로 만들었다.

수정 파일:

- `data/evaluation/question_concept_gold.seed.json`
- `data/evaluation/question_language_gold.seed.json`
- `src/evaluation/evaluate_concept_linking.py`
- `src/evaluation/evaluate_language_linking.py`
- `src/evaluation/compare_linking_strategies.py`

### 개선 효과

자동 연결 커버리지는 다음과 같이 개선되었다.

```text
364/380 (95.8%) -> 376/380 (98.95%)
```

코드형 문제 평가셋에서는 언어 대분류 오연결을 제거했다.

```text
언어 대분류 정확도: 85.21% -> 100.00%
코드/SQL 세부 단원 연결 정확도: 53.52% -> 64.79%
```

세부 단원 정확도는 아직 완전하지 않지만, 사용자가 체감하기 쉬운 “Java 문제인데 C 단원이 나오는” 수준의 대분류 오연결을 먼저 줄였다.

### 정리

기출-개념 자동 연결은 임베딩 점수 하나로 끝내기 어렵다. 날짜 정합성, 명시 언어 후보군 제한, BM25 키워드 보정을 함께 사용해야 GraphRAG의 엣지 품질을 안정적으로 높일 수 있다.

---

## 13. Reranker를 도입했지만 코드 문제 연결 정확도가 하락함

### 문제 상황

`bge-reranker-base` 모델을 별도 TEI reranker 컨테이너로 띄우고, vector/BM25로 만든 후보를 다시 재정렬하는 실험을 했다.

하지만 실제 평가 결과, reranker를 적용했을 때 코드/SQL 문제 연결 정확도가 오히려 떨어졌다.

코드에는 reranker 관련 클래스와 Docker profile이 남아 있지만, 기본 실행에서는 사용하지 않는 상태가 되었다.

### 원인

reranker는 후보 생성기가 아니라 후보 재정렬기이다.

즉, 정답 Concept가 후보군 안에 없으면 reranker가 정답을 새로 만들어낼 수 없다.

또한 `bge-reranker-base`는 문장 쌍 의미 관련도 재정렬에는 유용하지만, 이 프로젝트의 코드형 문제처럼 다음 기준을 함께 만족해야 하는 경우에는 기대만큼 강하지 않았다.

- 기출 날짜 일치
- 언어 대분류 일치
- 같은 언어 내부 세부 단원 구분
- 코드 키워드와 문제 의도 구분

추가로 reranker 컨테이너는 RAM/VRAM 사용량도 늘린다. 로컬 LLM, TEI 임베딩, Neo4j를 함께 사용하는 환경에서는 상시 실행 비용이 크다.

### 해결 방법

reranker 코드는 실험 가능한 옵션으로 남기되, 기본값은 비활성화했다.

수정 파일: `src/ingestion/loader.py`

```python
def _create_reranker(self):
    if os.getenv("USE_RERANKER", "false").lower() not in {"1", "true", "yes"}:
        print("Reranker disabled. Set USE_RERANKER=true to enable.")
        return None
```

Docker에서도 reranker는 별도 profile로 분리해 필요할 때만 켜도록 했다.

수정 파일: `docker-compose.yml`

```yaml
tei-reranker:
  profiles: ["reranker"]
```

그리고 링크 재생성 과정에서 reranker timeout 같은 실패가 발생해도 기존 관계가 먼저 삭제되지 않도록, 새 링크를 모두 준비한 뒤 마지막에 `VERIFIED_MENTIONS`를 교체하는 방식으로 바꿨다.

### 개선 효과

reranker를 “항상 쓰는 성능 개선 장치”가 아니라 “실험 가능한 재정렬 옵션”으로 분리했다.

기본 연결 품질은 검증된 hybrid/date/language 제한 로직으로 유지하고, reranker로 인한 메모리 사용량 증가와 정확도 하락을 피할 수 있다.

또한 실패 시 기존 `VERIFIED_MENTIONS`가 먼저 삭제되어 평가 결과가 전부 missing으로 나오는 위험을 줄였다.

### 정리

reranker는 무조건 성능을 올려주는 부품이 아니다. 후보 생성 품질이 충분하고 평가셋에서 개선이 확인될 때만 기본 경로에 넣어야 한다. 이 프로젝트에서는 실험 결과가 좋지 않았기 때문에 옵션으로 남기고 기본값에서는 제외했다.

---

## 14. 질문 검색에서 vector-only 검색이 단원명과 키워드 매칭을 놓침

### 문제 상황

사용자 질문을 Concept로 검색할 때 기존에는 TEI 임베딩 기반 vector 검색만 사용했다.

의미적으로 가까운 단원을 찾는 데는 효과가 있었지만, 다음과 같이 단원명이나 핵심 키워드가 중요한 질문에서는 1등 후보가 흔들릴 수 있었다.

- `화이트박스 테스트의 검증 기준`
- `Boundary Value Analysis`
- `EQUI JOIN`
- `Python range`
- `System.out.print`

특히 실제 답변에서는 top1 Concept의 연결 기출 문제만 가져오기 때문에, 검색 top1이 틀리면 기출 문제까지 잘못 출력될 수 있다.

### 원인

임베딩 검색은 문장 전체 의미를 기준으로 유사도를 계산한다.

따라서 “무슨 뜻인지”가 비슷한 단원을 찾는 데는 강하지만, 다음 신호를 항상 1등으로 끌어올리지는 못한다.

- 단원명과 거의 일치하는 키워드
- 영문 병기 또는 약어
- 코드/SQL 식별자
- `정의`, `종류`, `검증 기준`, `도구` 같은 검색 의도 구분

반대로 BM25는 단어 일치에는 강하지만, 의미적으로 풀어 쓴 질문에는 약하다.

따라서 질문 검색에는 vector와 BM25 중 하나만 쓰기보다, vector를 주력으로 두고 BM25를 보조 점수로 섞는 방식이 적합했다.

### 해결 방법

질문 검색 평가셋을 만들고 vector-only, BM25-only, hybrid 비율을 비교했다.

수정 파일: `src/evaluation/evaluate_query_retrieval.py`

```text
vector_only
bm25_only
hybrid_80_20
hybrid_60_40
hybrid_40_60
```

평가 결과 `vector 0.8 + BM25 0.2`가 가장 안정적이었다.

```text
vector_only   Top1 88.57%, Top3 97.14%, MRR 0.936
bm25_only     Top1 94.29%, Top3 97.14%, MRR 0.960
hybrid_80_20  Top1 94.29%, Top3 100.00%, MRR 0.967
```

이 결과를 runtime 검색에 적용했다.

수정 파일: `src/retrieval/graph.py`

```python
final_score = (
    vector_weight * normalized_vector.get(section_id, 0.0)
    + bm25_weight * normalized_bm25.get(section_id, 0.0)
)
```

최종 검색 흐름은 다음과 같다.

```text
정제된 검색어
-> TEI embedding
-> vector 후보 top100 조회
-> 같은 검색어로 BM25 점수 계산
-> vector_norm 0.8 + bm25_norm 0.2
-> 최종 top1 Concept 조회
-> top1 Concept 요약 + top1 연결 기출 3개 LLM 주입
```

평가 수치가 실제 앱 코드에도 그대로 적용되는지 확인하기 위해 `current_runtime` 전략도 추가했다.

수정 파일: `src/evaluation/evaluate_query_retrieval.py`

```python
strategies["current_runtime"] = build_runtime_candidates(
    retriever=retriever,
    query_vector=query_vector,
    query_text=case.query,
    top_k=final_top_k,
    candidate_top_k=vector_top_k,
)
```

### 개선 효과

질문 검색 품질이 vector-only 대비 개선되었다.

```text
Top1: 88.57% -> 94.29%
Top3: 97.14% -> 100.00%
MRR : 0.936 -> 0.967
```

실제 runtime 전략도 같은 수치로 검증되었다.

```text
current_runtime
Top1 94.29%, Top3 100.00%, Top5 100.00%, MRR 0.967
```

### 정리

질문 검색에서는 vector-only가 항상 최선이 아니다. 의미 유사도는 vector가 담당하고, 단원명/영문 병기/코드 키워드 매칭은 BM25가 보완하도록 hybrid 점수를 적용하면 검색 top1과 top3 품질을 함께 높일 수 있다.

---

## 15. 기출 더보기에서 LLM이 일부 문제를 누락하고도 다음 페이지로 넘어감

### 문제 상황

사용자가 `더 보여줘`를 입력하면 이전 검색 결과에서 다음 기출 3문제를 보여주도록 설계했다.

그러나 실제 응답에서는 LLM이 3문제 중 1문제만 출력하고 끝내는 경우가 있었다.

이때 기존 로직은 LLM이 실제로 몇 문제를 출력했는지 확인하지 않고 offset을 먼저 이동시켰다.

```text
1. offset 기준 다음 3문제 context 구성
2. LLM 답변 생성
3. offset을 3칸 이동
4. LLM이 1문제만 출력해도 시스템은 3문제를 모두 보여준 것으로 간주
```

그 결과 사용자가 다시 `더 보여줘`를 입력해도 누락된 문제가 다시 나오지 않고, 다음 묶음으로 넘어갈 수 있었다.

### 원인

기존 더보기 로직은 pagination 상태를 코드가 관리했지만, LLM 출력 결과를 검증하지 않았다.

따라서 다음 두 상태가 서로 불일치할 수 있었다.

```text
코드 상태: 3문제를 전달했으므로 3문제를 보여줬다고 판단
실제 응답: LLM이 1문제만 출력
```

특히 기출 문제는 원문, 정답, 이미지 토큰, 코드 블록이 포함될 수 있어 LLM이 임의로 일부만 출력하거나 요약할 가능성이 있었다.

### 해결 방법

LangGraph 답변 생성 흐름에 기출 문제 ID 검증 노드를 추가했다.

수정 파일: `src/llm/rag_graph.py`

```text
generate_answer
-> validate_question_output
-> 실패 시 retry_question_output
-> 재검증
-> 통과 시 finalize_answer
```

현재 페이지에 반드시 출력되어야 하는 문제 ID를 `expected_question_ids`로 저장한다.

```python
expected_question_ids = self._get_page_question_ids(
    filtered_results,
    question_offset=question_offset,
    question_limit=question_limit,
)
```

LLM 답변 생성 후, 응답 안에 expected ID가 모두 포함되었는지 검사한다.

```python
missing_question_ids = [
    question_id
    for question_id in expected_question_ids
    if not self._contains_question_id(response, question_id)
]
```

누락이 있으면 같은 context로 한 번 더 재생성한다.

수정 파일: `src/llm/prompts.py`

```text
[누락된 문제 ID]
2024_3_13

아래 규칙을 반드시 지켜 다시 답변하십시오.
1. [학습 지식]의 [실제 기출 문제] 섹션에 있는 문제를 모두 출력하십시오.
2. 누락된 문제 ID를 절대 빠뜨리지 마십시오.
```

중요한 점은 offset 이동 시점을 변경한 것이다.

기존에는 LLM 답변 생성 전에 offset을 이동했지만, 수정 후에는 검증을 통과한 뒤에만 `pending_page_state`를 실제 `question_page_store`에 반영한다.

```python
if state.get("question_output_valid", True):
    pending_page_state = state.get("pending_page_state")
    if pending_page_state is not None:
        self.question_page_store[state["session_id"]] = pending_page_state
```

재시도 후에도 누락이 남아 있으면 offset을 이동하지 않는다.

```text
누락 발생
-> 1회 재생성
-> 그래도 누락
-> 최종 답변에 검증 실패 안내
-> offset 유지
```

### 개선 효과

LLM이 일부 기출 문제를 출력하지 못해도 해당 문제가 조용히 사라지지 않는다.

사용자가 다시 `더 보여줘`를 입력하면 같은 페이지 범위가 다시 LLM에 전달되므로, 누락된 문제를 다시 볼 수 있다.

또한 검증 실패는 JSONL 로그로 남겨 나중에 어떤 질문과 어떤 문제 ID에서 실패했는지 확인할 수 있다.

수정 파일: `src/llm/rag_graph.py`

```text
logs/rag_validation_failures.jsonl
```

### 정리

LLM에게 "모두 출력하라"고 프롬프트로 지시하는 것만으로는 충분하지 않다. 기출 문제처럼 누락되면 사용자 학습 흐름이 깨지는 데이터는 코드가 기대 ID를 들고 있다가 실제 응답과 비교해야 한다. offset은 LLM 호출 시점이 아니라 검증 통과 시점에 이동해야 한다.

---

## 16. LLM이 GraphDB에 연결되지 않은 기출 문제 ID를 지어내 출력함

### 문제 상황

특정 단원에는 GraphDB 기준 연결 기출 문제가 7개뿐인데, 사용자가 `더 보여줘`를 반복하면 7개를 초과한 문제 ID가 출력되는 현상이 있었다.

예를 들어 `화이트박스 테스트의 검증 기준` 단원에서 다음과 같은 문제가 섞여 출력되었다.

```text
[문제 2020_3_12]
```

그러나 해당 문제는 실제로는 블랙박스 테스트 관련 문제였고, 현재 페이지 context에 포함된 문제도 아니었다.

즉, DB 연결 오류와 별개로 LLM이 허용되지 않은 문제 ID를 추가로 만들어 출력하는 hallucination이 발생했다.

### 원인

기존 검증은 "expected 문제 ID가 답변에 모두 들어있는지"만 확인했다.

```text
expected_question_ids = {2025_3_2, 2024_3_13, 2024_1_14}
```

답변이 아래처럼 나오면 기존 검증은 통과했다.

```text
2025_3_2
2024_3_13
2024_1_14
2020_3_12
```

이유는 expected ID 3개가 모두 포함되어 있기 때문이다.

하지만 실제로는 `2020_3_12`가 현재 페이지 context에 없는 허용 외 문제였으므로 실패로 처리해야 했다.

### 해결 방법

답변에서 `YYYY_R_N` 형태의 모든 문제 ID를 추출하는 검증을 추가했다.

수정 파일: `src/llm/rag_graph.py`

```python
def _extract_question_ids(self, response: str) -> list[str]:
    """답변에 등장한 `YYYY_R_N` 형태의 문제 ID를 등장 순서대로 추출합니다."""
    question_ids = re.findall(r"(?<![\w])\d{4}_\d+_\d+(?![\w])", response)
    return list(dict.fromkeys(question_ids))
```

추출한 actual ID와 현재 페이지의 expected ID를 비교한다.

```python
actual_question_ids = self._extract_question_ids(response)
expected_question_id_set = set(expected_question_ids)

unexpected_question_ids = [
    question_id
    for question_id in actual_question_ids
    if question_id not in expected_question_id_set
]
```

누락 ID와 허용 외 ID를 모두 검증 실패로 처리한다.

```python
is_valid = not missing_question_ids and not unexpected_question_ids
```

재시도 프롬프트도 확장했다.

수정 파일: `src/llm/prompts.py`

```text
[출력하면 안 되는 문제 ID]
2020_3_12

3. [학습 지식]에 없는 문제 ID나 [출력하면 안 되는 문제 ID]는 절대 출력하지 마십시오.
```

검증 실패 로그에는 실제 응답에서 추출된 문제 ID와 허용 외 문제 ID를 함께 기록한다.

```json
{
  "stage": "question_output",
  "missing_question_ids": [],
  "unexpected_question_ids": ["2020_3_12"],
  "actual_question_ids": ["2025_3_2", "2024_3_13", "2024_1_14", "2020_3_12"]
}
```

### 개선 효과

LLM이 GraphDB에 연결되지 않은 기출 문제를 임의로 추가해도 검증 단계에서 잡을 수 있다.

기출 문제 출력 범위가 현재 페이지 context로 제한되므로, 다음 문제가 줄어든다.

```text
연결 기출 수보다 더 많은 문제 출력
다른 단원의 문제 ID 혼입
출제 횟수와 연결 기출 수를 혼동해 문제를 추가 생성
```

재시도 후에도 허용 외 ID가 남아 있으면 offset을 이동하지 않으므로, 사용자는 같은 페이지 범위를 다시 요청할 수 있다.

### 정리

LLM 출력 검증은 "누락 방지"만으로는 부족하다. GraphRAG에서 주입한 context 밖의 문제 ID를 추가로 생성하는 경우도 검증해야 한다. expected ID와 actual ID를 집합 비교하면 기출 hallucination을 직접적으로 차단할 수 있다.
