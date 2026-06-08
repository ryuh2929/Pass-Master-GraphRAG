# Evaluation Datasets

GraphRAG 검색과 Question-Concept 연결 품질을 비교하기 위한 수동 검증 평가셋입니다.

## Files

- `question_concept_gold.seed.json`: 문제와 정답 Concept 연결을 검증하는 평가셋입니다.
- `question_language_gold.seed.json`: 코드/SQL 문제의 언어 힌트 추출을 검증하는 평가셋입니다.

## question_concept_gold.seed.json

- `problem_id`: 기출 문제 ID입니다. 예: `2025_1_15`
- `primary_concept`: 사람이 판단한 정답 Concept ID입니다.
- `labels`: 평가와 분석에 사용할 태그입니다.
- `review_status`: `confirmed` 또는 `needs_review`입니다.
- `note`: 판단 근거 또는 추가 확인 사항입니다.

## question_language_gold.seed.json

- `problem_id`: 기출 문제 ID입니다.
- `practical_date`: 실기 출제 날짜입니다. 예: `25.11`
- `expected_language`: 정답 언어 힌트입니다. `sql`, `c`, `java`, `python` 중 하나입니다.
- `source`: 언어명이 문제에 직접 나온 경우 `explicit`, 코드 형태로 판단한 경우 `code_marker`입니다.
- `has_malloc`: 문제 코드에 `malloc`이 등장하는지 여부입니다.
- `has_free`: 문제 코드에 `free`가 등장하는지 여부입니다.
- `review_status`: 현재는 전체 코드 문제를 `confirmed`로 둡니다.
- `evidence`: 검토 편의를 위한 문제 원문 앞부분입니다.

## Run

현재 Neo4j에 저장된 `VERIFIED_MENTIONS` 연결을 기준으로 Concept 연결 baseline을 측정합니다.

```powershell
uv run python src\evaluation\evaluate_linking.py
```

로컬 `uv` 캐시 오류가 나면 프로젝트 내부 캐시를 지정해서 실행합니다.

```powershell
$env:UV_CACHE_DIR=".uv-cache"; uv run python src\evaluation\evaluate_linking.py
```

검토 중인 샘플까지 포함하려면 다음 옵션을 사용합니다.

```powershell
uv run python src\evaluation\evaluate_linking.py --include-needs-review
```

정답 처리된 케이스까지 모두 보고 싶으면 다음 옵션을 사용합니다.

```powershell
uv run python src\evaluation\evaluate_linking.py --show-correct
```
