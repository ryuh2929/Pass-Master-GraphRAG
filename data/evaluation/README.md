# Question-Concept Evaluation Set

문제-Concept 연결 알고리즘을 비교하기 위한 수동 검증 평가셋입니다.

## 파일

- `question_concept_gold.seed.json`: 초기 seed 평가셋입니다. `review_status`가 `confirmed`인 항목만 기본 평가에 사용합니다.

## 필드

- `problem_id`: 기출 문제 ID입니다. 예: `2025_1_15`
- `primary_concept`: 사람이 판단한 가장 중요한 Concept ID입니다.
- `secondary_concepts`: 복합 문제에서 함께 관련될 수 있는 Concept ID 목록입니다.
- `labels`: 평가와 분석에 사용할 태그입니다.
- `review_status`: `confirmed` 또는 `needs_review`입니다.
- `note`: 판단 근거 또는 추가 확인 사항입니다.

## 평가 기준

- `primary_concept`가 현재 DB의 `VERIFIED_MENTIONS` 연결에 있으면 정답으로 봅니다.
- `primary_concept`는 없고 `secondary_concepts`만 맞으면 `secondary_only`로 분리합니다.
- 연결이 없으면 `missing_link`, 문제 노드가 없으면 `missing_question`으로 분리합니다.
- `needs_review` 항목은 사람이 확정하기 전까지 기본 정확도 계산에서 제외합니다.

## 실행 방법

현재 Neo4j에 저장된 `VERIFIED_MENTIONS` 연결을 기준으로 baseline을 측정합니다.

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
