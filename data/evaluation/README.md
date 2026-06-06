# Question-Concept Evaluation Set

문제-Concept 연결 알고리즘을 비교하기 위한 수동 검증 평가셋이다.

## 파일

- `question_concept_gold.seed.json`: 초기 seed 평가셋. `review_status`가 `confirmed`인 항목부터 평가에 사용한다.

## 필드

- `problem_id`: 기출 문제 ID. 예: `2025_1_15`
- `primary_concept`: 사람이 판단한 대표 Concept ID
- `secondary_concepts`: 복합 문제에서 함께 관련 있는 Concept ID 목록
- `labels`: 평가/분석용 태그
- `review_status`: `confirmed` 또는 `needs_review`
- `note`: 판단 근거 또는 추가 확인 사항

## 평가 기준

- `primary_concept`가 1순위 예측과 같으면 primary hit
- `primary_concept`가 top-k 후보에 있으면 top-k hit
- `secondary_concepts`는 복합 문제 후보 탐지 여부를 확인하는 데 사용한다.
- `needs_review` 항목은 사람이 확정하기 전까지 최종 정확도 계산에서 제외한다.
