# Pass-Master-GraphRAG
### uv 사용자
uv sync

### pip 사용자
pip install -r requirements.txt

---

### Docker 명령어
시작
```
docker-compose up
```
중지
```
docker-compose down
```
이미지 다운
```
docker-compose pull
```
기타
```
# 실행 중인 서비스 상태
docker-compose ps
# 로그
docker-compose logs
# 재시작
docker-compose restart
```

---

## 📝 트러블슈팅
#### **"중첩 구조의 데이터 손실 방지"**
- 현상: 부모 태그(Question)와 자식 태그(Answer)가 계층 구조로 얽혀 있어, 방문 처리(Visited Check) 시 하위 데이터가 누락되는 현상 발생.
- 해결: 성능 차이가 미미한 수준(O(N) 유지)이므로, 엄격한 순차 방문 대신 **계층적 재탐색을 허용**하여 데이터 추출의 완전성을 확보함.

---
## 모델별 성능 비교
#### BGE-M3 임베딩 모델 의미적 연결률
364/380 (95.8%)

- 멀티모달의 부재: "이미지(도표, 그래프)가 포함된 기출문제의 경우, 텍스트 임베딩만으로는 Concept 노드와의 의미적 연결에 한계가 있음."

- 날짜 기반 필터링의 엄격성: "유사도는 높으나 실기 출제 이력(practical_dates) 검증을 통과하지 못한 사례를 통해 데이터 정합성을 확보함."