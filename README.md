# Pass-Master-GraphRAG
### uv 사용자
uv sync

### pip 사용자
pip install -r requirements.txt

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

## 📝 트러블슈팅
#### **"중첩 구조의 데이터 손실 방지"**
- 현상: 부모 태그(Question)와 자식 태그(Answer)가 계층 구조로 얽혀 있어, 방문 처리(Visited Check) 시 하위 데이터가 누락되는 현상 발생.
- 해결: 성능 차이가 미미한 수준(O(N) 유지)이므로, 엄격한 순차 방문 대신 **계층적 재탐색을 허용**하여 데이터 추출의 완전성을 확보함.
