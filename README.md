# 🎓 Pass-Master-GraphRAG
> GraphRAG 기반 정보처리기사 자격증 실기 AI 튜터

**Pass-Master-GraphRAG**는 정보처리기사 실기 학습을 돕는 로컬 기반의 **GraphRAG** 시스템입니다. 단순한 문서 검색을 넘어, **Neo4j** 지식 그래프를 통해 개념 간의 유기적 관계를 파악하고 사용자에게 정확한 기출 데이터와 학습 맥락을 제공합니다.

---

## 🚀 Key Features

- **Graph-Centric Retrieval**: Neo4j Graph DB를 활용하여 단편적인 텍스트가 아닌, 개념(Entity)과 관계(Relation) 중심의 심층 답변을 생성합니다.
- **On-Premise Optimization**: NVIDIA RTX 4070(12GB VRAM) 환경에 최적화된 로컬 추론 엔진(Ollama)을 사용하여 데이터 보안과 빠른 응답성을 확보했습니다.
- **Intelligent Hybrid Routing**: 사용자의 입력 키워드(방금, 이거 등)를 분석하여 '이전 대화 활용'과 '신규 검색' 사이의 최적의 경로를 결정합니다.
- **Semantic Noise Cleaning**: 검색 전 LLM이 불용어(Stopwords)를 제거하고 핵심 전문 용어(명사)만 정제하여 벡터 검색의 정확도를 극대화합니다.
- **Interactive Answer Masking**: Streamlit UI에서 정답을 검은색 박스로 마스킹 처리하여, 사용자가 드래그를 통해 정답을 확인하는 능동적 학습 기능을 지원합니다.

---

## 🛠 Tech Stack

### 🔹 Language & Environment
* **Python 3.14.0+**: 메인 개발 언어 및 데이터 전처리 엔진
* **python-dotenv**: 보안을 위한 환경 변수 및 API Key 관리
* **pdfplumber**: PDF 텍스트 추출 및 데이터 구조화
* **uv**: 초고속 패키지 관리 및 일관된 가상환경 보장

### 🔹 AI & Data Pipeline
* **LLM (Ollama - Llama3-8B)**: 로컬 환경에서 구동되는 고성능 추론 모델
* **Embedding (HuggingFace/TEI)**: 지식 벡터화를 위한 로컬 임베딩 엔진
* **Graph DB (Neo4j)**: 개념 간의 관계형 지식 저장 및 복합 검색(Graph Search)
* **Orchestration (LangChain)**: LCEL을 활용한 지능형 RAG Chain 설계

### 🔹 Deployment & UI
* **Frontend (Streamlit)**: 마스킹 기능과 실시간 스트리밍 답변을 지원하는 웹 인터페이스
* **Version Control**: Git / GitHub를 통한 브랜치 기반 데이터 격리 및 형상 관리

---

## 🔄 System Workflow

1. **User Input**: 사용자가 질문을 입력합니다.
2. **Hard Logic Routing**: 입력어 내 특정 키워드(방금, 다시 등)를 검사하여 대화 맥락 활용 여부를 1차 판정합니다.
3. **Query Refinement**: 검색이 결정되면 LLM이 불용어(Stopwords)를 제거하고 검색용 핵심 키워드를 정제합니다.
4. **Graph-Vector Retrieval**: 정제된 키워드로 벡터 유사도 검색과 그래프 관계 탐색을 병행하여 Neo4j에서 관련 개념과 기출문제를 추출합니다.
5. **Answer Generation**: 추출된 지식과 필요시 대화 이력을 결합하여 'Pass-Master' 페르소나로 답변을 생성합니다.
6. **UI Rendering**: 정답 마스킹 처리된 답변을 Streamlit 화면에 송출합니다.

---

## 💻 Installation & Setup

### Prerequisites
- uv (권장)
- Docker

### Setup
```bash
### uv 사용자
uv sync

### pip 사용자
pip install -r requirements.txt
```

### Step 1: Docker Compose 실행
```bash
docker-compose up -d
```

모델 다운로드 및 neo4j 실행에 시간이 걸립니다.
neo4j 상태 로그 확인

```bash
docker-compose logs -f neo4j --tail 12
```

### Step 2: 모델 로드 (최초 1회)
```bash
docker exec -it ollama ollama pull llama3
```

### Step 3: 앱 실행
```bash
uv run streamlit run app.py
# 또는 가상환경 실행 후 실행
source .venv/Scripts/activate
streamlit run app.py
```
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
```bash
# 실행 중인 서비스 상태
docker-compose ps
# 로그
docker-compose logs
# 재시작
docker-compose restart
```

---

## 📝 Configuration (.env)
프로젝트 루트에 .env 파일을 생성하고 다음 정보를 설정해야 합니다.

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
LLM_MODEL=llama3:latest
OLLAMA_HOST=http://localhost:11434
```
---

## 🔗 neo4j Browser
http://localhost:7474/browser/

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

---
---

## 변경점
uv, docker, 온프레미스(하이브리드)

## 프로젝트 마일스톤
- langchain
- langgraph를 사용할 것(분기와 조건을 이용할 것)
- 이전 질문 기억하는 기능 (2~3개 정도)
- bm25 키워드 빈도 기반 가중치

## 결과 검증 방법
Rouge, BLEU
