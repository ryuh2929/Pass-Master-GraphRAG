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
- [uv](https://github.com/astral-sh/uv) (권장)
- [Docker](https://docs.docker.com/desktop/setup/install/windows-install/)

### ⚙️ Step 1: Setup
```bash
### uv 사용자
uv sync

### pip 사용자
pip install -r requirements.txt
```

### 🐳 Step 2: Docker Compose 실행
```bash
# openai 이용시
docker-compose up -d

# local 환경에서 ollama 이용시
docker-compose --profile local up -d
```

모델 다운로드 및 neo4j 실행에 시간이 걸립니다.
neo4j 상태 로그 확인

```bash
docker-compose logs -f neo4j --tail 12
```

### 🧠 Step 3: 모델 수동 로드 (도커를 실행했을 때 모델이 자동으로 다운되지 않았을 경우 실행)
```bash
docker exec -it ollama ollama pull llama3
```

### 🔑 Step 4: 환경 변수 설정 (.env)
프로젝트 루트에 .env 파일을 생성하고 다음 정보를 설정 (.env.example 수정 후 파일명 변경)

```
# llama3:latest | openai 중 선택
LLM_MODEL=llama3:latest
OLLAMA_HOST=http://localhost:11434

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# OpenAI (LLM_MODEL=openai 일 때만 필요)
OPENAI_API_KEY=your_api_key_here
```

### 🐉 Step 5: 앱 실행
```bash
### uv 사용자
uv run streamlit run app.py

### pip 사용자일 경우 (가상환경 실행 후 실행)
source .venv/Scripts/activate
streamlit run app.py
```

---

### 🐋 Docker 명령어
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
# 실행 중인 서비스 상태 확인
docker-compose ps
# 로그 확인
docker-compose logs
# 재시작
docker-compose restart
```

### 🕸️ neo4j 명령어
초기화
```
MATCH (n) DETACH DELETE n;
```
---

## 🔗 neo4j Browser 주소
http://localhost:7474/browser/

---

## 📝 트러블슈팅
#### **"중첩 구조의 데이터 손실 방지"**
- 현상: 크롤링 과정에서 웹 페이지의 부모 태그(Question)와 자식 태그(Answer)가 계층 구조로 얽혀 있어, **방문 처리(Visited Check) 시 하위 데이터가 누락**되는 현상 발생
- 해결: 성능 차이가 미미한 수준(O(N) 유지)이므로, 엄격한 순차 방문 대신 **계층적 재탐색을 허용**하여 데이터 추출의 완전성 확보

#### **"불용어로 인한 의미 왜곡 문제 해결을 위한 LLM 전처리 적용"**
- 현상: '알려줘' 같은 **불용어(Stopwords) 임베딩 벡터와 유사한 데이터가 답변으로 출력**되어 핵심 키워드에 관한 내용이 아닌 엉뚱한 답변이 출력되는 현상 발생
- 해결: 질문 입력 전에 **LLM을 통해 불용어 제거**하여 쿼리를 정제하는 단계 추가

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
- On-Premise 환경과 OpenAI-API 이용 환경 선택 가능(API 이용시 로컬 모델 다운로드 X)
- 이전 질문 기억하는 기능 (2~3개 정도)
- bm25 키워드 빈도 기반 가중치

## 결과 검증 방법 (모델 차이, 개선시 향상도)
Rouge, BLEU

## feat/openai-llm-switching 브랜치
llm_factory.py에서 .env의 llm 모델 (openai, ollama) 선택에 따라 llm 모델 불러오기
rag_chain 수정 (쓸모없어진 코드 나중에 정리)

docker-compose.yml에 profiles 추가 (profiles 있는 서비스는 명시적으로 호출할 때만 켜짐)

## 현재 JAVA 코드 문제가 Python 단원이랑 연결되어 있는 문제가 있음
두 노드의 연결에 관여하는 임베딩 모델의 벡터화 과정 문제로 bm25 기반으로 키워드 일치 여부와 키워드 빈도수에 따른 가중치 로직을 추가해서 해결할 것
- 전략 A: 하이브리드 가중치 조절 (추천)
BM25 점수와 벡터 점수를 7:3 정도로 섞으면, 특정 키워드(JAVA, Python, SQL 등)가 명시된 경우 엉뚱한 단원으로 튀는 것을 막을 수 있습니다.
- 전략 B: 하드 필터링 (Hard Filtering)
데이터 전처리 단계에서 Chapter 이름이나 Section 제목에 포함된 키워드를 추출하여, 문제 텍스트에 해당 키워드가 없을 경우 유사도 점수를 깎거나 제외하는 로직을 Cypher에 넣는 방식입니다.
- 전략 C: 프롬프트 앤서링 단계에서의 검증
RAG 답변 생성 시 LLM에게 다음과 같은 지침을 줍니다.
"참조된 컨텍스트 중 사용자의 질문과 프로그래밍 언어가 다른 정보는 무시하고 답변하세요."

## LLM 모델 변경
한국어 성능 문제, 토큰 생성 속도 문제

llama3 -> exaone3.5:7.8b -> gemma4:e4b

## Streamlit 페이지에서 로딩 문구 출력
콜백? generator + yeild? st.session_state + rerun?
