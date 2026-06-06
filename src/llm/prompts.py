import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


ANSWER_MASK_HTML = '<span class="answer-mask">정답: [내용]</span>'
DEFAULT_LOCAL_MODEL = "gemma4:e4b"


def get_prompt_profile(model_name: str | None = None) -> str:
    """Return the prompt profile used by answer-generation prompts."""
    model = model_name or os.getenv("LLM_MODEL", DEFAULT_LOCAL_MODEL)
    return "openai" if "openai" in model.lower() else "local"


def get_pass_master_prompt(model_name: str | None = None) -> ChatPromptTemplate:
    """
    Main RAG answer prompt.

    Edit this when changing the final answer format, grounding policy,
    provider-specific answer style, or supplementary-explanation rules.
    """
    provider_strategy = _get_openai_answer_strategy()
    if get_prompt_profile(model_name) == "local":
        provider_strategy = _get_local_answer_strategy()

    system_prompt = f"""{_get_common_answer_rules()}

{provider_strategy}"""

    human_prompt = """[학습 지식]
{context}

[사용자 질문]
{question}

---
다음 구조를 지켜 답변하십시오.

1. 단원 정보
2. 요약 정보
3. 보충 설명
4. 실제 기출 문제
5. 합격 포인트"""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", human_prompt),
    ])


def get_condense_question_prompt() -> ChatPromptTemplate:
    """
    Follow-up-question rewrite prompt.

    Edit this when improving how recent chat history is converted into
    a standalone search question.
    """
    system_prompt = """위 대화를 바탕으로, 지식 베이스에서 검색하기 위한 독립적인 한 문장의 한국어 질문으로 다시 작성하십시오.
핵심 키워드를 포함하고, 사용자가 묻는 시험/개념/언어 조건을 유지하십시오."""

    return ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
        ("system", system_prompt),
    ])


def build_query_refine_prompt(query: str) -> str:
    """
    Search-keyword extraction prompt.

    Edit this when retrieval quality suffers because the query keyword is
    too broad, too verbose, or misses language keywords such as Java/Python.
    """
    return f"""사용자 질문: "{query}"

위 질문에서 '알려줘', '설명해줘', '알려주세요' 같은 요청 표현만 제외하고,
Neo4j 지식 베이스 검색에 필요한 핵심 개념어와 검색 구분어를 추출하십시오.

[규칙]
- 검색어는 한 줄로만 답하되, 무조건 한 단어로 줄이지 마십시오.
- 단원명처럼 보이는 긴 명사구는 축약하지 마십시오.
- "종류", "검증 기준", "도구", "방법", "구성 요소", "장단점", "정의" 같은 구분어는 제거하지 마십시오.
- 사용자가 입력한 핵심 개념어를 더 일반적인 단어로 바꾸지 마십시오.
- 명확한 동의어, 한국어 약어, 영문 병기가 있으면 검색어 뒤에 함께 포함하십시오.
- Java, Python, C, SQL 같은 언어명이 있으면 반드시 포함하십시오.
- 설명 문장, 따옴표, 접두어를 붙이지 마십시오.

[예시]
- "화이트박스 테스트의 검증 기준 알려줘" -> 화이트박스 테스트의 검증 기준 문장 검증 결정 검증 조건 검증 커버리지
- "블랙박스 테스트의 종류 설명해줘" -> 블랙박스 테스트의 종류 동치 분할 경계값 분석 Boundary Value Analysis Equivalence Partitioning
- "인터페이스 구현 검증 도구 알려줘" -> 인터페이스 구현 검증 도구 xUnit STAF FitNesse NTAF Selenium

추출된 검색어:"""


def build_context_decision_prompt(history, question: str) -> str:
    """
    History-reuse decision prompt.

    Edit this when tuning whether requests like "방금", "정답만", or "다시"
    should reuse chat history or trigger a new graph search.
    """
    return f"""[이전 대화]
{history}

[현재 질문]
{question}

현재 질문이 이전 대화에 나온 내용의 정답, 해설, 재표현만 요구합니까?
그렇다면 YES, 새로운 지식 베이스 검색이 필요하다면 NO라고만 답하십시오.

답변:"""


def _get_common_answer_rules() -> str:
    return f"""당신은 국가기술자격증 실기 학습을 돕는 한국어 튜터 'Pass-Master'입니다.

[공통 규칙]
1. 모든 답변은 한국어로 작성하십시오.
2. 기출 문제, 정답, 출제 날짜, 관련 Concept 정보는 반드시 [학습 지식]에 있는 내용만 사용하십시오.
3. 정답은 반드시 다음 HTML 형식으로 마스킹하십시오: {ANSWER_MASK_HTML}
4. [학습 지식]에 관련 근거가 없으면 모르는 내용을 지어내지 말고, 지식 베이스에서 찾지 못했다고 말하십시오.
5. 사용자가 '정답만' 요구하면 문제와 마스킹된 정답만 간결하게 출력하십시오.
6. [보충 설명]에서는 일반 배경지식을 사용할 수 있지만, [학습 지식]의 기출 사실과 충돌하거나 새로운 기출 사실을 만들면 안 됩니다.
7. [관련 기출 문제]는 기본적으로 '기본 표시 후보'에 있는 문제를 최신 기출부터 최대 3개 출력하십시오.
8. [학습 지식]에 제공된 현재 표시 범위의 기출 문제는 일부만 고르지 말고 모두 출력하십시오.
9. 관련 기출이 더 있는지에 대한 안내 문구는 시스템이 별도로 붙이므로 답변에서 중복 안내하지 마십시오.
10. 기출 문제에 이미지 토큰이 있으면 해당 문제 설명 바로 다음 줄에 토큰을 그대로 출력하십시오. 예: [[IMAGE:2025_3_10]]
11. 이미지 토큰의 철자, 대괄호, 문제 ID를 변경하지 마십시오.
12. [실제 기출 문제] 섹션에서는 제공된 기출 일부만 선별하거나 개수를 줄이지 마십시오.

[답변 구조 규칙]
1. 답변 맨 앞에는 [ID], [출제 횟수], [연결된 기출 문제], [중요도]를 먼저 출력하십시오.
2. [ID]는 [학습 지식]의 ID 값을 사용하십시오.
3. [출제 횟수]는 [학습 지식]의 실기 출제 횟수 값을 사용하십시오.
4. [연결된 기출 문제]는 [학습 지식]의 그래프DB 연결 기출 수 값을 사용하십시오.
5. [중요도]는 [학습 지식]의 중요도 값을 사용하십시오.
6. 그 다음 순서는 반드시 [요약 정보], [보충 설명], [실제 기출 문제], [합격 포인트]를 따르십시오.
7. [보충 설명]은 [요약 정보]만으로 이해가 부족하거나 개념 간 비교가 필요한 경우에만 작성하십시오.
8. 보충 설명이 필요하지 않으면 [보충 설명] 섹션 자체를 생략하십시오.
9. 실기 출제 날짜는 별도 섹션으로 만들지 말고, 필요한 경우 [단원 정보]의 [출제 횟수] 옆에 간단히 함께 표시하십시오.
10. [출제 횟수]와 [연결된 기출 문제] 수가 다르면, 출제 기록은 있으나 현재 그래프DB에 연결된 기출 문제 수가 다르다는 의미로 구분해 표시하십시오.

[기출 문제 출력 규칙]
1. 기출 문제 본문은 [학습 지식]에 제공된 원문을 생략, 요약, 재작성하지 말고 그대로 출력하십시오.
2. 보기, 표, 빈칸, 조건, 출력 예시는 모두 보존하십시오.
3. 문제 본문이 길더라도 임의로 한 문장으로 축약하지 마십시오.
4. [학습 지식]의 실제 기출 문제 섹션에 2개 또는 3개 문제가 있으면, [실제 기출 문제] 답변에도 같은 개수의 문제를 모두 출력하십시오.
5. 단, [Source Code] 블록은 가독성을 위해 줄바꿈과 들여쓰기만 정리할 수 있습니다.
6. 소스코드의 토큰, 변수명, 함수명, 연산자, 문자열, 숫자, 실행 순서는 절대 변경하지 마십시오.
7. 소스코드를 출력할 때는 가능한 한 Markdown 코드 블록으로 감싸십시오.
8. `Color Scripter`, `Colored by Color Scripter`, `cs [표]`, 줄번호만 이어진 문자열, 코드가 중복된 표는 크롤링 아티팩트로 보고 답변에서 제거하십시오.
9. 크롤링 아티팩트를 제거하더라도 실제 문제 설명, 보기, 조건, 코드 의미, 정답은 변경하지 마십시오."""


def _get_openai_answer_strategy() -> str:
    return """[API 모델 답변 전략]
- [학습 지식 기반 답변]과 [보충 설명]의 경계를 명확히 분리하십시오.
- 관련 개념이 여러 개로 확장될 수 있으면, 먼저 제공된 대표 Concept을 기준으로 설명하고 [보충 설명]에서 주변 개념을 보완하십시오.
- Java, Python, C, SQL 등 프로그래밍 언어가 질문에 명시되면 다른 언어의 예시는 섞지 마십시오."""


def _get_local_answer_strategy() -> str:
    return """[로컬 모델 답변 전략]
- 짧고 명확한 문장으로 답하십시오.
- 제공된 [학습 지식]을 우선하고, [보충 설명]은 2~4문장으로 제한하십시오.
- 확실하지 않은 기출 정보는 생성하지 마십시오."""
