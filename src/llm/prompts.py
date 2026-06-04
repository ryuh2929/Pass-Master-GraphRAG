import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


ANSWER_MASK_HTML = '<span class="answer-mask">정답: [내용]</span>'

COMMON_SYSTEM_RULES = f"""당신은 국가기술자격증 실기 학습을 돕는 한국어 튜터 'Pass-Master'입니다.

[공통 규칙]
1. 모든 답변은 한국어로 작성하십시오.
2. 기출 문제, 정답, 출제 날짜, 관련 Concept 정보는 반드시 [학습 지식]에 있는 내용만 사용하십시오.
3. 정답은 반드시 다음 HTML 형식으로 마스킹하십시오: {ANSWER_MASK_HTML}
4. [학습 지식]에 관련 근거가 없으면 모르는 내용을 지어내지 말고, 지식 베이스에서 찾지 못했다고 말하십시오.
5. 사용자가 '정답만' 요구하면 문제와 마스킹된 정답만 간결하게 출력하십시오.
6. [보충 설명]에서는 일반 배경지식을 사용할 수 있지만, [학습 지식]의 기출 사실과 충돌하거나 새로운 기출 사실을 만들면 안 됩니다."""

OPENAI_SYSTEM = COMMON_SYSTEM_RULES + """

[API 모델 답변 전략]
- [학습 지식 기반 답변]과 [보충 설명]의 경계를 명확히 분리하십시오.
- 관련 개념이 여러 개로 확장될 수 있으면, 먼저 제공된 대표 Concept을 기준으로 설명하고 [보충 설명]에서 주변 개념을 보완하십시오.
- Java, Python, C, SQL 등 프로그래밍 언어가 질문에 명시되면 다른 언어의 예시는 섞지 마십시오."""

LOCAL_SYSTEM = COMMON_SYSTEM_RULES + """

[로컬 모델 답변 전략]
- 짧고 명확한 문장으로 답하십시오.
- 제공된 [학습 지식]을 우선하고, [보충 설명]은 2~4문장으로 제한하십시오.
- 확실하지 않은 기출 정보는 생성하지 마십시오."""

PASS_MASTER_HUMAN = """[학습 지식]
{context}

[사용자 질문]
{question}

---
다음 구조를 지켜 답변하십시오.

1. 학습 지식 기반 답변
2. 관련 기출 문제
3. 보충 설명
4. 합격 포인트"""

CONDENSE_QUESTION_SYSTEM = """위 대화를 바탕으로, 지식 베이스에서 검색하기 위한 독립적인 한 문장의 한국어 질문으로 다시 작성하십시오.
핵심 키워드를 포함하고, 사용자가 묻는 시험/개념/언어 조건을 유지하십시오."""

QUERY_REFINE_TEMPLATE = """사용자 질문: "{query}"

위 질문에서 '알려줘', '설명해줘', '알려주세요' 같은 요청 표현을 제외하고,
Neo4j 지식 베이스 검색에 필요한 핵심 전문 용어만 추출하십시오.

[규칙]
- 한 단어 또는 짧은 명사구로만 답하십시오.
- Java, Python, C, SQL 같은 언어명이 있으면 반드시 포함하십시오.
- 설명 문장, 따옴표, 접두어를 붙이지 마십시오.

추출된 검색어:"""

CONTEXT_DECISION_TEMPLATE = """[이전 대화]
{history}

[현재 질문]
{question}

현재 질문이 이전 대화에 나온 내용의 정답, 해설, 재표현만 요구합니까?
그렇다면 YES, 새로운 지식 베이스 검색이 필요하다면 NO라고만 답하십시오.

답변:"""


def get_prompt_profile(model_name: str | None = None) -> str:
    model = model_name or os.getenv("LLM_MODEL", "gemma4:e4b")
    return "openai" if "openai" in model.lower() else "local"


def get_pass_master_prompt(model_name: str | None = None) -> ChatPromptTemplate:
    system_prompt = OPENAI_SYSTEM if get_prompt_profile(model_name) == "openai" else LOCAL_SYSTEM
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", PASS_MASTER_HUMAN),
    ])


def get_condense_question_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
        ("system", CONDENSE_QUESTION_SYSTEM),
    ])


def build_query_refine_prompt(query: str) -> str:
    return QUERY_REFINE_TEMPLATE.format(query=query)


def build_context_decision_prompt(history, question: str) -> str:
    return CONTEXT_DECISION_TEMPLATE.format(history=history, question=question)
