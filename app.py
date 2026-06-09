from pathlib import Path
import base64
import mimetypes
import os
import re

import streamlit as st
from src.llm.rag_chain import PassMasterChain
from src.llm.rag_graph import PassMasterGraphChain

# 1. 페이지 설정 및 다크모드 최적화 커스텀 CSS
st.set_page_config(page_title="Pass-Master: 자격증 합격 튜터", layout="wide")

st.markdown("""
    <style>
    .answer-mask {
        color: black;
        background-color: black;
        border-radius: 4px;
        padding: 2px 4px;
        cursor: help;
    }
    .answer-mask:hover {
        /* 드래그를 유도하기 위한 힌트: 호버 시 살짝 투명하게 하거나 유지 */
        background-color: #222;
    }
    .stChatMessage {
        border-radius: 10px;
    }
    .question-image {
        display: block;
        max-height: 420px;
        max-width: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
        margin: 0.5rem 0 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_local_image(image_path: str, max_height: int = 420):
    """로컬 이미지를 max-height 기준으로 답변 안에 렌더링합니다."""
    path = Path(image_path)
    if not path.exists():
        return

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        (
            f'<img class="question-image" '
            f'style="max-height: {max_height}px;" '
            f'src="data:{mime_type};base64,{encoded_image}" />'
        ),
        unsafe_allow_html=True,
    )


def render_answer(content: str, images_by_question: dict, image_max_height: int = 420):
    """답변 Markdown을 출력하고 [[IMAGE:question_id]] 위치에 이미지를 삽입합니다."""
    if not isinstance(images_by_question, dict):
        images_by_question = {}

    token_pattern = re.compile(r"\[\[IMAGE:([^\]]+)\]\]")
    cursor = 0

    for match in token_pattern.finditer(content):
        before_token = content[cursor:match.start()]
        if before_token.strip():
            st.markdown(before_token, unsafe_allow_html=True)

        question_id = match.group(1).strip()
        for image_path in images_by_question.get(question_id, []):
            render_local_image(image_path, max_height=image_max_height)

        cursor = match.end()

    remaining_content = content[cursor:]
    if remaining_content.strip():
        st.markdown(remaining_content, unsafe_allow_html=True)

# 2. 체인 인스턴스 초기화 (세션 스테이트 활용하여 1회만 로드)
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 Pass-Master 엔진 로드 중..."):
        # LangGraph 체인을 기본값으로 사용합니다.
        # 기존 LangChain 체인으로 되돌려 비교해야 할 때만 USE_LANGGRAPH=false를 지정합니다.
        if os.getenv("USE_LANGGRAPH", "true").lower() in {"0", "false", "no"}:
            st.session_state.rag_chain = PassMasterChain()
        else:
            st.session_state.rag_chain = PassMasterGraphChain()
        st.session_state.messages = [] # 대화 기록 저장용

# 3. 사이드바 - 대화 관리
with st.sidebar:
    st.title("🛠 Pass-Master 설정")
    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        # 실제 체인 내부의 history_store는 session_id를 통해 관리되므로 
        # 여기서는 UI 표시용 메시지만 초기화하거나 session_id를 갱신합니다.
        st.rerun()
    
    st.info("💡 **Tip**: 정답이 보이지 않을 때는 검은색 박스를 마우스로 드래그하세요!")

# 4. 메인 화면 구성
st.title("🎓 Pass-Master : 자격증 합격 가이드")
st.caption("ADsP, 정보처리기사 등 국가기술자격증 데이터를 기반으로 답변합니다.")

# 5. 기존 대화 로그 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_answer(message["content"], message.get("images", {}))

# 6. 사용자 입력 및 추론
if prompt := st.chat_input("질문을 입력하세요 (예: 프로토콜 3요소 알려줘)"):
    # UI에 사용자 질문 즉시 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass-Master 답변 생성
    with st.chat_message("assistant"):
        response = "" # 결과값 담을 변수
        images_by_question = {}
        status_placeholder = st.empty()

        with status_placeholder.container():
            with st.status("🚀 분석 준비 중...", expanded=True) as status:
                stream = st.session_state.rag_chain.run_stream(prompt)

                for event in stream:
                    if isinstance(event, dict) and event.get("type") == "status":
                        status.update(label=event["message"])
                    elif isinstance(event, dict) and event.get("type") == "answer":
                        response = event["content"]
                        images_by_question = event.get("images", {})
                    else:
                        response = event

                status.update(label="✅ 분석 완료!", state="complete", expanded=False)

        status_placeholder.empty()
        
        # 3. 최종 결과 화면 출력
        render_answer(response, images_by_question)
            
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "images": images_by_question
    })
