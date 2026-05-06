import streamlit as st
from src.llm.rag_chain import PassMasterChain
import time

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
    </style>
    """, unsafe_allow_html=True)

# 2. 체인 인스턴스 초기화 (세션 스테이트 활용하여 1회만 로드)
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 Pass-Master 엔진 로드 중..."):
        st.session_state.rag_chain = PassMasterChain()
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
        st.markdown(message["content"], unsafe_allow_html=True)

# 6. 사용자 입력 및 추론
if prompt := st.chat_input("질문을 입력하세요 (예: 프로토콜 3요소 알려줘)"):
    # UI에 사용자 질문 즉시 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass-Master 답변 생성
    with st.chat_message("assistant"):
        # st.status를 사용하여 단계별 문구 변경
        with st.status("🚀 분석 시작...", expanded=True) as status:
            st.write("🔍 bge-m3 모델로 관련 지식을 검색하고 있습니다...")
            # 실제 내부 로직과 동기화 전까지는 시각적 피드백을 위해 status.update 활용 가능
            # 예: response = st.session_state.rag_chain.run(prompt, status=status) 
            
            # 아래는 UI 흐름을 보여주기 위한 시뮬레이션 문구 업데이트입니다.
            # 실제 PassMasterChain.run()이 내부에서 status를 제어하도록 수정하면 더욱 정확합니다.
            response = st.session_state.rag_chain.run(prompt) 
            
            status.update(label="✅ 분석 완료!", state="complete", expanded=False)
        # LLM 답변 내 HTML 태그가 동작하도록 unsafe_allow_html 적용
        st.markdown(response, unsafe_allow_html=True)
            
    st.session_state.messages.append({"role": "assistant", "content": response})