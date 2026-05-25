import os
import joblib

import pandas as pd
import streamlit as st

MODEL_PATH = "tomato.pkl"

st.set_page_config(
    page_title="토마토 착과율 예측",
    page_icon="🍅",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #fff8f0 0%, #f7f0f5 100%);
        color: #333333;
    }
    .stApp {
        background-image: linear-gradient(135deg, #ffe3d5 0%, #fff7f0 65%, #f2f0ff 100%);
    }
    .stButton>button {
        background-color: #e63946;
        color: white;
        font-weight: 600;
        border-radius: 999px;
        padding: 0.8rem 1.8rem;
    }
    .stButton>button:hover {
        background-color: #d62839;
    }
    .stMetric {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(255,255,255,0.85);
    }
    .css-1y0tads {
        background: rgba(255,255,255,0.75);
        box-shadow: 0 20px 45px rgba(0,0,0,0.1);
        border-radius: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            "모델 파일을 불러오는 중 문제가 발생했습니다. "
            "`tomato.pkl`이 sklearn joblib 형식인지 확인해주세요."
        ) from exc

st.header("🍅 토마토 착과율 예측기")
st.write(
    "환경 데이터를 입력하면 학습된 랜덤포레스트 모델을 통해 착과율을 빠르게 예측합니다."
)

with st.expander("사용 방법 안내", expanded=True):
    st.write(
        "1. 내부온도, 내부습도, 지온 값을 입력하세요.\n"
        "2. `예측 실행` 버튼을 누르면 착과율을 예측합니다.\n"
        "3. 결과는 소수점 1자리로 표시됩니다."
    )

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("입력값")
    with st.form(key="prediction_form"):
        temp = st.number_input("내부온도 (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        humidity = st.number_input("내부습도 (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        soil_temp = st.number_input("지온 (°C)", min_value=0.0, max_value=60.0, value=20.0, step=0.1)
        submit_button = st.form_submit_button("예측 실행")

with col2:
    st.subheader("모델 정보")
    st.write("학습된 랜덤포레스트 모델을 사용하여 착과율을 예측합니다.")
    st.markdown(
        "- 내부온도: 식물 생장에 중요한 요소\n"
        "- 내부습도: 수분 스트레스 예측에 유용\n"
        "- 지온: 뿌리 온도 상태 반영"
    )
    st.info("`tomato.pkl` 파일을 앱과 동일한 폴더에 두어야 합니다.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"모델 파일을 찾을 수 없습니다: `{MODEL_PATH}`\n"
        "같은 폴더에 `tomato.pkl`를 놓고 다시 실행하세요."
    )
else:
    try:
        rf_model = load_model(MODEL_PATH)
    except RuntimeError as exc:
        st.error(str(exc))
    else:
        if submit_button:
            input_data = pd.DataFrame(
                [[temp, humidity, soil_temp]],
                columns=["내부온도", "내부습도", "지온"],
            )
            predicted = rf_model.predict(input_data)
            score = float(predicted[0])

            st.metric(label="예측 착과율", value=f"{score:.1f}%")
            with st.expander("상세 결과 보기", expanded=True):
                st.write("입력 데이터")
                st.dataframe(input_data)
                st.write("모델 예측값")
                st.write(f"**{score:.1f}%**")
