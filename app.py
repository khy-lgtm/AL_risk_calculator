import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# =========================================================
# 상수 정의
# =========================================================
HIGH_RISK_THRESHOLD   = 0.30
MEDIUM_RISK_THRESHOLD = 0.15

# =========================================================
# ToString 클래스
# =========================================================
class ToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):   return X.astype(str)
    def get_feature_names_out(self, input_features=None): return input_features

# =========================================================
# 모델 로드
# =========================================================
st.set_page_config(page_title="AL Risk Calculator", layout="wide")

@st.cache_resource
def load_model():
    model             = joblib.load("al_model.pkl")
    expected_features = joblib.load("feature_list.pkl")
    return model, expected_features

try:
    model, expected_features = load_model()
except Exception as e:
    st.error(f"Model files not found: {e}")
    st.stop()

# =========================================================
# 비즈니스 로직 함수
# =========================================================
def build_input_dataframe(values: dict, expected_features: list) -> pd.DataFrame:
    """입력값을 DataFrame으로 변환하고 모델 요구 형식에 맞게 정렬"""
    df = pd.DataFrame([values])
    for col in expected_features:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected_features]

def get_risk_color(risk_prob: float) -> str:
    """위험도 확률에 따른 색상 코드 반환"""
    if risk_prob >= HIGH_RISK_THRESHOLD:
        return "#DC2626"
    elif risk_prob >= MEDIUM_RISK_THRESHOLD:
        return "#D97706"
    return "#16A34A"

def render_result_box(risk_prob: float):
    """예측 결과 박스 렌더링"""
    risk_color = get_risk_color(risk_prob)
    st.markdown(f"""
    <div style="background-color:#FEF2F2; border:2px solid #FEF2F2;
                padding:30px; border-radius:15px; text-align:center;">
        <h2 style="color:#991B1B; margin-top:0;">Predicted AL Risk</h2>
        <h1 style="font-size:60px; color:{risk_color};">{risk_prob:.1%}</h1>
        <p style="color:#4B5563; font-size:18px;">
            This risk score is based on the calibrated XGBoost model.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# UI 스타일
# =========================================================
st.markdown("""
<style>
html, body, .stApp {
    background-color: white !important;
    color: #111111 !important;
}
h1, h2, h3 {
    color: #111111 !important;
    margin-bottom: 20px !important;
}
label {
    color: #111111 !important;
    font-weight: 600 !important;
    margin-bottom: 5px !important;
}
.stNumberInput, .stSelectbox, .stTextInput, .stFileUploader {
    max-width: 100% !important;
    margin-bottom: 10px !important;
}
input {
    background-color: #F3F4F6 !important;
    color: #111111 !important;
    border: 1px solid #D1D5DB !important;
}
input:disabled {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    opacity: 1 !important;
}
div[data-baseweb="select"] {
    background-color: #F3F4F6 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 4px !important;
}
div[data-baseweb="select"] div {
    color: #111111 !important;
    background-color: transparent !important;
}
div[data-baseweb="select"] > div > div:nth-of-type(2) { display: none !important; }
div[data-baseweb="select"] svg                        { display: none !important; }
div[data-baseweb="select"]::after {
    content: "▼";
    position: absolute; right: 15px; top: 10px;
    font-size: 10px; color: #4B5563; pointer-events: none;
}
button[role="tab"] { color: #111111 !important; font-size: 18px !important; }
.stButton > button {
    background-color: #DC2626 !important; color: white !important;
    font-weight: bold; height: 55px; border-radius: 10px;
    border: none; font-size: 20px !important;
    margin-top: 20px !important; transition: 0.3s;
}
.stButton > button:hover {
    background-color: #991B1B !important; transform: scale(1.02);
}
.stDownloadButton > button {
    background-color: #2563EB !important; color: white !important;
    font-weight: bold; height: 55px; border-radius: 10px;
    border: none; font-size: 20px !important;
    margin-top: 20px !important; transition: 0.3s;
}
.stDownloadButton > button:hover {
    background-color: #1E40AF !important; transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown(
    "<h1 style='text-align:center;'>Rectal Surgery AL Risk Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# TAB
# =========================================================
tab1, tab2 = st.tabs([" Single Patient Prediction", "📂 Batch Processing"])

# =========================================================
# TAB 1: Single Patient
# =========================================================
with tab1:

    # 1. Patient Demographics
    st.subheader("1. Patient Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 20, 100, 65)
        sex = st.selectbox("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 24.0)
    with c2:
        asa = st.selectbox("ASA score", [1, 2, 3, 4])
        dm  = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
        cvd = st.selectbox("Cardiovascular morbidity", ["No", "Yes"])
    with c3:
        prior_abdominal  = st.selectbox("Prior abdominal surgery", ["No", "Yes"])
        prior_malignancy = st.selectbox("Prior malignancy", ["No", "Yes"])
        tm_height        = st.number_input("Tumor height from ARJ (cm)", 0.0, 20.0, 5.0)

    # 2. Tumor & Surgical Info
    st.subheader("2. Tumor & Surgical Info")
    c4, c5, c6 = st.columns(3)
    with c4:
        emvi = st.selectbox("EMVI", ["Negative", "Positive", "Unknown"])
        mrf  = st.selectbox("MRF Status", ["Negative", "Positive", "Threatened", "Unknown"])
    with c5:
        procedure_type = st.selectbox("Procedure Type", ["LAR", "uLAR", "APR", "Hartmann", "Other"])
        tech           = st.selectbox("Technique", ["Laparoscopic", "Robotic", "Open", "Transanal"])
    with c6:
        optime = st.number_input("Operation time (min)", 30, 600, 180)

    # 3. Neoadjuvant Treatment
    st.subheader("3. Neoadjuvant Treatment")
    c_neo_main, c_neo_sub = st.columns([1, 2])
    with c_neo_main:
        neoadj = st.selectbox("Neoadjuvant therapy", ["No", "Yes"])
    with c_neo_sub:
        if neoadj == "Yes":
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                tr  = st.selectbox("Tumor response",
                                   ["Partial response", "No response", "Complete response"])
                ycT = st.selectbox("ycT Stage", ["T3", "T0", "T1", "T2", "T4", "Unknown"])
            with nc2:
                ycN  = st.selectbox("ycN Stage", ["N0", "N1", "N2", "Unknown"])
                yMRF = st.selectbox("yMRF", ["No", "Yes", "Unknown"])
            with nc3:
                tts = st.number_input("Time to surgery (weeks)", 1.0, 30.0, 8.0)
            st.markdown("<div style='margin-bottom:120px;'></div>", unsafe_allow_html=True)
        else:
            tr, ycT, ycN, yMRF, tts = "Unknown", "Unknown", "Unknown", "Unknown", 0.0

    # 4. Intraoperative Details
    st.subheader("4. Intraoperative Details")
    c7, c8, c9 = st.columns(3)
    with c7:
        conversion = st.selectbox("Conversion to Open", ["No", "Yes"])
    with c8:
        stapler_used = st.selectbox("Stapler used", ["Yes", "No"])
    with c9:
        stapler_length = (
            st.selectbox("Stapler length (mm)", ["60", "45", "Unknown"])
            if stapler_used == "Yes" else "Unknown"
        )

    if stapler_used == "Yes":
        c10, _, __ = st.columns(3)
        with c10:
            stapler_reload = st.selectbox(
                "Stapler reload",
                ["No reload", "Purple", "Green", "Blue", "Black", "Unknown"]
            )
    else:
        stapler_reload = "Unknown"

    # 5. Laboratory Data
    st.subheader("5. Laboratory Data (Pre-operative)")

    lab_r1_c1, lab_r1_c2, lab_r1_c3 = st.columns(3)
    with lab_r1_c1:
        wbc = st.number_input("WBC (10³/µL)", 0.0, 30.0, 7.0)
    with lab_r1_c2:
        plt_val = st.number_input("PLT (10³/µL)", 0.0, 600.0, 250.0)
    with lab_r1_c3:
        gfr = st.number_input("GFR - MDRD (mL/min/1.73m²)", 5.0, 150.0, 90.0)

    lab_r2_c1, lab_r2_c2, _ = st.columns(3)
    with lab_r2_c1:
        hb = st.number_input("Hb (g/dL)", 0.0, 20.0, 13.0)
    with lab_r2_c2:
        glucose = st.number_input("Glucose (mg/dL)", 50.0, 500.0, 100.0)

    lab_r3_c1, lab_r3_c2, lab_r3_c3 = st.columns(3)
    with lab_r3_c1:
        alb = st.number_input("Albumin (g/dL)", 0.0, 6.0, 4.0)
    with lab_r3_c2:
        prot = st.number_input("Protein (g/dL)", 0.0, 10.0, 7.0)
    with lab_r3_c3:
        if alb > 0 and prot > alb:
            agr = alb / (prot - alb)
            st.text_input("AGR", value=f"{agr:.2f}", disabled=True)
        else:
            agr = np.nan
            st.text_input("AGR", value="N/A", disabled=True)

    # =========================================================
    # Calculate Button
    # =========================================================
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, __ = st.columns([1, 1, 1])
    with col_btn:
        predict_btn = st.button("Calculate", use_container_width=True)

    if predict_btn:
        input_values = {
            'Sex': sex, 'Age': age, 'ASA score': asa, 'BMI': bmi,
            'DM': dm, 'cardiovascular morbidity': cvd,
            'prior abdominal surgery': prior_abdominal,
            'prior malignancy': prior_malignancy,
            'tm height from ARJ': tm_height,
            'EMVI': emvi, 'MRF': mrf,
            'neoadj tx': neoadj, 'tumor_response': tr,
            'ycT': ycT, 'ycN': ycN, 'yMRF': yMRF,
            'Time to surgery': tts,
            'Type of technique': tech, 'procedure type': procedure_type,
            'optime': optime, 'conversion': conversion,
            'stapler used': stapler_used,
            'stapler length': stapler_length,
            'stapler reload': stapler_reload,
            'WBC': wbc, 'Hb': hb, 'PLT': plt_val,
            'glucose': glucose, 'AGR': agr, 'GFR1_MDRD': gfr
        }

        input_data = build_input_dataframe(input_values, expected_features)
        risk_prob  = model.predict_proba(input_data)[0][1]

        st.markdown("<hr>", unsafe_allow_html=True)
        _, res_col, __ = st.columns([1, 2, 1])
        with res_col:
            render_result_box(risk_prob)

# =========================================================
# TAB 2: Batch Prediction
# =========================================================
with tab2:
    st.subheader("Upload Patient Data for Batch Prediction")
    st.write("Please upload a CSV or Excel file. Columns should match the model's required features.")

    uploaded_file = st.file_uploader("Upload file (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            batch_df = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.endswith('.csv')
                else pd.read_excel(uploaded_file)
            )

            st.write("### Data Preview")
            st.dataframe(batch_df.head())

            _, col_batch, __ = st.columns([1, 1, 1])
            with col_batch:
                batch_predict_btn = st.button(
                    " Calculate Batch AL Risk", use_container_width=True)

            if batch_predict_btn:
                with st.spinner("Calculating predictions..."):
                    processed_df = batch_df.copy()
                    for col in expected_features:
                        if col not in processed_df.columns:
                            processed_df[col] = np.nan

                    probs = model.predict_proba(
                        processed_df[expected_features])[:, 1]
                    batch_df.insert(0, 'Predicted_AL_Risk (%)',
                                    (probs * 100).round(2))

                    st.markdown("""
                        <div style="background-color:#D1FAE5;
                                    border-left:5px solid #10B981;
                                    padding:15px; border-radius:5px;
                                    margin-bottom:20px;">
                            <span style="color:#4B5563; font-size:16px;
                                         font-weight:bold;">
                                ✅ Batch prediction completed successfully!
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(batch_df.head(10))
                    st.download_button(
                        label    = " Download Full Results as CSV",
                        data     = batch_df.to_csv(index=False).encode('utf-8-sig'),
                        file_name= "batch_prediction_results.csv",
                        mime     = "text/csv",
                    )

        except Exception as e:
            st.markdown(f"""
                <div style="background-color:#FEE2E2;
                            border-left:5px solid #EF4444;
                            padding:15px; border-radius:5px;
                            margin-bottom:20px;">
                    <span style="color:#4B5563; font-size:16px;
                                 font-weight:bold;">
                        ❌ Error processing the file: {e}
                    </span>
                </div>
            """, unsafe_allow_html=True)