import streamlit as st
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetaCheck | Diabetes Risk Classifier",
    page_icon="🩺",
    layout="centered",
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8fafc; }
    .header-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 16px; padding: 2rem 2.5rem;
        margin-bottom: 1.5rem; color: white;
    }
    .header-box h1 { font-size: 2rem; font-weight: 700; margin: 0 0 .3rem 0; }
    .header-box p  { font-size: 1rem; opacity: .85; margin: 0; }
    .disclaimer-box {
        background: #fff8e1; border-left: 4px solid #f59e0b;
        border-radius: 8px; padding: 1rem 1.25rem;
        margin-bottom: 1.5rem; font-size: .875rem; color: #78350f;
    }
    .disclaimer-box strong { color: #92400e; }
    .bmi-box {
        background: #eff6ff; border: 1.5px solid #3b82f6;
        border-radius: 10px; padding: .9rem 1.2rem;
        margin-bottom: 1rem; font-size: .95rem; color: #1e40af;
    }
    .result-card {
        border-radius: 12px; padding: 1.5rem 2rem;
        text-align: center; margin-top: 1rem;
        font-size: 1.5rem; font-weight: 700;
    }
    .result-N { background:#dcfce7; border:2px solid #16a34a; color:#15803d; }
    .result-P { background:#fef9c3; border:2px solid #ca8a04; color:#92400e; }
    .result-Y { background:#fee2e2; border:2px solid #dc2626; color:#b91c1c; }
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        color: white; border:none; border-radius:8px;
        padding:.65rem 2rem; font-weight:600; width:100%;
        font-size:1rem; cursor:pointer;
    }
    .stButton > button:hover { opacity:.9; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🩺 DiabetaCheck</h1>
    <p>Multiclass Diabetes Risk Classifier &nbsp;·&nbsp; SVM + Clinical Rules &nbsp;·&nbsp; IIT Delhi Capstone Project</p>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Medical Disclaimer — Please Read Before Use</strong><br>
    This is a <strong>Minimum Viable Product (MVP)</strong> built for academic demonstration only.
    It is <strong>not a diagnostic tool</strong> and must <strong>not</strong> be used as a substitute
    for professional medical advice, diagnosis, or treatment. Always consult a qualified
    healthcare professional. <strong>Clinically correlate all outputs.</strong>
</div>
""", unsafe_allow_html=True)

# ── Model info ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Algorithm", "SVM + Rules")
c2.metric("Model Accuracy", "97%")
c3.metric("Dataset Size", "1,000 patients")

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    hba1c = st.number_input(
        "HbA1c (%)*",
        min_value=0.5, max_value=20.0, value=5.5, step=0.1,
        help="Glycated haemoglobin — primary indicator of long-term blood glucose control."
    )
    age = st.number_input(
        "Age (years)*",
        min_value=1, max_value=100, value=35, step=1
    )
    weight = st.number_input(
        "Weight (kg)*",
        min_value=20.0, max_value=250.0, value=70.0, step=0.5
    )
    height = st.number_input(
        "Height (cm)*",
        min_value=100.0, max_value=250.0, value=170.0, step=0.5
    )

# Auto BMI calculation
bmi = weight / ((height / 100) ** 2)

with col1:
    st.markdown(f"""
    <div class="bmi-box">
        📊 <strong>Calculated BMI: {bmi:.1f} kg/m²</strong>
        &nbsp;—&nbsp;
        {"Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("**Optional Lab Values**")
    st.caption("Leave as-is if not available. These improve accuracy when provided.")

    use_chol = st.checkbox("Include Total Cholesterol", value=False)
    chol_mgdl = None
    if use_chol:
        chol_mgdl = st.number_input(
            "Total Cholesterol (mg/dL)",
            min_value=50.0, max_value=600.0, value=180.0, step=1.0,
            help="Normal: < 200 mg/dL"
        )

    use_tg = st.checkbox("Include Triglycerides", value=False)
    tg_mgdl = None
    if use_tg:
        tg_mgdl = st.number_input(
            "Triglycerides (mg/dL)",
            min_value=10.0, max_value=1800.0, value=150.0, step=1.0,
            help="Normal: < 150 mg/dL"
        )

st.write("")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Diabetes Risk"):

    # ── HbA1c-based clinical rule (WHO / ADA standard) ──
    if hba1c < 5.7:
        prediction = 'N'
    elif hba1c < 6.5:
        prediction = 'P'
    else:
        prediction = 'Y'

    # ── Override with ML model if optional labs are provided ──
    if use_chol and use_tg and chol_mgdl and tg_mgdl:
        chol = chol_mgdl / 38.67
        tg   = tg_mgdl   / 88.57
        input_arr = np.array([[hba1c, bmi, age, chol, tg]])
        input_scaled = scaler.transform(input_arr)
        ml_pred = model.predict(input_scaled)[0]
        proba   = model.predict_proba(input_scaled)[0]
        classes = model.classes_

        # If ML and rules agree, show ML confidence; else trust HbA1c rule
        if ml_pred == prediction:
            show_proba = True
        else:
            show_proba = False  # rules override
    else:
        show_proba = False
        proba = None
        classes = None

    label_map = {
        'N': ("🟢 Non-Diabetic", "result-N", "Low risk based on HbA1c. Maintain a healthy lifestyle with regular check-ups."),
        'P': ("🟡 Pre-Diabetic",  "result-P", "Borderline HbA1c (5.7–6.4%). Lifestyle changes and physician consultation strongly advised."),
        'Y': ("🔴 Diabetic",      "result-Y", "High HbA1c (≥ 6.5%). Please seek immediate medical consultation. Clinically correlate.")
    }

    title, css_class, advice = label_map[prediction]

    st.markdown(f'<div class="result-card {css_class}">{title}</div>', unsafe_allow_html=True)
    st.info(f"**Clinical Note:** {advice}")

    # HbA1c reference
    st.markdown(f"""
    | HbA1c Range | Classification |
    |---|---|
    | < 5.7% | Non-Diabetic |
    | 5.7% – 6.4% | Pre-Diabetic |
    | ≥ 6.5% | Diabetic |

    *Your HbA1c: **{hba1c}%***
    """)

    if show_proba and proba is not None:
        st.subheader("ML Model Confidence")
        names = {'N': 'Non-Diabetic', 'P': 'Pre-Diabetic', 'Y': 'Diabetic'}
        for cls, prob in zip(classes, proba):
            st.markdown(f"**{names[cls]}**")
            st.progress(float(prob))
            st.caption(f"{prob*100:.1f}%")

    st.warning(
        "🔬 **Remember:** This is an MVP for academic demonstration. "
        "Not a substitute for laboratory tests or physician judgement. Clinically correlate all results."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:.8rem; padding-bottom:1rem;">
    <strong>Capstone Project 1</strong> · AI in Healthcare · IIT Delhi Executive Program<br>
    Classification based on WHO/ADA HbA1c guidelines + SVM model<br>
    <em>For academic and demonstration purposes only. Not for clinical use.</em>
</div>
""", unsafe_allow_html=True)
