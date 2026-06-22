# 🩺 DiabetaCheck — Multiclass Diabetes Risk Classifier

**Capstone Project 1 | AI in Healthcare | IIT Delhi Executive Program**

A machine learning web application that classifies patients into three diabetes risk categories — **Non-Diabetic**, **Pre-Diabetic**, and **Diabetic** — using five key clinical biomarkers.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📌 Disclaimer

> ⚠️ **This is a Minimum Viable Product (MVP) built for academic demonstration only.**
> It is NOT a diagnostic tool and must NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always clinically correlate results with a qualified healthcare professional.

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Support Vector Machine (SVM) |
| Kernel | RBF |
| Regularization (C) | 10 |
| Accuracy | 95.5% |
| F1 Score (Weighted) | 0.9551 |
| ROC AUC | 0.9921 |
| Dataset Size | 1,000 patients |

### Input Features (Top 5 by importance)
1. **HbA1c** — Glycated haemoglobin (primary diabetes marker)
2. **BMI** — Body Mass Index
3. **Age** — Patient age in years
4. **Cholesterol** — Total serum cholesterol (mmol/L)
5. **Triglycerides** — Serum TG level (mmol/L)

### Output Classes
- 🟢 **N** — Non-Diabetic
- 🟡 **P** — Pre-Diabetic
- 🔴 **Y** — Diabetic

---

## 🛠️ Run Locally

```bash
git clone https://github.com/your-username/diabetacheck.git
cd diabetacheck
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
diabetacheck/
├── app.py              # Streamlit application
├── model.pkl           # Trained SVM model
├── scaler.pkl          # Fitted StandardScaler
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 📊 Dataset

**Source:** [Diabetes Dataset — Kaggle](https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-medical-features)  
1,000 patient records with demographic and biochemical markers.

---

*For academic and demonstration purposes only. Not for clinical use.*
