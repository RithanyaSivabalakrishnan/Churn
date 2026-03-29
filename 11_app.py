"""
Module 11: Streamlit Customer Churn Prediction App
====================================================
Run with:  streamlit run 11_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor · Telco",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    color: #ffffff;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
            
.stTabs [data-baseweb="tab-list"] span { color: white !important; font-weight: 600; }
.stTabs [data-baseweb="tab"] { color: white !important; }
.stTabs [data-baseweb="tab-list"] { background: #1e293b !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] span { 
    color: #60a5fa !important; font-weight: 700; 
}
            
[data-testid="stSidebar"] button {
    color: black !important;
    font-weight: 600;
}
[data-testid="stSidebar"] button[type="primary"] {
    color: black !important;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
}
/* Main area */
.stApp { background: #0f172a; color: #e2e8f0; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2844 100%);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    border: 1px solid #2d4a6e;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

/* Risk badge */
.risk-high { background: linear-gradient(135deg,#c0392b,#e74c3c); border-radius:12px; padding:16px 24px; color:#fff; font-size:2rem; font-weight:700; text-align:center; }
.risk-low  { background: linear-gradient(135deg,#1e8449,#27ae60); border-radius:12px; padding:16px 24px; color:#fff; font-size:2rem; font-weight:700; text-align:center; }

/* Section headers */
.section-header { 
    font-size: 1.3rem; font-weight: 700; 
    background: linear-gradient(90deg,#6366f1,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    missing = []
    for f in ["rf_model.pkl", "scaler.pkl", "feature_names.pkl"]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        return None, None, None, None
    rf           = joblib.load("rf_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    dummy_cols   = joblib.load("dummy_columns.pkl") if os.path.exists("dummy_columns.pkl") else []
    return rf, scaler, feature_names, dummy_cols

rf, scaler, feature_names, dummy_cols = load_artefacts()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — user inputs
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📋 Customer Details")
st.sidebar.markdown("---")

# Demographics
st.sidebar.markdown("### 👤 Demographics")
gender          = st.sidebar.selectbox("Gender", ["Male", "Female"], key="gender")
senior          = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
partner         = st.sidebar.selectbox("Has Partner", ["Yes", "No"], key="partner")
dependents      = st.sidebar.selectbox("Has Dependents", ["Yes", "No"], key="dependents")

# Services
st.sidebar.markdown("### 📡 Services")
tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12, key="tenure")
phone_service   = st.sidebar.selectbox("Phone Service", ["Yes", "No"], key="phone")
multiple_lines  = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"], key="multi")
internet        = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="internet")
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"], key="security")
online_backup   = st.sidebar.selectbox("Online Backup",   ["Yes", "No"], key="backup")
device_protect  = st.sidebar.selectbox("Device Protection",["Yes", "No"], key="device")
tech_support    = st.sidebar.selectbox("Tech Support",    ["Yes", "No"], key="tech")
streaming_tv    = st.sidebar.selectbox("Streaming TV",    ["Yes", "No"], key="tv")
streaming_movies= st.sidebar.selectbox("Streaming Movies",["Yes", "No"], key="movies")

# Account
st.sidebar.markdown("### 💳 Account")
contract        = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
paperless       = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"], key="paper")
payment         = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
], key="payment")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5, key="monthly")
total_charges   = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                          value=float(monthly_charges * tenure), key="total")

predict_btn = st.sidebar.button("🔮 Predict Churn Risk", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Build input row
# ─────────────────────────────────────────────────────────────────────────────
def build_input_df():
    raw = {
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "Tenure_Ratio":     tenure / max(monthly_charges, 1),
        # Categorical dummies (match get_dummies on training set)
        "gender_Male":              1 if gender == "Male" else 0,
        "Partner_Yes":              1 if partner == "Yes" else 0,
        "Dependents_Yes":           1 if dependents == "Yes" else 0,
        "PhoneService_Yes":         1 if phone_service == "Yes" else 0,
        "MultipleLines_Yes":        1 if multiple_lines == "Yes" else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No":       1 if internet == "No" else 0,
        "OnlineSecurity_Yes":       1 if online_security == "Yes" else 0,
        "OnlineBackup_Yes":         1 if online_backup == "Yes" else 0,
        "DeviceProtection_Yes":     1 if device_protect == "Yes" else 0,
        "TechSupport_Yes":          1 if tech_support == "Yes" else 0,
        "StreamingTV_Yes":          1 if streaming_tv == "Yes" else 0,
        "StreamingMovies_Yes":      1 if streaming_movies == "Yes" else 0,
        "Contract_One year":        1 if contract == "One year" else 0,
        "Contract_Two year":        1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes":     1 if paperless == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":        1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check":            1 if payment == "Mailed check" else 0,
    }

    row = pd.DataFrame([raw])

    # Align columns to training feature set
    if feature_names:
        for col in feature_names:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_names]

    # Scale numeric columns using fitted scaler
    if scaler is not None:
        num_cols_in_scaler = [
            c for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Tenure_Ratio"]
            if c in row.columns
        ]
        try:
            row[num_cols_in_scaler] = scaler.transform(row[num_cols_in_scaler])
        except Exception:
            pass  # If scaler was fit on different columns, skip gracefully

    return row

# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📞 Telco Customer Churn Predictor")
st.markdown("*Powered by Random Forest + SHAP Explainability*")
st.markdown("---")

tabs = st.tabs(["🔮 Prediction", "📊 EDA", "📈 Model Performance", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    if rf is None:
        st.error("⚠️  Model files not found. Please run the pipeline first: `python 12_run_all.py`")
    elif predict_btn:
        input_df = build_input_df()
        prob     = rf.predict_proba(input_df)[0][1]
        label    = "High Risk 🔴" if prob >= 0.5 else "Low Risk 🟢"

        # ── Risk display ───────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            css_cls = "risk-high" if prob >= 0.5 else "risk-low"
            st.markdown(f'<div class="{css_cls}">{label}<br><span style="font-size:1.1rem">{prob:.1%} churn probability</span></div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # ── Probability gauge ──────────────────────────────────────
        st.markdown('<p class="section-header">Churn Probability Gauge</p>', unsafe_allow_html=True)
        fig_gauge, ax_g = plt.subplots(figsize=(8, 1.2), facecolor="#0f172a")
        ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
        ax_g.barh(0, 1, color="#1e293b", height=0.5)
        color = "#e74c3c" if prob >= 0.5 else "#27ae60"
        ax_g.barh(0, prob, color=color, height=0.5)
        ax_g.axvline(0.5, color="#94a3b8", linewidth=1.5, linestyle="--")
        ax_g.set_yticks([]); ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_g.set_xticklabels(["0%","25%","50%","75%","100%"], color="#94a3b8")
        ax_g.spines[:].set_visible(False)
        ax_g.tick_params(colors="#94a3b8")
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close()

        # ── SHAP force plot ────────────────────────────────────────
        st.markdown('<p class="section-header">SHAP Feature Contributions</p>', unsafe_allow_html=True)
        try:
            explainer   = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(input_df)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values

            sv_series = pd.Series(sv[0], index=input_df.columns).abs().nlargest(10)

            fig_shap, ax_shap = plt.subplots(figsize=(9, 4), facecolor="#0f172a")
            colors = ["#e74c3c" if v > 0 else "#27ae60"
                      for v in pd.Series(sv[0], index=input_df.columns).loc[sv_series.index]]
            sv_series.sort_values().plot(kind="barh", color=colors[::-1], ax=ax_shap)
            ax_shap.set_title("Top 10 Feature Impact on Prediction", color="#e2e8f0", fontsize=12)
            ax_shap.set_facecolor("#0f172a")
            ax_shap.tick_params(colors="#94a3b8")
            ax_shap.spines[:].set_color("#2d4a6e")
            ax_shap.set_xlabel("|SHAP value|", color="#94a3b8")
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")

        # ── Summary card ───────────────────────────────────────────
        st.markdown('<p class="section-header">Customer Summary</p>', unsafe_allow_html=True)
        summary_data = {
            "Tenure": f"{tenure} months",
            "Contract": contract,
            "Monthly Charges": f"${monthly_charges:.2f}",
            "Internet": internet,
            "Tech Support": tech_support,
            "Churn Probability": f"{prob:.1%}",
        }
        c1, c2, c3 = st.columns(3)
        for i, (k, v) in enumerate(summary_data.items()):
            col = [c1, c2, c3][i % 3]
            col.markdown(f'<div class="metric-card"><h4 style="color:#94a3b8;margin:0">{k}</h4>'
                         f'<p style="font-size:1.4rem;font-weight:700;margin:8px 0 0 0">{v}</p></div>',
                         unsafe_allow_html=True)
    else:
        st.info("👈  Configure the customer details in the sidebar and click **Predict Churn Risk**.")
        st.markdown("""
        ### How it works
        1. **Fill in** the customer's account and service details on the left.
        2. **Click** the *Predict Churn Risk* button.
        3. The Random Forest model outputs a churn probability.
        4. **SHAP** explains which factors drove the prediction.
        """)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Exploratory Data Analysis")

    eda_plots = {
        "Contract Type vs Churn": "eda_tenure.png",
        "Monthly Charges & Tenure vs Churn": "eda_charges.png",
        "Correlation Heatmap": "eda_heatmap.png",
        "SHAP Summary (Global)": "shap_summary.png",
    }

    available = {k: v for k, v in eda_plots.items() if os.path.exists(v)}
    if available:
        for title, path in available.items():
            st.markdown(f"#### {title}")
            st.image(path, width="stretch")
            st.markdown("---")
    else:
        st.warning("No EDA plots found. Run `python 02_eda.py` first.")

    # Live dataset stats if CSV present
    if os.path.exists("cleaned_data.csv"):
        df_live = pd.read_csv("cleaned_data.csv")
        st.markdown("### Live Dataset Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{len(df_live):,}")
        c2.metric("Churned", f"{df_live['Churn'].sum():,}")
        c3.metric("Churn Rate", f"{df_live['Churn'].mean():.1%}")
        c4.metric("Features", str(df_live.shape[1] - 1))

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Model Evaluation Results")
    if os.path.exists("eval_results.csv"):
        eval_df = pd.read_csv("eval_results.csv")
        st.dataframe(eval_df.style.highlight_max(
            subset=["Accuracy", "F1 Score", "ROC-AUC"],
            color="#1e4d2b"
        ), use_container_width=True)
    else:
        st.info("Run `python 09_eval.py` to generate evaluation results.")

    if os.path.exists("roc_curves.png"):
        st.image("roc_curves.png", caption="ROC Curves — Logistic Regression vs Random Forest",
                 width="stretch")

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — About
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    ### About This App

    This interactive dashboard demonstrates an end-to-end **Customer Churn Prediction** pipeline
    built for the Telco Customer dataset from IBM.

    | Component | Detail |
    |---|---|
    | Dataset | Telco Customer Churn (IBM / Kaggle) |
    | Baseline | Logistic Regression |
    | Best Model | Random Forest (100 trees) |
    | Imbalance Handling | SMOTE |
    | Explainability | SHAP TreeExplainer |
    | UI Framework | Streamlit |

    ### Pipeline Modules
    | # | Module | Description |
    |---|---|---|
    | 01 | Setup | Install packages, load dataset |
    | 02 | EDA | Visualisations & insights |
    | 03 | Clean | Null handling, encoding target |
    | 04 | Preprocess | One-hot encoding, scaling, feature engineering |
    | 05 | Balance | SMOTE oversampling |
    | 06 | Split | 80/20 stratified train-test split |
    | 07 | LogReg | Logistic Regression training |
    | 08 | RF | Random Forest training |
    | 09 | Eval | Model comparison & ROC curves |
    | 10 | SHAP | SHAP global explanations |
    | 11 | App | This Streamlit UI |
    | 12 | Runner | Master script to run all modules |

    ### Run the full pipeline
    ```bash
    python 12_run_all.py
    streamlit run 11_app.py
    ```
    """)
