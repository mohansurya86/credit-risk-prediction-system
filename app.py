"""
Credit Risk Prediction System
------------------------------
Streamlit app that loads a trained XGBoost pipeline and predicts a loan
applicant's probability of default from their financial profile.

Run:
    streamlit run app.py
"""

import io
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "credit_risk_pipeline.joblib")
FI_PATH = os.path.join(os.path.dirname(__file__), "models", "feature_importance.csv")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "loan_data.csv")

st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / CSS
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-navy: #0b0f1e;
    --bg-navy-2: #121734;
    --accent-purple: #7c3aed;
    --accent-blue: #2563eb;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --glass: rgba(255, 255, 255, 0.06);
    --glass-border: rgba(255, 255, 255, 0.12);
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 20%, #1e1b4b 0%, transparent 45%),
                radial-gradient(circle at 85% 10%, #0f2b46 0%, transparent 40%),
                radial-gradient(circle at 50% 90%, #1a1035 0%, transparent 50%),
                linear-gradient(160deg, var(--bg-navy) 0%, var(--bg-navy-2) 100%);
    background-attachment: fixed;
    color: #e5e7eb;
}

.bubble {
    position: fixed;
    border-radius: 50%;
    filter: blur(2px);
    opacity: 0.18;
    z-index: 0;
    animation: floatY 9s ease-in-out infinite;
    pointer-events: none;
}
@keyframes floatY {
    0%, 100% { transform: translateY(0px) translateX(0px); }
    50% { transform: translateY(-30px) translateX(15px); }
}
.b1 { width: 140px; height: 140px; top: 8%; left: 4%; background: radial-gradient(circle, var(--accent-purple), transparent 70%); animation-delay: 0s; }
.b2 { width: 90px; height: 90px; top: 65%; left: 9%; background: radial-gradient(circle, var(--accent-cyan), transparent 70%); animation-delay: 2s; }
.b3 { width: 180px; height: 180px; top: 15%; right: 6%; background: radial-gradient(circle, var(--accent-blue), transparent 70%); animation-delay: 1s; }
.b4 { width: 110px; height: 110px; top: 75%; right: 10%; background: radial-gradient(circle, var(--accent-green), transparent 70%); animation-delay: 3s; }

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #22d3ee, #a78bfa);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shine 6s linear infinite;
    margin-bottom: 0.2rem;
}
@keyframes shine {
    to { background-position: 300% center; }
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #9ca3af;
    margin-bottom: 1.6rem;
    font-weight: 500;
}

.glass-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    backdrop-filter: blur(10px);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    text-align: center;
}
.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(124, 58, 237, 0.25);
    border-color: rgba(167, 139, 250, 0.5);
}
.stat-icon { font-size: 1.9rem; margin-bottom: 0.3rem; }
.stat-value { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.5rem; color: #f3f4f6; }
.stat-label { font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; }

.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1.3rem;
    border-radius: 999px;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    animation: fadeUp 0.5s ease;
}
.risk-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
.risk-mod { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
.risk-high { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.4); animation: fadeUp 0.5s ease, pulse 1.8s ease-in-out infinite; }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.35); }
    50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-header {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #e5e7eb;
    margin: 1.2rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.stButton > button {
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue));
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
    padding: 0.7rem 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.35);
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 26px rgba(124, 58, 237, 0.5);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
}

.mini-metric {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 0.8rem;
    text-align: center;
    backdrop-filter: blur(8px);
    animation: fadeUp 0.6s ease;
}
.mini-metric .val { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.25rem; color: #f3f4f6; }
.mini-metric .lbl { font-size: 0.78rem; color: #9ca3af; }

.share-box {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(10px);
}
.share-box code {
    color: #a78bfa;
}

hr { border-color: rgba(255,255,255,0.08) !important; }
</style>

<div class="bubble b1"></div>
<div class="bubble b2"></div>
<div class="bubble b3"></div>
<div class="bubble b4"></div>
"""


@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load the trained model: {e}")
        return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


@st.cache_data
def load_feature_importance():
    if os.path.exists(FI_PATH):
        try:
            return pd.read_csv(FI_PATH)
        except Exception:
            return None
    return None


@st.cache_data
def load_dataset_stats():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            return {"rows": len(df), "default_rate": float(df["default"].mean())}
        except Exception:
            return None
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must mirror src/train_model.py::engineer_features exactly."""
    df = df.copy()
    monthly_income = df["annual_income"] / 12.0
    df["monthly_installment"] = df["loan_amount"] / df["loan_term_months"]
    df["debt_to_income_ratio"] = (
        (df["existing_debt"] * 0.03) + df["monthly_installment"]
    ) / (monthly_income + 1.0)
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1.0)
    df["disposable_income_ratio"] = 1.0 - df["debt_to_income_ratio"].clip(upper=2.0)
    return df


def risk_band(prob: float) -> tuple[str, str, str]:
    """Returns (label, css_class, hex_color)."""
    if prob < 0.10:
        return "Low Risk", "risk-low", "#10b981"
    elif prob < 0.30:
        return "Moderate Risk", "risk-mod", "#f59e0b"
    else:
        return "High Risk", "risk-high", "#ef4444"


def make_gauge(prob: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 42, "color": "#f3f4f6", "family": "Poppins"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7280", "tickfont": {"color": "#9ca3af"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 10], "color": "rgba(16,185,129,0.18)"},
                {"range": [10, 30], "color": "rgba(245,158,11,0.18)"},
                {"range": [30, 100], "color": "rgba(239,68,68,0.18)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


def make_radar(input_row: pd.Series) -> go.Figure:
    dti = min(input_row["debt_to_income_ratio"] / 0.6, 1.0)
    lti = min(input_row["loan_to_income_ratio"] / 1.0, 1.0)
    credit_risk = 1 - min(max((input_row["credit_score"] - 300) / 550, 0), 1)
    late_pay_risk = min(input_row["num_late_payments"] / 10, 1.0)
    tenure_risk = 1 - min(input_row["employment_years"] / 15, 1.0)

    categories = ["Debt-to-Income", "Loan-to-Income", "Credit Score", "Late Payments", "Job Tenure"]
    values = [dti, lti, credit_risk, late_pay_risk, tenure_risk]
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(124, 58, 237, 0.35)",
        line=dict(color="#a78bfa", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#9ca3af"),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        height=320,
        margin=dict(l=40, r=40, t=30, b=20),
    )
    return fig


def make_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    top = fi_df.head(12).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker=dict(
            color=top["importance"],
            colorscale=[[0, "#2563eb"], [0.5, "#7c3aed"], [1, "#22d3ee"]],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def make_default_donut(default_rate: float) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Repaid", "Defaulted"],
        values=[1 - default_rate, default_rate],
        hole=0.65,
        marker=dict(colors=["#10b981", "#ef4444"]),
        textinfo="percent",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(text="Training<br>Data", showarrow=False, font=dict(size=14, color="#9ca3af"))],
    )
    return fig


def generate_pdf_report(applicant: dict, prob: float, band: str) -> bytes:
    """Builds a shareable PDF risk report for a single prediction."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(11, 15, 30)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_text_color(167, 139, 250)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, "Credit Risk Prediction Report", ln=True)

    pdf.set_text_color(156, 163, 175)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_draw_color(80, 80, 100)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    band_colors = {"Low Risk": (52, 211, 153), "Moderate Risk": (251, 191, 36), "High Risk": (248, 113, 113)}
    r, g, b = band_colors.get(band, (255, 255, 255))
    pdf.set_text_color(r, g, b)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Result: {band}  ({prob:.1%} probability of default)", ln=True)
    pdf.ln(6)

    pdf.set_text_color(229, 231, 235)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Applicant Profile", ln=True)
    pdf.set_font("Helvetica", "", 10)

    field_labels = {
        "age": "Age",
        "employment_status": "Employment Status",
        "employment_years": "Years Employed",
        "home_ownership": "Home Ownership",
        "annual_income": "Annual Income ($)",
        "credit_score": "Credit Score",
        "credit_history_years": "Credit History (years)",
        "num_late_payments": "Late Payments (2yr)",
        "existing_debt": "Existing Debt ($)",
        "loan_amount": "Loan Amount ($)",
        "loan_term_months": "Loan Term (months)",
        "loan_purpose": "Loan Purpose",
    }
    for key, label in field_labels.items():
        value = applicant.get(key, "-")
        pdf.set_text_color(156, 163, 175)
        pdf.cell(70, 7, label, border=0)
        pdf.set_text_color(229, 231, 235)
        pdf.cell(0, 7, str(value), ln=True)

    pdf.ln(4)
    pdf.set_text_color(229, 231, 235)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Ratios", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for label, value in [
        ("Debt-to-Income Ratio", f"{applicant.get('debt_to_income_ratio', 0):.1%}"),
        ("Loan-to-Income Ratio", f"{applicant.get('loan_to_income_ratio', 0):.1%}"),
        ("Monthly Installment", f"${applicant.get('monthly_installment', 0):,.0f}"),
    ]:
        pdf.set_text_color(156, 163, 175)
        pdf.cell(70, 7, label, border=0)
        pdf.set_text_color(229, 231, 235)
        pdf.cell(0, 7, value, ln=True)

    pdf.ln(10)
    pdf.set_text_color(107, 114, 128)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, "This report is generated by an automated machine-learning model for "
                          "informational purposes and does not constitute a final credit decision.")

    return bytes(pdf.output())


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    pipeline = load_pipeline()
    metrics = load_metrics()
    fi_df = load_feature_importance()
    ds_stats = load_dataset_stats()

    # ---------------- HERO ----------------
    st.markdown('<div class="hero-title">💳 Credit Risk Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">AI-powered loan default prediction using XGBoost, '
        'financial ratio engineering, and real-time credit risk scoring.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="glass-card"><div class="stat-icon">🤖</div>'
            '<div class="stat-value">XGBoost</div><div class="stat-label">Model</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        auc_val = f"{metrics['roc_auc']:.2f}" if metrics else "—"
        st.markdown(
            f'<div class="glass-card"><div class="stat-icon">🎯</div>'
            f'<div class="stat-value">{auc_val}</div><div class="stat-label">ROC-AUC</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        rows_val = f"{ds_stats['rows']:,}" if ds_stats else "—"
        st.markdown(
            f'<div class="glass-card"><div class="stat-icon">📊</div>'
            f'<div class="stat-value">{rows_val}</div><div class="stat-label">Training Records</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="glass-card"><div class="stat-icon">⚡</div>'
            '<div class="stat-value">Real-time</div><div class="stat-label">Scoring</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    if pipeline is None:
        st.error(
            "No trained model found at `models/credit_risk_pipeline.joblib`. "
            "Run the training pipeline first:\n\n"
            "```\npython data/generate_data.py\npython src/train_model.py\n```"
        )
        st.stop()

    with st.sidebar:
        st.markdown("### 💳 Credit Risk AI")
        st.caption("Predict loan default probability from applicant financials in real time.")
        st.divider()
        if metrics:
            st.markdown("**Held-out Test Metrics**")
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
            if "cv" in metrics:
                st.divider()
                st.markdown("**5-Fold Cross-Validation**")
                cv = metrics["cv"]
                st.caption(f"ROC-AUC: {cv['roc_auc_mean']:.3f} ± {cv['roc_auc_std']:.3f}")
                st.caption(f"Accuracy: {cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}")
                st.caption(f"F1: {cv['f1_mean']:.3f} ± {cv['f1_std']:.3f}")
        st.divider()
        st.caption("Built with Streamlit + XGBoost + Plotly")

    tab_predict, tab_insights, tab_share = st.tabs(
        ["🔍  Predict Risk", "📊  Model Insights", "🌐  Share & Deploy"]
    )

    # ---------------- PREDICT TAB ----------------
    with tab_predict:
        st.markdown('<div class="section-header">🧾 Applicant Profile</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Personal**")
            age = st.number_input("Age", min_value=18, max_value=90, value=35)
            employment_status = st.selectbox(
                "Employment status", ["EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED"]
            )
            employment_years = st.number_input(
                "Years in current employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5
            )
            home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])

        with col2:
            st.markdown("**💰 Income & Credit**")
            annual_income = st.number_input(
                "Annual income ($)", min_value=1000, max_value=2_000_000, value=60000, step=1000
            )
            credit_score = st.slider("Credit score", 300, 850, 680)
            credit_history_years = st.number_input(
                "Credit history length (years)", min_value=0.0, max_value=50.0, value=8.0, step=0.5
            )
            num_late_payments = st.number_input(
                "Late payments (last 2 years)", min_value=0, max_value=30, value=0
            )
            existing_debt = st.number_input(
                "Existing outstanding debt ($)", min_value=0, max_value=1_000_000, value=5000, step=500
            )

        with col3:
            st.markdown("**🏦 Loan Request**")
            loan_amount = st.number_input(
                "Loan amount requested ($)", min_value=500, max_value=500_000, value=15000, step=500
            )
            loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 84], index=2)
            loan_purpose = st.selectbox(
                "Loan purpose",
                ["DEBT_CONSOLIDATION", "EDUCATION", "MEDICAL", "HOME_IMPROVEMENT", "BUSINESS", "OTHER"],
            )

        st.write("")
        predict_clicked = st.button("✨ Predict Default Risk", type="primary", use_container_width=True)

        if predict_clicked:
            try:
                raw_input = {
                    "age": age,
                    "annual_income": annual_income,
                    "employment_years": employment_years,
                    "employment_status": employment_status,
                    "home_ownership": home_ownership,
                    "credit_score": credit_score,
                    "credit_history_years": credit_history_years,
                    "num_late_payments": num_late_payments,
                    "existing_debt": existing_debt,
                    "loan_amount": loan_amount,
                    "loan_term_months": loan_term_months,
                    "loan_purpose": loan_purpose,
                }
                input_df = pd.DataFrame([raw_input])
                input_df = engineer_features(input_df)
                prob_default = float(pipeline.predict_proba(input_df)[0, 1])
                band, css_class, color = risk_band(prob_default)
            except Exception as e:
                st.error(f"⚠️ Something went wrong while generating the prediction: {e}")
                st.info("Double-check your inputs, or retrain the model if the input columns changed.")
                st.stop()

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📈 Prediction Result</div>', unsafe_allow_html=True)

            icon = "✅" if band == "Low Risk" else ("⚠️" if band == "Moderate Risk" else "🚨")
            st.markdown(
                f'<span class="risk-badge {css_class}">{icon} {band} &nbsp;•&nbsp; {prob_default:.1%} probability of default</span>',
                unsafe_allow_html=True,
            )

            st.write("")
            g1, g2 = st.columns([1, 1])
            with g1:
                st.plotly_chart(make_gauge(prob_default, color), use_container_width=True)
            with g2:
                st.plotly_chart(make_radar(input_df.iloc[0]), use_container_width=True)

            st.markdown('<div class="section-header">🔎 Key Ratios</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f'<div class="mini-metric">💳 <div class="val">{input_df["debt_to_income_ratio"].iloc[0]:.1%}</div>'
                    f'<div class="lbl">Debt-to-Income</div></div>', unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="mini-metric">🏠 <div class="val">{input_df["loan_to_income_ratio"].iloc[0]:.1%}</div>'
                    f'<div class="lbl">Loan-to-Income</div></div>', unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    f'<div class="mini-metric">📅 <div class="val">${input_df["monthly_installment"].iloc[0]:,.0f}</div>'
                    f'<div class="lbl">Monthly Installment</div></div>', unsafe_allow_html=True,
                )

            st.write("")
            if band == "High Risk":
                st.warning("🚨 This applicant profile carries **elevated default risk**. Consider additional collateral, a co-signer, or a smaller loan amount.")
            elif band == "Moderate Risk":
                st.info("⚠️ This applicant profile carries **moderate risk**. A standard review is recommended.")
            else:
                st.success("✅ This applicant profile carries **low default risk** based on the model.")

            st.write("")
            try:
                report_bytes = generate_pdf_report(
                    {**raw_input, **input_df.iloc[0].to_dict()}, prob_default, band
                )
                st.download_button(
                    "📄 Download PDF Risk Report",
                    data=report_bytes,
                    file_name=f"credit_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"(PDF report unavailable: {e}. Run `pip install fpdf2` to enable it.)")

    # ---------------- INSIGHTS TAB ----------------
    with tab_insights:
        st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)

        if metrics:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="glass-card"><div class="stat-icon">🎯</div>'
                    f'<div class="stat-value">{metrics["roc_auc"]:.3f}</div><div class="stat-label">Test ROC-AUC</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="glass-card"><div class="stat-icon">✅</div>'
                    f'<div class="stat-value">{metrics["accuracy"]:.3f}</div><div class="stat-label">Test Accuracy</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="glass-card"><div class="stat-icon">⚖️</div>'
                    f'<div class="stat-value">{metrics["f1_score"]:.3f}</div><div class="stat-label">Test F1 Score</div></div>',
                    unsafe_allow_html=True,
                )

            if "cv" in metrics:
                st.write("")
                st.markdown(f'<div class="section-header">🔁 {metrics.get("cv_folds", 5)}-Fold Cross-Validation (train set)</div>', unsafe_allow_html=True)
                cv = metrics["cv"]
                cvm1, cvm2, cvm3 = st.columns(3)
                with cvm1:
                    st.markdown(
                        f'<div class="glass-card"><div class="stat-icon">🎯</div>'
                        f'<div class="stat-value">{cv["roc_auc_mean"]:.3f} ± {cv["roc_auc_std"]:.3f}</div>'
                        f'<div class="stat-label">CV ROC-AUC</div></div>', unsafe_allow_html=True,
                    )
                with cvm2:
                    st.markdown(
                        f'<div class="glass-card"><div class="stat-icon">✅</div>'
                        f'<div class="stat-value">{cv["accuracy_mean"]:.3f} ± {cv["accuracy_std"]:.3f}</div>'
                        f'<div class="stat-label">CV Accuracy</div></div>', unsafe_allow_html=True,
                    )
                with cvm3:
                    st.markdown(
                        f'<div class="glass-card"><div class="stat-icon">⚖️</div>'
                        f'<div class="stat-value">{cv["f1_mean"]:.3f} ± {cv["f1_std"]:.3f}</div>'
                        f'<div class="stat-label">CV F1 Score</div></div>', unsafe_allow_html=True,
                    )
                st.caption(
                    "Cross-validated scores are averaged across multiple train/validation folds, "
                    "giving a more robust estimate of real-world performance than a single train/test split."
                )

            if "best_params" in metrics:
                with st.expander("⚙️ Best hyperparameters found by RandomizedSearchCV"):
                    st.json(metrics["best_params"])
        else:
            st.caption("Run `src/train_model.py` to generate evaluation metrics.")

        st.write("")
        i1, i2 = st.columns([1.4, 1])
        with i1:
            st.markdown('<div class="section-header">🏆 Top Risk Drivers</div>', unsafe_allow_html=True)
            if fi_df is not None:
                st.plotly_chart(make_importance_chart(fi_df), use_container_width=True)
            else:
                st.caption("Run `src/train_model.py` to generate feature importance.")
        with i2:
            st.markdown('<div class="section-header">🥯 Training Data Balance</div>', unsafe_allow_html=True)
            if ds_stats:
                st.plotly_chart(make_default_donut(ds_stats["default_rate"]), use_container_width=True)
            else:
                st.caption("Run `data/generate_data.py` to generate the dataset.")

        st.divider()
        st.caption(
            "Model: XGBoost classifier, tuned via RandomizedSearchCV with 5-fold stratified "
            "cross-validation, trained on engineered ratios (debt-to-income, loan-to-income, "
            "installment burden) plus raw demographic, income, and credit-history fields, with "
            "median/mode imputation, standard scaling, and one-hot encoding handled inside the "
            "same scikit-learn pipeline."
        )

    # ---------------- SHARE & DEPLOY TAB ----------------
    with tab_share:
        st.markdown('<div class="section-header">🌐 Share This App</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="share-box">
Right now this app runs on <code>localhost</code>, which only you can open. To get a real
public link you can share (with a recruiter, on LinkedIn, etc.), deploy it for free on
<b>Streamlit Community Cloud</b>. It takes about 5 minutes:
</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**Steps to deploy:**")
        st.markdown(
            """
1. Push this whole project folder to a **public GitHub repository**
   (make sure `models/credit_risk_pipeline.joblib`, `data/loan_data.csv`, `requirements.txt`,
   and `.streamlit/config.toml` are all included — don't `.gitignore` the model file).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"**, select your repository, branch, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs everything from `requirements.txt` automatically.
5. You'll get a live link like `https://your-app-name.streamlit.app` — that's your shareable link.
            """
        )
        st.write("")
        st.markdown('<div class="section-header">⚙️ Project Settings Included</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(
                '<div class="glass-card" style="text-align:left;">'
                '<b>🎨 .streamlit/config.toml</b><br>'
                '<span style="color:#9ca3af;font-size:0.85rem;">Native dark theme so built-in '
                'widgets match the custom design, both locally and once deployed.</span></div>',
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                '<div class="glass-card" style="text-align:left;">'
                '<b>🧹 .gitignore</b><br>'
                '<span style="color:#9ca3af;font-size:0.85rem;">Keeps venv/, cache files, and IDE '
                'settings out of your GitHub repo before deploying.</span></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.info(
            "💡 Tip: since Streamlit Cloud rebuilds your app from GitHub, the trained model file "
            "must be committed to the repo (it's small — a few MB). If you'd rather not commit a "
            "binary file, you can instead have the cloud app run `train_model.py` once on startup."
        )


if __name__ == "__main__":
    main()
