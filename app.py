import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.model import FEATURES, demo_data, train

st.set_page_config(page_title="Sentinel | Fraud Intelligence", page_icon="🛡️", layout="wide")
st.title("🛡️ Sentinel — Financial Fraud Intelligence")
st.caption("Explainable machine-learning screening for transaction-risk prioritisation")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload labelled CSV (optional)", type="csv", help="Required target: is_fraud. Fraud = 1; legitimate = 0.")
    st.divider()
    st.header("Decision policy")
    threshold = st.slider("Escalation threshold", .01, .99, .50, .01, help="Transactions at or above this risk score are sent for review.")

@st.cache_data(show_spinner=False)
def load_default():
    return demo_data()

try:
    data = pd.read_csv(uploaded) if uploaded else load_default()
    model, x_test, y_test, probabilities, metrics = train(data)
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()

predictions = (probabilities >= threshold).astype(int)
fraud_rate = data.is_fraud.mean()
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)

if not uploaded:
    st.info("Demo mode uses a synthetic, imbalanced dataset. Upload real labelled data for a genuine evaluation.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions analysed", f"{len(data):,}")
c2.metric("Fraud prevalence", f"{fraud_rate:.2%}")
c3.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", help="Best headline metric for rare-fraud detection.")
c4.metric("Recall at threshold", f"{recall:.1%}", help="Share of fraud cases captured.")

left, right = st.columns(2)
with left:
    st.subheader("Precision–recall trade-off")
    curve = pd.DataFrame({"Recall": metrics["recall"], "Precision": metrics["precision"]})
    st.plotly_chart(px.line(curve, x="Recall", y="Precision", template="plotly_dark"), use_container_width=True)
    st.caption("PR-AUC is preferred to accuracy: a naive model can be 98% accurate while missing nearly all fraud.")
with right:
    st.subheader("Decision outcomes")
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    matrix = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"])
    st.dataframe(matrix, use_container_width=True)
    st.metric("Precision at threshold", f"{precision:.1%}", help="Share of alerts that are truly fraudulent.")

st.subheader("Live transaction screening")
with st.form("transaction"):
    a, b, c, d = st.columns(4)
    values = {
        "amount": a.number_input("Amount ($)", 0.0, value=275.0),
        "hour": b.number_input("Hour (0–23)", 0, 23, value=2),
        "account_age_days": c.number_input("Account age (days)", 0.0, value=45.0),
        "transactions_24h": d.number_input("Transactions in 24h", 0.0, value=12.0),
        "distance_from_home_km": a.number_input("Distance from home (km)", 0.0, value=350.0),
        "merchant_risk_score": b.number_input("Merchant risk (0–100)", 0.0, 100.0, value=70.0),
        "device_trust_score": c.number_input("Device trust (0–100)", 0.0, 100.0, value=25.0),
    }
    submitted = st.form_submit_button("Screen transaction", type="primary")

if submitted:
    risk = float(model.predict_proba(pd.DataFrame([values])[FEATURES])[:, 1][0])
    if risk >= threshold:
        st.error(f"🔴 REVIEW REQUIRED — predicted fraud risk: **{risk:.1%}**")
    else:
        st.success(f"🟢 LOW RISK — predicted fraud risk: **{risk:.1%}**")
    st.progress(risk, text=f"Risk score {risk:.1%} · escalation threshold {threshold:.1%}")

with st.expander("Model governance and limitations"):
    st.markdown("""
    - The model ranks risk; it does **not** autonomously block transactions.
    - The threshold expresses an operations trade-off: higher recall catches more fraud but generates more false alerts.
    - Before production use: monitor data drift, segment fairness, calibration, false-positive costs, latency and privacy controls.
    - Retrain on time-based splits with real, leakage-audited labels. Never treat this synthetic demo's metrics as business performance.
    """)
