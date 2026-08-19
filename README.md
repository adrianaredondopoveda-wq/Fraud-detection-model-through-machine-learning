# Sentinel — Financial Fraud Detection

An interview-ready Streamlit project for ML-driven transaction-risk screening.

## Features

- Reproducible, imbalanced demo data — works immediately on Streamlit Cloud.
- Leakage-safe stratified train/test split and a `Pipeline` with imputation, robust scaling and class-weighted logistic regression.
- PR-AUC and precision/recall metrics, appropriate for rare-event classification.
- Adjustable operational threshold, confusion matrix and a live transaction-screening form.
- CSV upload path for real labelled data. Required columns:

`amount, hour, account_age_days, transactions_24h, distance_from_home_km, merchant_risk_score, device_trust_score, is_fraud`

The target uses `0 = legitimate` and `1 = fraud`.

## Deploy to Streamlit Cloud

Keep this exact structure in GitHub:

```text
repo/
├── app.py
├── requirements.txt
└── src/
    ├── __init__.py
    └── model.py
```

Set `app.py` as the Streamlit entrypoint. Streamlit Cloud installs `requirements.txt` automatically.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
