"""Reproducible fraud-detection training and inference utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

FEATURES = ["amount", "hour", "account_age_days", "transactions_24h", "distance_from_home_km", "merchant_risk_score", "device_trust_score"]


def demo_data(rows: int = 12_000, seed: int = 42) -> pd.DataFrame:
    """A realistic, intentionally imbalanced synthetic demo dataset (not production data)."""
    x, y = make_classification(
        n_samples=rows, n_features=len(FEATURES), n_informative=6, n_redundant=0,
        weights=[0.985, 0.015], class_sep=1.3, flip_y=0.004, random_state=seed,
    )
    frame = pd.DataFrame(x, columns=FEATURES)
    frame["amount"] = np.exp(frame["amount"] + 3.5).clip(1, 15_000)
    frame["hour"] = ((frame["hour"] * 3 + 12) % 24).round().astype(int)
    frame["account_age_days"] = np.exp(frame["account_age_days"] + 5).clip(1, 8_000)
    frame["transactions_24h"] = np.exp(frame["transactions_24h"] + 1).clip(0, 250).round()
    frame["distance_from_home_km"] = np.exp(frame["distance_from_home_km"] + 2).clip(0, 5_000)
    frame["merchant_risk_score"] = ((frame["merchant_risk_score"] + 3) * 16.7).clip(0, 100)
    frame["device_trust_score"] = ((frame["device_trust_score"] + 3) * 16.7).clip(0, 100)
    frame["is_fraud"] = y
    return frame


def train(frame: pd.DataFrame):
    if not set(FEATURES + ["is_fraud"]).issubset(frame.columns):
        raise ValueError("Dataset must include: " + ", ".join(FEATURES + ["is_fraud"]))
    x, y = frame[FEATURES], frame["is_fraud"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, stratify=y, random_state=42)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42)),
    ])
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    return pipeline, x_test, y_test, probabilities, {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }
