"""
app.py
------
Flask web app that serves the trained credit-risk model.

Folder layout expected (see README.md for the full explanation):

credit_risk_app/
├── data/                     raw xlsx files (only needed for training)
├── model/
│   ├── train_model.py
│   ├── model.pkl             <- created by train_model.py
│   ├── scaler.pkl            <- created by train_model.py
│   ├── label_encoder.pkl     <- created by train_model.py
│   └── feature_info.json     <- created by train_model.py
└── app/
    ├── app.py                <- this file
    ├── templates/index.html
    └── static/style.css

Run locally with:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained artifacts once, at startup (not on every request).
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

with open(os.path.join(MODEL_DIR, "feature_info.json")) as f:
    FEATURE_INFO = json.load(f)

NUMERIC_FEATURES = FEATURE_INFO["numeric_features"]
CATEGORICAL_FEATURES = FEATURE_INFO["categorical_features"]   # {col: [options]}
EDUCATION_OPTIONS = FEATURE_INFO["education_options"]
EDUCATION_MAP = FEATURE_INFO["education_map"]
SCALE_COLUMNS = FEATURE_INFO["scale_columns"]
DUMMY_PREFIX_COLUMNS = FEATURE_INFO["dummy_prefix_columns"]
FINAL_FEATURE_ORDER = FEATURE_INFO["final_feature_order"]
CLASSES = FEATURE_INFO["classes"]

RISK_LABELS = {
    "P1": "Excellent (lowest risk)",
    "P2": "Good (low risk)",
    "P3": "Fair (moderate-to-high risk)",
    "P4": "Poor (highest risk)",
}


def build_feature_vector(form: dict) -> pd.DataFrame:
    """Turn raw form input into the exact one-row DataFrame the model expects."""
    row = {}

    # Numeric fields
    for col in NUMERIC_FEATURES:
        raw = form.get(col, "0")
        try:
            row[col] = float(raw)
        except (TypeError, ValueError):
            row[col] = 0.0

    # EDUCATION (ordinal, not one-hot)
    education_choice = form.get("EDUCATION", "GRADUATE")
    row["EDUCATION"] = EDUCATION_MAP.get(education_choice, 3)

    df_row = pd.DataFrame([row])

    # One-hot encode the remaining categorical fields exactly like training did
    for col in DUMMY_PREFIX_COLUMNS:
        choice = form.get(col, CATEGORICAL_FEATURES[col][0])
        for option in CATEGORICAL_FEATURES[col]:
            df_row[f"{col}_{option}"] = 1 if choice == option else 0

    # Scale the numeric columns that were scaled at training time
    present_scale_cols = [c for c in SCALE_COLUMNS if c in df_row.columns]
    if present_scale_cols:
        df_row[present_scale_cols] = scaler.transform(df_row[present_scale_cols])

    # Align to the exact column order the model was trained on; any column
    # the model expects but that we didn't build gets filled with 0.
    df_row = df_row.reindex(columns=FINAL_FEATURE_ORDER, fill_value=0)
    return df_row


@app.route("/")
def home():
    return render_template(
        "index.html",
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        education_options=EDUCATION_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = build_feature_vector(request.form)
        pred_idx = int(model.predict(X)[0])
        pred_class = label_encoder.inverse_transform([pred_idx])[0]
        probs = model.predict_proba(X)[0]
        prob_map = {
            label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }
        result = {
            "prediction": pred_class,
            "risk_label": RISK_LABELS.get(pred_class, pred_class),
            "probabilities": prob_map,
        }
        return render_template(
            "index.html",
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            education_options=EDUCATION_OPTIONS,
            result=result,
            form_values=request.form,
        )
    except Exception as e:
        return render_template(
            "index.html",
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            education_options=EDUCATION_OPTIONS,
            error=str(e),
            form_values=request.form,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint, e.g. for curl / Postman / a separate frontend."""
    data = request.get_json(force=True)
    X = build_feature_vector(data)
    pred_idx = int(model.predict(X)[0])
    pred_class = label_encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(X)[0]
    prob_map = {
        label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2)
        for i, p in enumerate(probs)
    }
    return jsonify({"prediction": pred_class, "probabilities": prob_map})


if __name__ == "__main__":
    # debug=True is fine for local development only; turned off in production
    # (see README.md "Deployment" section).
    app.run(debug=True)
