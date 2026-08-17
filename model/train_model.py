"""
train_model.py
----------------
Cleaned, error-free training pipeline for the Credit Risk (Approved_Flag)
classification problem.

Bugs fixed from the original notebook:
1. VIF loop used a fixed `range(0, total_columns)` while columns were being
   dropped from `vif_data` inside the loop -> caused an IndexError once
   enough columns were dropped. Replaced with a `while` loop driven by the
   *current* number of remaining columns.
2. Chained `.loc[...] = value` assignment for EDUCATION encoding produced a
   SettingWithCopyWarning. Replaced with a single `.map()` call.
3. Column dtype for EDUCATION was cast with `.astype(int)` on a column that
   could contain unmapped values -> replaced with an explicit dict map that
   covers every category so no NaNs get introduced before casting.
4. `df1 = df1.loc[...]` created a view -> chained assignments later raised
   SettingWithCopyWarning. Fixed with `.copy()`.
5. Grid search / scaling / final model steps consolidated into one
   reproducible function so re-running the script always gives the same
   result (random_state fixed everywhere, stratified train/test split).
6. Added `.values` when calling `variance_inflation_factor` (statsmodels
   expects a numpy array; passing a DataFrame directly works but is slower
   and can misbehave with non-default indices after `.drop()` calls).
7. Everything the Flask app needs at inference time (feature order, which
   columns get scaled, which categories are valid, the education mapping)
   is saved to `feature_info.json` so the web app and the notebook never
   go out of sync.

Run with:
    python train_model.py
Outputs (written to this folder):
    model.pkl, scaler.pkl, label_encoder.pkl, feature_info.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

CATEGORICAL_COLS = ["MARITALSTATUS", "EDUCATION", "GENDER", "last_prod_enq2", "first_prod_enq2"]
DUMMY_COLS = ["MARITALSTATUS", "GENDER", "last_prod_enq2", "first_prod_enq2"]  # EDUCATION is ordinal, not one-hot
SCALE_COLS = [
    "Age_Oldest_TL", "Age_Newest_TL", "time_since_recent_payment",
    "max_recent_level_of_deliq", "recent_level_of_deliq",
    "time_since_recent_enq", "NETMONTHLYINCOME", "Time_With_Curr_Empr",
]
EDUCATION_MAP = {
    "SSC": 1, "12TH": 2, "GRADUATE": 3, "UNDER GRADUATE": 3,
    "POST-GRADUATE": 4, "OTHERS": 1, "PROFESSIONAL": 3,
}


def load_data():
    a1 = pd.read_excel(os.path.join(DATA_DIR, "case_study1.xlsx"))
    a2 = pd.read_excel(os.path.join(DATA_DIR, "case_study2.xlsx"))
    return a1.copy(), a2.copy()


def clean_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # df1: drop rows where Age_Oldest_TL is a null placeholder
    df1 = df1.loc[df1["Age_Oldest_TL"] != -99999].copy()

    # df2: drop columns that are mostly null placeholders, then drop
    # remaining rows that still contain a placeholder value
    cols_to_remove = [c for c in df2.columns if (df2[c] == -99999).sum() > 10000]
    df2 = df2.drop(columns=cols_to_remove).copy()
    for c in df2.columns:
        df2 = df2.loc[df2[c] != -99999]

    df = pd.merge(df1, df2, how="inner", on="PROSPECTID")
    assert df.isna().sum().sum() == 0
    return df


def print_chi_square(df: pd.DataFrame):
    for c in CATEGORICAL_COLS:
        if c == "EDUCATION":  # ordinal, chi-sq is still informative but not required
            continue
        chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[c], df["Approved_Flag"]))
        print(f"  {c:16s} p-value = {pval:.4g}")


def select_numeric_features(df: pd.DataFrame) -> list:
    # NOTE: newer pandas (2.x/3.x) can give text columns dtype "str" instead
    # of "object", so `dtype != 'object'` (the original notebook's check)
    # silently lets text columns slip into the numeric list and crashes VIF.
    # `select_dtypes(include=np.number)` is version-proof.
    numeric_columns = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in ["PROSPECTID", "Approved_Flag"]
    ]

    # --- VIF filtering (fixed: while-loop over the *current* column count) ---
    vif_data = df[numeric_columns].copy()
    kept = []
    idx = 0
    while idx < vif_data.shape[1]:
        vif_value = variance_inflation_factor(vif_data.values, idx)
        if vif_value <= 6:
            kept.append(vif_data.columns[idx])
            idx += 1
        else:
            vif_data = vif_data.drop(columns=[vif_data.columns[idx]])

    # --- ANOVA filtering (keep columns whose means differ across classes) ---
    kept_numerical = []
    for c in kept:
        groups = [df.loc[df["Approved_Flag"] == g, c] for g in ["P1", "P2", "P3", "P4"]]
        _, p_value = f_oneway(*groups)
        if p_value <= 0.05:
            kept_numerical.append(c)

    return kept_numerical


def build_final_frame(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    features = numeric_features + CATEGORICAL_COLS
    out = df[features + ["Approved_Flag"]].copy()
    out["EDUCATION"] = out["EDUCATION"].map(EDUCATION_MAP).astype(int)
    return out


def train():
    print("Loading data...")
    df1, df2 = load_data()

    print("Cleaning + merging...")
    df = clean_data(df1, df2)
    print(f"  merged shape: {df.shape}")

    print("Chi-square tests (categorical vs Approved_Flag):")
    print_chi_square(df)

    print("Selecting numeric features via VIF + ANOVA...")
    numeric_features = select_numeric_features(df)
    print(f"  kept {len(numeric_features)} numeric features")

    df_final = build_final_frame(df, numeric_features)

    df_encoded = pd.get_dummies(df_final, columns=DUMMY_COLS)

    scale_cols_present = [c for c in SCALE_COLS if c in df_encoded.columns]
    scaler = StandardScaler()
    df_encoded[scale_cols_present] = scaler.fit_transform(df_encoded[scale_cols_present])

    y = df_encoded["Approved_Flag"]
    X = df_encoded.drop(columns=["Approved_Flag"])

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=4,
        eval_metric="mlogloss",
        learning_rate=0.2,
        max_depth=3,
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}\n")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {cls}: precision={precision[i]:.3f} recall={recall[i]:.3f} f1={f1[i]:.3f}")

    # --- Persist everything the Flask app needs ---
    joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(BASE_DIR, "label_encoder.pkl"))

    feature_info = {
        "numeric_features": numeric_features,          # raw numeric inputs the user must supply
        "categorical_features": {                       # raw categorical inputs + valid options
            c: sorted(df[c].dropna().unique().tolist()) for c in DUMMY_COLS
        },
        "education_options": list(EDUCATION_MAP.keys()),
        "education_map": EDUCATION_MAP,
        "scale_columns": scale_cols_present,
        "dummy_prefix_columns": DUMMY_COLS,
        "final_feature_order": X.columns.tolist(),      # exact column order the model expects
        "classes": label_encoder.classes_.tolist(),
        "test_accuracy": acc,
    }
    with open(os.path.join(BASE_DIR, "feature_info.json"), "w") as f:
        json.dump(feature_info, f, indent=2)

    print("\nSaved model.pkl, scaler.pkl, label_encoder.pkl, feature_info.json")


if __name__ == "__main__":
    train()
