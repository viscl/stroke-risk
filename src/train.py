import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import calibrate_models, cross_validate_models, train_models
from neural_net import NeuralNetClassifier

DATA_PATH = "data/brfss_cleaned.csv"
MODEL_DIR = "models"

FEATURE_COLUMNS = [
    "age_group",
    "bmi",
    "smoking_status",
    "exercise",
    "alcohol",
    "diabetes",
    "heart_disease",
    "sex",
    "race",
]
NUMERIC_COLUMNS = ["bmi", "heart_disease"]
CATEGORICAL_COLUMNS = [
    "age_group",
    "smoking_status",
    "exercise",
    "alcohol",
    "diabetes",
    "sex",
    "race",
]
LABEL_COLUMN = "stroke"
ANALYSIS_COLUMN = "age_is_young"


def get_feature_names(preprocessor):
    names = []
    for name, _, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
    return names


def load_data():
    return pd.read_csv(DATA_PATH)


def encode_data(df):
    X_raw = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].values

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLUMNS),
        ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
    ])
    X = preprocessor.fit_transform(X_raw)
    feature_names = get_feature_names(preprocessor)

    return X, y, preprocessor, feature_names


def _print_metrics(label, metrics, n, n_pos):
    print(f"  {label}  (n={n}, strokes={n_pos})")
    print(f"    AUC: {metrics['auc']:.4f}  Prec: {metrics['precision']:.4f}  Rec: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")


def _metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading BRFSS data...")
    df = load_data()
    print(f"  {len(df)} rows")

    print("\nEncoding features...")
    X, y, preprocessor, feature_names = encode_data(df)
    age_is_young = df[ANALYSIS_COLUMN].values
    print(f"  Encoded feature count: {len(feature_names)}")

    print("\n--- 5-Fold Cross-Validation (SMOTE per fold) ---")
    cv_results = cross_validate_models(X, y, n_splits=5)
    for model_name in ["xgb", "lr", "rf"]:
        parts = [f"  {model_name.upper()}:"]
        for metric, (mean, std) in cv_results[model_name].items():
            parts.append(f"{metric}={mean:.4f}+-{std:.4f}")
        print("  |  ".join(parts))

    print("\nSplitting for final evaluation...")
    X_train, X_test, y_train, y_test, young_train, young_test = train_test_split(
        X, y, age_is_young, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"  Train (after SMOTE): {len(y_train)}  |  Test: {len(y_test)}")
    print(f"  Train stroke rate: {y_train.mean():.2%}")

    print("\nTraining final models...")
    results = train_models(X_train, y_train, X_test, y_test, random_state=42)

    print("\n--- Platt scaling calibration (XGBoost + Random Forest) ---")
    calib_xgb, calib_rf = calibrate_models(
        results["xgb"], results["rf"], X_train, y_train
    )

    print("\n--- Training neural network ---")
    nn_model = NeuralNetClassifier(input_dim=X_train.shape[1], random_state=42)
    nn_model.fit(X_train, y_train)
    print("  Trained.")

    print("\nOverall metrics (threshold=0.5):")
    models = [
        ("XGBoost (calibrated)", calib_xgb),
        ("Logistic Regression", results["lr"]),
        ("Random Forest (calibrated)", calib_rf),
        ("Neural Network", nn_model),
    ]

    all_metrics = {}
    for label, model in models:
        y_prob = model.predict_proba(X_test)[:, 1]
        m = _metrics(y_test, y_prob)
        all_metrics[label] = {"overall": m}
        _print_metrics(label, m, len(y_test), int(y_test.sum()))

    print("\n--- Metrics by age group (young=True) ---")
    young_mask = young_test == True
    for label, model in models:
        y_prob = model.predict_proba(X_test[young_mask])[:, 1]
        m = _metrics(y_test[young_mask], y_prob)
        all_metrics[label]["young"] = m
        _print_metrics(label, m, int(young_mask.sum()), int(y_test[young_mask].sum()))

    print("\n--- Metrics by age group (young=False) ---")
    old_mask = young_test == False
    for label, model in models:
        y_prob = model.predict_proba(X_test[old_mask])[:, 1]
        m = _metrics(y_test[old_mask], y_prob)
        all_metrics[label]["older"] = m
        _print_metrics(label, m, int(old_mask.sum()), int(y_test[old_mask].sum()))

    print("\nSaving models...")
    joblib.dump(calib_xgb, os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(results["lr"], os.path.join(MODEL_DIR, "lr_model.joblib"))
    joblib.dump(calib_rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    nn_model.save(os.path.join(MODEL_DIR, "nn_model.pt"))

    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  Saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
