import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from neural_net import NeuralNetClassifier

DATA_PATH = "data/brfss_cleaned.csv"
MODEL_DIR = "models"

FEATURE_COLUMNS = [
    "age_group", "bmi", "smoking_status", "exercise", "alcohol",
    "diabetes", "heart_disease", "sex", "race",
]
LABEL_COLUMN = "stroke"
ANALYSIS_COLUMN = "age_is_young"

THRESHOLD_RANGE = np.arange(0.10, 0.91, 0.01)
DEFAULT_THRESHOLD = 0.5
PRECISION_FLOOR = 0.10

MODELS = [
    ("XGBoost (calibrated)", "xgb_model.joblib", "xgb"),
    ("Logistic Regression", "lr_model.joblib", "lr"),
    ("Random Forest (calibrated)", "rf_model.joblib", "rf"),
    ("Neural Network", "nn_model.pt", "nn"),
]


def _load_models():
    loaded = []
    for label, filename, key in MODELS:
        path = os.path.join(MODEL_DIR, filename)
        if key == "nn":
            model = NeuralNetClassifier.load(path)
        else:
            model = joblib.load(path)
        loaded.append((label, model, key))
    return loaded


def _best_threshold(y_true, y_prob, young_mask):
    y_young = y_true[young_mask]
    p_young = y_prob[young_mask]

    best_t = THRESHOLD_RANGE[0]
    best_rec = -1.0
    floor_met = False

    fallback_t = THRESHOLD_RANGE[0]
    fallback_rec = -1.0

    for t in THRESHOLD_RANGE:
        y_pred = (p_young >= t).astype(int)
        prec = precision_score(y_young, y_pred, zero_division=0)
        rec = recall_score(y_young, y_pred)

        if rec > fallback_rec:
            fallback_rec = rec
            fallback_t = float(t)

        if prec >= PRECISION_FLOOR and rec > best_rec:
            best_rec = rec
            best_t = float(t)
            floor_met = True

    if floor_met:
        return best_t, best_rec, ""
    else:
        return fallback_t, fallback_rec, " ⚠ precision floor not met"


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows")

    print("Loading preprocessor...")
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))

    X = preprocessor.transform(df[FEATURE_COLUMNS])
    y = df[LABEL_COLUMN].values
    age_is_young = df[ANALYSIS_COLUMN].values

    _, X_test, _, y_test, _, young_test = train_test_split(
        X, y, age_is_young, test_size=0.2, random_state=42, stratify=y
    )

    n_young = int(young_test.sum())
    n_young_pos = int(y_test[young_test].sum())
    n_older = int((~young_test).sum())
    n_older_pos = int(y_test[~young_test].sum())

    print(f"\nTest set: {len(y_test)} total | "
          f"Young: {n_young} ({n_young_pos} strokes) | "
          f"Older: {n_older} ({n_older_pos} strokes)\n")

    models = _load_models()

    results = []
    header = (
        f"{'Model':<25} {'Threshold':<12} {'Young Rec':<11} {'Young Prec':<11} {'Older Rec':<11}  Note"
    )
    sep = "-" * len(header)

    print("Young Patient Fairness Threshold Tuning")
    print("=" * len(header))
    print()
    print(header)
    print(sep)

    for label, model, key in models:
        y_prob = model.predict_proba(X_test)[:, 1]

        young_mask = young_test == True

        y_young = y_test[young_mask]
        p_young = y_prob[young_mask]

        y_older = y_test[~young_mask]
        p_older = y_prob[~young_mask]

        # Default threshold (0.50)
        default_pred = (p_young >= DEFAULT_THRESHOLD).astype(int)
        default_young_rec = recall_score(y_young, default_pred)
        default_young_prec = precision_score(y_young, default_pred, zero_division=0)
        older_default_pred = (p_older >= DEFAULT_THRESHOLD).astype(int)
        default_older_rec = recall_score(y_older, older_default_pred)

        print(
            f"{label:<25} def {DEFAULT_THRESHOLD:.2f}  "
            f"{default_young_rec:<11.4f} {default_young_prec:<11.4f} "
            f"{default_older_rec:<11.4f}"
        )

        # Optimal threshold
        opt_t, opt_rec, warn = _best_threshold(y_test, y_prob, young_mask)
        opt_pred = (p_young >= opt_t).astype(int)
        opt_young_prec = precision_score(y_young, opt_pred, zero_division=0)
        older_opt_pred = (p_older >= opt_t).astype(int)
        opt_older_rec = recall_score(y_older, older_opt_pred)

        print(
            f"{'':<25} opt {opt_t:<8.2f}  "
            f"{opt_rec:<11.4f} {opt_young_prec:<11.4f} "
            f"{opt_older_rec:<11.4f} {warn}"
        )
        print()

        results.append({"model": key, "optimal_threshold": round(opt_t, 2)})

    thresholds = {
        "young_optimized": {r["model"]: r["optimal_threshold"] for r in results},
        "default": DEFAULT_THRESHOLD,
    }

    out_path = os.path.join(MODEL_DIR, "thresholds.json")
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved thresholds to {out_path}")


if __name__ == "__main__":
    main()
