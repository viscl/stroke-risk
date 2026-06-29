import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

DATA_PATH = "data/brfss_cleaned.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

FEATURE_COLUMNS = [
    "age_group", "bmi", "smoking_status", "exercise", "alcohol",
    "diabetes", "heart_disease", "sex", "race",
]
LABEL_COLUMN = "stroke"
ANALYSIS_COLUMN = "age_is_young"


def _reconstruct_test_set():
    df = pd.read_csv(DATA_PATH)
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
    X = preprocessor.transform(df[FEATURE_COLUMNS])
    y = df[LABEL_COLUMN].values
    age_is_young = df[ANALYSIS_COLUMN].values

    _, X_test, _, y_test, _, young_test = train_test_split(
        X, y, age_is_young, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test, young_test


def _load_feature_names():
    path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(path) as f:
        return json.load(f)


def _load_xgb_model():
    calib_xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    raw_xgb = calib_xgb.calibrated_classifiers_[0].estimator
    return raw_xgb


def _mean_abs_shap(shap_values):
    keys = sorted(shap_values.keys(), key=lambda k: np.abs(shap_values[k]).mean(), reverse=True)
    return {k: round(float(np.abs(shap_values[k]).mean()), 4) for k in keys}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reconstructing test set...")
    X_test, y_test, young_test = _reconstruct_test_set()
    young_mask = young_test == True

    n_young = int(young_mask.sum())
    n_young_pos = int(y_test[young_mask].sum())
    n_older = int((~young_mask).sum())
    n_older_pos = int(y_test[~young_mask].sum())

    print(f"  {len(y_test)} total | "
          f"Young: {n_young} ({n_young_pos} strokes) | "
          f"Older: {n_older} ({n_older_pos} strokes)")

    feature_names = _load_feature_names()
    print(f"  Feature count: {len(feature_names)}")

    print("\nLoading XGBoost model and computing SHAP values...")
    raw_xgb = _load_xgb_model()
    explainer = shap.TreeExplainer(raw_xgb)
    shap_values_full = explainer(X_test)

    sv_young = shap_values_full.values[young_mask]
    sv_older = shap_values_full.values[~young_mask]
    X_young = X_test[young_mask]
    X_older = X_test[~young_mask]

    mean_abs_young = np.abs(sv_young).mean(axis=0)
    mean_abs_older = np.abs(sv_older).mean(axis=0)

    young_idx = np.argsort(mean_abs_young)[::-1]
    older_idx = np.argsort(mean_abs_older)[::-1]

    print()
    header = (
        f"  {'Top 5 for YOUNG stroke risk':<42} "
        f"{'Top 5 for OLDER stroke risk':<42}"
    )
    print(header)
    print("  " + "-" * 84)
    for rank in range(5):
        yi = young_idx[rank]
        oi = older_idx[rank]
        print(
            f"  {feature_names[yi]:<30} {mean_abs_young[yi]:>7.4f}    "
            f"{feature_names[oi]:<30} {mean_abs_older[oi]:>7.4f}"
        )

    print("\nGenerating summary plots...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_young, X_young, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_young.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/shap_young.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_older, X_older, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_older.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/shap_older.png")

    young_dict = {name: round(float(val), 4) for name, val in zip(feature_names, mean_abs_young)}
    older_dict = {name: round(float(val), 4) for name, val in zip(feature_names, mean_abs_older)}

    young_sorted = dict(sorted(young_dict.items(), key=lambda x: x[1], reverse=True))
    older_sorted = dict(sorted(older_dict.items(), key=lambda x: x[1], reverse=True))

    comparison = {"young": young_sorted, "older": older_sorted}
    json_path = os.path.join(OUTPUT_DIR, "shap_comparison.json")
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved {json_path}")


if __name__ == "__main__":
    main()
