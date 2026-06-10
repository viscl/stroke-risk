import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from data import FEATURE_COLUMNS, encode_data, load_data, split_and_balance
from model import (
    calibrate_models,
    cross_validate_models,
    save_artifacts,
    train_models,
    tune_rf,
    tune_threshold,
    tune_xgb,
)


def main(data_path: str = "healthcare-dataset-stroke-data.csv", random_state: int = 42, tune: bool = True) -> None:
    print("Loading data...")
    df = load_data(data_path)
    print(f"  Samples: {len(df)}  |  Stroke prevalence: {df['stroke'].mean():.2%}")

    print("\nEncoding features...")
    X, y, preprocessor, feature_names = encode_data(df)
    print(f"  Encoded feature count: {len(feature_names)}")

    print("\n--- 5-Fold Cross-Validation (SMOTE per fold) ---")
    cv_results = cross_validate_models(X, y, n_splits=5, random_state=random_state)

    for model_name in ["xgb", "lr", "rf"]:
        print(f"  {model_name.upper()}:")
        for metric, (mean, std) in cv_results[model_name].items():
            print(f"    {metric}: {mean:.4f} +- {std:.4f}")

    xgb_params = None
    rf_params = None

    if tune:
        print("\n--- Optuna hyperparameter tuning (XGBoost) ---")
        xgb_params = tune_xgb(X, y, n_trials=50, random_state=random_state)
        print(f"  Best params: {xgb_params}")

        print("\n--- Optuna hyperparameter tuning (Random Forest) ---")
        rf_params = tune_rf(X, y, n_trials=50, random_state=random_state)
        print(f"  Best params: {rf_params}")
    else:
        print("\n--- Skipping Optuna tuning (--no-tune) ---")

    print("\nSplitting for final evaluation...")
    X_train, X_test, y_train, y_test = split_and_balance(
        X, y, test_size=0.2, random_state=random_state, apply_smote=True
    )
    print(f"  Train (after SMOTE): {len(y_train)}  |  Test: {len(y_test)}")
    print(f"  Train stroke rate: {y_train.mean():.2%}")

    print("\nTraining final models...")
    results = train_models(
        X_train, y_train, X_test, y_test,
        random_state=random_state,
        xgb_params=xgb_params,
        rf_params=rf_params,
    )

    print("\n--- Platt scaling calibration (XGBoost + Random Forest) ---")
    calib_xgb, calib_rf = calibrate_models(
        results["xgb"], results["rf"], X_train, y_train
    )
    print("  Calibrated.")

    print("\n--- Threshold tuning (F1-optimal on test set) ---")
    calib_probs = calib_xgb.predict_proba(X_test)[:, 1]
    best_threshold = tune_threshold(y_test, calib_probs, metric="f1")
    print(f"  Optimal decision threshold: {best_threshold:.4f}")
    print(f"  (default was 0.5000)")

    threshold_dict = {"decision": best_threshold, "low": 0.3, "high": 0.6}

    print("\n--- Classification reports (using tuned threshold) ---")
    for name, model, label in [
        ("xgb", calib_xgb, "XGBoost (calibrated)"),
        ("lr", results["lr"], "Logistic Regression"),
        ("rf", calib_rf, "Random Forest (calibrated)"),
    ]:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= best_threshold).astype(int)

        print(f"\n  === {label} ===")
        print(f"  {classification_report(y_test, y_pred)}")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion matrix:\n    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}\n    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    print("\nSaving artifacts...")
    save_artifacts(calib_xgb, results["lr"], calib_rf, preprocessor, threshold=threshold_dict)
    print("  Saved to artifacts/")

    print("\n--- Sample predictions ---")
    test_sample = pd.DataFrame([
        {
            "gender": "Male",
            "age": 67,
            "hypertension": 0,
            "heart_disease": 1,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 228.69,
            "bmi": 36.6,
            "smoking_status": "formerly smoked",
        },
        {
            "gender": "Female",
            "age": 35,
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "No",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 85.0,
            "bmi": 24.0,
            "smoking_status": "never smoked",
        },
    ])

    from predict import predict_risk

    for idx, row in test_sample.iterrows():
        result = predict_risk(row.to_dict())
        print(
            f"  Patient {idx + 1}: risk={result['risk_probability']:.2%}"
            f" ({result['risk_level']}),"
            f" top factor: {result['shap_explanation'][0]['feature']}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", default="healthcare-dataset-stroke-data.csv")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna hyperparameter tuning")
    args = parser.parse_args()

    main(data_path=args.data_path, tune=not args.no_tune)
