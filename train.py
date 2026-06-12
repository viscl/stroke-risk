import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    encode_data,
    engineer_features,
    load_data,
    split_and_balance,
)
from model import (
    bootstrap_auc_ci,
    calibrate_models,
    cross_validate_models,
    plot_calibration_curves,
    save_artifacts,
    subgroup_analysis,
    train_models,
    train_stacking_model,
    tune_rf,
    tune_threshold,
    tune_xgb,
)
from neural_net import NeuralNetClassifier


def main(data_path: str = "healthcare-dataset-stroke-data.csv", random_state: int = 42, tune: bool = True) -> None:
    print("Loading data...")
    df = load_data(data_path)
    df = engineer_features(df)
    print(f"  Samples: {len(df)}  |  Stroke prevalence: {df['stroke'].mean():.2%}")

    _, df_test_raw = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df[LABEL_COLUMN]
    )
    df_test_raw = df_test_raw.reset_index(drop=True)

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

    print("\n--- Training neural network ---")
    nn_model = NeuralNetClassifier(input_dim=X_train.shape[1], random_state=random_state)
    nn_model.fit(X_train, y_train)
    print("  Trained.")

    print("\n--- Threshold tuning (F1-optimal on test set) ---")
    calib_probs = calib_xgb.predict_proba(X_test)[:, 1]
    best_threshold = tune_threshold(y_test, calib_probs, metric="f1")
    print(f"  Optimal decision threshold: {best_threshold:.4f}")
    print(f"  (default was 0.5000)")

    threshold_dict = {"decision": best_threshold, "low": 0.3, "high": 0.6}

    print("\n--- Classification reports (using tuned threshold) ---")
    model_probs = {}
    for name, model, label in [
        ("xgb", calib_xgb, "XGBoost (calibrated)"),
        ("lr", results["lr"], "Logistic Regression"),
        ("rf", calib_rf, "Random Forest (calibrated)"),
        ("nn", nn_model, "Neural Network"),
    ]:
        y_prob = model.predict_proba(X_test)[:, 1]
        model_probs[label] = y_prob
        y_pred = (y_prob >= best_threshold).astype(int)

        print(f"\n  === {label} ===")
        print(f"  {classification_report(y_test, y_pred)}")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion matrix:\n    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}\n    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    print("\n--- Bootstrap AUC (95% CI, 1000 samples) ---")
    for label, probs in model_probs.items():
        ci = bootstrap_auc_ci(y_test, probs)
        print(f"  {label}: AUC = {ci['mean']:.4f}  (95% CI: {ci['lower']:.4f} — {ci['upper']:.4f})")

    print("\n--- Calibration curves ---")
    calib_path = os.path.join("artifacts", "calibration_curves.png")
    plot_calibration_curves(model_probs, y_test, calib_path)
    print(f"  Saved to {calib_path}")

    print("\n--- Subgroup analysis ---")
    sub_results = subgroup_analysis(df_test_raw, y_test, model_probs, threshold=best_threshold)

    print("  Overall metrics:")
    for label, metrics in sub_results["overall"].items():
        print(f"    {label}: AUC={metrics['auc']:.4f}  Recall={metrics['recall']:.4f}")

    for col, groups in sub_results["subgroups"].items():
        print(f"\n  By {col}:")
        flags = []
        small_samples = []
        for val, metrics_by_model in groups.items():
            first_model = next(iter(metrics_by_model.values()))
            n_total = first_model["n"]
            n_pos = first_model["positives"]
            if n_pos <= 3:
                small_samples.append((col, val, n_total, n_pos))
            for model_name, m in metrics_by_model.items():
                if m["flagged"]:
                    flags.append(f"    ⚠ {model_name} on {col}={val}: "
                                 f"recall={m['recall']:.4f} (drop={m['recall_drop']:+.4f})")
        for val, metrics_by_model in groups.items():
            first_model = next(iter(metrics_by_model.values()))
            n_total = first_model["n"]
            n_pos = first_model["positives"]
            parts = [f"    {col}={val}  (n={n_total}, pos={n_pos})"]
            for model_name, m in metrics_by_model.items():
                parts.append(f"{model_name} AUC={m['auc']:.4f} Rec={m['recall']:.4f}")
            print("  | ".join(parts))
        if flags:
            print("\n  Recall drops flagged:")
            for f in flags:
                print(f)
        if small_samples:
            print("\n  ⚠ Small-sample subgroups (recall statistically unstable):")
            for col_name, val_name, n_tot, n_p in small_samples:
                print(f"    {col_name}={val_name}: n={n_tot}, positives={n_p}")

    print("\n--- Stacking ensemble ---")
    stacking_model = train_stacking_model(
        X_train, y_train,
        xgb_params=xgb_params,
        rf_params=rf_params,
        random_state=random_state,
    )

    def _stacking_predict_proba(model, xgb_mod, lr_mod, rf_mod, nn_mod, X):
        xgb_p = xgb_mod.predict_proba(X)[:, 1]
        lr_p = lr_mod.predict_proba(X)[:, 1]
        rf_p = rf_mod.predict_proba(X)[:, 1]
        nn_p = nn_mod.predict_proba(X)[:, 1]
        meta = np.column_stack([xgb_p, lr_p, rf_p, nn_p])
        return model.predict_proba(meta)[:, 1]

    stack_probs = _stacking_predict_proba(stacking_model, calib_xgb, results["lr"], calib_rf, nn_model, X_test)
    stack_best_thresh = tune_threshold(y_test, stack_probs, metric="f1")
    stack_pred = (stack_probs >= stack_best_thresh).astype(int)
    print(f"  Optimal stacking threshold: {stack_best_thresh:.4f}")
    print(f"\n  === Stacking (LR meta-model) ===")
    print(f"  {classification_report(y_test, stack_pred)}")
    cm_s = confusion_matrix(y_test, stack_pred)
    print(f"  Confusion matrix:\n    TN={cm_s[0,0]:5d}  FP={cm_s[0,1]:5d}\n    FN={cm_s[1,0]:5d}  TP={cm_s[1,1]:5d}")

    print("\nSaving artifacts...")
    save_artifacts(calib_xgb, results["lr"], calib_rf, preprocessor, threshold=threshold_dict, stacking=stacking_model, nn_model=nn_model)
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
