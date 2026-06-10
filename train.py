import sys

import pandas as pd

from data import FEATURE_COLUMNS, encode_data, load_data, split_and_balance
from model import cross_validate_models, save_artifacts, train_models


def main(data_path: str = "healthcare-dataset-stroke-data.csv", random_state: int = 42) -> None:
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
            print(f"    {metric}: {mean:.4f} ± {std:.4f}")

    print("\nSplitting for final evaluation...")
    X_train, X_test, y_train, y_test = split_and_balance(
        X, y, test_size=0.2, random_state=random_state, apply_smote=True
    )
    print(f"  Train (after SMOTE): {len(y_train)}  |  Test: {len(y_test)}")
    print(f"  Train stroke rate: {y_train.mean():.2%}")

    print("\nTraining final models (XGBoost + Logistic Regression + Random Forest)...")
    results = train_models(X_train, y_train, X_test, y_test, random_state=random_state)

    for name, label in [("xgb", "XGBoost"), ("lr", "Logistic Regression"), ("rf", "Random Forest")]:
        print(f"\n  --- {label} ---")
        for metric, value in results[f"{name}_metrics"].items():
            print(f"    {metric}: {value:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(results["xgb"], results["lr"], results["rf"], preprocessor)
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
    data_path = sys.argv[1] if len(sys.argv) > 1 else "healthcare-dataset-stroke-data.csv"
    main(data_path)
