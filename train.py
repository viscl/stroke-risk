import pandas as pd

from data import load_data, preprocess_data
from model import save_artifacts, train_models


def main(data_path: str | None = None, random_state: int = 42) -> None:
    print("Loading data...")
    X, y = load_data(data_path)

    print(f"  Samples: {len(y)}  |  Stroke risk prevalence: {y.mean():.2%}")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, random_state=random_state
    )

    print(f"  Train: {len(y_train)}  |  Test: {len(y_test)}")

    print("Training models (XGBoost + Logistic Regression)...")
    results = train_models(X_train, y_train, X_test, y_test, random_state=random_state)

    print("\n--- XGBoost ---")
    for metric, value in results["xgb_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print("\n--- Logistic Regression ---")
    for metric, value in results["lr_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(results["xgb"], results["lr"], scaler)
    print("  Saved to artifacts/\n")

    test_sample = pd.DataFrame(
        {
            "age": [62, 45],
            "systolic_bp": [148.0, 118.0],
            "diastolic_bp": [92.0, 76.0],
            "heart_rate_variability": [22.0, 48.0],
            "BMI": [31.2, 24.1],
            "cholesterol": [245.0, 178.0],
            "atrial_fibrillation": [1, 0],
            "sleep_hours": [5.5, 7.8],
            "activity_level": [2.0, 7.5],
            "diabetes": [1, 0],
            "smoking": [1, 0],
            "family_history": [1, 0],
        }
    )

    print("Sample predictions from predict_risk:")
    from predict import predict_risk

    for idx, row in test_sample.iterrows():
        result = predict_risk(row.to_dict())
        print(
            f"  Patient {idx + 1}: risk={result['risk_probability']:.2%}"
            f" ({result['risk_level']}),"
            f" top factor: {result['shap_explanation'][0]['feature']}"
        )


if __name__ == "__main__":
    main()
