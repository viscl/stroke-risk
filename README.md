# Stroke Risk Early Detection

An ML pipeline for early detection of stroke risk, inspired by a personal family experience with stroke. The goal is to provide an accessible, interpretable screening tool that can flag elevated risk before a crisis occurs.

## Features

12 clinical and lifestyle features:

- **Age** — years
- **Systolic BP / Diastolic BP** — blood pressure (mmHg)
- **Heart Rate Variability** — ms
- **BMI** — body mass index
- **Cholesterol** — mg/dL
- **Atrial Fibrillation** — 0 or 1
- **Sleep Hours** — hours per night
- **Activity Level** — 1 (sedentary) to 10 (highly active)
- **Diabetes** — 0 or 1
- **Smoking** — 0 or 1
- **Family History** — 0 or 1

Binary label: **stroke_risk** (0 = low risk, 1 = elevated risk).

## Tech Stack

| Component | Library |
|---|---|
| Gradient boosting | XGBoost |
| Linear baseline | Logistic Regression (scikit-learn) |
| Class imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| Preprocessing | scikit-learn StandardScaler |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/<your-org>/stroke-risk.git
cd stroke-risk
pip install -r requirements.txt

# 2. Train models (uses synthetic data by default, or pass a CSV path)
python train.py

# 3. Predict
python -c "
from predict import predict_risk

patient = {
    'age': 62,
    'systolic_bp': 148.0,
    'diastolic_bp': 92.0,
    'heart_rate_variability': 22.0,
    'BMI': 31.2,
    'cholesterol': 245.0,
    'atrial_fibrillation': 1,
    'sleep_hours': 5.5,
    'activity_level': 2.0,
    'diabetes': 1,
    'smoking': 1,
    'family_history': 1,
}

result = predict_risk(patient)
print(f\"Risk: {result['risk_probability']:.2%} ({result['risk_level']})\")
for feat in result['shap_explanation'][:3]:
    print(f\"  {feat['feature']}: SHAP={feat['shap_value']:+.4f}\")
"
```

## Risk Levels

| Probability | Level |
|---|---|
| < 0.3 | Low |
| 0.3 – 0.6 | Moderate |
| \> 0.6 | High |

## Using Your Own Data

Supply a CSV with the required columns and a `stroke_risk` label:

```bash
python train.py /path/to/your_data.csv
```

## License

MIT
