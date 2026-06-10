# Stroke Risk Early Detection

An ML pipeline for early detection of stroke risk, inspired by a personal family experience with stroke. The goal is to provide an accessible, interpretable screening tool that can flag elevated risk before a crisis occurs.

Trained on the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (5,110 patient records, ~5% stroke prevalence).

## Features

10 clinical and demographic features from the Kaggle dataset:

| Feature | Type | Values |
|---|---|---|
| gender | categorical | Male, Female, Other |
| age | numeric | years |
| hypertension | binary | 0, 1 |
| heart_disease | binary | 0, 1 |
| ever_married | categorical | Yes, No |
| work_type | categorical | Private, Self-employed, Govt_job, children, Never_worked |
| Residence_type | categorical | Urban, Rural |
| avg_glucose_level | numeric | mg/dL |
| bmi | numeric | kg/m^2 (N/A imputed with median) |
| smoking_status | categorical | never smoked, formerly smoked, smokes, Unknown |

Binary label: **stroke** (0 = no stroke, 1 = stroke).

## Tech Stack

| Component | Library |
|---|---|
| Gradient boosting | XGBoost, Random Forest |
| Linear baseline | Logistic Regression (scikit-learn) |
| Class imbalance | SMOTE per fold (imbalanced-learn) |
| Cross-validation | 5-fold stratified |
| Explainability | SHAP TreeExplainer |
| Preprocessing | scikit-learn ColumnTransformer + OneHotEncoder + StandardScaler |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/<your-org>/stroke-risk.git
cd stroke-risk
pip install -r requirements.txt

# 2. Train models
python train.py

# 3. Predict
python -c "
from predict import predict_risk

patient = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 0,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'smoking_status': 'formerly smoked',
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
| 0.3 - 0.6 | Moderate |
| > 0.6 | High |

## Using Your Own Data

Supply a CSV with the required columns and a `stroke` label:

```bash
python train.py /path/to/your_data.csv
```

## License

MIT
