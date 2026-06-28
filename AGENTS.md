# AGENTS.md — Stroke Risk

BRFSS-only pipeline. No tests, no CI, no linter/typecheck config.

## Commands

```bash
# One-time setup
pip install -r requirements.txt          # torch is ~2.5 GB — plan accordingly

# All commands run from the stroke-risk/ directory
cd stroke-risk

# Train (~5 min on 8-core CPU)
python src/train.py
```

Prediction is not yet implemented — `src/predict.py` will be built in a
future step.

## Required order

`src/train.py` must complete before prediction works — it writes
`models/{xgb,lr,rf}_model.joblib`, `preprocessor.joblib`,
`nn_model.pt`, `feature_names.json`, and `metrics.json`.

## Pipeline

| Data | Features | Artifact dir |
|---|---|---|
| `data/brfss_cleaned.csv` (CDC BRFSS 2022, 457k) | bmi, smoking_status, exercise, alcohol, diabetes, heart_disease, sex, race, age_group | `models/` |

## Pipeline flow

```
load_data → encode_data → cross_validate_models (5-fold CV, SMOTE per fold)
  → SMOTE train split → train_models (XGB+LR+RF)
  → calibrate_models (Platt scaling for XGB+RF)
  → train_nn (NeuralNetClassifier: MLP 128→64, ReLU+Dropout, Adam, early stopping)
  → metrics by age group → save models
```

## Source layout

```
stroke-risk/
  src/
    train.py          # Main entrypoint — orchestrates full training run
    model.py          # _build_models, cross_validate_models, train_models, calibrate_models
    neural_net.py     # StrokeMLP + NeuralNetClassifier (torch)
    data/
      brfss.py        # BRFSS XPT → CSV cleaning script
  data/
    brfss_cleaned.csv # Cleaned BRFSS dataset
    raw/              # Source XPT (gitignored)
  models/             # Trained artifacts (*.joblib gitignored; metrics.json, feature_names.json tracked)
  requirements.txt
  AGENTS.md
```

## Gotchas

- All commands run from `stroke-risk/`, not from `src/`. `src/train.py`
  uses relative paths (`data/brfss_cleaned.csv`, `models/`) that break if
  cwd is different.
- `nn_model.pt` is a `torch.save` dict checkpoint (not TorchScript).
  `NeuralNetClassifier.load()` requires `weights_only=False` — don't change
  that or it will break loading.
- `src/train.py` has no feature engineering step — BRFSS features are used
  directly from the CSV. The old Kaggle pipeline's `engineer_features()`
  and its interaction/binned features do not apply here.
- `data/raw/LLCP2022.XPT` is gitignored. Only `data/brfss_cleaned.csv` is
  tracked.
- Prediction (`predict_risk`) is not yet built. Do not try to load
  `models/` artifacts for inference until `src/predict.py` exists.
