# AGENTS.md — Stroke Risk

Flat single-package ML project: data → train → predict. No tests, no CI, no linter/typecheck config.

## Commands

```bash
# Train with Optuna tuning (default; ~5 min)
python train.py                          # uses healthcare-dataset-stroke-data.csv
python train.py /path/to/your_data.csv   # custom data: must have all FEATURE_COLUMNS + stroke label

# Train without tuning (skips Optuna, uses default hyperparams)
python train.py --no-tune

# Predict (requires artifacts/ from training)
python -c "from predict import predict_risk; ..."
```

## Required order

`train.py` must complete before `predict_risk` works — it writes `artifacts/{xgb,lr,rf}_model.joblib`, `preprocessor.joblib`, and `threshold.json`. The `artifacts/` directory is gitignored but pre-populated in this repo with trained models.

## Pipeline flow

```
load_data → engineer_features → encode_data → cross_validate_models (baseline CV)
  → tune_xgb + tune_rf (Optuna, 50 trials each, F1-optimized via SMOTE-per-fold CV)
  → split_and_balance (SMOTE train split)
  → train_models (with tuned params)
  → calibrate_models (Platt scaling via CalibratedClassifierCV(cv=5) for XGB+RF)
  → train_nn (NeuralNetClassifier: MLP 128→64, ReLU+Dropout, Adam, early stopping)
  → train_stacking_model (5-fold CV out-of-fold calibrated probs → LR meta-model, 4 base models)
  → tune_threshold (F1-optimal on test set)
  → classification_report + confusion_matrix
  → save_artifacts (calibrated xgb, lr, calibrated rf, stacking, nn, preprocessor, threshold.json)
```

## Gotchas

- `artifacts/scaler.joblib` is stale — not saved or loaded by current code. Ignore it.
- `Residence_type` is the canonical column name (capital `R`, lowercase `_type`). All other column names are lowercase_with_underscores.
- `load_data` imputes missing BMI with the median *before* encoding. The preprocessing pipeline also has a median imputer, so non-BMI numeric NaNs are still handled.
- `engineer_features(df)` must be called after `load_data` and before `encode_data`. It adds 5 numeric + 2 categorical interaction/binned features and updates `FEATURE_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS` (idempotent). `predict_risk` also calls it before transforming.
- `artifacts/threshold.json` stores the tuned decision threshold (`{"decision": X, "low": 0.3, "high": 0.6}`). Predict falls back to `decision=0.5` if the file is missing.

## SMOTE convention

- **CV (5-fold stratified):** `cross_validate_models` and `tune_xgb`/`tune_rf` use `imblearn.pipeline.Pipeline` to apply SMOTE *per fold*, avoiding data leakage.
- **Final training:** `train.py` calls `split_and_balance(apply_smote=True)`, which SMOTEs the train split *once* before passing to `train_models`. `train_models` itself does not apply SMOTE internally.

## Predict module

- Lazy-loads models and threshold into module globals on first `predict_risk()` call — no explicit init step.
- Accepts a single `dict` or `list[dict]`; returns the same shape.
- Models saved to artifacts are calibrated `CalibratedClassifierCV` wrappers (XGB, RF). SHAP explainer extracts the raw XGBoost from the wrapper via `_xgb.calibrated_classifiers_[0].estimator`.
- `risk_label` uses the tuned `decision` threshold from `threshold.json`, not hardcoded 0.5.

## Stacking ensemble

- `train_stacking_model` uses 5-fold CV to generate out-of-fold calibrated predictions from all 4 base models as features for a LogisticRegression meta-model (avoids data leakage).
- Manual Platt scaling per fold (80/20 train/calibration split + sigmoid fitting) avoids the `CalibratedClassifierCV(cv='prefit')` incompatibility.
- `predict_risk` output includes `stack_probability` field (`None` if `stacking_model.joblib` is missing).
- Base models at prediction time are the externally-calibrated ones (XGB, RF via `CalibratedClassifierCV(cv=5)`, LR raw, NN raw).
- The stacking meta-model's `n_features_in_` attribute is used at prediction time to decide 3 vs 4 input features for backward compatibility.
