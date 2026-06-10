import os
from typing import Any

import numpy as np
import pandas as pd
import shap

from data import get_feature_names

DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def _load_artifacts():
    from model import load_artifacts

    return load_artifacts(DEFAULT_ARTIFACTS_DIR)


_xgb, _lr, _rf, _preprocessor = None, None, None, None
_threshold = None
_explainer = None
_feature_names = None


def _ensure_loaded():
    global _xgb, _lr, _rf, _preprocessor, _threshold, _explainer, _feature_names
    if _xgb is None:
        _xgb, _lr, _rf, _preprocessor, _threshold = _load_artifacts()
        raw_xgb = _xgb.calibrated_classifiers_[0].estimator
        _explainer = shap.TreeExplainer(raw_xgb)
        _feature_names = get_feature_names(_preprocessor)


def _risk_level(prob: float) -> str:
    low = _threshold.get("low", 0.3)
    high = _threshold.get("high", 0.6)
    if prob < low:
        return "Low"
    if prob < high:
        return "Moderate"
    return "High"


def predict_risk(
    patient: dict[str, Any] | list[dict[str, Any]],
) -> dict | list[dict]:
    _ensure_loaded()

    single_input = isinstance(patient, dict)
    patients = [patient] if single_input else patient

    X_raw = pd.DataFrame(patients)
    X_encoded = _preprocessor.transform(X_raw)

    xgb_probs = _xgb.predict_proba(X_encoded)[:, 1]
    lr_probs = _lr.predict_proba(X_encoded)[:, 1]
    rf_probs = _rf.predict_proba(X_encoded)[:, 1]

    decision = _threshold.get("decision", 0.5)

    results = []
    for i, p in enumerate(patients):
        xgb_prob = float(xgb_probs[i])
        shap_values = _explainer(X_encoded[i : i + 1])

        feature_contributions = []
        for j, name in enumerate(_feature_names):
            feature_contributions.append(
                {
                    "feature": name,
                    "value": float(X_encoded[i, j]),
                    "shap_value": float(shap_values.values[0, j]),
                }
            )
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        results.append(
            {
                "risk_probability": round(xgb_prob, 4),
                "risk_level": _risk_level(xgb_prob),
                "risk_label": int(xgb_prob >= decision),
                "lr_probability": round(float(lr_probs[i]), 4),
                "rf_probability": round(float(rf_probs[i]), 4),
                "shap_explanation": feature_contributions,
            }
        )

    return results[0] if single_input else results
