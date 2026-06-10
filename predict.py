import os
from typing import Any

import numpy as np
import shap

from data import FEATURE_COLUMNS

DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

_LOW_THRESHOLD = 0.3
_HIGH_THRESHOLD = 0.6


def _load_artifacts():
    from model import load_artifacts

    return load_artifacts(DEFAULT_ARTIFACTS_DIR)


_xgb, _lr, _scaler = None, None, None
_explainer = None


def _ensure_loaded():
    global _xgb, _lr, _scaler, _explainer
    if _xgb is None:
        _xgb, _lr, _scaler = _load_artifacts()
        _explainer = shap.TreeExplainer(_xgb)


def _risk_level(prob: float) -> str:
    if prob < _LOW_THRESHOLD:
        return "Low"
    if prob < _HIGH_THRESHOLD:
        return "Moderate"
    return "High"


def predict_risk(
    patient: dict[str, Any] | list[dict[str, Any]],
) -> dict | list[dict]:
    _ensure_loaded()

    single_input = isinstance(patient, dict)
    patients = [patient] if single_input else patient

    X_raw = np.array([[p[col] for col in FEATURE_COLUMNS] for p in patients])
    X_scaled = _scaler.transform(X_raw)

    xgb_probs = _xgb.predict_proba(X_scaled)[:, 1]
    lr_probs = _lr.predict_proba(X_scaled)[:, 1]

    results = []
    for i, p in enumerate(patients):
        xgb_prob = float(xgb_probs[i])
        shap_values = _explainer(X_scaled[i : i + 1])

        feature_contributions = []
        for j, col in enumerate(FEATURE_COLUMNS):
            feature_contributions.append(
                {
                    "feature": col,
                    "value": p[col],
                    "shap_value": float(shap_values.values[0, j]),
                }
            )
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        results.append(
            {
                "risk_probability": round(xgb_prob, 4),
                "risk_level": _risk_level(xgb_prob),
                "risk_label": int(xgb_prob >= 0.5),
                "lr_probability": round(float(lr_probs[i]), 4),
                "shap_explanation": feature_contributions,
            }
        )

    return results[0] if single_input else results
