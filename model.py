import json
import os

import joblib
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

DEFAULT_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)

DEFAULT_RF_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
)

OPTUNA_METRIC = "f1"


def _build_models(
    random_state: int = 42,
    xgb_params: dict | None = None,
    rf_params: dict | None = None,
):
    xgb_kwargs = {**DEFAULT_XGB_PARAMS, "random_state": random_state}
    if xgb_params:
        xgb_kwargs.update(xgb_params)
    xgb = XGBClassifier(**xgb_kwargs)

    lr = LogisticRegression(max_iter=2000, random_state=random_state)

    rf_kwargs = {**DEFAULT_RF_PARAMS, "random_state": random_state}
    if rf_params:
        rf_kwargs.update(rf_params)
    rf = RandomForestClassifier(**rf_kwargs)

    return xgb, lr, rf


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    xgb_params: dict | None = None,
    rf_params: dict | None = None,
) -> dict:
    xgb, lr, rf = _build_models(random_state, xgb_params=xgb_params, rf_params=rf_params)

    xgb.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    def _metrics(y_true, y_pred, y_prob):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
        }

    return {
        "xgb": xgb,
        "lr": lr,
        "rf": rf,
        "xgb_metrics": _metrics(y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1]),
        "lr_metrics": _metrics(y_test, lr.predict(X_test), lr.predict_proba(X_test)[:, 1]),
        "rf_metrics": _metrics(y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1]),
    }


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    xgb, lr, rf = _build_models(random_state)

    models = {"xgb": xgb, "lr": lr, "rf": rf}
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for name, clf in models.items():
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", clf),
        ])
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {}
        for metric in scoring:
            key = f"test_{metric}"
            results[name][metric] = (float(scores[key].mean()), float(scores[key].std()))

    return results


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_xgb(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "eval_metric": "logloss",
            "random_state": random_state,
        }
        clf = XGBClassifier(**params)
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", clf),
        ])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=OPTUNA_METRIC, n_jobs=-1)
        return float(scores["test_score"].mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    # Map back to canonical keys
    return {
        "n_estimators": best["n_estimators"],
        "max_depth": best["max_depth"],
        "learning_rate": best["learning_rate"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "min_child_weight": best["min_child_weight"],
        "gamma": best["gamma"],
    }


def tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    def objective(trial):
        class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": class_weight,
            "random_state": random_state,
        }
        clf = RandomForestClassifier(**params)
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", clf),
        ])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=OPTUNA_METRIC, n_jobs=-1)
        return float(scores["test_score"].mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    return {
        "n_estimators": best["n_estimators"],
        "max_depth": best["max_depth"],
        "min_samples_split": best["min_samples_split"],
        "min_samples_leaf": best["min_samples_leaf"],
        "class_weight": best["class_weight"],
    }


# ---------------------------------------------------------------------------
# Platt scaling (calibration)
# ---------------------------------------------------------------------------

def calibrate_models(
    xgb: XGBClassifier,
    rf: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[CalibratedClassifierCV, CalibratedClassifierCV]:
    calib_xgb = CalibratedClassifierCV(xgb, method="sigmoid", cv=5)
    calib_xgb.fit(X_train, y_train)

    calib_rf = CalibratedClassifierCV(rf, method="sigmoid", cv=5)
    calib_rf.fit(X_train, y_train)

    return calib_xgb, calib_rf


def _platt_calibrate(estimator, X_train, y_train, X_calib, y_calib, random_state):
    estimator.fit(X_train, y_train)
    raw = estimator.predict_proba(X_calib)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000, random_state=random_state)
    platt.fit(raw, y_calib)
    return platt


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

def train_stacking_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xgb_params: dict | None = None,
    rf_params: dict | None = None,
    random_state: int = 42,
    n_splits: int = 5,
) -> LogisticRegression:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    meta_X_parts = []
    meta_y_parts = []

    for fold_idx, (inner_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        fold_seed = random_state + fold_idx

        X_inner, X_val = X_train[inner_idx], X_train[val_idx]
        y_inner, y_val = y_train[inner_idx], y_train[val_idx]

        smote = SMOTE(random_state=fold_seed)
        X_inner_bal, y_inner_bal = smote.fit_resample(X_inner, y_inner)

        X_t, X_c, y_t, y_c = train_test_split(
            X_inner_bal, y_inner_bal,
            test_size=0.2, stratify=y_inner_bal,
            random_state=fold_seed,
        )

        xgb, lr, rf = _build_models(
            random_state=fold_seed,
            xgb_params=xgb_params,
            rf_params=rf_params,
        )

        platt_xgb = _platt_calibrate(xgb, X_t, y_t, X_c, y_c, fold_seed)
        platt_rf = _platt_calibrate(rf, X_t, y_t, X_c, y_c, fold_seed)
        lr.fit(X_t, y_t)

        xgb_raw = xgb.predict_proba(X_val)[:, 1].reshape(-1, 1)
        rf_raw = rf.predict_proba(X_val)[:, 1].reshape(-1, 1)
        lr_raw = lr.predict_proba(X_val)[:, 1].reshape(-1, 1)

        xgb_cal = platt_xgb.predict_proba(xgb_raw)[:, 1]
        rf_cal = platt_rf.predict_proba(rf_raw)[:, 1]

        meta_X_parts.append(np.column_stack([xgb_cal, rf_cal, lr_raw.flatten()]))
        meta_y_parts.append(y_val)

    meta_X = np.vstack(meta_X_parts)
    meta_y = np.concatenate(meta_y_parts)

    stacking = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    stacking.fit(meta_X, meta_y)

    return stacking


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    step: float = 0.01,
) -> float:
    best_threshold = 0.5
    best_score = -1.0

    if metric == "f1":
        scorer = f1_score
    elif metric == "precision":
        scorer = precision_score
    elif metric == "recall":
        scorer = recall_score
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    for t in np.arange(0.01, 1.0, step):
        y_pred = (y_prob >= t).astype(int)
        score = scorer(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    xgb,
    lr,
    rf,
    preprocessor,
    threshold: dict | None = None,
    stacking=None,
    dir_path: str = DEFAULT_ARTIFACTS_DIR,
) -> None:
    os.makedirs(dir_path, exist_ok=True)
    joblib.dump(xgb, os.path.join(dir_path, "xgb_model.joblib"))
    joblib.dump(lr, os.path.join(dir_path, "lr_model.joblib"))
    joblib.dump(rf, os.path.join(dir_path, "rf_model.joblib"))
    joblib.dump(preprocessor, os.path.join(dir_path, "preprocessor.joblib"))

    if threshold is None:
        threshold = {"decision": 0.5, "low": 0.3, "high": 0.6}

    with open(os.path.join(dir_path, "threshold.json"), "w") as f:
        json.dump(threshold, f)

    if stacking is not None:
        joblib.dump(stacking, os.path.join(dir_path, "stacking_model.joblib"))


def load_artifacts(dir_path: str = DEFAULT_ARTIFACTS_DIR):
    xgb = joblib.load(os.path.join(dir_path, "xgb_model.joblib"))
    lr = joblib.load(os.path.join(dir_path, "lr_model.joblib"))
    rf = joblib.load(os.path.join(dir_path, "rf_model.joblib"))
    preprocessor = joblib.load(os.path.join(dir_path, "preprocessor.joblib"))

    threshold_path = os.path.join(dir_path, "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = json.load(f)
    else:
        threshold = {"decision": 0.5, "low": 0.3, "high": 0.6}

    return xgb, lr, rf, preprocessor, threshold
