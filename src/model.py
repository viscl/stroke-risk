import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

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
