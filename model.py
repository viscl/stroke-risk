import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict:
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
    )
    xgb.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)

    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]

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
        "xgb_metrics": _metrics(y_test, xgb_preds, xgb_probs),
        "lr_metrics": _metrics(y_test, lr_preds, lr_probs),
    }


def save_artifacts(
    xgb: XGBClassifier,
    lr: LogisticRegression,
    scaler,
    dir_path: str = DEFAULT_ARTIFACTS_DIR,
) -> None:
    os.makedirs(dir_path, exist_ok=True)
    joblib.dump(xgb, os.path.join(dir_path, "xgb_model.joblib"))
    joblib.dump(lr, os.path.join(dir_path, "lr_model.joblib"))
    joblib.dump(scaler, os.path.join(dir_path, "scaler.joblib"))


def load_artifacts(dir_path: str = DEFAULT_ARTIFACTS_DIR) -> tuple:
    xgb = joblib.load(os.path.join(dir_path, "xgb_model.joblib"))
    lr = joblib.load(os.path.join(dir_path, "lr_model.joblib"))
    scaler = joblib.load(os.path.join(dir_path, "scaler.joblib"))
    return xgb, lr, scaler
