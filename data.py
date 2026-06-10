import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

FEATURE_COLUMNS = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate_variability",
    "BMI",
    "cholesterol",
    "atrial_fibrillation",
    "sleep_hours",
    "activity_level",
    "diabetes",
    "smoking",
    "family_history",
]

LABEL_COLUMN = "stroke_risk"


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(35, 90, n_samples)
    atrial_fibrillation = rng.binomial(1, 0.15, n_samples)
    diabetes = rng.binomial(1, 0.20, n_samples)
    smoking = rng.binomial(1, 0.25, n_samples)
    family_history = rng.binomial(1, 0.18, n_samples)

    systolic_bp = 110 + 0.6 * age + rng.normal(0, 12, n_samples) + atrial_fibrillation * 8 + diabetes * 6
    diastolic_bp = 70 + 0.15 * age + rng.normal(0, 8, n_samples) + atrial_fibrillation * 3
    systolic_bp = np.clip(systolic_bp, 90, 220)
    diastolic_bp = np.clip(diastolic_bp, 50, 140)

    heart_rate_variability = 50 - 0.3 * age + rng.normal(0, 10, n_samples) - diabetes * 5 - smoking * 4
    heart_rate_variability = np.clip(heart_rate_variability, 10, 120)

    BMI = 22 + 0.12 * age + rng.normal(0, 4, n_samples) + diabetes * 2.5
    BMI = np.clip(BMI, 16, 50)

    cholesterol = 160 + 0.5 * age + rng.normal(0, 30, n_samples) + smoking * 10 + diabetes * 8
    cholesterol = np.clip(cholesterol, 100, 350)

    sleep_hours = 7.5 - 0.02 * age + rng.normal(0, 1.2, n_samples)
    sleep_hours = np.clip(sleep_hours, 3, 12)

    activity_level = rng.integers(1, 11, n_samples).astype(float)
    activity_level = np.clip(activity_level - 0.03 * age + rng.normal(0, 1.8, n_samples), 1, 10)

    logit = (
        -8.0
        + 0.06 * age
        + 0.02 * systolic_bp
        + 0.01 * diastolic_bp
        - 0.03 * heart_rate_variability
        + 0.04 * BMI
        + 0.005 * cholesterol
        + 1.2 * atrial_fibrillation
        - 0.15 * sleep_hours
        - 0.12 * activity_level
        + 0.9 * diabetes
        + 0.7 * smoking
        + 0.8 * family_history
    )
    prob = 1 / (1 + np.exp(-logit))
    stroke_risk = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "age": age,
            "systolic_bp": systolic_bp.round(1),
            "diastolic_bp": diastolic_bp.round(1),
            "heart_rate_variability": heart_rate_variability.round(2),
            "BMI": BMI.round(1),
            "cholesterol": cholesterol.round(1),
            "atrial_fibrillation": atrial_fibrillation,
            "sleep_hours": sleep_hours.round(1),
            "activity_level": activity_level.round(1),
            "diabetes": diabetes,
            "smoking": smoking,
            "family_history": family_history,
            "stroke_risk": stroke_risk,
        }
    )
    return df


def load_data(path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if path is not None:
        df = pd.read_csv(path)
        _validate_columns(df)
    else:
        df = generate_synthetic_data()

    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values

    return X, y


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler


def _validate_columns(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
