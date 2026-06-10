import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

LABEL_COLUMN = "stroke"

NUMERIC_COLUMNS = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
CATEGORICAL_COLUMNS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.drop(columns=["id"], errors="ignore")

    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    bmi_median = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(bmi_median)

    _validate_columns(df)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    new_numeric = [
        "glucose_hypertension",
        "glucose_bmi_ratio",
        "age_hypertension",
        "age_glucose",
        "vascular_risk_score",
    ]
    new_categorical = ["age_group", "bmi_category"]

    for col in new_numeric:
        if col not in NUMERIC_COLUMNS:
            NUMERIC_COLUMNS.append(col)
            FEATURE_COLUMNS.append(col)

    for col in new_categorical:
        if col not in CATEGORICAL_COLUMNS:
            CATEGORICAL_COLUMNS.append(col)
            FEATURE_COLUMNS.append(col)

    df["glucose_hypertension"] = df["avg_glucose_level"] * df["hypertension"]
    df["glucose_bmi_ratio"] = df["avg_glucose_level"] / df["bmi"].clip(lower=1e-6)
    df["age_hypertension"] = df["age"] * df["hypertension"]
    df["age_glucose"] = df["age"] * df["avg_glucose_level"]
    df["vascular_risk_score"] = (
        df["hypertension"]
        + df["heart_disease"]
        + (df["smoking_status"] == "smokes").astype(int)
    )

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, float("inf")],
        labels=["young", "middle", "senior"],
    )
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=["underweight", "normal", "overweight", "obese"],
    )

    return df


def encode_data(df: pd.DataFrame):
    X_raw = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].values

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLUMNS),
        ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
    ])

    X = preprocessor.fit_transform(X_raw)
    feature_names = get_feature_names(preprocessor)

    return X, y, preprocessor, feature_names


def split_and_balance(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor) -> list[str]:
    names = []
    for name, _, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
    return names


def _validate_columns(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
