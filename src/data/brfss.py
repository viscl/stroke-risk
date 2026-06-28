"""Clean and recode BRFSS 2022 XPT data for stroke risk analysis."""

import pandas as pd

SOURCE_COLUMNS = [
    "CVDSTRK3",
    "_AGEG5YR",
    "_BMI5",
    "_SMOKER3",
    "EXERANY2",
    "DRNKANY6",
    "DIABETE4",
    "SEXVAR",
    "_IMPRACE",
    "CVDCRHD4",
]

AGE_GROUP_MAP = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80-84",
    14: "85+",
}

SMOKING_MAP = {1: "Current", 2: "Current", 3: "Former", 4: "Never"}

EXERCISE_MAP = {1: "Yes", 2: "No"}

ALCOHOL_MAP = {1: "Yes", 2: "No"}

DIABETES_MAP = {
    1: "Yes",
    2: "Yes (gestational)",
    3: "No",
    4: "Pre-diabetes",
}

SEX_MAP = {1: "Male", 2: "Female"}

RACE_MAP = {
    1: "White",
    2: "Black",
    3: "American Indian/Alaska Native",
    4: "Asian",
    5: "Native Hawaiian/Pacific Islander",
    6: "Multiracial/Other",
}


def _safe_map(series, mapping, default="Unknown"):
    return series.map(mapping).fillna(default)


def main():
    print("Loading BRFSS data...")
    df = pd.read_sas("data/raw/LLCP2022.XPT")
    df = df[SOURCE_COLUMNS].copy()
    print(f"  Raw: {len(df)} rows")

    print("\nCleaning...")
    df = df.dropna(subset=["CVDSTRK3"])
    df = df[~df["CVDSTRK3"].isin([7, 9])]
    print(f"  After dropping invalid stroke: {len(df)} rows")

    df["stroke"] = df["CVDSTRK3"].map({1: 1, 2: 0})
    df["age_group"] = _safe_map(df["_AGEG5YR"], AGE_GROUP_MAP)
    df["age_is_young"] = df["_AGEG5YR"] <= 4
    df["bmi"] = pd.to_numeric(df["_BMI5"], errors="coerce") / 100.0
    df["smoking_status"] = _safe_map(df["_SMOKER3"], SMOKING_MAP)
    df["exercise"] = _safe_map(df["EXERANY2"], EXERCISE_MAP)
    df["alcohol"] = _safe_map(df["DRNKANY6"], ALCOHOL_MAP)
    df["diabetes"] = _safe_map(df["DIABETE4"], DIABETES_MAP)
    df["sex"] = _safe_map(df["SEXVAR"], SEX_MAP)
    df["race"] = _safe_map(df["_IMPRACE"], RACE_MAP)
    df["heart_disease"] = df["CVDCRHD4"].map({1: 1, 2: 0})

    output_cols = [
        "stroke",
        "age_group",
        "age_is_young",
        "bmi",
        "smoking_status",
        "exercise",
        "alcohol",
        "diabetes",
        "heart_disease",
        "sex",
        "race",
    ]
    df = df[output_cols]

    out_path = "data/brfss_cleaned.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {out_path}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Stroke prevalence: {df['stroke'].mean():.2%}")


if __name__ == "__main__":
    main()

    # Temporary analysis: stroke breakdown by age
    df = pd.read_csv("data/brfss_cleaned.csv")
    total_stroke = df["stroke"].sum()
    young_stroke = df[df["age_is_young"]]["stroke"].sum()
    old_stroke = df[~df["age_is_young"]]["stroke"].sum()
    young_prev = df[df["age_is_young"]]["stroke"].mean()
    old_prev = df[~df["age_is_young"]]["stroke"].mean()

    print(f"\n--- Stroke breakdown by age ---")
    print(f"  Total stroke cases: {total_stroke}")
    print(f"  Stroke cases (young):        {young_stroke}")
    print(f"  Stroke cases (older):         {old_stroke}")
    print(f"  Stroke prevalence (young):    {young_prev:.2%}")
    print(f"  Stroke prevalence (older):     {old_prev:.2%}")
