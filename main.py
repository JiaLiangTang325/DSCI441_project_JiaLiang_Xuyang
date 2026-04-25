from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "data" / "global_air_pollution_dataset.csv"

AQI_CATEGORY_ORDER = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]

AQI_CATEGORY_COLORS = {
    "Good": "#2f9e44",
    "Moderate": "#f59f00",
    "Unhealthy for Sensitive Groups": "#f08c00",
    "Unhealthy": "#e03131",
    "Very Unhealthy": "#9c36b5",
    "Hazardous": "#5f3dc4",
}

AQI_CATEGORY_DESCRIPTIONS = {
    "Good": "Air quality is satisfactory.",
    "Moderate": "Air quality is acceptable for most people.",
    "Unhealthy for Sensitive Groups": "Sensitive groups may experience health effects.",
    "Unhealthy": "Some members of the general public may experience health effects.",
    "Very Unhealthy": "Health alert conditions are possible.",
    "Hazardous": "Emergency health conditions are possible.",
}

POLLUTANT_VALUE_COLS = [
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value",
]

MODEL_FEATURE_COLS = [
    "Country",
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value",
    "Dominant Pollutant",
]


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the global air pollution dataset from disk."""
    return pd.read_csv(path)


def clean_air_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the AQI dataset and add the dominant pollutant feature."""
    cleaned = df.drop_duplicates().copy()

    text_cols = [
        "Country",
        "City",
        "AQI Category",
        "CO AQI Category",
        "Ozone AQI Category",
        "NO2 AQI Category",
        "PM2.5 AQI Category",
    ]
    for col in text_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("string").str.strip()

    numeric_cols = ["AQI Value", *POLLUTANT_VALUE_COLS]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["Country", "City", "AQI Category", *numeric_cols])
    cleaned = cleaned[cleaned["AQI Category"].isin(AQI_CATEGORY_ORDER)].copy()
    cleaned["Dominant Pollutant"] = cleaned[POLLUTANT_VALUE_COLS].idxmax(axis=1)
    cleaned["Dominant Pollutant"] = cleaned["Dominant Pollutant"].str.replace(" AQI Value", "", regex=False)

    return cleaned.reset_index(drop=True)


def get_summary_metrics(df: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "rows": int(len(df)),
        "countries": int(df["Country"].nunique()),
        "cities": int(df["City"].nunique()),
        "mean_aqi": float(df["AQI Value"].mean()),
        "median_aqi": float(df["AQI Value"].median()),
        "most_common_category": str(df["AQI Category"].mode().iloc[0]),
    }


def category_counts(df: pd.DataFrame) -> pd.Series:
    return df["AQI Category"].value_counts().reindex(AQI_CATEGORY_ORDER, fill_value=0)


def country_summary(df: pd.DataFrame, min_records: int = 30) -> pd.DataFrame:
    summary = (
        df.groupby("Country")
        .agg(
            records=("AQI Value", "size"),
            mean_aqi=("AQI Value", "mean"),
            median_aqi=("AQI Value", "median"),
            max_aqi=("AQI Value", "max"),
        )
        .reset_index()
    )
    summary = summary[summary["records"] >= min_records]
    return summary.sort_values("mean_aqi", ascending=False).reset_index(drop=True)


def spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in POLLUTANT_VALUE_COLS:
        corr, p_value = stats.spearmanr(df["AQI Value"], df[col])
        rows.append(
            {
                "Pollutant AQI": col.replace(" AQI Value", ""),
                "Spearman correlation": corr,
                "p-value": p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("Spearman correlation", ascending=False)


def bootstrap_mean_aqi_ci(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    values = df["AQI Value"].to_numpy()
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()

    alpha = 1 - confidence
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def train_aqi_classifier(df: pd.DataFrame, random_state: int = 42) -> dict[str, object]:
    model_df = df[MODEL_FEATURE_COLS + ["AQI Category"]].copy()
    x = model_df[MODEL_FEATURE_COLS]
    y = model_df["AQI Category"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["Country", "Dominant Pollutant"]),
            ("numeric", "passthrough", POLLUTANT_VALUE_COLS),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    labels = [label for label in AQI_CATEGORY_ORDER if label in y_test.unique()]

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "test_size": len(y_test),
    }


def predict_aqi_category(
    pipeline: Pipeline,
    country: str,
    co_aqi: float,
    ozone_aqi: float,
    no2_aqi: float,
    pm25_aqi: float,
) -> tuple[str, pd.DataFrame]:
    values = {
        "CO AQI Value": co_aqi,
        "Ozone AQI Value": ozone_aqi,
        "NO2 AQI Value": no2_aqi,
        "PM2.5 AQI Value": pm25_aqi,
    }
    dominant = max(values, key=values.get).replace(" AQI Value", "")
    sample = pd.DataFrame(
        [
            {
                "Country": country,
                **values,
                "Dominant Pollutant": dominant,
            }
        ]
    )
    prediction = str(pipeline.predict(sample)[0])

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(sample)[0]
        classes = pipeline.classes_
        probability_df = (
            pd.DataFrame({"AQI Category": classes, "Probability": probabilities})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
    else:
        probability_df = pd.DataFrame({"AQI Category": [prediction], "Probability": [1.0]})

    return prediction, probability_df
