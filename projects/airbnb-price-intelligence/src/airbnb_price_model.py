from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "reports"
DATA_URL = "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-10-01/data/listings.csv.gz"

COLUMNS = [
    "id",
    "neighbourhood_cleansed",
    "room_type",
    "price",
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "reviews_per_month",
    "host_is_superhost",
]


def load_listings(sample_size: int | None = 30000) -> pd.DataFrame:
    listings = pd.read_csv(
        DATA_URL,
        usecols=lambda column: column in COLUMNS,
        low_memory=False,
    )
    if sample_size is not None and len(listings) > sample_size:
        listings = listings.sample(sample_size, random_state=42)
    return listings


def parse_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["price"] = parse_price(clean["price"])

    numeric_columns = [
        "accommodates",
        "bedrooms",
        "beds",
        "minimum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "reviews_per_month",
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean["host_is_superhost"] = clean["host_is_superhost"].fillna("unknown")
    clean["neighbourhood_cleansed"] = clean["neighbourhood_cleansed"].fillna("Unknown")
    clean["room_type"] = clean["room_type"].fillna("Unknown")

    clean = clean[
        clean["price"].between(30, 1000)
        & clean["accommodates"].between(1, 12)
        & clean["minimum_nights"].between(1, 60)
    ].copy()

    for column in numeric_columns:
        clean[column] = clean[column].fillna(clean[column].median())

    clean["log_price"] = np.log1p(clean["price"])
    return clean


def train_price_model(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series, np.ndarray]:
    feature_columns = [
        "neighbourhood_cleansed",
        "room_type",
        "host_is_superhost",
        "accommodates",
        "bedrooms",
        "beds",
        "minimum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "reviews_per_month",
    ]
    categorical_features = ["neighbourhood_cleansed", "room_type", "host_is_superhost"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_columns],
        df["log_price"],
        test_size=0.2,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, x_test, y_test, predictions


def write_reports(df: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    actual_price = np.expm1(y_test)
    predicted_price = np.expm1(predictions)

    rmse = float(np.sqrt(mean_squared_error(actual_price, predicted_price)))
    mae = float(mean_absolute_error(actual_price, predicted_price))
    r2 = float(r2_score(y_test, predictions))

    metrics = [
        "Airbnb Price Intelligence",
        "",
        f"Rows used after cleaning: {len(df):,}",
        f"Median nightly price: ${df['price'].median():.0f}",
        f"RMSE: ${rmse:.2f}",
        f"MAE: ${mae:.2f}",
        f"R2 on log price: {r2:.3f}",
    ]
    (REPORT_DIR / "model_metrics.txt").write_text("\n".join(metrics), encoding="utf-8")

    df.groupby("neighbourhood_cleansed")["price"].agg(["count", "median", "mean"]).sort_values(
        "median", ascending=False
    ).head(20).round(2).to_csv(REPORT_DIR / "top_neighborhood_prices.csv")

    df.groupby("room_type")["price"].agg(["count", "median", "mean"]).sort_values(
        "median", ascending=False
    ).round(2).to_csv(REPORT_DIR / "room_type_prices.csv")


def main() -> None:
    listings = clean_listings(load_listings())
    _, _, y_test, predictions = train_price_model(listings)
    write_reports(listings, y_test, predictions)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
