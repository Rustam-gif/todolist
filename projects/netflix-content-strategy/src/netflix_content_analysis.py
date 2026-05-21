from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "reports"
DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv"


def load_netflix_titles() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


def first_value(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return str(value).split(",")[0].strip() or "Unknown"


def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["date_added"] = pd.to_datetime(clean["date_added"], errors="coerce")
    clean["year_added"] = clean["date_added"].dt.year
    clean["main_country"] = clean["country"].apply(first_value)
    clean["main_genre"] = clean["listed_in"].apply(first_value)
    clean["duration_value"] = (
        clean["duration"].astype(str).str.extract(r"(\d+)").astype(float)
    )
    text_columns = ["title", "description", "listed_in", "rating", "country"]
    clean["text_features"] = clean[text_columns].fillna("").agg(" ".join, axis=1)
    return clean


def write_summary_tables(df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df["type"].value_counts().rename_axis("type").reset_index(name="titles").to_csv(
        REPORT_DIR / "content_type_mix.csv", index=False
    )
    df["main_country"].value_counts().head(15).rename_axis("country").reset_index(
        name="titles"
    ).to_csv(REPORT_DIR / "top_countries.csv", index=False)
    df["main_genre"].value_counts().head(20).rename_axis("genre").reset_index(
        name="titles"
    ).to_csv(REPORT_DIR / "top_genres.csv", index=False)
    df.dropna(subset=["year_added"]).groupby(["year_added", "type"]).size().reset_index(
        name="titles_added"
    ).to_csv(REPORT_DIR / "yearly_additions.csv", index=False)


def train_type_classifier(df: pd.DataFrame) -> tuple[float, str]:
    model_data = df.dropna(subset=["type", "text_features"]).copy()
    x_train, x_test, y_train, y_test = train_test_split(
        model_data["text_features"],
        model_data["type"],
        test_size=0.2,
        random_state=42,
        stratify=model_data["type"],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return accuracy, report


def write_model_report(df: pd.DataFrame, accuracy: float, report: str) -> None:
    total_titles = len(df)
    movie_share = (df["type"].eq("Movie").mean()) if total_titles else 0
    tv_share = (df["type"].eq("TV Show").mean()) if total_titles else 0
    top_country = df["main_country"].value_counts().idxmax()
    top_genre = df["main_genre"].value_counts().idxmax()

    lines = [
        "Netflix Content Strategy Analysis",
        "",
        f"Total titles: {total_titles:,}",
        f"Movie share: {movie_share:.1%}",
        f"TV show share: {tv_share:.1%}",
        f"Top country: {top_country}",
        f"Top genre: {top_genre}",
        f"Text classifier accuracy: {accuracy:.3f}",
        "",
        "Classification report:",
        report,
    ]
    (REPORT_DIR / "model_metrics.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    titles = clean_titles(load_netflix_titles())
    write_summary_tables(titles)
    accuracy, report = train_type_classifier(titles)
    write_model_report(titles, accuracy, report)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
