from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "reports"

SEASON_URLS = {
    "2021-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
}


def load_matches() -> pd.DataFrame:
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    optional = ["HS", "AS", "HST", "AST", "HC", "AC"]
    frames: list[pd.DataFrame] = []

    for season, url in SEASON_URLS.items():
        frame = pd.read_csv(url)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{season} is missing required columns: {missing}")

        columns = required + [column for column in optional if column in frame.columns]
        frame = frame[columns].copy()
        frame["season"] = season
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    matches["date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
    matches = matches.dropna(subset=["date", "FTHG", "FTAG", "FTR"])
    matches = matches.sort_values("date").reset_index(drop=True)
    return matches


def result_points(goals_for: float, goals_against: float) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def summarize_history(history: list[dict[str, float]], window: int = 5) -> dict[str, float]:
    recent = history[-window:]
    if not recent:
        return {
            "points_per_match": 1.0,
            "goals_for": 1.2,
            "goals_against": 1.2,
            "goal_diff": 0.0,
            "matches_played": 0.0,
        }

    goals_for = np.array([match["goals_for"] for match in recent], dtype=float)
    goals_against = np.array([match["goals_against"] for match in recent], dtype=float)
    points = np.array([match["points"] for match in recent], dtype=float)

    return {
        "points_per_match": float(points.mean()),
        "goals_for": float(goals_for.mean()),
        "goals_against": float(goals_against.mean()),
        "goal_diff": float((goals_for - goals_against).mean()),
        "matches_played": float(len(history)),
    }


def build_form_features(matches: pd.DataFrame) -> pd.DataFrame:
    histories: dict[str, list[dict[str, float]]] = defaultdict(list)
    feature_rows: list[dict[str, object]] = []

    for _, match in matches.iterrows():
        home_team = str(match["HomeTeam"])
        away_team = str(match["AwayTeam"])
        home_stats = summarize_history(histories[home_team])
        away_stats = summarize_history(histories[away_team])

        feature_rows.append(
            {
                "date": match["date"],
                "season": match["season"],
                "home_team": home_team,
                "away_team": away_team,
                "home_form_points": home_stats["points_per_match"],
                "away_form_points": away_stats["points_per_match"],
                "home_form_goals_for": home_stats["goals_for"],
                "away_form_goals_for": away_stats["goals_for"],
                "home_form_goals_against": home_stats["goals_against"],
                "away_form_goals_against": away_stats["goals_against"],
                "home_form_goal_diff": home_stats["goal_diff"],
                "away_form_goal_diff": away_stats["goal_diff"],
                "home_matches_played": home_stats["matches_played"],
                "away_matches_played": away_stats["matches_played"],
                "target": match["FTR"],
            }
        )

        home_goals = float(match["FTHG"])
        away_goals = float(match["FTAG"])
        histories[home_team].append(
            {
                "goals_for": home_goals,
                "goals_against": away_goals,
                "points": result_points(home_goals, away_goals),
            }
        )
        histories[away_team].append(
            {
                "goals_for": away_goals,
                "goals_against": home_goals,
                "points": result_points(away_goals, home_goals),
            }
        )

    return pd.DataFrame(feature_rows)


def train_model(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series, np.ndarray]:
    feature_columns = [
        "season",
        "home_team",
        "away_team",
        "home_form_points",
        "away_form_points",
        "home_form_goals_for",
        "away_form_goals_for",
        "home_form_goals_against",
        "away_form_goals_against",
        "home_form_goal_diff",
        "away_form_goal_diff",
        "home_matches_played",
        "away_matches_played",
    ]
    categorical_features = ["season", "home_team", "away_team"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    split_index = int(len(data) * 0.8)
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )

    model.fit(train[feature_columns], train["target"])
    predictions = model.predict(test[feature_columns])
    return model, test[feature_columns], test["target"], predictions


def write_reports(y_test: pd.Series, predictions: np.ndarray, data: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = y_test.value_counts(normalize=True).max()
    accuracy = accuracy_score(y_test, predictions)
    labels = ["H", "D", "A"]

    metrics = [
        "Football Match Outcome Prediction",
        "",
        f"Rows used: {len(data):,}",
        f"Temporal test rows: {len(y_test):,}",
        f"Majority-class baseline accuracy: {baseline:.3f}",
        f"Model accuracy: {accuracy:.3f}",
        "",
        "Classification report:",
        classification_report(y_test, predictions, labels=labels, zero_division=0),
    ]

    (REPORT_DIR / "model_metrics.txt").write_text("\n".join(metrics), encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")


def main() -> None:
    matches = load_matches()
    data = build_form_features(matches)
    _, _, y_test, predictions = train_model(data)
    write_reports(y_test, predictions, data)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
