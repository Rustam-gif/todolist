from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_DIR / "reports"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_churn_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


def clean_churn_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean = df.copy()
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    clean["target_churn"] = clean["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    clean = clean.drop(columns=["customerID", "Churn"])
    x = clean.drop(columns=["target_churn"])
    y = clean["target_churn"]
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_features = x.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = x.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "dummy_baseline": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_and_compare(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    models = build_models(build_preprocessor(x_train))
    rows: list[dict[str, float | str]] = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        rows.append({"model": name, **metrics})

    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return models, comparison


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        importance_type = "random_forest_importance"
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
        importance_type = "absolute_logistic_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
                "importance_type": importance_type,
            }
        )
        .sort_values("importance", ascending=False)
        .head(30)
    )


def write_reports(
    models: dict[str, Pipeline],
    comparison: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    y: pd.Series,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_model_name = str(comparison.iloc[0]["model"])
    best_model = models[best_model_name]
    best_predictions = best_model.predict(x_test)

    pd.DataFrame(
        confusion_matrix(y_test, best_predictions),
        index=["actual_not_churned", "actual_churned"],
        columns=["predicted_not_churned", "predicted_churned"],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    get_feature_importance(best_model).to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )

    lines = [
        "Customer Churn ML Pipeline",
        "",
        f"Rows: {len(y):,}",
        f"Overall churn rate: {y.mean():.1%}",
        f"Best model by ROC-AUC: {best_model_name}",
        "",
        "Model comparison:",
        comparison.round(3).to_string(index=False),
    ]
    (REPORT_DIR / "model_metrics.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    x, y = clean_churn_data(load_churn_data())
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    models, comparison = train_and_compare(x_train, x_test, y_train, y_test)
    write_reports(models, comparison, x_test, y_test, y)
    print(f"Reports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
