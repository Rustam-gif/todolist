# Data Science Portfolio Projects

A collection of reproducible data science and machine learning projects focused on practical business questions, clean analysis, model evaluation, and clear interpretation.

> Repository note: this repo is currently named `todolist`. For a stronger employer-facing profile, rename it to `data-science-portfolio` or `data-science-projects` in GitHub settings.

## Projects

| Project | Focus | Skills shown |
| --- | --- | --- |
| [Football Match Outcome Prediction](projects/football-match-outcome-prediction) | Predict English Premier League match results from historical form | pandas, feature engineering, classification, temporal validation |
| [Netflix Content Strategy Analysis](projects/netflix-content-strategy) | Analyze Netflix catalog trends and classify content type from metadata | EDA, text features, logistic regression, content analytics |
| [Airbnb Price Intelligence](projects/airbnb-price-intelligence) | Estimate NYC Airbnb listing prices and identify pricing drivers | data cleaning, regression, categorical encoding, model evaluation |
| [Customer Churn ML Pipeline](projects/customer-churn-ml-pipeline) | Predict telecom customer churn and explain the strongest churn drivers | preprocessing pipelines, imbalanced classification, ROC-AUC, feature importance |

## Why this repository matters

Each project is built around a realistic question an analyst or data scientist might answer at work:

- Which football teams are more likely to win based on recent form?
- What patterns define the Netflix catalog?
- Which listing features influence Airbnb prices?
- Which customers are most likely to churn?

The goal is not only to train models, but to show the full workflow: data loading, cleaning, feature engineering, baseline comparison, model evaluation, and business interpretation.

## How to run

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a project script:

```bash
python projects/football-match-outcome-prediction/src/football_match_model.py
python projects/netflix-content-strategy/src/netflix_content_analysis.py
python projects/airbnb-price-intelligence/src/airbnb_price_model.py
python projects/customer-churn-ml-pipeline/src/customer_churn_model.py
```

Each script creates a `reports/` folder inside its project with model metrics and summary outputs.

## Data sources

- Football match results: Football-Data.co.uk public CSV files
- Netflix catalog: TidyTuesday Netflix titles dataset
- Airbnb listings: Inside Airbnb public New York City listings data
- Customer churn: IBM Telco Customer Churn sample dataset

The scripts download public datasets at runtime. If a source is temporarily unavailable, download the dataset manually and update the file path in the relevant script.

## Skills represented

- Python data analysis with pandas and NumPy
- Data cleaning and feature engineering
- Classification and regression modeling with scikit-learn
- Model evaluation with accuracy, precision, recall, F1, ROC-AUC, RMSE, MAE, and R2
- Baseline comparison and interpretable reporting
- Writing project documentation for technical and non-technical readers

## Next improvements

- Add Jupyter notebooks with charts for each project
- Add Streamlit dashboards for interactive exploration
- Add screenshots/GIFs to each project README
- Add tests for cleaning and feature engineering functions
- Rename this repo to a portfolio-focused name
