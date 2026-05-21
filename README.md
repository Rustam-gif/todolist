# Data Science Portfolio Projects

This repository rebuilds several lost data science projects as clean, reproducible portfolio case studies. Each project is designed to show a specific part of the data science workflow: data loading, cleaning, feature engineering, exploratory analysis, modeling, evaluation, and business interpretation.

> Repository note: this repo was originally named `todolist`. For a stronger GitHub profile, rename it to `data-science-portfolio` or `data-science-projects` in GitHub settings.

## Projects

| Project | Focus | Skills shown |
| --- | --- | --- |
| Football Match Outcome Prediction | Predict English Premier League match results from historical form | pandas, feature engineering, classification, temporal validation |
| Netflix Content Strategy Analysis | Analyze Netflix catalog trends and classify content type from metadata | EDA, text features, logistic regression, content analytics |
| Airbnb Price Intelligence | Estimate NYC Airbnb listing prices and identify pricing drivers | data cleaning, regression, categorical encoding, model evaluation |
| Customer Churn ML Pipeline | Predict telecom customer churn and explain the strongest churn drivers | preprocessing pipelines, imbalanced classification, ROC-AUC, feature importance |

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

## Portfolio positioning

These projects are intentionally practical. They are meant to show that I can take a real-world question, structure the data, build a baseline model, evaluate it honestly, and explain what the results mean for a product or business decision.

## Next improvements

- Add Jupyter notebooks with charts for each project
- Add Streamlit dashboards for interactive exploration
- Add screenshots/GIFs to each project README
- Add tests for cleaning and feature engineering functions
- Rename this repo to a portfolio-focused name
