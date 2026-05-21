# Football Match Outcome Prediction

This project predicts English Premier League match outcomes using historical match results and rolling team form. It is designed to show practical feature engineering, temporal validation, and classification model evaluation.

## Business question

Can recent team form help predict whether the home team wins, the match ends in a draw, or the away team wins?

## Data

The script downloads public Premier League CSV files from Football-Data.co.uk. Each row is a completed match with teams, final score, result, and available match statistics.

## Method

1. Load several seasons of match data.
2. Sort matches by date to avoid future data leakage.
3. Build rolling pre-match features for each team:
   - recent points per match
   - recent goals scored
   - recent goals conceded
   - recent goal difference
   - matches played so far
4. Train a logistic regression classifier with one-hot encoded teams and seasons.
5. Evaluate against a simple majority-class baseline.

## Skills demonstrated

- pandas data cleaning
- feature engineering from time-ordered sports data
- classification modeling
- temporal train/test split
- honest baseline comparison

## Run

```bash
python projects/football-match-outcome-prediction/src/football_match_model.py
```

Outputs are written to `projects/football-match-outcome-prediction/reports/`.
