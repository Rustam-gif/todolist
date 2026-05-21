# Airbnb Price Intelligence

This project estimates Airbnb listing prices in New York City and identifies the features most useful for pricing decisions.

## Business question

Which listing characteristics are associated with higher nightly prices, and how accurately can a baseline model estimate price?

## Data

The script downloads public New York City listings data from Inside Airbnb. The dataset includes price, room type, neighborhood, review data, availability, host attributes, and listing capacity fields.

## Method

1. Load public NYC Airbnb listing data.
2. Clean currency fields and numeric listing attributes.
3. Remove extreme price outliers for a more realistic baseline model.
4. One-hot encode categorical features and scale numeric features.
5. Train a ridge regression model on log price.
6. Report RMSE, MAE, R2, and summary pricing tables.

## Skills demonstrated

- real-world data cleaning
- categorical encoding
- regression modeling
- price modeling with log transforms
- business-friendly metric reporting

## Run

```bash
python projects/airbnb-price-intelligence/src/airbnb_price_model.py
```

Outputs are written to `projects/airbnb-price-intelligence/reports/`.
