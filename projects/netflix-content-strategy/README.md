# Netflix Content Strategy Analysis

This project analyzes the Netflix catalog and trains a simple text classifier to distinguish movies from TV shows using title metadata, descriptions, genres, ratings, and countries.

## Business question

What types of content dominate the catalog, how has the catalog changed over time, and can metadata patterns separate movies from TV shows?

## Data

The script downloads the TidyTuesday Netflix titles dataset, which contains title-level metadata such as type, country, date added, release year, rating, duration, genre labels, and descriptions.

## Method

1. Clean dates, countries, genres, ratings, and duration fields.
2. Produce summary tables for content mix, countries, genres, and yearly additions.
3. Build a text feature from title, description, genre, rating, and country.
4. Train a TF-IDF + logistic regression classifier.
5. Report classification quality and key catalog summaries.

## Skills demonstrated

- exploratory data analysis
- text feature extraction with TF-IDF
- classification model pipeline
- content analytics and interpretation

## Run

```bash
python projects/netflix-content-strategy/src/netflix_content_analysis.py
```

Outputs are written to `projects/netflix-content-strategy/reports/`.
