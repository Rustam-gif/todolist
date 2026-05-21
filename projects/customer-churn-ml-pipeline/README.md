# Customer Churn ML Pipeline

This project predicts telecom customer churn using a full supervised machine learning workflow. It is designed to show practical classification skills, model comparison, class imbalance handling, and interpretable feature importance.

## Business question

Which customers are most likely to leave, and which product or billing factors are most associated with churn?

## Data

The script downloads the IBM Telco Customer Churn sample dataset. Each row represents one customer, with demographic, account, service, billing, tenure, and churn fields.

## Method

1. Load and clean the customer dataset.
2. Convert the churn label into a binary target.
3. Fix numeric fields such as `TotalCharges`.
4. Split the data using stratification to preserve churn rate.
5. Build reusable preprocessing for numeric and categorical variables.
6. Compare:
   - dummy majority-class baseline
   - logistic regression
   - random forest classifier
7. Evaluate with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
8. Export feature importance from the best tree-based model.

## Skills demonstrated

- end-to-end ML pipeline design
- missing-value handling
- one-hot encoding
- class imbalance handling
- baseline comparison
- classification metrics
- model interpretation

## Run

```bash
python projects/customer-churn-ml-pipeline/src/customer_churn_model.py
```

Outputs are written to `projects/customer-churn-ml-pipeline/reports/`.

## Portfolio interpretation

This project is useful for data science interviews because churn prediction is a common business use case. It connects model performance to a real decision: identifying customers who may need retention campaigns before they cancel.
