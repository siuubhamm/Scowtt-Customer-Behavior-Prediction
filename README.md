# Predicting Customer Conversion Propensity and Order Value

## Overview

This project focuses on modeling **customer purchasing behavior** on an e-commerce platform using historical transactional and behavioral data. The goal is to identify high-value customers by estimating:

1. **Conversion Propensity** – the probability that a customer will place an order in a future time window.
2. **Order Value** – the expected monetary value of a customer’s order.

Together, these models enable expected value–based customer ranking, which can be used for marketing prioritization, targeting and revenue optimization.

---

## Objectives

- Predict the probability of a customer placing an order within a specified future horizon (e.g., 90 days).
- Predict the **average order value** for customers who convert.
- Combine both stages to estimate **expected customer value**.

---

## Datasets

The following datasets were used to construct the modeling pipeline:

| Dataset | Description |
|-------|-------------|
| `olist_closed_deals_dataset` | Information about marketing deals that resulted in closed sales |
| `olist_customers_dataset` | Demographic information about customers |
| `olist_geolocation_dataset` | ZIP code prefix to geographic metadata mapping |
| `olist_marketing_qualified_leads_dataset` | Marketing-qualified leads from campaigns |
| `olist_order_items_dataset` | Item-level information for each order |
| `olist_order_payments_dataset` | Payment information associated with orders |
| `olist_order_reviews_dataset` | Customer reviews for purchased items |
| `olist_orders_dataset` | Core order-level transactional data |
| `olist_products_dataset` | Product catalog metadata |
| `olist_sellers_dataset` | Seller-level information |

---

## Data Processing & Master Table

To enable customer-level modeling, all relevant datasets were aggregated into a **master table** with the following characteristics:

- **Grain:** Customer ID × Year–Month
- Each row represents a customer with activity in a given month.
- Lower-level entities (items, payments, reviews) are progressively aggregated up to the customer level.

The flow of data is as per the Data Model shown here:
![Data Model](Data%20Model.png)

The arrows show the flow of feature and overlapt percentage between the tables being joined.


### Excluded Tables

The following datasets were excluded from modeling due to limited relevance to customer purchase behavior:

- **Closed Deals Dataset** – primarily seller/campaign-focused
- **Marketing Qualified Leads Dataset** – more relevant to seller acquisition than customer behavior

---

## Data Quality & Data Integrity Report

A full **Data Integrity and Data Quality (DIDQ)** analysis was performed across all datasets.  
The full report is available here:
[Data Integrity and Data Quality (DIDQ) Report](Reports/didq_report.xls)

### DIDQ Report Structure

The **Data Integrity and Data Quality (DIDQ) Report** is organized to enable quick inspection and validation of each dataset:

- **One worksheet per dataset**  
  Each source table is analyzed independently and written to its own Excel sheet.

- **Column-level quality metrics**  
  For every column in a dataset, the following checks and summaries are provided:
  - Null value counts and percentages
  - Duplicate value counts (where applicable)
  - Number of unique values
  - Most frequent and least frequent values (categorical features)
  - Statistical summaries for numerical features, including:
    - Mean
    - Minimum
    - Maximum
    - Variance

- **Consistent schema across sheets**  
  All worksheets follow the same structure, making it easy to compare data quality issues across different tables.

This structure allows systematic identification of missing data patterns, outliers, and potential integrity issues before feature engineering and modeling.

### Key Findings

- No primary keys contain null values.
- No dataset-level duplicates were detected.
- Expected null patterns were observed:
  - Order timestamps vary by order status.
  - Not all orders receive reviews.

All such cases were handled explicitly during feature engineering.

---

## Feature Engineering

- **98 customer-level features** were engineered.
- Feature categories include:
  - Purchase frequency and recency
  - Payment behavior
  - Order value aggregates
  - Review statistics
  - Product and seller diversity
- Features become progressively richer as data is aggregated upward through the schema.

A complete list of features, source tables, and aggregation logic is documented in the **Feature Engineering Sheet**.

---

## Feature Validation

To ensure correctness and prevent leakage:
- Feature values were manually validated for multiple customer IDs.
- Temporal alignment was verified to ensure no future information leaks into historical features.

---

## Feature Selection

To improve stability and interpretability:
- Pairwise Pearson correlation was computed across numeric features.
- A threshold of **0.7** was applied.
- One feature from each highly correlated pair was removed prior to modeling.

---

## Modeling Approach

A **two-stage modeling framework** was adopted.

### Stage 1: Conversion Propensity

- **Target:** Whether a customer places an order in the next 90 days.
- **Models evaluated:**
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting (XGBoost)
  - Neural Network baseline
- **Metric:** PR-AUC (chosen due to strong class imbalance)

### Stage 2: Order Value Regression

- **Target:** Average order value for customers who convert.
- **Models evaluated:**
  - Linear (Ridge) Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Target values were log-transformed to handle heavy skew.
- **Metrics:** MAE and RMSE (reported on original value scale).

---

## Key Results (High-Level)

- Linear models perform poorly for both propensity and value prediction.
- Tree-based models capture non-linear effects and perform substantially better.
- Random Forest and XGBoost outperform linear baselines for value prediction.
- Conversion propensity remains a challenging, low-signal problem due to extreme class imbalance and temporal drift.

---

## Key Takeaways

- Customer purchase behavior is **highly non-linear** and temporally sensitive.
- Feature engineering and correct time alignment are more impactful than model complexity alone.
- A two-stage **propensity × value** framework provides a principled way to estimate expected customer value.

---

## Future Work

- Rolling backtests across multiple snapshot months
- Expected value modeling at scale
- Calibration of propensity scores
- Feature attribution using SHAP
- Production-ready scoring and monitoring pipeline

---

## Notes

This project emphasizes **correct temporal modeling**, **leakage prevention**, and **interpretability**, making it suitable for real-world deployment and business decision-making.
