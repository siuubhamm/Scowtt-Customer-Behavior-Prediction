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
- The number of customers who have done more than one purchase ar ~3%, indicating a class imbalance.

All such cases were handled explicitly during feature engineering.

---

## Feature Engineering

A diverse set of customer-level features was engineered to capture historical behavior, spending patterns, and preferences.  
Features were derived through systematic aggregation of lower-level transactional data and can be broadly categorized as follows.

**Detailed documentation for every engineered feature — including source tables, aggregation logic, and definitions — is available in the  
[Feature Engineering File](Feature%20Engineering.xlsx).**

### 1. Count-Based (Numerical) Features
These features capture **how often** a customer interacts with the platform.

- Number of orders placed
- Number of products purchased
- Number of sellers interacted with
- Number of reviews submitted

**Aggregation logic:**  
Counts are computed by aggregating order- and item-level records grouped by `customer_unique_id` and time window.

---

### 2. Monetary (Total / Aggregate) Features
These features quantify **how much** a customer spends or pays through different channels.

- Total order value
- Total payment value
- Payment value split by payment type (credit card, boleto, voucher, etc.)
- Total freight cost

**Aggregation logic:**  
Monetary values are summed over all relevant transactions for a customer within the aggregation period.

---

### 3. Pivoted Status-Based Features
These features capture **order lifecycle behavior** by splitting aggregates across different order states.

- Number of orders by status (approved, shipped, delivered, canceled, etc.)
- Total order value by status
- Total freight value by status
- Average order size by status

**Aggregation logic:**  
Order-level metrics are first grouped by `(customer_unique_id, order_status)` and then pivoted into separate columns per status.

---

### 4. Recency and Temporal Features
These features describe **how recently** a customer interacted with the platform.

- Days since last order purchased
- Days since last order shipped
- Days since last review created

**Aggregation logic:**  
Recency is calculated as the difference between the snapshot date and the most recent relevant timestamp per customer.

---

### 5. Preference-Based Features
These features summarize **customer preferences** inferred from historical behavior.

- Preferred product category
- Preferred product category (English mapping)
- Most frequent payment method

**Aggregation logic:**  
Preferences are derived using **mode-based aggregation**, selecting the most frequently occurring value per customer.

---

### 6. Ratio and Average Features
These features normalize totals to reflect **behavioral intensity** rather than scale.

- Average order size
- Average order price
- Average freight value
- Average review score

**Aggregation logic:**  
Computed as ratios of aggregated totals (e.g., total value divided by number of orders).

---

Together, these feature groups provide a comprehensive representation of customer behavior across frequency, monetary value, recency, lifecycle dynamics and preferences, forming the foundation for both propensity and order value modeling.

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

## Label Creation

To ensure realistic and leakage-free modeling, both propensity and order value labels are created using a **future-looking time horizon** relative to a fixed snapshot date.

### Snapshot and Horizon Definition

- A **snapshot date** is defined for each modeling run (e.g., end of a given month).
- A fixed **prediction horizon** of 90 days is used.
- Only information available **on or before the snapshot date** is used for feature computation.
- Labels are derived strictly from customer activity **after the snapshot date and within the horizon window**.

---

### 1. Conversion Propensity Label

The propensity label indicates whether a customer places **at least one order** within the future prediction horizon.

**Label definition:**

- `y_propensity = 1`  
  If the customer places one or more orders within the next 90 days after the snapshot date.
- `y_propensity = 0`  
  Otherwise.

**Computation logic:**

- Orders are filtered to those with purchase timestamps:
  - greater than the snapshot date, and
  - less than or equal to snapshot date + 90 days.
- Orders are mapped to customers using `customer_id → customer_unique_id`.
- A binary indicator is created per customer based on the presence of at least one order in the horizon.

This formulation captures **future purchase intent**, not historical frequency.

---

### 2. Order Value Label

The order value label represents the **monetary value of customer purchases** within the future horizon, conditional on conversion.

**Label definition:**

- `y_value` is defined as the **average order value** generated by a customer within the prediction horizon.
- For modeling stability, the target is log-transformed:
  ```text
  y_value_log = log(1 + y_value)

### Computation logic:

Item-level aggregation

From the **olist_order_items_dataset**, item-level price and freight_value are summed.

These values are first aggregated at the order_id level, producing a total order value per order.

Order-level values are joined with the orders dataset to associate orders with customers.

All orders placed by a customer within the future horizon are aggregated at the
**customer_unique_id** level (sum or average, depending on modeling objective).

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

## Key Results

## Modeling Results

### Stage 1: Conversion Propensity (Classification)

| Model | Best Hyperparameters | Validation PR-AUC |
|------|----------------------|------------------|
| Logistic Regression | C=1.0, penalty=l1, class_weight=balanced | 0.0128 |
| Random Forest Classifier | n_estimators=500, max_depth=5, min_samples_leaf=1 | 0.0182 |
| XGBoost Classifier | n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=1.0 | 0.0684 |
| Neural Network (MLP) | 2 hidden layers, dropout=0.2, Adam (lr=1e-3) | ~0.01–0.03 |

**Metric:** PR-AUC (chosen due to extreme class imbalance)

---

### Stage 2: Order Value Prediction (Regression)

| Model | Best Hyperparameters | MAE | RMSE |
|------|----------------------|-----|------|
| Ridge Regression | alpha=10.0 | 40282.20 | 598085.32 |
| Random Forest Regressor | n_estimators=300, max_depth=None, min_samples_leaf=5, max_features=0.7 | 17.17 | 143.74 |
| XGBoost Regressor | n_estimators=800, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8 | 23.21 | 173.66 |

**Metric:** MAE and RMSE (evaluated on original value scale after inverse log transform)

**Key Insights**:

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
