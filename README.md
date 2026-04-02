# Telco Customer Churn Prediction
### End-to-end Data Science Project | Classification | Explainable AI

---

## Business Problem

A telecom company is losing **26.5% of its customers** every period. Acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a machine learning system that:

1. **Predicts** which customers are likely to churn before they leave
2. **Explains** the specific reasons driving each customer's risk
3. **Identifies** the highest-risk customer segments for targeted intervention

---

## Key Findings

| Finding | Detail |
|---|---|
| Churn rate | 26.5% of customers churned |
| Highest risk segment | Month-to-month contract + Fiber optic + tenure < 6 months |
| Contract impact | Month-to-month → 43% churn vs Two-year → 3% churn |
| Price signal | Churners pay ~$15/month MORE than retained customers |
| Tenure cliff | Customers in first 6 months churn at 2× the average rate |
| Top protective factor | Long-tenure + annual/two-year contract |

---

## Results

| Model | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|
| Random Forest | 0.67 | 0.76 | 0.71 | 0.84 |
| **XGBoost** | **0.71** | **0.78** | **0.74** | **0.87** |

**XGBoost** selected as the final model based on superior F1 and AUC scores.

---

## Example: Why a Specific Customer is High Risk

Using SHAP values, the model explains every individual prediction:

```
Customer #412  —  Predicted churn probability: 91.3%

Pushing TOWARD churn:
  Contract: Month-to-month     +0.412
  Internet: Fiber optic        +0.287
  high_spend_new               +0.198
  tenure (very low)            +0.156
  Payment: Electronic check    +0.089

Pushing AWAY from churn:
  has_multiple_services        -0.043
```

This turns a black-box model into an actionable business tool.

---

## Project Structure

```
telco-churn/
│
├── data/
│   ├── telco_churn.csv              ← raw data (never modified)
│   └── telco_churn_cleaned.csv      ← cleaned, pre-scaling version
│
├── notebooks/
│   └── 01_churn_analysis.ipynb      ← full analysis notebook
│
├── outputs/
│   ├── figures/                     ← all saved charts
│   │   ├── 02_churn_overview.png
│   │   ├── 03_tenure_churn.png
│   │   ├── 04_contract_churn.png
│   │   ├── 05_charges_churn.png
│   │   ├── 06_correlation_heatmap.png
│   │   ├── 07_churner_profile.png
│   │   ├── 08_model_comparison.png
│   │   └── 09_shap_analysis.png
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── shap_explainer.pkl
│   ├── scaler.pkl
│   └── model_results.csv
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Cleaning
- Converted `TotalCharges` from string to float (11 rows had blank values — new customers with $0 charges, filled with 0)
- Dropped `customerID` (identifier, no predictive signal)
- Binary encoded Yes/No columns
- One-hot encoded multi-category columns (Contract, InternetService, PaymentMethod, etc.)

### 2. Feature Engineering
Four domain-informed features were engineered:

| Feature | Formula | Reasoning |
|---|---|---|
| `charges_per_tenure` | MonthlyCharges / (tenure + 1) | Normalises cost by loyalty — high value signals risk |
| `is_new_customer` | tenure < 6 → 1 else 0 | Captures the churn cliff in the first 6 months |
| `has_multiple_services` | PhoneService AND charges > median | More services = higher switching cost |
| `high_spend_new` | charges > $65 AND tenure < 12 | Highest risk profile: expensive + not yet loyal |

### 3. Handling Class Imbalance
Applied **SMOTE** (Synthetic Minority Oversampling Technique) to the training set only, balancing from 73/27 to 50/50. The test set was kept at its original imbalanced distribution to reflect real-world conditions.

### 4. Scaling
Applied `StandardScaler` fitted on training data only — preventing data leakage. Continuous features scaled: `tenure`, `MonthlyCharges`, `TotalCharges`, `charges_per_tenure`.

### 5. Modeling
- **Random Forest:** 200 trees, max_depth=10, class_weight='balanced'
- **XGBoost:** 300 trees, learning_rate=0.05, scale_pos_weight adjusted for imbalance
- Both trained on SMOTE-balanced training data, evaluated on original test set

### 6. Explainability
SHAP (SHapley Additive exPlanations) values computed for every prediction, enabling:
- Global feature importance ranking
- Per-customer explanation of prediction drivers
- Business-actionable insight delivery

---

## Business Recommendations

Based on the model findings:

1. **Target month-to-month customers** in months 1–6 with retention offers — this is the highest-risk window
2. **Incentivise annual contracts** — the single strongest predictor of retention
3. **Flag fiber optic + high charges customers** — they churn at 2× the rate of DSL customers
4. **Review pricing for new customers** — paying premium prices before feeling value is the core churn driver
5. **Use the SHAP score per customer** to personalise retention outreach — not all churners leave for the same reason

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing, modeling, evaluation |
| XGBoost | Gradient boosted model |
| Imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| Matplotlib / Seaborn | Visualisation |

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/telco-churn.git
cd telco-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch notebook
jupyter notebook notebooks/01_churn_analysis.ipynb
```

Run all cells top to bottom. Dataset downloads automatically in Cell 2.

---

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 21 features.  
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

*Project completed as part of a Data Science portfolio — Business domain*
