# CreditIQ | Model Performance & Underwriting Risk Report

## Executive Summary
This report documents the design, validation, and deployment of **CreditIQ**, an enterprise machine learning credit risk decisioning system developed for consumer loan portfolios. The system predicts applicant default probability (`TARGET = 1`) on unsecured cash and revolving loans using applicant demographics, financial leverage ratios, and multi-bureau external ratings.

Operating on an actual portfolio of **8,643 loan applications** with an observed **7.7% default rate** (~12:1 negative-to-positive class imbalance), the system achieved:
- **Test ROC-AUC:** `0.7659` (High tabular discriminatory power).
- **Cost-Optimal Decision Cutoff:** `0.610` (yielding a **9.9% reduction in total expected portfolio loss** compared to standard uncalibrated 0.50 cutoff).
- **Recall on Defaulters:** `59.4%` at balanced sensitivity, capturing the majority of credit losses before disbursement.

---

## Model Benchmark & Evaluation Matrix

All models were evaluated on an identical **20% stratified holdout test split** (1,729 applications) containing 133 actual defaulters and 1,596 repaid loans.

| Model Candidate | Algorithm Family | Test ROC-AUC | Test PR-AUC | Default Recall | Precision | Brier Score |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Tuned)** | **Bagged Decision Trees** | **0.7659** | **0.2522** | **59.4%** | **21.5%** | **0.1872** |
| Logistic Regression | L2-Regularized GLM | 0.7596 | 0.2378 | 66.2% | 15.1% | 0.2140 |
| Hist Gradient Boosting | Histogram Tree Boosting | 0.7407 | 0.2310 | 49.6% | 21.4% | 0.1985 |

### Benchmark Observations
1. **Tree Bagging Superiority under Moderate Sample Size:** `RandomForestClassifier` with `class_weight='balanced'` achieved the highest stability and area under the curve, avoiding overfitting while capturing non-linear feature interactions between debt burden and bureau scores.
2. **Logistic Regression Baseline Strength:** The linear model achieved `0.7596` ROC-AUC, confirming that the engineered financial ratios (`DEBT_TO_INCOME`, `ANNUITY_TO_INCOME`, `EXT_SOURCE_MEAN`) cleanly linearize key credit signals.

---

## Cost-Sensitive Business Optimization

In consumer credit underwriting, misclassifications exhibit severe financial asymmetry:
- **False Negative (Default Missed):** High loss ($L_{FN} \approx \$5.0$ units) due to principal loss and collection costs.
- **False Positive (Safe Applicant Rejected):** Moderate loss ($L_{FP} \approx \$1.0$ unit) representing lost interest margin and customer friction.

$$\text{Total Expected Loss} = 5.0 \times \text{FN} + 1.0 \times \text{FP}$$

### Threshold Optimization Analysis
- **Standard Naive Cutoff ($T = 0.50$):** Expected portfolio loss = **595.0 units**.
- **Cost-Optimal Cutoff ($T^* = 0.610$):** Expected portfolio loss = **536.0 units**.
- **Net Portfolio Savings:** **9.92% direct reduction in loan loss expenses**.

---

## Primary Credit Risk Drivers (Explainability)

Feature attribution analysis demonstrates that domain-engineered indicators dominate raw demographic attributes:

1. **`EXT_SOURCE_MEAN` & `EXT_SOURCE_2/3`:** Multi-bureau historical credit bureau repayment scores provide the highest predictive information gain.
2. **`DEBT_TO_INCOME` ($Credit / Income$):** Ratios exceeding 4.0x indicate heavy debt leverage and significantly spike default probability.
3. **`CREDIT_ANNUITY_RATIO` ($Credit / Annuity$):** Measures loan duration and repayment strain.
4. **`EMPLOYED_YEARS` / `EMPLOYED_TO_AGE_RATIO`:** Applicant employment stability and tenure heavily mitigate default probability.
5. **`AGE_YEARS`:** Younger credit applicants (<25 years) exhibit higher delinquency volatility than mature borrowers (>38 years).

---

## Regulatory Compliance & Model Governance (FCRA & Fair Lending)

1. **Adverse Action Reason Codes:** The inference engine emits transparent, interpretable reasons (`key_risk_factors`) for every declined or referred decision, ensuring compliance with the Equal Credit Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA).
2. **Audit Logging:** Every scoring event, feature input vector, predicted probability, and underwriter override is committed to the MySQL `underwriting_decisions` table with immutable timestamps.
3. **Protected Attribute Disparate Impact:** Sensitive attributes (`CODE_GENDER`, family status) are isolated from primary underwriting rule weights to guarantee non-discriminatory lending practices.
