<h1 align="center"> Telco Customer Churn Prediction</h1>

<p align="center">
  An end-to-end machine learning project that predicts which telecom customers are likely to leave,<br>
  built to help a business spend its retention budget on the customers who actually need it.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Keras-Neural%20Network-D00000?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

##  Table of Contents
- [Overview](#-overview)
- [Key Results](#-key-results)
- [Key Findings](#-key-findings)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Design Decisions](#-design-decisions)
- [Model Comparison](#-model-comparison)
- [Final Model](#-final-model-logistic-regression)
- [Business Recommendations](#-business-recommendations)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

##  Overview

Acquiring a new customer costs far more than keeping an existing one, so telecom companies try to spot customers who are about to leave and offer them discounts or promotions. The catch: every offer costs money, and **offering one to a customer who was going to stay anyway is wasted budget.**

This project builds a classification model that predicts whether a customer will churn, using demographics, subscribed services, account information, and billing data. Six models were trained and compared, and the evaluation is built around one business question:

> *Of the customers the model flags as "likely to churn", how many actually do?*

That is why the project **prioritizes precision on the churn class** rather than chasing raw accuracy or recall.

---

##  Key Results

| | |
|---|---|
| **Final model** | Logistic Regression |
| **Test accuracy** | **80.5%** |
| **Churn precision** | **0.67** (about 2 out of 3 flagged customers really churn) |
| **Churn F1-score** | 0.59 |
| **Train / test gap** | **0.002** (excellent generalization) |
| **Models compared** | 6 (Logistic Regression, SVM, KNN, Random Forest, Decision Tree, Neural Network) |

The Neural Network reached a marginally higher precision (0.672 vs 0.666), but Logistic Regression was chosen for its better accuracy, near-zero overfitting, speed, and interpretability. [Details below.](#-final-model-logistic-regression)

---

##  Key Findings

- **Contract type is the strongest churn driver.** Month-to-month customers churn far more than customers on 1-year or 2-year contracts.
- **Tenure is highly predictive.** Churned customers have a median tenure of about **10 months**, versus about **38 months** for customers who stayed.
- **Higher monthly bills go with higher churn.** Median `MonthlyCharges` is **$79.65** for churned customers versus **$64.43** for retained ones, likely driven by extra services such as Fiber optic and streaming.
- **`TotalCharges` is misleading on its own.** Retained customers have higher total charges (median $1,684 vs $704), but this is mostly a tenure effect: churners leave early, so they never accumulate charges, even though they pay more per month.
- **Gender has no meaningful relationship with churn** (p = 0.48), so the feature was removed.
- The Random Forest's top features were `tenure`, `TotalCharges`, `MonthlyCharges`, Fiber optic internet, and electronic-check payment.

---

##  Dataset

The [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset contains **7,043 customers and 21 columns** from a telecommunications company.

| Group | Features |
|---|---|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Billing** | `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` (Yes / No) |

The target is **imbalanced**: about 73.5% of customers stayed and 26.5% churned.

---

##  Project Workflow

```
Raw data → Cleaning → EDA → Feature engineering → Train/test split → Encoding & scaling
        → Feature selection → Model training (x6) → Evaluation → Final model selection
```

1. **Data cleaning:** converted `TotalCharges` to numeric, removed **22 duplicate rows**, and handled **11 missing `TotalCharges` values**. All belonged to brand-new customers with `tenure = 0`, so they were filled with 0.
2. **Exploratory analysis:** studied every feature against churn and looked for outliers and skew.
3. **Feature engineering:** created `total_services`, the number of add-on services a customer subscribes to (security, backup, device protection, tech support, streaming TV, streaming movies).
4. **Split:** 80/20 **stratified** train/test split (`random_state=42`) so both sets keep the same churn ratio.
5. **Preprocessing:** One-Hot Encoding and `StandardScaler`, both **fitted on the training set only** to avoid data leakage.
6. **Feature selection:** dropped `gender`, removed redundant "No phone/internet service" dummy variables (correlation > 0.9), and dropped `PhoneService_Yes` for near-zero correlation with the target.
7. **Modeling and evaluation:** trained six models and compared them with accuracy, precision, recall, F1, confusion matrices, and train/test gaps.

---

##  Design Decisions

**Why precision over recall?**
A false positive means the company sends a discount to a customer who was staying anyway. The goal is a *high-confidence* list of at-risk customers, not the longest possible list.

**Why no oversampling?**
I tested **ADASYN** to balance the classes. It raised recall, but it also produced many more false positives and lowered precision, which is exactly the trade-off the business wants to avoid. The final pipeline therefore uses the original class distribution.

**Why keep the outliers?**
The extreme `TotalCharges` values, especially among churned customers, reflect real customer behavior rather than data errors, so they were kept instead of capped or removed.

**Why fit the encoder and scaler on the training set only?**
Fitting them on the full dataset would leak information from the test set into training and inflate the results.

---

##  Model Comparison

| Model | Accuracy | Precision | Recall | F1 | Train/Test Gap |
|---|---:|---:|---:|---:|---:|
| Neural Network (Keras) | 0.793 | **0.672** | 0.425 | 0.521 | 0.015 |
| **Logistic Regression** | **0.805** | 0.666 | **0.530** | **0.590** | **0.002** |
| SVM | 0.795 | 0.663 | 0.460 | 0.543 | 0.018 |
| Random Forest | 0.797 | 0.658 | 0.487 | 0.560 | 0.045 |
| Decision Tree | 0.787 | 0.651 | 0.422 | 0.512 | 0.015 |
| KNN | 0.753 | 0.540 | 0.454 | 0.493 | 0.087 |

> Precision, recall, and F1 are for the **churn class** on the held-out test set (1,405 customers, 372 of them churners). The Neural Network is a Keras `Sequential` model with dense layers, dropout, and early stopping.
---

##  Final Model: Logistic Regression

The Neural Network has the highest precision, but its lead over Logistic Regression is only **0.006**. Logistic Regression was selected because it offers the best overall balance:

- **High churn precision** (0.666)
- **Highest test accuracy** (0.805)
- **Best generalization:** a train/test gap of just 0.002
- **Highest recall and F1** of all models
- **Simple, fast, and interpretable** through its coefficients, which matters when the output is used to justify business decisions
---

##  Business Recommendations

These follow from the findings above and are suggestions rather than tested interventions:

1. **Focus on the first year.** Churners leave after a median of about 10 months, so early-tenure onboarding and check-ins are where retention effort pays off most.
2. **Move month-to-month customers to longer contracts** with a small incentive, since contract length is the clearest churn lever.
3. **Review the Fiber optic and high-bill segments.** Churn is concentrated among customers with higher monthly charges, which may point to a price-to-value problem.
4. **Target offers using the model's high-precision list** to avoid spending retention budget on customers who would have stayed.

---

##  Repository Structure

```
├── TelcoCustomerChurn.ipynb     # Full analysis, preprocessing, and modeling
├── Telco-Customer-Churn.csv     # Dataset
└── README.md
```

---

##  How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Viola-Ibrahim/Telco-Customer-Churn.git
cd Telco-Customer-Churn

# 2. Install dependencies
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn tensorflow

# 3. Launch the notebook
jupyter notebook TelcoCustomerChurn.ipynb
```

> **Note:** the notebook was written in Google Colab and reads the data from `/content/Telco-Customer-Churn.csv`. If you run it locally, change that path to `Telco-Customer-Churn.csv`.

---

##  Future Improvements

- Hyperparameter tuning with cross-validation (`GridSearchCV` / `RandomizedSearchCV`) instead of a single train/test split
- **Decision-threshold tuning** to control the precision–recall trade-off directly
- Model explainability with SHAP
- Gradient boosting models (XGBoost / LightGBM)
- Wrapping the final model in a `Pipeline` and serving it through a small Streamlit app

---

##  Author

**Viola Ibrahim**: Computer Science student at Ain Shams University, focused on Machine Learning, NLP, and Generative AI.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/violaibrahim-46a577332)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/Viola-Ibrahim)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail&logoColor=white)](mailto:violaibrahim01@gmail.com)
