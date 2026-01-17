## 1. Problem Statement

Financial institutions incur significant losses due to vehicle loan defaults, particularly when borrowers fail to pay the first EMI on the due date. The objective of this assignment is to build and evaluate multiple machine learning classification models to predict the probability of a borrower defaulting on the first EMI of a vehicle loan. Accurate prediction helps institutions minimize credit risk while ensuring that creditworthy applicants are not unnecessarily rejected.

---

## 2. Dataset Description 

- **Dataset Type:** Vehicle Loan Default Prediction
- **Source:** Public dataset (Kaggle)
- **Problem Type:** Binary Classification
- **Target Variable:** `loan_default`
  - `1` → Borrower defaulted on first EMI
  - `0` → Borrower did not default

### Dataset Information:

The dataset contains:

- **Loanee Information:** Demographic data such as age and identity-related attributes
- **Loan Information:** Loan amount, disbursal details, loan-to-value ratio, etc.
- **Bureau Data & Credit History:** Bureau score, number of active accounts, repayment history, and other credit indicators
- **Total Records:** ~2.3 lakh
- **Number of Features:** More than 12 (after preprocessing)
- **Class Distribution:** Imbalanced (default rate ≈ 22%)

---

## 3. Machine Learning Models and Evaluation Metrics 

The following six classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### Model Comparison Table

| ML Model                 | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.5897   | 0.6288 | 0.2857    | 0.5932 | 0.3856 | 0.1508 |
| Decision Tree            | 0.5957   | 0.5977 | 0.2758    | 0.5304 | 0.3629 | 0.1206 |
| KNN                      | 0.3505   | 0.5434 | 0.2294    | 0.8444 | 0.3608 | 0.0596 |
| Naive Bayes              | 0.4165   | 0.5160 | 0.2252    | 0.6920 | 0.3398 | 0.0280 |
| Random Forest (Ensemble) | 0.5966   | 0.6047 | 0.2764    | 0.5305 | 0.3635 | 0.1216 |
| XGBoost (Ensemble)       | 0.3245   | 0.5952 | 0.2338    | 0.9274 | 0.3734 | 0.1010 |

---

## 4. Observations on Model Performance  

| ML Model            | Observation                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved the highest AUC score, indicating better generalization compared to other models.              |
| Decision Tree       | Performed reasonably but showed limited generalization on high-dimensional data.                        |
| KNN                 | Achieved very high recall but low precision, resulting in many false positives.                         |
| Naive Bayes         | Performed close to random due to the strong independence assumption on one-hot encoded features.        |
| Random Forest       | Improved stability over a single decision tree but struggled with sparse, high-dimensional data.        |
| XGBoost             | Achieved high recall but low accuracy due to aggressive positive class prediction at default threshold. |

---

## 5. Streamlit Web Application  

An interactive Streamlit application was developed and deployed using **Streamlit Community Cloud** with the following features:

- CSV file upload option (test data only)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix / classification report

**Live Streamlit App Link:**
