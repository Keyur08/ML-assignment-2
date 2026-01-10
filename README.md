# Vehicle Loan Default Prediction

## Problem Statement

Predict whether a borrower will default on their vehicle loan in the first EMI payment using machine learning classification models.

## Dataset Description

- **Source**: Kaggle - L&T Vehicle Loan Default Dataset
- **Instances**: 233,155 loan records
- **Features**: 42 attributes
- **Target**: loan_default (Binary: 0=No Default, 1=Default)
- **Class Distribution**: 78% Non-default, 22% Default (Imbalanced)

**Feature Categories**:

- Demographic: Age, income, employment type
- Loan Details: Disbursed amount, asset cost, LTV ratio
- Credit Bureau: Credit scores, active accounts, credit history

## Models Used

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.5649   | 0.6220 | 0.2786    | 0.6319 | 0.3867 | 0.1470 |
| Decision Tree       | 0.6216   | 0.5935 | 0.2803    | 0.4742 | 0.3523 | 0.1168 |
| K-Nearest Neighbors | 0.5621   | 0.5523 | 0.2462    | 0.4933 | 0.3284 | 0.0619 |
| Naive Bayes         | 0.3532   | 0.5935 | 0.2352    | 0.8795 | 0.3712 | 0.0914 |
| Random Forest       | 0.6253   | 0.6186 | 0.2885    | 0.4954 | 0.3646 | 0.1336 |
| XGBoost             | 0.3725   | 0.6024 | 0.2412    | 0.8811 | 0.3787 | 0.1145 |

## Model Performance Observations

| ML Model Name       | Observation                                                                                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Best overall performer with highest F1 score (0.3867) and AUC (0.6220). Good balance between precision and recall. Suitable for interpretable loan decisions.                                 |
| Decision Tree       | Achieved highest accuracy (0.6216) but lower recall. Tends to favor majority class. Simple and interpretable but prone to overfitting.                                                        |
| K-Nearest Neighbors | Weakest performer with lowest AUC (0.5523). Distance-based approach struggles with high-dimensional loan data and imbalanced classes.                                                         |
| Naive Bayes         | Highest recall (0.8795) catches most defaults but produces many false positives. Independence assumption between features may not hold for loan data.                                         |
| Random Forest       | Strong ensemble performer with balanced metrics. Handles feature interactions well. Slightly better AUC (0.6186) than Decision Tree.                                                          |
| XGBoost             | Highest recall (0.8811) similar to Naive Bayes. Effective at identifying defaults but lower precision. Gradient boosting handles imbalanced data effectively with scale_pos_weight parameter. |

## Key Insights

- **Class Imbalance Handling**: Applied SMOTE oversampling and class_weight='balanced' to address 78-22 split
- **Trade-off**: Lower accuracy but significantly improved recall and F1 scores - critical for loan default detection
- **Best Models**: Logistic Regression and Random Forest offer best balance for production use
- **High Recall Models**: Naive Bayes and XGBoost catch more defaults at cost of false positives

## Deployment

- **Live App**: [Your Streamlit URL]
- **GitHub**: [Your Repo URL]
