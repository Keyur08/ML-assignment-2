import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report
)
from pathlib import Path

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.subheader("Classification Models Performance Evaluation")

@st.cache_resource
def load_resources():
    models = {}
    model_names = [
        'logistic_regression', 'decision_tree', 'knn',
        'naive_bayes', 'random_forest', 'xgboost'
    ]
    
    for name in model_names:
        models[name] = joblib.load(f'model/saved_models/{name}.pkl')
    
    preprocessor = joblib.load('model/saved_models/preprocessor.pkl')
    metrics = pd.read_csv('model/saved_models/metrics_report.csv', index_col=0)
    
    return models, preprocessor, metrics

try:
    models_dict, preprocessor, metrics_df = load_resources()
    resources_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    resources_loaded = False

if resources_loaded:
    with st.sidebar:
        st.header("Settings")
        
        # ---- Download sample test CSV ----
        test_csv_path = Path("data/test.csv")

        if test_csv_path.exists():
            with open(test_csv_path, "rb") as f:
                st.download_button(
                    label="Download Sample Test CSV",
                    data=f,
                    file_name="test.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Sample test.csv not found at data/test.csv")

        model_options = {
            'Logistic Regression': 'logistic_regression',
            'Decision Tree': 'decision_tree',
            'K-Nearest Neighbors': 'knn',
            'Naive Bayes': 'naive_bayes',
            'Random Forest': 'random_forest',
            'XGBoost': 'xgboost'
        }
        
        selected_model_name = st.selectbox(
            "Select Model",
            list(model_options.keys())
        )
        model_key = model_options[selected_model_name]
        selected_model = models_dict[model_key]
        
        # Data source
        data_source = st.radio(
            "Test Data Source",
            ["Upload CSV"]
        )
    
    tab1, tab2 = st.tabs(["Model Testing", "Performance Comparison"])
    
    with tab1:
        st.header("Model Testing and Evaluation")
        
        test_data = None
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload test CSV file", type=['csv'])
            if uploaded_file is not None:
                test_data = pd.read_csv(uploaded_file)
                if len(test_data) > 5000:
                    test_data = test_data.head(5000)
                    st.warning("Using first 5000 rows only.")
        
        
        if test_data is not None:
            st.write(f"Test data loaded: {test_data.shape[0]} rows, {test_data.shape[1]} columns")
            
            if st.button("Run Evaluation"):
                with st.spinner("Evaluating model..."):
                    if 'loan_default' in test_data.columns:
                        y_true = test_data['loan_default']
                        X_test = test_data.drop(columns=['loan_default'])
                        has_labels = True
                    else:
                        X_test = test_data
                        y_true = None
                        has_labels = False
                    
                    X_processed = preprocessor.transform(X_test)
                    
                    if selected_model_name == "Naive Bayes" and hasattr(X_processed, "toarray"):
                        X_processed = X_processed.toarray()
                    
                    # Predict
                    y_pred = selected_model.predict(X_processed)
                    
                    # Get probabilities if available
                    if hasattr(selected_model, 'predict_proba'):
                        y_proba = selected_model.predict_proba(X_processed)[:, 1]
                    else:
                        y_proba = None
                    
                    # Display results
                    st.subheader(f"Results for {selected_model_name}")
                    
                    # Basic counts
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(y_pred))
                    with col2:
                        st.metric("Predicted Default", sum(y_pred == 1))
                    with col3:
                        st.metric("Predicted Non-Default", sum(y_pred == 0))
                    
                    if has_labels and y_true is not None:
                        st.subheader("Evaluation Metrics")
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        mcc = matthews_corrcoef(y_true, y_pred)
                        
                        if y_proba is not None:
                            auc = roc_auc_score(y_true, y_proba)
                        else:
                            auc = "N/A"
                        
                        # Display metrics
                        metrics_cols = st.columns(6)
                        metrics_data = [
                            ("Accuracy", f"{accuracy:.4f}"),
                            ("AUC", f"{auc:.4f}" if auc != "N/A" else auc),
                            ("Precision", f"{precision:.4f}"),
                            ("Recall", f"{recall:.4f}"),
                            ("F1-Score", f"{f1:.4f}"),
                            ("MCC", f"{mcc:.4f}")
                        ]
                        
                        for col, (name, value) in zip(metrics_cols, metrics_data):
                            with col:
                                st.metric(name, value)
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax.text(j, i, str(cm[i, j]), ha='center', va='center')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])
                        ax.set_xticklabels(['Non-Default', 'Default'])
                        ax.set_yticklabels(['Non-Default', 'Default'])
                        st.pyplot(fig)
                        
                        # Classification Report
                        st.subheader("Classification Report")
                        report = classification_report(y_true, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)
    
    with tab2:
        st.header("Model Performance Comparison")
        
        display_df = metrics_df.copy()
        display_df.index = [
            'Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
            'Naive Bayes', 'Random Forest', 'XGBoost'
        ]
        
        display_df = display_df.rename(columns={
            'accuracy': 'Accuracy',
            'auc': 'AUC',
            'precision': 'Precision', 
            'recall': 'Recall',
            'f1': 'F1-Score',
            'mcc': 'MCC'
        })
        
        observations = {
            'Logistic Regression': 'Baseline model with reasonable performance.',
            'Decision Tree': 'Interpretable but prone to overfitting.',
            'K-Nearest Neighbors': 'Sensitive to data scaling and dimensionality.',
            'Naive Bayes': 'Fast but assumes feature independence.',
            'Random Forest': 'Robust ensemble method with good performance.',
            'XGBoost': 'Advanced boosting algorithm with best overall metrics.'
        }
        
        display_df['Observation'] = [observations[model] for model in display_df.index]
        
        cols = ['Observation'] + [col for col in display_df.columns if col != 'Observation']
        display_df = display_df[cols]
        
        for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score', 'MCC']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df, use_container_width=True)

else:
    st.warning("Please ensure all model files are available in the model/saved_models directory.")
    st.write("Required files:")
    st.write("- logistic_regression.pkl")
    st.write("- decision_tree.pkl")
    st.write("- knn.pkl")
    st.write("- naive_bayes.pkl")
    st.write("- random_forest.pkl")
    st.write("- xgboost.pkl")
    st.write("- preprocessor.pkl")
    st.write("- metrics_report.csv")