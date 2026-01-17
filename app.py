import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Predictor", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_trained_models():
    model_paths = {
        'Logistic Regression': 'model/saved_models/logistic_regression.pkl',
        'Decision Tree': 'model/saved_models/decision_tree.pkl',
        'K-Nearest Neighbors': 'model/saved_models/knn.pkl',
        'Naive Bayes': 'model/saved_models/naive_bayes.pkl',
        'Random Forest': 'model/saved_models/random_forest.pkl',
        'XGBoost': 'model/saved_models/xgboost.pkl'
    }

    loaded_models = {}
    for name, path in model_paths.items():
        loaded_models[name] = joblib.load(path)

    preprocessor = joblib.load('model/saved_models/preprocessor.pkl')
    return loaded_models, preprocessor

def generate_confusion_heatmap(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    return fig

def main():
    st.markdown('<p class="main-header"> Vehicle Loan Default Prediction System</p>', unsafe_allow_html=True)

    models_dict, preprocessing_preprocessor = load_trained_models()

    st.sidebar.title(" Configuration Panel")
    selected_model = st.sidebar.selectbox(
        "Choose Classification Model:",
        list(models_dict.keys()),
        index=0
    )

    st.sidebar.markdown("---")

    tab1, tab2 = st.tabs([" Model Evaluation", " Performance Metrics"])

    with tab1:
        st.subheader("Upload Test Dataset")
        uploaded_csv = st.file_uploader("Choose CSV file", type=['csv'], key="csv_uploader")

        if uploaded_csv is not None:
            test_df = pd.read_csv(uploaded_csv)
            st.success(f" Dataset loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

            with st.expander(" View Dataset Preview"):
                st.dataframe(test_df.head(10))

            if st.button(" Run Prediction Analysis", type="primary"):
                with st.spinner("Processing predictions..."):

                    if 'loan_default' in test_df.columns:
                        y_test = test_df['loan_default']
                        test_df_copy = test_df.drop(columns=['loan_default'])
                        has_labels = True
                    else:
                        test_df_copy = test_df.copy()
                        has_labels = False

                    X_test_processed = preprocessing_preprocessor.transform(test_df_copy)

                    selected_clf = models_dict[selected_model]
                    predictions = selected_clf.predict(X_test_processed)

                    if hasattr(selected_clf, 'predict_proba'):
                        pred_probs = selected_clf.predict_proba(X_test_processed)[:, 1]
                    else:
                        pred_probs = None

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        default_count = np.sum(predictions == 1)
                        st.metric(" Predicted Defaults", default_count)

                    with col2:
                        safe_count = np.sum(predictions == 0)
                        st.metric(" Predicted Safe", safe_count)

                    with col3:
                        default_rate = (default_count / len(predictions)) * 100
                        st.metric(" Default Rate", f"{default_rate:.2f}%")

                    if has_labels:
                        st.markdown("---")
                        st.subheader(" Model Performance Metrics")

                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            acc = accuracy_score(y_test, predictions)
                            st.metric("Accuracy", f"{acc:.4f}")

                        with col2:
                            prec = precision_score(y_test, predictions, zero_division=0)
                            st.metric("Precision", f"{prec:.4f}")

                        with col3:
                            rec = recall_score(y_test, predictions, zero_division=0)
                            st.metric("Recall", f"{rec:.4f}")

                        with col4:
                            f1 = f1_score(y_test, predictions, zero_division=0)
                            st.metric("F1-Score", f"{f1:.4f}")

                        st.markdown("---")
                        st.subheader(" Confusion Matrix Analysis")

                        cm = confusion_matrix(y_test, predictions)
                        fig = generate_confusion_heatmap(cm, selected_model)
                        st.pyplot(fig)

                        with st.expander(" Detailed Classification Report"):
                            report = classification_report(y_test, predictions, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose())

    with tab2:
        st.subheader(" Comparative Model Performance")

        metrics_df = pd.read_csv('model/saved_models/metrics_report.csv', index_col=0)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))

        st.markdown("---")
        st.subheader(" Metrics Visualization")

        metric_choice = st.selectbox(
            "Select metric to visualize:",
            ['accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']
        )

        fig = go.Figure(data=[
            go.Bar(x=metrics_df.index, y=metrics_df[metric_choice],
                   marker_color='indianred')
        ])
        fig.update_layout(
            title=f'{metric_choice.upper()} Comparison Across Models',
            xaxis_title='Model',
            yaxis_title=metric_choice.upper()
        )
        st.plotly_chart(fig, width="stretch")

if __name__ == "__main__":
    main()
