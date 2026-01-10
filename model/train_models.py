import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE
from data_preprocessing import load_and_prepare

class ModelTrainingPipeline:
    def __init__(self):
        # Calculate scale_pos_weight for XGBoost (approximation)
        self.model_registry = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced',
                solver='saga'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=10,
                class_weight='balanced',
                min_samples_split=100
            ),
            'knn': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            'naive_bayes': GaussianNB(var_smoothing=1e-8),
            'random_forest': RandomForestClassifier(
                n_estimators=150, 
                random_state=42,
                class_weight='balanced',
                max_depth=15,
                min_samples_split=50
            ),
            'xgboost': xgb.XGBClassifier(
                eval_metric='logloss', 
                random_state=42,
                scale_pos_weight=3.5,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=150
            )
        }
        self.performance_metrics = {}
        
    def calculate_metrics(self, y_actual, y_predicted, y_proba=None):
        metrics_dict = {
            'accuracy': accuracy_score(y_actual, y_predicted),
            'precision': precision_score(y_actual, y_predicted, zero_division=0),
            'recall': recall_score(y_actual, y_predicted, zero_division=0),
            'f1': f1_score(y_actual, y_predicted, zero_division=0),
            'mcc': matthews_corrcoef(y_actual, y_predicted)
        }
        
        if y_proba is not None:
            metrics_dict['auc'] = roc_auc_score(y_actual, y_proba)
        else:
            metrics_dict['auc'] = 0.0
            
        return metrics_dict
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        os.makedirs('model/saved_models', exist_ok=True)
        
        # Apply SMOTE only to training data
        print("Applying SMOTE for class balancing...")
        smote_sampler = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote_sampler.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_balanced.shape}")
        
        for model_name, model_obj in self.model_registry.items():
            print(f"Training {model_name}...")
            
            model_obj.fit(X_train_balanced, y_train_balanced)
            y_pred = model_obj.predict(X_test)
            
            # Get probability for AUC
            if hasattr(model_obj, 'predict_proba'):
                y_proba = model_obj.predict_proba(X_test)[:, 1]
            else:
                y_proba = None
            
            metrics = self.calculate_metrics(y_test, y_pred, y_proba)
            self.performance_metrics[model_name] = metrics
            
            # Save model
            joblib.dump(model_obj, f'model/saved_models/{model_name}.pkl')
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        self.save_metrics_report()
    
    def save_metrics_report(self):
        df_metrics = pd.DataFrame(self.performance_metrics).T
        df_metrics = df_metrics.round(4)
        df_metrics.to_csv('model/saved_models/metrics_report.csv')
        print("\nMetrics saved to model/saved_models/metrics_report.csv")
        print(df_metrics)

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoders = load_and_prepare('data/train.csv')
    
    # Save scaler and encoders for app.py
    joblib.dump(scaler, 'model/saved_models/scaler.pkl')
    joblib.dump(encoders, 'model/saved_models/encoders.pkl')
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Class distribution - Default: {sum(y_train)}, Non-default: {len(y_train)-sum(y_train)}")
    
    pipeline = ModelTrainingPipeline()
    pipeline.train_all_models(X_train, X_test, y_train, y_test)
    
    print("\nTraining complete! Models saved in model/saved_models/")