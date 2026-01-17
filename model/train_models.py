import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from imblearn.over_sampling import SMOTE
import xgboost as xgb

from data_preprocessing import load_and_prepare


# -------------------------------------------------
# SAFE GPU DETECTION FOR XGBOOST
# -------------------------------------------------
def get_xgb_tree_method():
    """
    Always use CPU method for reliability on Windows.
    Change to 'gpu_hist' only if you have confirmed CUDA support.
    """
    # For Windows/CPU stability, always use CPU method
    return "hist"  # Force CPU histogram method


class ModelTrainingPipeline:
    def __init__(self):
        self.models = {}
        self.metrics = {}

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred)
        }

    def train_all_models(self, X_train, X_test, y_train, y_test):
        os.makedirs("model/saved_models", exist_ok=True)

        print("Applying SMOTE on training data...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        tree_method = get_xgb_tree_method()
        print(f"XGBoost tree_method set to: {tree_method}")

        # -------------------------------------------------
        # ALL 6 MANDATORY MODELS
        # -------------------------------------------------
        self.models = {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                solver="saga",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=50,
                class_weight="balanced",
                random_state=42
            ),
            "knn": KNeighborsClassifier(
                n_neighbors=7,
                weights="distance"
            ),
            "naive_bayes": GaussianNB(),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=18,
                min_samples_split=40,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),
            "xgboost": xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=300,  # Reduced for faster training
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                scale_pos_weight=pos_weight,
                tree_method=tree_method,
                random_state=42,
                n_jobs=-1
            )
        }

        best_auc = 0.0
        best_model_name = None

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # -------------------------------------------------
            # SPECIAL SAFE HANDLING FOR NAIVE BAYES
            # -------------------------------------------------
            if name == "naive_bayes":
                print("⚠️ Training Naive Bayes on subset (memory-safe mode)")

                max_samples = 30000
                if X_train.shape[0] > max_samples:
                    idx = np.random.choice(X_train.shape[0], max_samples, replace=False)
                    X_nb = X_train[idx]
                    y_nb = y_train.iloc[idx]
                else:
                    X_nb = X_train
                    y_nb = y_train

                X_nb = X_nb.toarray()
                X_test_eval = X_test.toarray()

                model.fit(X_nb, y_nb)
                y_proba = model.predict_proba(X_test_eval)[:, 1]

            else:
                model.fit(X_train_bal, y_train_bal)
                y_proba = model.predict_proba(X_test)[:, 1]

            y_pred = (y_proba >= 0.5).astype(int)

            metrics = self.calculate_metrics(y_test, y_pred, y_proba)
            self.metrics[name] = metrics

            print(
                f"{name.upper()} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"AUC: {metrics['auc']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"Recall: {metrics['recall']:.4f}"
            )

            joblib.dump(model, f"model/saved_models/{name}.pkl")

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_model_name = name

        # Save best model
        joblib.dump(
            self.models[best_model_name],
            "model/saved_models/best_model.pkl"
        )

        print(f"\nBest Model: {best_model_name} (AUC={best_auc:.4f})")
        self.save_metrics()

    def save_metrics(self):
        df = pd.DataFrame(self.metrics).T.round(4)
        df.to_csv("model/saved_models/metrics_report.csv")
        print("\nMetrics saved to model/saved_models/metrics_report.csv")
        print(df)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare(
        "data/train.csv"
    )

    os.makedirs("model/saved_models", exist_ok=True)
    joblib.dump(preprocessor, "model/saved_models/preprocessor.pkl")

    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")
    print(f"Default rate: {y_train.mean():.4f}")

    trainer = ModelTrainingPipeline()
    trainer.train_all_models(X_train, X_test, y_train, y_test)

    print("\nTraining complete. All models saved.")