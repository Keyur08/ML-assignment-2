import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class LoanDataHandler:
    def __init__(self, filepath: str):
        self.raw_data = pd.read_csv(filepath)
        self.preprocessor = None

    def drop_unnecessary(self):
        drop_cols = []
        for col in self.raw_data.columns:
            if 'ID' in col.upper() or self.raw_data[col].nunique() <= 1:
                drop_cols.append(col)

        if drop_cols:
            self.raw_data.drop(columns=drop_cols, inplace=True)

    def build_preprocessor(self, X: pd.DataFrame):
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ]
        )


    def prepare_features(self):
        self.drop_unnecessary()

        target_col = 'loan_default'
        if target_col not in self.raw_data.columns:
            raise ValueError("Target column 'loan_default' not found")

        X = self.raw_data.drop(columns=[target_col])
        y = self.raw_data[target_col].astype(int)

        self.build_preprocessor(X)

        return X, y

    def split_and_transform(self, test_fraction=0.2, random_seed=42):
        X, y = self.prepare_features()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_fraction,
            random_state=random_seed,
            stratify=y
        )

        # Fit only on training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        return X_train_processed, X_test_processed, y_train, y_test, self.preprocessor


    def create_sample_test_data(self, sample_size=5000, random_state=42):
        """Create a smaller test dataset for Streamlit deployment"""
        X, y = self.prepare_features()
        
        # Stratified sampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        
        _, X_sample, _, y_sample = train_test_split(
            X, y, 
            test_size=sample_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Add target back for test data
        test_sample = X_sample.copy()
        test_sample['loan_default'] = y_sample
        
        return test_sample

def load_and_prepare(filepath='data/train.csv'):
    processor = LoanDataHandler(filepath)

    X_train, X_test, y_train, y_test, preprocessor = processor.split_and_transform()

    return X_train, X_test, y_train, y_test, preprocessor


