import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class LoanDataHandler:
    def __init__(self, filepath):
        self.raw_data = pd.read_csv(filepath)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def handle_missing_vals(self):
        # Numeric columns - fill with median
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.raw_data[col].isnull().any():
                self.raw_data[col].fillna(self.raw_data[col].median(), inplace=True)
        
        # Categorical - fill with mode
        cat_cols = self.raw_data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.raw_data[col].isnull().any():
                self.raw_data[col].fillna(self.raw_data[col].mode()[0], inplace=True)
    
    def encode_categories(self):
        cat_cols = self.raw_data.select_dtypes(include=['object']).columns
        
        for col in cat_cols:
            le = LabelEncoder()
            self.raw_data[col] = le.fit_transform(self.raw_data[col].astype(str))
            self.label_encoders[col] = le
    
    def drop_unnecessary(self):
        # Drop ID columns or columns with single value
        drop_cols = []
        for col in self.raw_data.columns:
            if 'ID' in col.upper() or self.raw_data[col].nunique() == 1:
                drop_cols.append(col)
        
        if drop_cols:
            self.raw_data.drop(columns=drop_cols, inplace=True)
    
    def prepare_features(self):
        self.handle_missing_vals()
        self.drop_unnecessary()
        self.encode_categories()
        
        # Separate features and target
        target_col = 'loan_default'
        X = self.raw_data.drop(columns=[target_col])
        y = self.raw_data[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y
    
    def split_data(self, test_fraction=0.2, random_seed=42):
        X, y = self.prepare_features()
        return train_test_split(X, y, test_size=test_fraction, 
                                random_state=random_seed, stratify=y)

def load_and_prepare(filepath='data/train.csv'):
    processor = LoanDataHandler(filepath)
    X_train, X_test, y_train, y_test = processor.split_data()
    return X_train, X_test, y_train, y_test, processor.scaler, processor.label_encoders