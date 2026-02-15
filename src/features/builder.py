"""Feature engineering module."""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureBuilder:
    """Handles feature engineering and preprocessing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with new features
        """
        df = df.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Create interaction terms for top numeric features
        if len(numeric_cols) >= 2:
            top_cols = numeric_cols[:2]
            df[f'{top_cols[0]}_x_{top_cols[1]}'] = df[top_cols[0]] * df[top_cols[1]]
        
        # Create binned features for numeric columns
        for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
            df[f'{col}_binned'] = pd.cut(df[col], bins=5, labels=False)
        
        return df
    
    def preprocess(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> tuple:
        """
        Preprocess features using scaling and encoding.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Preprocessed X_train and X_test (if provided)
        """
        X_train = X_train.copy()
        
        # Handle numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        # Handle categorical columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            encoder = LabelEncoder()
            X_train[col] = encoder.fit_transform(X_train[col].astype(str))
            self.encoders[col] = encoder
        
        self.is_fitted = True
        
        if X_test is not None:
            X_test = X_test.copy()
            
            # Apply same scaling to test
            if len(numeric_cols) > 0:
                X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            
            # Apply same encoding to test
            for col in categorical_cols:
                if col in self.encoders:
                    # Handle unseen categories
                    X_test[col] = X_test[col].astype(str).map(
                        lambda x: self.encoders[col].transform([x])[0] 
                        if x in self.encoders[col].classes_ else -1
                    )
            
            return X_train, X_test
        
        return X_train
