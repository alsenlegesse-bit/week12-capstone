"""Model training module with production-ready code."""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

class ModelTrainer:
    """Handles model training, evaluation, and saving."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize model trainer.
        
        Args:
            model_params: Dictionary of model parameters
        """
        self.model_params = model_params or {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        }
        self.model = RandomForestClassifier(**self.model_params)
        self.is_trained = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def save_model(self, filepath: Path) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
