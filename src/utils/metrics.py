"""Business metrics calculation module."""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import confusion_matrix

def calculate_business_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Calculate business-focused metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary of business metrics
    """
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Business assumptions
    cost_per_false_positive = 100  # $100 cost for false alarm
    cost_per_false_negative = 500  # $500 cost for missed detection
    value_per_true_positive = 1000  # $1000 value for correct detection
    
    # Calculate costs and savings
    total_cost = (fp * cost_per_false_positive) + (fn * cost_per_false_negative)
    total_value = tp * value_per_true_positive
    potential_savings = total_value - total_cost
    
    # Accuracy-based metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Financial impact per prediction
    avg_impact_per_prediction = potential_savings / len(y_test)
    
    return {
        "potential_savings": float(potential_savings),
        "avg_impact_per_prediction": float(avg_impact_per_prediction),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "cost_efficiency": float(total_value / (total_cost + 1))  # Avoid division by zero
    }

def calculate_roi(initial_investment: float, savings: float) -> float:
    """
    Calculate Return on Investment.
    
    Args:
        initial_investment: Cost to implement solution
        savings: Annual savings from solution
        
    Returns:
        ROI percentage
    """
    return ((savings - initial_investment) / initial_investment) * 100
