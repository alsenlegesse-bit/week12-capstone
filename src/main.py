"""Main pipeline for the machine learning project."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
from src.data.loader import DataLoader, DataConfig
from src.features.builder import FeatureBuilder
from src.models.trainer import ModelTrainer
from src.utils.metrics import calculate_business_metrics

def main() -> Dict[str, Any]:
    """
    Main execution pipeline.
    
    Returns:
        Dictionary with results and metrics
    """
    # Configuration
    data_path = Path("data/sample_data.csv")
    config = DataConfig(
        data_path=data_path,
        target_column="target",
        test_size=0.2,
        random_state=42
    )
    
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader(config)
    data = loader.load_data()
    print(f"Data loaded: {data.shape}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = loader.split_data()
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # 3. Feature Engineering
    print("Creating features...")
    feature_builder = FeatureBuilder()
    
    # Create new features
    X_train = feature_builder.create_features(X_train)
    X_test = feature_builder.create_features(X_test)
    
    # Preprocess
    X_train, X_test = feature_builder.preprocess(X_train, X_test)
    
    # 4. Train Model
    print("Training model...")
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # 6. Business Metrics
    business_metrics = calculate_business_metrics(
        trainer, X_test, y_test
    )
    
    # Combine results
    results = {
        "model_metrics": metrics,
        "business_metrics": business_metrics,
        "data_summary": loader.get_data_summary(),
        "feature_importance": dict(zip(
            X_train.columns,
            trainer.model.feature_importances_
        ))
    }
    
    print("\n=== Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Potential Savings: ${business_metrics['potential_savings']:,.2f}")
    
    return results

if __name__ == "__main__":
    results = main()
EOFcat > src/utils/metrics.py << 'EOF'
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
