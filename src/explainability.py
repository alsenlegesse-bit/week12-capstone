#!/usr/bin/env python
"""Model explainability module using SHAP."""

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_shap_explanations():
    """Generate SHAP explanations and save visualizations."""
    print("Generating SHAP explanations...")
    
    # Create shap_plots directory
    os.makedirs('shap_plots', exist_ok=True)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.6, n_samples),
        'loan_amount': np.random.normal(20000, 10000, n_samples),
        'employment_length': np.random.randint(0, 30, n_samples),
        'previous_defaults': np.random.randint(0, 3, n_samples)
    })
    
    # Create target
    data['target'] = (data['debt_to_income'] > 0.4).astype(int)
    
    # Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    X = data.drop('target', axis=1)
    y = data['target']
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample
    X_sample = X.head(100)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Create visualizations
    print("Generating SHAP plots...")
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot - Feature Impact")
    plt.tight_layout()
    plt.savefig('shap_plots/summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Summary plot saved")
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig('shap_plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance plot saved")
    
    print("\nSHAP visualizations saved to shap_plots/ directory")
    print("Files created:")
    for f in os.listdir('shap_plots'):
        print(f"  - shap_plots/{f}")
    
    return shap_values

if __name__ == "__main__":
    generate_shap_explanations()
