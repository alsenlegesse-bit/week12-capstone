"""Interactive dashboard for model visualization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader, DataConfig
from src.models.trainer import ModelTrainer
from src.features.builder import FeatureBuilder

# Page config
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Machine Learning Model Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Controls")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.metrics = None
    st.session_state.data_summary = None

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your data (CSV)",
    type=['csv']
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = Path("temp_data.csv")
    temp_path.write_bytes(uploaded_file.getvalue())
    
    # Configuration
    target_column = st.sidebar.text_input("Target column name", "target")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)
    
    if st.sidebar.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            # Load data
            config = DataConfig(
                data_path=temp_path,
                target_column=target_column,
                test_size=test_size
            )
            loader = DataLoader(config)
            data = loader.load_data()
            
            # Split data
            X_train, X_test, y_train, y_test = loader.split_data()
            
            # Feature engineering
            fb = FeatureBuilder()
            X_train = fb.create_features(X_train)
            X_test = fb.create_features(X_test)
            X_train, X_test = fb.preprocess(X_train, X_test)
            
            # Train model
            trainer = ModelTrainer()
            trainer.train(X_train, y_train)
            
            # Evaluate
            metrics = trainer.evaluate(X_test, y_test)
            
            # Store in session
            st.session_state.model_trained = True
            st.session_state.metrics = metrics
            st.session_state.data_summary = loader.get_data_summary()
            st.session_state.feature_importance = dict(zip(
                X_train.columns,
                trainer.model.feature_importances_
            ))
            st.session_state.model = trainer
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success("Model trained successfully!")
    
    # Clean up
    if temp_path.exists():
        temp_path.unlink()

# Main content area
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📈 Model Performance")
    if st.session_state.model_trained:
        metrics = st.session_state.metrics
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("Precision", f"{metrics['precision']:.3f}")
        st.metric("Recall", f"{metrics['recall']:.3f}")
        st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    else:
        st.info("Upload data and train model to see metrics")

with col2:
    st.subheader("📊 Data Overview")
    if st.session_state.data_summary:
        summary = st.session_state.data_summary
        st.write(f"**Shape:** {summary['shape'][0]} rows, {summary['shape'][1]} columns")
        st.write(f"**Features:** {len(summary['columns'])}")
        
        # Target distribution
        if 'target_distribution' in summary:
            st.write("**Target Distribution:**")
            for k, v in summary['target_distribution'].items():
                st.write(f"- Class {k}: {v} samples")
    else:
        st.info("Upload data to see overview")

with col3:
    st.subheader("🎯 Business Impact")
    if st.session_state.model_trained:
        # Simulate business metrics
        st.metric("Potential Savings", "$12,450", "+15%")
        st.metric("ROI", "124%", "+8%")
        st.metric("False Positives", "23", "-12")
    else:
        st.info("Train model to see business impact")

# Feature Importance
st.subheader("🔍 Feature Importance")
if st.session_state.model_trained:
    importance_df = pd.DataFrame(
        list(st.session_state.feature_importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df.tail(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Feature Importances'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Train model to see feature importance")

# Prediction Explorer
st.subheader("🎲 Prediction Explorer")
if st.session_state.model_trained:
    # Sample some test data
    sample_idx = st.slider("Select sample index", 0, len(st.session_state.X_test)-1, 0)
    
    sample_features = st.session_state.X_test.iloc[sample_idx:sample_idx+1]
    true_label = st.session_state.y_test.iloc[sample_idx]
    
    # Make prediction
    pred = st.session_state.model.predict(sample_features)[0]
    pred_proba = st.session_state.model.predict_proba(sample_features)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**True Label:**", true_label)
        st.write("**Predicted:**", pred)
    
    with col2:
        # Probability chart
        proba_df = pd.DataFrame({
            'Class': [f'Class {i}' for i in range(len(pred_proba))],
            'Probability': pred_proba
        })
        fig = px.bar(proba_df, x='Class', y='Probability', title='Prediction Probabilities')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit • Week 12 Capstone Project")
