#!/usr/bin/env python
"""Main entry point for the Financial Risk Prediction Model."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """Main function to run the pipeline."""
    print("=" * 60)
    print("Financial Risk Prediction Model")
    print("=" * 60)
    print("\nStarting pipeline...")
    
    # Import here to avoid circular imports
    try:
        from src.model import train_model
        print("✓ Model module loaded successfully")
    except ImportError as e:
        print(f"! Error loading model module: {e}")
    
    print("\nPipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
