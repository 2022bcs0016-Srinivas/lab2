# Lab 4: Training Script for Wine Quality Prediction
# Student: Srinivas Raghav V C
# Roll No: 2022BCS0016

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

# Configuration
MODEL_TYPE = "Lasso"
TEST_SIZE = 0.2
RANDOM_STATE = 42
APPLY_SCALING = True
ROLL_NO = "2022BCS0016"
STUDENT_NAME = "Srinivas Raghav V C"

def main():
    print("=" * 60)
    print(f"Student: {STUDENT_NAME}")
    print(f"Roll Number: {ROLL_NO}")
    print("=" * 60)
    print(f"Starting training with {MODEL_TYPE}...")
    
    # Create artifacts directory
    os.makedirs('app/artifacts', exist_ok=True)
    
    # Load dataset
    df = pd.read_csv('data/winequality-red.csv', sep=';')
    print(f"Dataset shape: {df.shape}")
    print(f"Training samples: {len(df) - int(len(df) * TEST_SIZE)}")
    print(f"Test samples: {int(len(df) * TEST_SIZE)}")
    
    # Prepare features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Preprocessing
    if APPLY_SCALING:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Train model
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{ROLL_NO} Training Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R2 Score: {r2:.6f}")
    
    # Save model
    joblib.dump(model, 'app/artifacts/model.joblib')
    joblib.dump(scaler, 'app/artifacts/scaler.joblib')
    
    # Save metrics
    metrics = {
        "roll_no": ROLL_NO,
        "student_name": STUDENT_NAME,
        "model_type": MODEL_TYPE,
        "mse": round(mse, 6),
        "r2_score": round(r2, 6),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open('app/artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModel and metrics saved to app/artifacts/")
    print(f"{ROLL_NO} - Training completed successfully!")
    
    return metrics

if __name__ == "__main__":
    main()
