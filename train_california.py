"""
Lab 8: Training script for California Housing dataset
Student: Srinivas Raghav V C
Roll Number: 2022BCS0016
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train model on California Housing dataset')
    parser.add_argument('--data-path', type=str, default='data/california_housing.csv',
                        help='Path to the California Housing dataset')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Student: Srinivas Raghav V C")
    print("Roll Number: 2022BCS0016")
    print("=" * 50)
    print(f"Training with data from: {args.data_path}")
    
    # Load dataset
    df = pd.read_csv(args.data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocessing
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
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print("=" * 50)
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save trained model and metrics
    joblib.dump(model, 'output/model.joblib')
    joblib.dump(scaler, 'output/scaler.joblib')
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'dataset_shape': list(df.shape),
        'student_name': 'Srinivas Raghav V C',
        'roll_no': '2022BCS0016'
    }
    
    with open('output/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model and metrics saved successfully to output/")
    return metrics

if __name__ == "__main__":
    main()
