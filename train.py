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

def main():
    print(f"Starting training with {MODEL_TYPE}...")
    
    # Load dataset
    df = pd.read_csv('data/winequality-red.csv', sep=';')
    
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
    
    # Print metrics
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save trained model and metrics in root directory
    joblib.dump(model, 'model.joblib')
    with open('metrics.json', 'w') as f:
        json.dump({'mse': mse, 'r2': r2}, f)
    
    print("Model and metrics saved successfully in root directory.")

if __name__ == "__main__":
    main()
