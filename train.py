import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import joblib
import json
import os

# Configuration
MODEL_TYPE = "Ridge"
TEST_SIZE = 0.2
RANDOM_STATE = 42
APPLY_SCALING = True

def main():
    print(f"Starting training with {MODEL_TYPE}...")
    print(f"Configuration: test_size={TEST_SIZE}, scaling={APPLY_SCALING}")

    # Load dataset
    df = pd.read_csv('data/winequality-red.csv', sep=';')
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

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
        print("Applied StandardScaler preprocessing")

    # Train model
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    print("Model training completed")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print(f"\n{'='*50}")
    print(f"EVALUATION METRICS")
    print(f"{'='*50}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"{'='*50}\n")

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Save trained model
    model_path = 'output/model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save metrics to JSON
    results = {
        'model_type': MODEL_TYPE,
        'test_size': TEST_SIZE,
        'scaling': APPLY_SCALING,
        'mse': float(mse),
        'r2_score': float(r2)
    }

    results_path = 'output/results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
