# Lab 8: California Housing Training Script
# Student: Srinivas Raghav V C
# Roll No: 2022BCS0016

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
ROLL_NO = "2022BCS0016"
STUDENT_NAME = "Srinivas Raghav V C"


def main():
    print("=" * 60)
    print("Lab 8: California Housing Training")
    print(f"Student: {STUDENT_NAME}")
    print(f"Roll Number: {ROLL_NO}")
    print("=" * 60)

    os.makedirs('app/artifacts', exist_ok=True)

    data_path = 'data/housing.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            'data/housing.csv is missing. Create dataset versions with create_datasets.py '
            'or pull the tracked file with DVC before training.'
        )

    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset rows: {len(df)}")

    df = df.dropna()

    target_col = None
    for col in ['median_house_value', 'MedHouseVal', 'target', 'price']:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        target_col = df.columns[-1]

    print(f"Target column: {target_col}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    X = df[numeric_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{ROLL_NO} Training Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R2 Score: {r2:.6f}")
    print(f"  Dataset Size: {len(df)}")

    joblib.dump(model, 'app/artifacts/california_model.joblib')
    joblib.dump(scaler, 'app/artifacts/california_scaler.joblib')

    metrics = {
        'roll_no': ROLL_NO,
        'student_name': STUDENT_NAME,
        'mse': round(mse, 6),
        'rmse': round(rmse, 6),
        'r2_score': round(r2, 6),
        'dataset_size': len(df),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
    }

    with open('app/artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved to app/artifacts/metrics.json")
    print(f"{ROLL_NO} - Training completed successfully!")

    return metrics


if __name__ == '__main__':
    main()
