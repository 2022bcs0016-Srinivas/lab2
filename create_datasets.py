# Lab 8: Create Dataset Versions
# Student: Srinivas Raghav V C
# Roll No: 2022BCS0016

import pandas as pd
import os

ROLL_NO = "2022BCS0016"
STUDENT_NAME = "Srinivas Raghav V C"

def create_partial_dataset(input_path, output_path, n_rows=5000):
    """Create partial dataset with first n_rows"""
    print(f"{ROLL_NO} - Creating partial dataset with {n_rows} rows...")
    
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        partial = df.head(n_rows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        partial.to_csv(output_path, index=False)
        print(f"Partial dataset saved: {output_path}")
        print(f"Shape: {partial.shape}")
        return True
    else:
        print(f"Input file not found: {input_path}")
        return False

def create_sample_datasets():
    """Create sample datasets for testing"""
    import numpy as np
    
    print(f"{ROLL_NO} - Creating sample California Housing datasets...")
    
    os.makedirs('data', exist_ok=True)
    
    # Create full dataset (20640 rows like real California Housing)
    np.random.seed(42)
    n_samples = 20640
    
    data = {
        'longitude': np.random.uniform(-124.35, -114.31, n_samples),
        'latitude': np.random.uniform(32.54, 41.95, n_samples),
        'housing_median_age': np.random.randint(1, 52, n_samples),
        'total_rooms': np.random.randint(2, 40000, n_samples),
        'total_bedrooms': np.random.randint(1, 7000, n_samples),
        'population': np.random.randint(3, 40000, n_samples),
        'households': np.random.randint(1, 7000, n_samples),
        'median_income': np.random.uniform(0.5, 15, n_samples),
        'median_house_value': np.random.uniform(15000, 500000, n_samples)
    }
    
    df_full = pd.DataFrame(data)
    df_full.to_csv('data/housing_full.csv', index=False)
    print(f"Full dataset created: data/housing_full.csv ({len(df_full)} rows)")
    
    # Create partial dataset (first 5000 rows)
    df_partial = df_full.head(5000)
    df_partial.to_csv('data/housing.csv', index=False)
    print(f"Partial dataset created: data/housing.csv ({len(df_partial)} rows)")
    
    print(f"\n{ROLL_NO} - Dataset creation completed!")
    return df_full, df_partial

if __name__ == "__main__":
    create_sample_datasets()
