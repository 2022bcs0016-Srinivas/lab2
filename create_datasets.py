"""
Lab 8: Create partial and full datasets for DVC versioning
Student: Srinivas Raghav V C
Roll Number: 2022BCS0016
"""

import pandas as pd
import os

def main():
    print("=" * 50)
    print("Student: Srinivas Raghav V C (2022BCS0016)")
    print("=" * 50)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download URL
    url = "https://www.kaggle.com/api/v1/datasets/download/camnugent/california-housing-prices"
    
    print("Please download the dataset from:")
    print(url)
    print()
    
    # If the full dataset exists, create versions
    full_data_path = 'california_housing_full.csv'
    
    if os.path.exists(full_data_path):
        df_full = pd.read_csv(full_data_path)
        print(f"Full dataset loaded: {df_full.shape}")
        
        # Version 1: First 5000 rows
        df_v1 = df_full.head(5000)
        df_v1.to_csv('data/california_housing.csv', index=False)
        print(f"Version 1 created: {df_v1.shape} rows")
        
        # Version 2: Full dataset
        # df_v2 = df_full
        # df_v2.to_csv('data/california_housing.csv', index=False)
        # print(f"Version 2 created: {df_v2.shape} rows")
    else:
        print("Full dataset not found. Please download it first.")
        print("Expected file: california_housing_full.csv")

if __name__ == "__main__":
    main()
