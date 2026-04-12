# Lab 8: Create Dataset Versions
# Student: Srinivas Raghav V C
# Roll No: 2022BCS0016

import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

ROLL_NO = "2022BCS0016"
STUDENT_NAME = "Srinivas Raghav V C"


def create_partial_dataset(input_path, output_path, n_rows=5000):
    """Create partial dataset with first n_rows."""
    print(f"{ROLL_NO} - Creating partial dataset with {n_rows} rows...")

    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        partial = df.head(n_rows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        partial.to_csv(output_path, index=False)
        print(f"Partial dataset saved: {output_path}")
        print(f"Shape: {partial.shape}")
        return True

    print(f"Input file not found: {input_path}")
    return False


def create_dataset_versions():
    """Fetch the canonical California Housing dataset and materialize v1/v2 CSVs."""
    print(f"{ROLL_NO} - Fetching California Housing dataset...")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)
    df_full = dataset.frame

    full_path = data_dir / "housing_full.csv"
    version1_path = data_dir / "housing.csv"

    df_full.to_csv(full_path, index=False)
    df_full.head(5000).to_csv(version1_path, index=False)

    print(f"Full dataset created: {full_path} ({len(df_full)} rows)")
    print(f"Version 1 dataset created: {version1_path} ({len(df_full.head(5000))} rows)")
    print(f"\n{ROLL_NO} - Dataset creation completed!")
    return df_full, df_full.head(5000)


if __name__ == "__main__":
    create_dataset_versions()
