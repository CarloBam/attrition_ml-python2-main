from typing import Any, Dict
import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)

def add_random_columns(df: pd.DataFrame, num_columns: int) -> pd.DataFrame:
    """Add random columns to the DataFrame."""
    for i in range(num_columns):
        column_name = f'random_col_{i+1}'
        df[column_name] = np.random.rand(len(df))
    return df

def generate_new_sample_data(num_rows: int, num_random_columns: int) -> pd.DataFrame:
    """Generate a new synthetic dataset with random columns."""
    # Generate base synthetic data
    rng = np.random.default_rng()
    departments = ["Sales", "Engineering", "Customer Success", "Operations", "Marketing", "Finance", "HR", "Legal"]
    
    df = pd.DataFrame({
        "employee_id": np.arange(1, num_rows + 1),
        "age": rng.integers(21, 63, num_rows),
        "tenure_months": rng.integers(3, 180, num_rows),
        "salary": rng.normal(450_000, 120_000, num_rows).clip(120_000, 1_200_000),
        "department": rng.choice(departments, num_rows),
        "attrition": rng.choice([0, 1], num_rows)  # 0 = Stay, 1 = Leave
    })
    
    # Add random columns
    df = add_random_columns(df, num_random_columns)
    
    return df

def create_sample_dataset(filepath: str, num_rows: int = 2000, num_random_columns: int = 5) -> None:
    """Create and save a new sample dataset."""
    df = generate_new_sample_data(num_rows, num_random_columns)
    save_data(df, filepath)