import pytest
import pandas as pd
from src.generate_sample import make_synthetic

def test_make_synthetic():
    # Test with default parameters
    df = make_synthetic()
    assert df.shape[0] == 2000  # Check number of rows
    assert 'employee_id' in df.columns  # Check for employee_id column
    assert 'attrition_12m' in df.columns  # Check for attrition_12m column
    assert df['age'].min() >= 21  # Check minimum age
    assert df['age'].max() <= 62  # Check maximum age
    assert df['salary'].min() >= 120000  # Check minimum salary
    assert df['salary'].max() <= 1200000  # Check maximum salary

def test_make_synthetic_with_custom_size():
    # Test with custom parameters
    df = make_synthetic(n=5000)
    assert df.shape[0] == 5000  # Check number of rows

def test_attrition_distribution():
    # Test the distribution of attrition
    df = make_synthetic()
    attrition_rate_12m = df['attrition_12m'].mean()
    assert 0 <= attrition_rate_12m <= 1  # Check that attrition rate is between 0 and 1

def test_column_types():
    # Test the data types of generated columns
    df = make_synthetic()
    assert df['employee_id'].dtype == 'int64'
    assert df['age'].dtype == 'int64'
    assert df['salary'].dtype == 'float64'
    assert df['attrition_12m'].dtype == 'int64'  # Assuming attrition is binary (0 or 1)

def test_randomness():
    # Test that multiple calls produce different results
    df1 = make_synthetic(seed=1)
    df2 = make_synthetic(seed=2)
    assert not df1.equals(df2)  # Check that different seeds produce different data