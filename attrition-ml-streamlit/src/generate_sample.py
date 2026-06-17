from sklearn.utils import shuffle
import numpy as np
import pandas as pd

def generate_synthetic_data(n=2000, seed=42):
    """Generate synthetic employee attrition data with additional variables."""
    rng = np.random.default_rng(seed)
    departments = ["Sales", "Engineering", "Customer Success", "Operations", "Marketing", "Finance", "HR", "Legal"]
    grades = ["A", "B", "C", "D"]
    locations = ["HQ", "Cape Town", "Johannesburg", "Remote"]
    gender = ["Male", "Female"]

    df = pd.DataFrame({
        "employee_id": np.arange(1, n + 1),
        "age": rng.integers(21, 63, n),
        "tenure_months": rng.integers(3, 180, n),
        "salary": rng.normal(450_000, 120_000, n).clip(120_000, 1_200_000),
        "compa_ratio": rng.uniform(0.7, 1.3, n),
        "overtime_hours_month": rng.integers(0, 45, n),
        "performance_rating": rng.choice([2, 3, 4, 5], n, p=[0.1, 0.35, 0.4, 0.15]),
        "engagement_score": rng.normal(65, 15, n).clip(10, 100),
        "manager_tenure_months": rng.integers(1, 180, n),
        "commute_km": rng.integers(0, 60, n),
        "leave_balance_days": rng.integers(0, 35, n),
        "training_hours_last_yr": rng.integers(0, 80, n),
        "num_promotions": rng.integers(0, 5, n),
        "pay_change_12m_pct": rng.normal(3, 6, n).clip(-15, 25),
        "market_delta_pct": rng.normal(0, 8, n).clip(-20, 30),
        "remote_ratio": rng.integers(0, 100, n),
        "absences_12m": rng.integers(0, 20, n),
        "department": rng.choice(departments, n),
        "job_role": rng.choice(["Analyst", "Manager", "Specialist", "Lead"], n),
        "grade": rng.choice(grades, n, p=[0.35, 0.4, 0.2, 0.05]),
        "location": rng.choice(locations, n, p=[0.3, 0.35, 0.2, 0.15]),
        "gender": rng.choice(gender, n),
        "work_from_home_days": rng.integers(0, 5, n),  # New variable
        "last_promotion_months": rng.integers(0, 24, n)  # New variable
    })

    # Add snapshot date
    start = pd.Timestamp("2024-01-01")
    df["snapshot_date"] = [start + pd.offsets.MonthBegin(int(i % 18)) for i in range(n)]

    # Generate attrition based on risk factors
    z = (
        0.7 * (df["overtime_hours_month"] / 40) +
        0.5 * ((80 - df["engagement_score"]) / 80) +
        0.4 * (df["commute_km"] / 60) +
        0.3 * ((10 - df["pay_change_12m_pct"]) / 20) +
        0.2 * (df["compa_ratio"] < 0.9).astype(int) +
        0.2 * (df["manager_tenure_months"] < 12).astype(int) +
        0.15 * (df["remote_ratio"] < 20).astype(int) +
        0.1 * (df["absences_12m"] > 8).astype(int)
    )
    p12 = 1 / (1 + np.exp(-(z - 1.1)))
    p3 = 1 / (1 + np.exp(-(z - 1.6)))

    df["attrition_12m"] = (rng.random(n) < p12).astype(int)
    df["attrition_3m"] = (rng.random(n) < p3).astype(int)

    return shuffle(df)  # Shuffle the dataset for randomness

def save_sample_data(filepath='data/sample_attrition_data.csv', n=2000):
    """Generate and save synthetic data to a CSV file."""
    df = generate_synthetic_data(n)
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")