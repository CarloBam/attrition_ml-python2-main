from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def create_derived_features(df):
    """Create derived features for the dataset."""
    # Example: Create interaction terms and polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    # Assuming 'age', 'tenure_months', and 'salary' are relevant columns
    if all(col in df.columns for col in ['age', 'tenure_months', 'salary']):
        interaction_terms = poly.fit_transform(df[['age', 'tenure_months', 'salary']])
        interaction_df = pd.DataFrame(interaction_terms, columns=poly.get_feature_names_out(['age', 'tenure_months', 'salary']))
        
        # Concatenate the new features with the original dataframe
        df = pd.concat([df, interaction_df], axis=1)
    
    # Example: Create age bands
    if 'age' in df.columns:
        df['age_band'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
    
    # Example: Create tenure bands
    if 'tenure_months' in df.columns:
        df['tenure_band'] = pd.cut(df['tenure_months'], bins=[0, 12, 36, 60, 999], 
                                   labels=['<1yr', '1-3yrs', '3-5yrs', '5+yrs'])
    
    # Example: Create salary bands
    if 'salary' in df.columns:
        df['salary_band'] = pd.qcut(df['salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    return df

def select_features(df, target_col):
    """Select relevant features for modeling."""
    feature_cols = [col for col in df.columns if col != target_col and col != 'employee_id']
    return feature_cols