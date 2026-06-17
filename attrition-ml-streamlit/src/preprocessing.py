from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def clean_data(df):
    """Clean the dataset by handling missing values and encoding categorical variables."""
    # Handle missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill for simplicity
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

def apply_column_configuration(df, config):
    """Apply user-defined column configurations to the dataframe."""
    df_clean = df.copy()
    
    # Filter out rejected columns
    accepted_cols = [col for col, settings in config.items() 
                     if settings.get('include', True) and col in df.columns]
    
    df_clean = df_clean[accepted_cols]
    
    # Apply data type conversions
    for col, settings in config.items():
        if col not in df_clean.columns:
            continue
        
        dtype = settings.get('dtype', 'numeric')
        
        try:
            if dtype == 'numeric':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            elif dtype == 'categorical':
                df_clean[col] = df_clean[col].astype(str)
            elif dtype == 'ordinal':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            elif dtype == 'boolean':
                df_clean[col] = df_clean[col].astype(float).astype('Int64')
            elif dtype == 'date':
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        except Exception as e:
            print(f"Could not convert {col} to {dtype}: {str(e)}")
    
    return df_clean

def preprocess_data(file_path, config):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    df = clean_data(df)
    df = apply_column_configuration(df, config)
    return df