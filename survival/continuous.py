import pandas as pd

try:
    from lifelines import CoxPHFitter
except Exception:
    CoxPHFitter = None


def prepare_survival_dataframe(df: pd.DataFrame, duration_col: str, event_col: str, id_col: str = 'employee_id') -> pd.DataFrame:
    """Prepare a dataframe for survival modeling.

    - Ensures `duration_col` and `event_col` exist and are numeric.
    - If `duration_col` not found, attempts to compute from dates.
    Returns cleaned dataframe with `id_col`, `duration_col`, `event_col` and covariates.
    """
    df = df.copy()

    if duration_col not in df.columns:
        # try compute from tenure_months or from dates
        if 'tenure_months' in df.columns:
            df[duration_col] = pd.to_numeric(df['tenure_months'], errors='coerce')
        elif 'hire_date' in df.columns and 'termination_date' in df.columns:
            df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
            df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
            snapshot = pd.to_datetime(df.get('snapshot_date', pd.Timestamp.now()))
            df[duration_col] = ((df['termination_date'].fillna(snapshot) - df['hire_date']).dt.days / 30.0).round(1)
        else:
            df[duration_col] = 0
    else:
        df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')

    if event_col not in df.columns:
        if 'termination_date' in df.columns:
            df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
            snapshot = pd.to_datetime(df.get('snapshot_date', pd.Timestamp.now()))
            df[event_col] = (~df['termination_date'].isna()) & (df['termination_date'] <= snapshot)
            df[event_col] = df[event_col].astype(int)
        else:
            df[event_col] = 0
    else:
        df[event_col] = df[event_col].astype(int)

    return df


def fit_coxph(df: pd.DataFrame, duration_col: str, event_col: str, feature_cols: list):
    """Fit a Cox Proportional Hazards model using lifelines.

    Returns fitted CoxPHFitter.
    """
    if CoxPHFitter is None:
        raise ImportError("lifelines is not installed. Install lifelines to use CoxPH models.")

    cph = CoxPHFitter()
    cols = [duration_col, event_col] + feature_cols
    # ensure columns exist
    used = [c for c in cols if c in df.columns]
    if duration_col not in used or event_col not in used:
        raise ValueError("Duration or event columns missing from dataframe for CoxPH fitting.")

    data = df[used].dropna()
    cph.fit(data, duration_col=duration_col, event_col=event_col)
    return cph


def predict_survival_function(model, X: pd.DataFrame, times) -> pd.DataFrame:
    """Given a fitted lifelines model and covariates X, return survival probabilities at specified times.

    Returns a DataFrame where rows correspond to X and columns to times.
    """
    if CoxPHFitter is None:
        raise ImportError("lifelines is not installed. Install lifelines to use CoxPH predictions.")

    # lifelines' predict_survival_function returns survival curves for each row
    sf = model.predict_survival_function(X, times=times)
    # Transpose to have rows=observations, cols=times
    return sf.T
