import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def compute_tenure_and_event(df: pd.DataFrame, start_col: str = None, end_col: str = None, event_indicator_col: str = None, tenure_col: str = "tenure_months", event_col: str = "event") -> pd.DataFrame:
    """Compute tenure and event flag for survival modeling.

    Behavior:
    - If `tenure_col` exists in `df`, uses that as duration (expected in months).
    - If `event_indicator_col` exists, uses that as event flag (1=event,0=censor).
    - If `start_col` and `end_col` are provided and present in `df`, computes duration in months between them and sets event flag accordingly.

    Returns a copy of `df` with numeric `tenure_col` and binary `event_col`.
    """
    df = df.copy()

    # Determine tenure
    if tenure_col in df.columns:
        df[tenure_col] = pd.to_numeric(df[tenure_col], errors='coerce')
    else:
        # Try to compute from dates
        if start_col in df.columns and end_col in df.columns:
            df[start_col] = pd.to_datetime(df[start_col], errors='coerce')
            df[end_col] = pd.to_datetime(df[end_col], errors='coerce')
            # If end is NaT, use snapshot_date or today
            snapshot = pd.to_datetime(df.get('snapshot_date', pd.Timestamp.now()))
            df[tenure_col] = ((df[end_col].fillna(snapshot) - df[start_col]).dt.days / 30.0).round(1)
        else:
            # fallback: set tenure to zero to avoid crashes
            df[tenure_col] = 0

    # Determine event
    if event_indicator_col and event_indicator_col in df.columns:
        df[event_col] = df[event_indicator_col].astype(int)
    else:
        # Try to infer from termination date
        if 'termination_date' in df.columns:
            df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
            snapshot = pd.to_datetime(df.get('snapshot_date', pd.Timestamp.now()))
            df[event_col] = (~df['termination_date'].isna()) & (df['termination_date'] <= snapshot)
            df[event_col] = df[event_col].astype(int)
        else:
            # Default: treat everyone as censored
            df[event_col] = 0

    return df


def expand_person_period(df: pd.DataFrame, duration_col: str = 'tenure_months', event_col: str = 'event', interval: int = 1, id_col: str = 'employee_id', max_periods: int = 60) -> pd.DataFrame:
    """Expand each subject into person-period rows.

    - `interval` is the step in months (default 1 month).
    - `max_periods` limits horizon (default 60 months).

    Returns a DataFrame with columns: id_col, period (1..), event (0/1), and duplicated covariates.
    """
    rows = []
    feature_cols = [c for c in df.columns if c not in [id_col, duration_col, event_col]]

    for _, row in df.iterrows():
        eid = row.get(id_col, None)
        duration = int(np.nan_to_num(row.get(duration_col, 0)))
        duration = max(0, duration)
        duration = min(duration, max_periods)
        event_flag = int(row.get(event_col, 0))

        # Determine number of periods to create
        n_periods = duration if duration > 0 else (1 if event_flag == 0 else 1)
        n_periods = max(1, n_periods)

        for p in range(1, n_periods + 1):
            # event occurs only on the period equal to duration when event_flag==1
            ev = 1 if (event_flag == 1 and p == duration) else 0
            entry = {
                id_col: eid,
                'period': p,
                'event': ev
            }
            # copy static features
            for f in feature_cols:
                entry[f] = row.get(f)
            rows.append(entry)

    pp = pd.DataFrame(rows)
    return pp


def fit_discrete_hazard_model(pp_df: pd.DataFrame, feature_cols: list, period_col: str = 'period', event_col: str = 'event', model_type: str = 'logistic'):
    """Fit a discrete-time hazard model on person-period data.

    Returns a dict with 'model' and 'feature_names'.
    """
    # If no features provided, use period as the only feature
    if not feature_cols or len(feature_cols) == 0:
        if 'period' in pp_df.columns:
            feature_cols = ['period']
        else:
            # Fallback: use all numeric columns except event
            feature_cols = [c for c in pp_df.select_dtypes(include=[np.number]).columns if c != event_col]
    
    # Ensure feature_cols are in dataframe
    feature_cols = [c for c in feature_cols if c in pp_df.columns]
    
    if len(feature_cols) == 0:
        raise ValueError("No valid features found for hazard model training.")
    
    X = pp_df[feature_cols].copy()
    y = pp_df[event_col].copy()

    # Simple categorical encoding
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype(str).fillna('NA')
        X[col] = X[col].factorize()[0]

    if model_type == 'logistic':
        clf = LogisticRegression(max_iter=2000)
    else:
        clf = GradientBoostingClassifier(n_estimators=100)

    clf.fit(X, y)

    return {'model': clf, 'feature_names': feature_cols}


def predict_hazard(model_dict, pp_df: pd.DataFrame, times=None):
    """Predict hazards for person-period rows (returns DataFrame with predicted hazard per row).

    If `times` provided, it is ignored for a row-wise prediction; aggregation by id/time is up to caller.
    """
    model = model_dict['model'] if isinstance(model_dict, dict) and 'model' in model_dict else model_dict
    feature_cols = model_dict.get('feature_names', pp_df.columns.tolist()) if isinstance(model_dict, dict) else pp_df.columns.tolist()

    X = pp_df[feature_cols].copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype(str).fillna('NA')
        X[col] = X[col].factorize()[0]

    probs = model.predict_proba(X)[:, 1]
    out = pp_df.copy()
    out['hazard'] = probs
    return out


def hazard_to_survival(hazards: np.ndarray) -> np.ndarray:
    """Convert a sequence of discrete hazards to survival probabilities.

    hazards: array-like of hazards per period (values in [0,1]).
    returns survival array S(t) where S(0)=1 and S(t)=prod_{i=1..t} (1-h_i)
    """
    hazards = np.asarray(hazards)
    survival = np.cumprod(1 - hazards)
    return survival
