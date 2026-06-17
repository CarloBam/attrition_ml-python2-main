import pandas as pd


def validate_survival_inputs(df: pd.DataFrame, duration_col: str, event_col: str, id_col: str = 'employee_id') -> None:
    """Simple validations for survival inputs. Raises ValueError if checks fail."""
    if duration_col not in df.columns:
        raise ValueError(f"Duration column '{duration_col}' not found in dataframe.")
    if event_col not in df.columns:
        raise ValueError(f"Event column '{event_col}' not found in dataframe.")
    if id_col not in df.columns:
        # it's acceptable if id_col missing, but warn via exception if needed
        pass


def summarize_person_period(pp_df: pd.DataFrame, id_col: str = 'employee_id') -> pd.DataFrame:
    """Return summary of person-period table: counts and events per id."""
    grp = pp_df.groupby(id_col, observed=False).agg({'period': 'max', 'event': 'sum'}).reset_index()
    grp.columns = [id_col, 'max_period', 'events']
    return grp
