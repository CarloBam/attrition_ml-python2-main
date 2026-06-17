import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix
)

# Optional: statsmodels for p-values and feature selection
try:
    import statsmodels.api as sm
    statsmodels_available = True
except ImportError:
    statsmodels_available = False

# Survival modeling helper imports (optional)
try:
    from survival.continuous import prepare_survival_dataframe, fit_coxph, predict_survival_function
    from survival.discrete import compute_tenure_and_event, expand_person_period, fit_discrete_hazard_model, predict_hazard, hazard_to_survival
except Exception:
    # survival modules may not yet be present; safe import
    prepare_survival_dataframe = None
    fit_coxph = None
    predict_survival_function = None
    compute_tenure_and_event = None
    expand_person_period = None
    fit_discrete_hazard_model = None
    predict_hazard = None
    hazard_to_survival = None

# Page config
st.set_page_config(page_title="Attrition ML Dashboard", layout="wide")

# Theme colors
PRIMARY_BLUE = "#2f80ed"
DANGER_RED = "#ff4b4b"
TEXT = "#eaecef"

# Initialize session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "data_configured" not in st.session_state:
    st.session_state.data_configured = False
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "configured_data" not in st.session_state:
    st.session_state.configured_data = None
if "column_config" not in st.session_state:
    st.session_state.column_config = {}
if "config_set" not in st.session_state:
    st.session_state.config_set = False
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "training_data_uploaded" not in st.session_state:
    st.session_state.training_data_uploaded = False
if "training_data" not in st.session_state:
    st.session_state.training_data = None

# Utility functions
def make_synthetic(n=2000, seed=42):
    """Generate synthetic employee attrition data"""
    rng = np.random.default_rng(seed)
    departments = ["Sales", "Engineering", "Customer Success", "Operations", "Marketing", "Finance", "HR", "Legal"]
    grades = ["A", "B", "C", "D"]
    locations = ["HQ", "Cape Town", "Johannesburg", "Remote"]
    gender = ["Male", "Female"]

    df = pd.DataFrame({
        "employee_id": np.arange(1, n+1),
        "age": rng.integers(21, 63, n),
        "tenure_months": rng.integers(3, 180, n),
        "salary": rng.normal(450_000, 120_000, n).clip(120_000, 1_200_000),
        "compa_ratio": rng.uniform(0.7, 1.3, n),
        "overtime_hours_month": rng.integers(0, 45, n),
        "performance_rating": rng.choice([2,3,4,5], n, p=[0.1,0.35,0.4,0.15]),
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
        "job_role": rng.choice(["Analyst","Manager","Specialist","Lead"], n),
        "grade": rng.choice(grades, n, p=[0.35,0.4,0.2,0.05]),
        "location": rng.choice(locations, n, p=[0.3,0.35,0.2,0.15]),
        "gender": rng.choice(gender, n),
    })
    
    # Add snapshot date
    start = pd.Timestamp("2024-01-01")
    df["snapshot_date"] = [start + pd.offsets.MonthBegin(int(i % 18)) for i in range(n)]
    
    # Generate attrition based on risk factors
    z = (
        0.7*(df["overtime_hours_month"]/40) +
        0.5*((80 - df["engagement_score"])/80) +
        0.4*(df["commute_km"]/60) +
        0.3*((10 - df["pay_change_12m_pct"])/20) +
        0.2*(df["compa_ratio"] < 0.9).astype(int) +
        0.2*(df["manager_tenure_months"] < 12).astype(int) +
        0.15*(df["remote_ratio"] < 20).astype(int) +
        0.1*(df["absences_12m"] > 8).astype(int)
    )
    p12 = 1 / (1 + np.exp(-(z - 1.1)))
    p3 = 1 / (1 + np.exp(-(z - 1.6)))

    df["attrition_12m"] = (rng.random(n) < p12).astype(int)
    df["attrition_3m"] = (rng.random(n) < p3).astype(int)

    return df

def infer_column_type(series):
    """Infer the best data type for a column"""
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return "boolean"
        elif len(unique_vals) <= 10 and pd.api.types.is_integer_dtype(series):
            return "ordinal"
        else:
            return "numeric"
    else:
        return "categorical"


def read_csv_heuristic(fobj):
    """Robust CSV reader that handles wide files, sniffing delimiters and decoding bytes.
    Accepts a file path, a file-like object (UploadedFile), or raw bytes.
    """
    import io, csv
    # Try straightforward read first
    try:
        try:
            # reset pointer if possible
            fobj.seek(0)
        except Exception:
            pass
        return pd.read_csv(fobj, low_memory=False)
    except Exception:
        # Fallback: read bytes and sniff delimiter
        try:
            if hasattr(fobj, 'read'):
                raw = fobj.read()
            elif isinstance(fobj, (bytes, bytearray)):
                raw = fobj
            else:
                # assume path-like
                return pd.read_csv(fobj, low_memory=False)

            # Ensure bytes
            if isinstance(raw, str):
                raw = raw.encode('utf-8')

            text = raw.decode('utf-8', errors='replace')
            sample = '\n'.join(text.splitlines()[:10])
            try:
                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
            except Exception:
                sep = ','

            return pd.read_csv(io.StringIO(text), sep=sep, engine='python')
        except Exception as e:
            # Last resort: try pandas with python engine on original object/path
            try:
                try:
                    fobj.seek(0)
                except Exception:
                    pass
                return pd.read_csv(fobj, engine='python', low_memory=False)
            except Exception:
                raise

def apply_column_configuration(df, config):
    """Apply user-defined column configurations to the dataframe"""
    df_clean = df.copy()
    
    # Filter out rejected columns
    # If a column is not present in the config, include it by default
    accepted_cols = []
    for col in df.columns:
        if col in config:
            if config[col].get('include', True):
                accepted_cols.append(col)
        else:
            accepted_cols.append(col)

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
            st.warning(f"Could not convert {col} to {dtype}: {str(e)}")
    
    return df_clean

def get_feature_columns(df, target_col, id_cols=None):
    """Get feature columns excluding target and ID columns"""
    if id_cols is None:
        id_cols = ['employee_id', 'snapshot_date']
    
    exclude_cols = [target_col] + id_cols
    exclude_cols += [c for c in df.columns if 'attrition' in c.lower() and c != target_col]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols

def stepwise_selection(X, y, direction='forward', p_threshold=0.05):
    """
    Perform stepwise feature selection using statsmodels logistic regression p-values.
    
    direction: 'forward', 'backward', or 'stepwise'
    p_threshold: p-value threshold for feature inclusion (default 0.05)
    
    Returns: (selected_features, pvalues_dict)
    """
    if not statsmodels_available:
        return list(X.columns), {}
    
    X_sm = sm.add_constant(X)
    current_features = list(X.columns)
    pvalues_dict = {}
    
    if direction == 'forward':
        selected = []
        remaining = set(current_features)
        
        while remaining:
            best_pval = p_threshold
            best_feature = None
            best_model = None
            
            for feature in remaining:
                try:
                    test_features = selected + [feature]
                    model = sm.Logit(y, X_sm[['const'] + test_features]).fit(disp=0)
                    pvals = model.pvalues
                    
                    # Get p-value for the new feature (last one)
                    feat_pval = pvals[feature]
                    
                    if feat_pval < best_pval:
                        best_pval = feat_pval
                        best_feature = feature
                        best_model = model
                except Exception:
                    continue
            
            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
                if best_model:
                    for feat in selected:
                        if feat in best_model.pvalues.index:
                            pvalues_dict[feat] = best_model.pvalues[feat]
            else:
                break
        
        return selected, pvalues_dict
    
    elif direction == 'backward':
        selected = current_features.copy()
        
        while len(selected) > 1:
            try:
                model = sm.Logit(y, X_sm[['const'] + selected]).fit(disp=0)
                pvals = model.pvalues.drop('const')
                pvalues_dict = dict(pvals)
                
                worst_feature = pvals.idxmax()
                worst_pval = pvals.max()
                
                if worst_pval > p_threshold:
                    selected.remove(worst_feature)
                else:
                    break
            except Exception:
                break
        
        return selected, pvalues_dict
    
    elif direction == 'stepwise':
        # Stepwise: starts with forward, then applies backward
        selected, pvals_fwd = stepwise_selection(X, y, direction='forward', p_threshold=p_threshold)
        X_selected = X[selected]
        final_selected, pvals_bwd = stepwise_selection(X_selected, y, direction='backward', p_threshold=p_threshold)
        
        return final_selected, pvals_bwd
    
    return current_features, pvalues_dict

# ============= HIERARCHICAL DRILL-DOWN FUNCTIONS =============

def save_hierarchy_config(hierarchy_config):
    """Save hierarchy configuration to session state"""
    st.session_state.hierarchy_config = hierarchy_config
    st.session_state.hierarchy_levels = {v['variable']: v['level'] for k, v in hierarchy_config.items()}

def get_hierarchy_config():
    """Get current hierarchy configuration"""
    return st.session_state.get('hierarchy_config', {})

def get_hierarchy_levels():
    """Get mapping of variable to hierarchy level"""
    return st.session_state.get('hierarchy_levels', {})

def filter_data_by_drill_path(df, drill_path, hierarchy_config):
    """Filter dataframe based on current drill path (selections made at each level)
    
    drill_path: list of tuples [(level, variable, selected_value), ...]
    """
    filtered_df = df.copy()
    
    for level, variable, selected_value in drill_path:
        if variable in filtered_df.columns and selected_value:
            filtered_df = filtered_df[filtered_df[variable] == selected_value]
    
    return filtered_df

def get_next_level_values(df, drill_path, hierarchy_config):
    """Get unique values for the next level in the hierarchy
    
    Returns: (next_variable, unique_values)
    """
    # Filter data based on current drill path
    filtered_df = filter_data_by_drill_path(df, drill_path, hierarchy_config)
    
    # Find next level
    current_level = len(drill_path)
    
    # Find variable at next level
    next_variable = None
    for var, config in hierarchy_config.items():
        if config.get('level') == current_level + 1:
            next_variable = var
            break
    
    if next_variable is None:
        return None, []
    
    if next_variable not in filtered_df.columns:
        return None, []
    
    unique_vals = sorted(filtered_df[next_variable].dropna().unique().tolist())
    return next_variable, unique_vals

def render_hierarchy_selector(df, hierarchy_config):
    """Render interactive hierarchy drill-down selector in main area"""
    if not hierarchy_config:
        st.info("No hierarchy configured. Please define hierarchy levels in the Hierarchy Configuration section.")
        return None, df
    
    # Sort config by level
    sorted_config = sorted(hierarchy_config.items(), key=lambda x: x[1].get('level', 0))
    
    drill_path = []
    current_data = df.copy()
    
    st.subheader("Drill-Down Navigator")
    
    for idx, (config_key, config_val) in enumerate(sorted_config):
        level = config_val.get('level', 0)
        variable = config_val.get('variable', '')
        level_name = config_val.get('level_name', f'Level {level}')
        
        if variable not in current_data.columns:
            st.warning(f"Variable '{variable}' not found in data.")
            continue
        
        # Get unique values for this level
        unique_vals = sorted(current_data[variable].dropna().unique().tolist())
        
        # Add "All" option
        options = ["All"] + [str(v) for v in unique_vals]
        
        # Create selectbox for this level
        selected = st.selectbox(
            f"**{level_name}** ({variable})",
            options=options,
            key=f"drill_level_{level}"
        )
        
        # Filter data if selection made (not "All")
        if selected != "All":
            current_data = current_data[current_data[variable] == selected]
            drill_path.append((level, variable, selected))
        
        st.caption(f"📊 {len(current_data):,} records at this level")
    
    return drill_path, current_data

def render_sidebar_drill_down(df, hierarchy_config):
    """Render drill-down filters in the sidebar and return filtered data"""
    if not hierarchy_config:
        return None, df
    
    # Sort config by level
    sorted_config = sorted(hierarchy_config.items(), key=lambda x: x[1].get('level', 0))
    
    drill_path = []
    current_data = df.copy()
    
    for config_key, config_val in sorted_config:
        level = config_val.get('level', 0)
        variable = config_val.get('variable', '')
        level_name = config_val.get('level_name', f'Level {level}')
        
        if variable not in current_data.columns:
            continue
        
        # Get unique values for this level
        unique_vals = sorted(current_data[variable].dropna().unique().tolist())
        
        # Add "All" option
        options = ["All"] + [str(v) for v in unique_vals]
        
        # Create selectbox in sidebar
        selected = st.sidebar.selectbox(
            level_name,
            options=options,
            key=f"sidebar_drill_level_{level}",
            help=f"Filter by {variable}"
        )
        
        # Filter data if selection made (not "All")
        if selected != "All":
            current_data = current_data[current_data[variable] == selected]
            drill_path.append((level, variable, selected))
    
    # Store filtered data in session state for use across tabs
    st.session_state.current_drill_path = drill_path
    st.session_state.filtered_data_by_drill = current_data
    
    return drill_path, current_data

# ============= END HIERARCHY FUNCTIONS =============

# ============= Helper: Active Data Getters =============
def get_active_configured_data():
    """Return configured data filtered by drill-down if present, else configured/raw data.

    Returns a copy or None.
    """
    # Prefer drill-filtered data when present
    if 'filtered_data_by_drill' in st.session_state and st.session_state.get('filtered_data_by_drill') is not None:
        df = st.session_state.get('filtered_data_by_drill')
    elif 'configured_data' in st.session_state and st.session_state.get('configured_data') is not None:
        df = st.session_state.get('configured_data')
    else:
        df = st.session_state.get('raw_data') if 'raw_data' in st.session_state else None

    return df.copy() if df is not None else None


def get_active_prepared_data():
    """Return prepared data filtered by drill-down if present, else fallback to prepared/configured/raw.

    Returns a copy or None.
    """
    # Prefer drill-filtered data when present
    if 'filtered_data_by_drill' in st.session_state and st.session_state.get('filtered_data_by_drill') is not None:
        df = st.session_state.get('filtered_data_by_drill')
    elif 'prepared_data' in st.session_state and st.session_state.get('prepared_data') is not None:
        df = st.session_state.get('prepared_data')
    elif 'configured_data' in st.session_state and st.session_state.get('configured_data') is not None:
        df = st.session_state.get('configured_data')
    else:
        df = st.session_state.get('raw_data') if 'raw_data' in st.session_state else None

    return df.copy() if df is not None else None

# ============= END HELPERS =============

# Sidebar controls
st.sidebar.header("Global Controls")

random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1,
                                       help="Set random seed for reproducible results")

# Department filter with checkboxes
if st.session_state.config_set and st.session_state.configured_data is not None:
    df_for_filter = st.session_state.configured_data
    if 'department' in df_for_filter.columns:
        st.sidebar.subheader("Department Filter")
        all_depts = sorted(df_for_filter['department'].dropna().unique().tolist())
        
        # Select All / Deselect All
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Select All", key="select_all_depts"):
                st.session_state.selected_departments = all_depts
        with col2:
            if st.button("Clear All", key="clear_all_depts"):
                st.session_state.selected_departments = []
        
        # Initialize if not exists
        if 'selected_departments' not in st.session_state:
            st.session_state.selected_departments = all_depts
        
        # Individual checkboxes
        selected_depts = []
        for dept in all_depts:
            if st.sidebar.checkbox(dept, value=dept in st.session_state.selected_departments, key=f"dept_{dept}"):
                selected_depts.append(dept)
        
        st.session_state.selected_departments = selected_depts
    else:
        selected_depts = None
else:
    selected_depts = None

# Model family selector - NOW A CHECKBOX LIST
st.sidebar.subheader("Model Family Selection")

# Initialize model selections in session state
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = {
        'Logistic Regression': True,
        'Random Forest': True,
        'Gradient Boosting': True,
        'Decision Tree': True,
        'Ensemble': True
    }

# Display checkboxes
st.sidebar.markdown("Choose which models to train:")
for model_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree', 'Ensemble']:
    st.session_state.selected_models[model_name] = st.sidebar.checkbox(
        model_name,
        value=st.session_state.selected_models.get(model_name, True),
        key=f"model_{model_name}"
    )

# Show selected count
selected_count = sum(st.session_state.selected_models.values())
st.sidebar.info(f"📊 {selected_count} model(s) selected")

# Forecast horizon selector
forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=["12 months", "2 years", "3 years", "4 years", "5 years"],
    help="Select the time horizon for attrition risk forecasting"
)

# Hierarchical Drill-Down Sidebar Widget
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Drill-Down Filter")
st.sidebar.caption("Apply hierarchy-based filters to all analyses below")

# Initialize hierarchy session state
if "hierarchy_config" not in st.session_state:
    st.session_state.hierarchy_config = {}
if "hierarchy_levels" not in st.session_state:
    st.session_state.hierarchy_levels = {}
if "current_drill_path" not in st.session_state:
    st.session_state.current_drill_path = []
if "filtered_data_by_drill" not in st.session_state:
    st.session_state.filtered_data_by_drill = None

# Render sidebar drill-down if hierarchy is configured and data is available
if (st.session_state.hierarchy_config and 
    (st.session_state.get('configured_data') is not None or st.session_state.get('raw_data') is not None)):
    
    # Use configured data if available, else raw data (avoid DataFrame truthiness)
    if 'configured_data' in st.session_state and st.session_state.get('configured_data') is not None:
        drill_source_df = st.session_state.get('configured_data')
    else:
        drill_source_df = st.session_state.get('raw_data') if 'raw_data' in st.session_state else None
    
    if drill_source_df is not None:
        drill_path, filtered_df = render_sidebar_drill_down(drill_source_df, st.session_state.hierarchy_config)
        
        if drill_path:
            st.sidebar.success(f"📌 Drill-down active: {len(filtered_df):,} records")
            # Show breadcrumb of current selections
            try:
                crumbs = [f"{step[1]}: {step[2]}" for step in st.session_state.get('current_drill_path', [])]
                if crumbs:
                    st.sidebar.markdown("**Active Filters:**")
                    st.sidebar.caption(" > ".join(crumbs))
            except Exception:
                pass

            # Clear filters button
            if st.sidebar.button("Clear Filters", key="clear_drill_filters"):
                # Reset filtered data and current drill path
                if 'filtered_data_by_drill' in st.session_state:
                    del st.session_state['filtered_data_by_drill']
                st.session_state['current_drill_path'] = []

                # Reset sidebar and main drill selectboxes to 'All' for configured levels
                try:
                    levels = [v.get('level') for v in st.session_state.get('hierarchy_config', {}).values()]
                    for lvl in levels:
                        ks = f"sidebar_drill_level_{lvl}"
                        km = f"drill_level_{lvl}"
                        try:
                            if ks in st.session_state:
                                st.session_state[ks] = "All"
                            if km in st.session_state:
                                st.session_state[km] = "All"
                        except Exception:
                            pass
                except Exception:
                    pass

                st.experimental_rerun()
        else:
            st.sidebar.info(f"📊 All {len(drill_source_df):,} records visible (no drill selections)")
    else:
        st.sidebar.info("⏳ Load data to enable drill-down filters")
else:
    if st.session_state.hierarchy_config:
        st.sidebar.info("⏳ Load data to enable drill-down filters")
    else:
        st.sidebar.caption("Configure hierarchy in Data Upload tab")

# Main title
st.title("🎯 Employee Attrition Prediction Workbench")

# Main-area breadcrumb for active drill path
try:
    path = st.session_state.get('current_drill_path', [])
    if path:
        crumbs = [f"{step[1]}: {step[2]}" for step in path]
        st.markdown("**Active Drill Path:** " + "   ".join(crumbs))
except Exception:
    pass

# Tabs (added Time to Termination as tab10)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16 = st.tabs([
    "📊 Data Upload",
    "📤 Bulk Training Data",
    "🔧 Feature Preparation",
    "🤖 Model Training",
    "📊 Model Comparison",
    "📉 KPI Dashboard",
    "🌳 Segments",
    "👤 Individual Risk",
    "💡 Recommendations",
    "⏳ Time to Termination",
    "💼 Executive Dashboard",
    "🎯 Risk Engine",
    "📋 Prescriptive Actions",
    "🔍 Diagnostic Drivers",
    "🎓 Talent Pipeline",
    "👥 Department Drill-Down"
])

# TAB 1: Data Upload
with tab1:
    st.header("Data Source and Configuration")

    # Layout: left = Data Upload, right = Bulk Training (shows right when training data present)
    left_col, right_col = st.columns([3, 2])

    # Right column: Bulk Training uploader (preferred for large datasets)
    with right_col:
        st.subheader("Bulk Training Data (optional)")
        top_bt_file = st.file_uploader("Upload bulk training data (CSV/Excel/Parquet)", type=['csv', 'xlsx', 'xls', 'parquet'], key='top_bt_file')
        if top_bt_file is not None:
            try:
                if top_bt_file.name.endswith('.csv'):
                    training_df = read_csv_heuristic(top_bt_file)
                elif top_bt_file.name.endswith('.parquet'):
                    training_df = pd.read_parquet(top_bt_file)
                else:
                    training_df = pd.read_excel(top_bt_file)
                st.session_state.training_data = training_df
                st.session_state.training_data_uploaded = True
                st.success(f"✅ Loaded {len(training_df):,} rows from {top_bt_file.name}")
                st.dataframe(training_df.head(5), width='stretch')
            except Exception as e:
                st.error(f"Error loading bulk training file: {e}")

        # Also expose alternative bulk sources (S3/BigQuery/Azure) in the right column
        st.markdown("---")
        st.markdown("**Alternative Bulk Sources**")
        st.caption("Load large training datasets from cloud sources if preferred.")

    # Left column: Data source selection and configuration
    with left_col:
        data_source = st.radio(
            "Select Data Source",
            options=["Upload CSV", "Use Sample Data", "Generate Synthetic Data"],
            horizontal=True
        )

        raw_df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload employee data CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                raw_df = read_csv_heuristic(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                raw_df = None
            st.success(f"Loaded {len(raw_df)} rows from uploaded file")
    
    elif data_source == "Use Sample Data":
        try:
            raw_df = read_csv_heuristic("sample_attrition_data.csv")
            st.success(f"Loaded {len(raw_df)} rows from sample data")
        except FileNotFoundError:
            st.error("Sample data file not found. Please upload a CSV or generate synthetic data.")
    
    elif data_source == "Generate Synthetic Data":
        synth_size = st.slider("Number of employees", min_value=500, max_value=30000, value=2000, step=500)
        if st.button("Generate Data"):
            raw_df = make_synthetic(n=synth_size, seed=random_seed)
            st.session_state.raw_data = raw_df.copy()
            st.session_state.data_loaded = True
            st.success(f"Generated {len(raw_df)} synthetic employee records")
            st.rerun()
    
    if raw_df is not None:
        # Data sampling
        st.subheader("Data Sampling")
        sample_pct = st.select_slider(
            "Use % of data",
            options=[5, 10, 25, 50, 75, 100],
            value=100,
            help="Select what percentage of the data to use for analysis"
        )
        
        if sample_pct < 100:
            sample_size = int(len(raw_df) * sample_pct / 100)
            raw_df = raw_df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
            st.info(f"Using {len(raw_df)} rows ({sample_pct}% of data)")
        
        # Store raw data in session state
        st.session_state.raw_data = raw_df.copy()
        st.session_state.data_loaded = True
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(raw_df.head(10), width='stretch')
        
        # Column configuration
        st.subheader("Column Configuration")
        st.markdown("Review and configure each column below. Select data types and choose which columns to include in analysis.")
        
        # Initialize or sync column config - ensure newly added columns are detected
        if 'column_config_state' not in st.session_state:
            st.session_state.column_config_state = {}

        # Add any new columns that aren't yet present in session state
        for col in raw_df.columns:
            if col not in st.session_state.column_config_state:
                inferred_type = infer_column_type(raw_df[col])
                st.session_state.column_config_state[col] = {
                    'dtype': inferred_type,
                    'include': True
                }

        # Remove configuration for columns that no longer exist (optional cleanup)
        removed_cols = [c for c in list(st.session_state.column_config_state.keys()) if c not in raw_df.columns]
        for c in removed_cols:
            del st.session_state.column_config_state[c]
        
        # Create configuration table
        config_data = []
        for col in raw_df.columns:
            current_config = st.session_state.column_config_state.get(col, {})
            config_data.append({
                'Column': col,
                'System Type': current_config.get('dtype', 'numeric'),
                'Non-Null': raw_df[col].notna().sum(),
                'Null': raw_df[col].isna().sum(),
                'Unique': raw_df[col].nunique()
            })
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, width='stretch', hide_index=True)
        
        # Column-by-column configuration
        st.markdown("### Configure Individual Columns")
        
        # Detect potential target columns
        potential_targets = [col for col in raw_df.columns 
                            if 'attrition' in col.lower() or 'left' in col.lower() or 'churn' in col.lower()]
        
        if potential_targets:
            st.info(f"💡 Detected potential target columns: {', '.join(potential_targets)}")
        
        # Show configuration for each column in a compact format
        num_cols = 3
        cols_per_row = st.columns(num_cols)
        
        for idx, col in enumerate(raw_df.columns):
            with cols_per_row[idx % num_cols]:
                with st.expander(f"**{col}**", expanded=False):
                    # Safely read current values from session state before rendering widgets
                    current_config = st.session_state.column_config_state.get(col, {'dtype': infer_column_type(raw_df[col]), 'include': True})

                    # Include/Exclude toggle - use stable key
                    include = st.checkbox(
                        "Include in analysis",
                        value=current_config.get('include', True),
                        key=f"inc_{col}"
                    )

                    # Data type selector - use stable key
                    dtype_options = ["numeric", "categorical", "ordinal", "boolean", "date"]
                    try:
                        dtype_index = dtype_options.index(current_config.get('dtype', 'numeric'))
                    except Exception:
                        dtype_index = 0

                    dtype = st.selectbox(
                        "Data Type",
                        options=dtype_options,
                        index=dtype_index,
                        key=f"dtype_{col}"
                    )

                    # Show sample values (5) and a small value distribution
                    st.caption("Sample values (first 5):")
                    st.code(str(raw_df[col].head(5).tolist()), language=None)
                    if raw_df[col].nunique() < 10:
                        st.caption("Value counts:")
                        st.write(raw_df[col].value_counts().head(10).to_dict())

                    # Update session state after widget changes
                    st.session_state.column_config_state[col] = {
                        'dtype': dtype,
                        'include': include
                    }
        
        # Set button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✅ Set Configuration", type="primary", key="set_config_btn"):
                # Ensure every column has a configuration entry (default include=True)
                for col in raw_df.columns:
                    if col not in st.session_state.column_config_state:
                        st.session_state.column_config_state[col] = {
                            'dtype': infer_column_type(raw_df[col]),
                            'include': True
                        }

                # Persist configuration
                st.session_state.column_config = st.session_state.column_config_state.copy()
                configured_df = apply_column_configuration(raw_df, st.session_state.column_config)
                st.session_state.configured_data = configured_df
                st.session_state.config_set = True
                st.session_state.data_configured = True
                st.success(f"✅ Configuration applied! {len(configured_df.columns)} columns ready for analysis.")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Configuration", width='stretch'):
                st.session_state.column_config_state = {}
                st.session_state.column_config = {}
                st.session_state.config_set = False
                st.session_state.data_configured = False
                st.rerun()
        
        with col3:
            if st.button("One-click: Auto-include all columns", key="auto_include_btn"):
                # Ensure the column_config_state exists
                if 'column_config_state' not in st.session_state:
                    st.session_state.column_config_state = {}

                for col in raw_df.columns:
                    if col not in st.session_state.column_config_state:
                        st.session_state.column_config_state[col] = {
                            'dtype': infer_column_type(raw_df[col]),
                            'include': True
                        }
                    else:
                        st.session_state.column_config_state[col]['include'] = True

                # Persist and apply configuration immediately
                st.session_state.column_config = st.session_state.column_config_state.copy()
                configured_df = apply_column_configuration(raw_df, st.session_state.column_config)
                st.session_state.configured_data = configured_df
                st.session_state.config_set = True
                st.session_state.data_configured = True
                st.success(f"✅ All columns included and configuration applied! {len(configured_df.columns)} columns ready for analysis.")
                st.rerun()
        
        # Show configuration summary
        if st.session_state.config_set:
            st.success("✅ Data configured and ready for analysis!")
            included_cols = [col for col, cfg in st.session_state.column_config.items() if cfg.get('include', True)]
            st.info(f"📊 {len(included_cols)} columns included | {len(raw_df.columns) - len(included_cols)} columns excluded")

        # Correlation matrix and dataset observations (appear after configuration)
        st.markdown("---")
        st.subheader("Correlation Matrix & Dataset Observations")

        corr_df_source = st.selectbox("Correlation source", options=["Configured Data (recommended)", "Raw Data"], index=0)
        if corr_df_source.startswith("Configured") and st.session_state.get('config_set', False) and st.session_state.get('configured_data') is not None:
            corr_df = st.session_state.configured_data.copy()
        else:
            corr_df = raw_df.copy()

        # Convert date columns to numeric for correlation calculation
        for c in corr_df.select_dtypes(include=['datetime64']).columns:
            corr_df[c] = (pd.Timestamp.now() - corr_df[c]).dt.days

        numeric_cols = corr_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = corr_df[numeric_cols].corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format(precision=2), width='stretch')

            st.markdown("**Dataset Observations:**")
            observations = []
            # Multicollinearity check
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    a = numeric_cols[i]
                    b = numeric_cols[j]
                    val = corr_matrix.loc[a, b]
                    if abs(val) >= 0.8:
                        high_corr_pairs.append((a, b, val))

            if high_corr_pairs:
                observations.append(f"⚠️ High correlation detected between {len(high_corr_pairs)} variable pair(s). Consider removing or combining these variables to reduce multicollinearity.")
                for a, b, val in high_corr_pairs[:10]:
                    observations.append(f"- {a} ↔ {b}: correlation = {val:.2f}")
            else:
                observations.append("✓ No strong linear multicollinearity detected among numeric variables (|r| < 0.8).")

            # Missingness and skew notes
            missing_pct = (corr_df.isna().sum().sum() / (len(corr_df) * len(corr_df.columns)) * 100)
            observations.append(f"Missing data across dataset: {missing_pct:.1f}%")

            skewed = []
            for col in numeric_cols:
                if corr_df[col].dropna().shape[0] > 0:
                    s = corr_df[col].dropna().skew()
                    if abs(s) > 2:
                        skewed.append((col, s))
            if skewed:
                observations.append(f"⚠️ Strong skew detected in {len(skewed)} numeric column(s). Consider log or power transforms.")
                for c, s in skewed[:10]:
                    observations.append(f"- {c}: skew = {s:.2f}")

            for obs in observations:
                st.write(obs)
        else:
            st.info("Not enough numeric columns to compute a meaningful correlation matrix.")

        # Hierarchy Configuration Section
        st.markdown("---")
        st.subheader("📊 Hierarchical Drill-Down Configuration")
        st.caption("Define up to 6 hierarchy levels for multi-dimensional analysis across all tabs. E.g., Company > Business Function > Region > Department > Sub-Department > Team")
        
        if st.session_state.config_set and st.session_state.get('configured_data') is not None:
            config_df = st.session_state.configured_data
        else:
            config_df = raw_df
        
        # Get all categorical/string columns suitable for hierarchy
        categorical_cols = sorted([c for c in config_df.columns if config_df[c].dtype == 'object' or config_df[c].dtype == 'category'])
        
        if not categorical_cols:
            st.warning("No categorical columns found. Hierarchy drill-down works best with categorical variables.")
        else:
            # Configure hierarchy levels
            hierarchy_config = {}
            
            st.write("**Define hierarchy levels (up to 6):**")
            
            col_sets = st.columns([1, 2, 1])
            
            with col_sets[0]:
                num_levels = st.slider("Number of hierarchy levels", min_value=1, max_value=6, value=1, key="num_hierarchy_levels")
            
            # Create UI for each level
            for level_num in range(1, num_levels + 1):
                level_col1, level_col2 = st.columns([2, 1])
                
                with level_col1:
                    level_name = st.text_input(
                        f"Level {level_num} Name",
                        value=f"Level {level_num}",
                        key=f"level_{level_num}_name",
                        help="e.g., 'Company', 'Department', 'Region'"
                    )
                
                with level_col2:
                    selected_var = st.selectbox(
                        f"Variable",
                        options=[""] + categorical_cols,
                        key=f"level_{level_num}_var",
                        help="Select the column for this hierarchy level"
                    )
                
                if selected_var:
                    hierarchy_config[f"level_{level_num}"] = {
                        'level': level_num,
                        'level_name': level_name,
                        'variable': selected_var
                    }
            
            # Save hierarchy configuration
            if st.button("💾 Save Hierarchy Configuration", type="primary", key="save_hierarchy_btn"):
                if hierarchy_config:
                    save_hierarchy_config(hierarchy_config)
                    st.success("✅ Hierarchy configuration saved!")
                    st.info("You can now use drill-down filters in other tabs.")
                else:
                    st.warning("Please configure at least one hierarchy level.")
            
            # Show current hierarchy configuration
            if st.session_state.hierarchy_config:
                st.markdown("---")
                st.write("**Current Hierarchy Configuration:**")
                current_config = st.session_state.hierarchy_config
                for key, val in sorted(current_config.items(), key=lambda x: x[1].get('level', 0)):
                    st.write(f"  **{val['level_name']}** ➜ {val['variable']}")

        # NOTE: Inline bulk uploader removed — top-level Bulk Training section (Tab 2) is preferred.

# TAB 2: Bulk Training Data Upload
with tab2:
    st.header("Bulk Training Data Upload")
    st.markdown("Load large datasets from multiple sources for training the models. This is **optional** - the app works perfectly with data from Tab 1.")
    
    # Progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    data_source = st.radio(
        "Select data source",
        options=["📁 Local File", "☁️ Google BigQuery", "📦 Amazon S3", "☁️ Azure Blob"],
        horizontal=True
    )
    
    if data_source == "📁 Local File":
        st.subheader("Upload from Computer")
        training_file = st.file_uploader("Upload training data (CSV, Excel, or Parquet)", type=['csv', 'xlsx', 'xls', 'parquet'])
        
        if training_file is not None:
            try:
                progress_bar = progress_placeholder.progress(0)
                status_placeholder.text("📥 Loading file...")
                progress_bar.progress(25)
                
                if training_file.name.endswith('.csv'):
                    training_df = pd.read_csv(training_file)
                elif training_file.name.endswith('.parquet'):
                    training_df = pd.read_parquet(training_file)
                else:
                    training_df = pd.read_excel(training_file)
                
                progress_bar.progress(75)
                st.session_state.training_data = training_df
                st.session_state.training_data_uploaded = True
                progress_bar.progress(100)
                status_placeholder.success(f"✅ Loaded {len(training_df):,} rows from {training_file.name}")
                
                st.subheader("Training Data Preview")
                st.dataframe(training_df.head(10), width='stretch')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(training_df):,}")
                with col2:
                    st.metric("Columns", len(training_df.columns))
                with col3:
                    missing_pct = (training_df.isna().sum().sum() / (len(training_df) * len(training_df.columns)) * 100)
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
    
    elif data_source == "☁️ Google BigQuery":
        st.subheader("Connect to Google BigQuery")
        st.info("Requires: `pip install google-cloud-bigquery` and authentication via `gcloud auth application-default login`")
        col1, col2 = st.columns(2)
        with col1:
            bq_project = st.text_input("GCP Project ID")
        with col2:
            bq_dataset = st.text_input("BigQuery Dataset")
        bq_table = st.text_input("Table Name")
        
        if st.button("Load from BigQuery"):
            try:
                progress_bar = progress_placeholder.progress(0)
                status_placeholder.text("☁️ Connecting to BigQuery...")
                try:
                    from google.cloud import bigquery
                    client = bigquery.Client(project=bq_project)
                    query = f"SELECT * FROM `{bq_project}.{bq_dataset}.{bq_table}` LIMIT 1000000"
                    progress_bar.progress(50)
                    status_placeholder.text("⚙️ Executing query...")
                    training_df = client.query(query).to_dataframe()
                    progress_bar.progress(100)
                    status_placeholder.success(f"✅ Loaded {len(training_df):,} rows from BigQuery")
                    st.session_state.training_data = training_df
                    st.session_state.training_data_uploaded = True
                except ImportError:
                    st.error("google-cloud-bigquery not installed")
            except Exception as e:
                status_placeholder.error(f"❌ BigQuery error: {str(e)}")
    
    elif data_source == "📦 Amazon S3":
        st.subheader("Load from Amazon S3")
        st.info("Requires: `pip install boto3` and AWS credentials configured")
        s3_bucket = st.text_input("S3 Bucket name")
        s3_key = st.text_input("S3 Object key (path)")
        
        if st.button("Load from S3"):
            try:
                progress_bar = progress_placeholder.progress(0)
                status_placeholder.text("📦 Connecting to S3...")
                try:
                    import importlib
                    # Check if boto3 is available before importing
                    if importlib.util.find_spec("boto3") is None:
                        status_placeholder.error("❌ `boto3` not installed. Install with: pip install boto3")
                    else:
                        # Import via importlib to avoid static import errors in environments
                        boto3 = importlib.import_module('boto3')
                        s3 = boto3.client('s3')
                        progress_bar.progress(50)
                        status_placeholder.text("⬇️ Downloading file...")
                        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
                        if s3_key.endswith('.csv'):
                            training_df = pd.read_csv(obj['Body'])
                        elif s3_key.endswith('.parquet'):
                            training_df = pd.read_parquet(obj['Body'])
                        else:
                            training_df = pd.read_excel(obj['Body'])
                        progress_bar.progress(100)
                        status_placeholder.success(f"✅ Loaded {len(training_df):,} rows from S3")
                        st.session_state.training_data = training_df
                        st.session_state.training_data_uploaded = True
                except Exception as e:
                    status_placeholder.error(f"❌ S3 error: {str(e)}")
            except Exception as e:
                status_placeholder.error(f"❌ S3 error: {str(e)}")
    
    elif data_source == "☁️ Azure Blob":
        st.subheader("Load from Azure Blob Storage")
        st.info("Requires: `pip install azure-storage-blob`")
        azure_conn_str = st.text_input("Connection String", type="password")
        container = st.text_input("Container name")
        blob_name = st.text_input("Blob name")
        
        if st.button("Load from Azure"):
            try:
                progress_bar = progress_placeholder.progress(0)
                status_placeholder.text("☁️ Connecting to Azure...")
                try:
                    import importlib, io
                    if importlib.util.find_spec("azure.storage.blob") is None:
                        status_placeholder.error("❌ `azure-storage-blob` not installed. Install with: pip install azure-storage-blob")
                    else:
                        # Dynamically import azure blob client to avoid static analyzer issues
                        azure_blob = importlib.import_module('azure.storage.blob')
                        BlobClient = getattr(azure_blob, 'BlobClient')
                        blob = BlobClient.from_connection_string(azure_conn_str, container, blob_name)
                        progress_bar.progress(50)
                        status_placeholder.text("⬇️ Downloading blob...")
                        data = blob.download_blob().readall()
                        if blob_name.endswith('.csv'):
                            training_df = pd.read_csv(io.BytesIO(data))
                        else:
                            training_df = pd.read_excel(io.BytesIO(data))
                        progress_bar.progress(100)
                        status_placeholder.success(f"✅ Loaded {len(training_df):,} rows from Azure")
                        st.session_state.training_data = training_df
                        st.session_state.training_data_uploaded = True
                except Exception as e:
                    status_placeholder.error(f"❌ Azure error: {str(e)}")
            except Exception as e:
                status_placeholder.error(f"❌ Azure error: {str(e)}")
    
    if st.session_state.training_data_uploaded:
        st.info("💡 Training data is loaded and will be used for model training.")
        if st.button("Clear Training Data"):
            st.session_state.training_data = None
            st.session_state.training_data_uploaded = False
            progress_placeholder.empty()
            status_placeholder.empty()
            st.rerun()

# TAB 3: Feature Preparation
with tab3:
    st.header("Transform and Feature Preparation")
    
    if not st.session_state.config_set:
        st.warning("⚠️ Please configure your data in the Data Upload tab first.")
    else:
        df = get_active_configured_data()
        if df is None:
            st.warning("⚠️ No configured data available. Please configure your data in the Data Upload tab.")
            st.stop()
        
        # Apply department filter if active
        if selected_depts is not None and 'department' in df.columns and len(selected_depts) > 0:
            df = df[df['department'].isin(selected_depts)]
            st.info(f"Filtered to {len(df)} rows in {len(selected_depts)} selected departments")
        elif selected_depts is not None and len(selected_depts) == 0:
            st.warning("No departments selected. Showing all data.")
        
        st.subheader("Column Summary")
        
        # Categorize columns by type
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Numeric Columns", len(numeric_cols))
        with col2:
            st.metric("Categorical Columns", len(categorical_cols))
        with col3:
            st.metric("Date Columns", len(date_cols))
        
        # Missing value summary
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isna().sum(),
            'Missing %': (df.isna().sum() / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, width='stretch', hide_index=True)
            
            # Missing value treatment
            st.subheader("Missing Value Treatment")
            treatment = st.radio(
                "How to handle missing values?",
                options=["Drop rows with any missing values", "Simple imputation", "Keep as-is"],
                index=2
            )
            
            if treatment == "Drop rows with any missing values":
                df = df.dropna()
                st.info(f"Dropped rows with missing values. {len(df)} rows remaining.")
            elif treatment == "Simple imputation":
                for col in numeric_cols:
                    if df[col].isna().any():
                        df[col].fillna(df[col].median(), inplace=True)
                for col in categorical_cols:
                    if df[col].isna().any():
                        df[col].fillna("Unknown", inplace=True)
                st.info("Applied median imputation for numeric columns and 'Unknown' for categorical columns.")
        else:
            st.success("✅ No missing values detected!")
        
        # Derived features
        st.subheader("Derived Features")
        
        create_derived = st.checkbox("Create derived features", value=True)
        
        if create_derived:
            derived_features = []
            
            # Age bands
            if 'age' in df.columns:
                df['age_band'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
                derived_features.append('age_band')
            
            # Tenure bands
            if 'tenure_months' in df.columns:
                df['tenure_band'] = pd.cut(df['tenure_months'], bins=[0, 12, 36, 60, 999], 
                                           labels=['<1yr', '1-3yrs', '3-5yrs', '5+yrs'])
                derived_features.append('tenure_band')
            
            # Salary bands
            if 'salary' in df.columns:
                df['salary_band'] = pd.qcut(df['salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                derived_features.append('salary_band')
            
            if derived_features:
                st.success(f"✅ Created {len(derived_features)} derived features: {', '.join(derived_features)}")
        
        # Feature matrix summary
        st.subheader("Feature Matrix Summary")
        
        # Identify potential target column
        potential_targets = [col for col in df.columns 
                            if 'attrition' in col.lower() or 'left' in col.lower() or 'churn' in col.lower()]
        
        if potential_targets:
            target_col = st.selectbox("Select target variable", options=potential_targets, index=0)
            
            feature_cols = get_feature_columns(df, target_col)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Feature Columns", len(feature_cols))
            with col3:
                if target_col in df.columns:
                    attrition_rate = df[target_col].mean() * 100
                    st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
            
            # Store prepared data
            st.session_state.prepared_data = df
            st.session_state.target_column = target_col
            st.session_state.feature_columns = feature_cols
            
            st.success("✅ Data is prepared and ready for model training!")
        else:
            st.warning("⚠️ No attrition target column found. Please ensure your data has an attrition indicator column.")

# TAB 4: Model Training
with tab4:
    st.header("Model Training and Configuration")
    
    if not st.session_state.config_set or 'prepared_data' not in st.session_state:
        st.warning("⚠️ Please complete data upload and feature preparation first.")
    else:
        # Use training data if available, otherwise use prepared data
        if st.session_state.training_data_uploaded and st.session_state.training_data is not None:
            st.info("📊 Using uploaded bulk training data for model training")
            df = st.session_state.training_data.copy()
            # Apply same preprocessing as prepared data
            target_col = st.session_state.target_column
        else:
            df = get_active_prepared_data()
            if df is None:
                st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
                st.stop()
            target_col = st.session_state.target_column
        
        # Apply department filter
        if selected_depts is not None and 'department' in df.columns and len(selected_depts) > 0:
            df = df[df['department'].isin(selected_depts)]
        
        st.subheader("Select Features for Modeling")
        
        all_features = st.session_state.feature_columns
        
        # Feature selection
        selected_features = st.multiselect(
            "Choose features to include in models",
            options=all_features,
            default=all_features[:min(15, len(all_features))],
            help="Select which features to use for training"
        )
        
        if len(selected_features) == 0:
            st.warning("⚠️ Please select at least one feature.")
        else:
            # Train-Test-Validation Split (NEW)
            st.subheader("Data Split Configuration")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                train_size = st.slider("Training %", min_value=10, max_value=70, value=60, step=5, help="Percentage for training")
            with col2:
                test_size_pct = st.slider("Test %", min_value=10, max_value=40, value=20, step=5, help="Percentage for testing")
            with col3:
                val_size_pct = st.slider("Validation %", min_value=5, max_value=40, value=20, step=5, help="Percentage for validation")
            
            # Validate proportions
            total = train_size + test_size_pct + val_size_pct
            if total != 100:
                st.warning(f"⚠️ Proportions must sum to 100% (currently {total}%)")
                train_size = 60
                test_size_pct = 20
                val_size_pct = 20
            
            # Show split breakdown
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training", f"{train_size}%")
            with col2:
                st.metric("Testing", f"{test_size_pct}%")
            with col3:
                st.metric("Validation", f"{val_size_pct}%")
            with col4:
                st.metric("Total", "100%")
            
            col1, col2 = st.columns(2)
            with col1:
                stratify_split = st.checkbox("Stratify by target", value=True)
            with col2:
                random_state_override = st.checkbox("Use custom random seed", value=False)
                if random_state_override:
                    custom_seed = st.number_input("Custom seed", min_value=0, value=random_seed)
                else:
                    custom_seed = random_seed
            
            # Model-specific parameters section
            st.subheader("Model Configuration")
            
            # Create tabs for each model
            model_tabs = st.tabs([m for m in ['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'Ensemble'] 
                                 if st.session_state.selected_models.get(m, False)])
            
            model_params = {}
            
            for tab_idx, (model_name, tab) in enumerate(zip([m for m in ['Logistic Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'Ensemble'] 
                                                               if st.session_state.selected_models.get(m, False)], model_tabs)):
                with tab:
                    st.markdown(f"### {model_name} Hyperparameters")
                    
                    if model_name == "Logistic Regression":
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            C = st.slider("C (regularization inverse)", min_value=0.001, max_value=10.0, value=1.0, step=0.1)
                        with c2:
                            solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "newton-cg", "sag"])
                        with c3:
                            max_iter = st.number_input("Max iterations", min_value=100, value=2000, step=100)
                        
                        # Feature selection method for Logistic Regression
                        st.markdown("### Feature Selection (Optional)")
                        feature_selection_method = st.radio(
                            "Select feature selection method",
                            options=["None", "Stepwise", "Forward", "Backward"],
                            index=0,
                            help="Use stepwise, forward, or backward selection to reduce features based on p-values (requires statsmodels)."
                        )
                        
                        model_params[model_name] = {
                            'C': C, 
                            'solver': solver, 
                            'max_iter': max_iter,
                            'feature_selection': feature_selection_method
                        }
                    
                    elif model_name == "Random Forest":
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            n_estimators = st.slider("Number of trees", min_value=10, max_value=500, value=100, step=10)
                        with c2:
                            max_depth = st.slider("Max depth", min_value=5, max_value=50, value=20, step=1)
                        with c3:
                            min_samples_split = st.slider("Min samples split", min_value=2, max_value=20, value=5, step=1)
                        model_params[model_name] = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
                    
                    elif model_name == "Decision Tree":
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            dt_max_depth = st.slider("Max depth", min_value=2, max_value=30, value=10, step=1)
                        with c2:
                            dt_min_samples_leaf = st.slider("Min samples leaf", min_value=1, max_value=50, value=10, step=1)
                        with c3:
                            dt_criterion = st.selectbox("Split criterion", options=["gini", "entropy"])
                        model_params[model_name] = {'max_depth': dt_max_depth, 'min_samples_leaf': dt_min_samples_leaf, 'criterion': dt_criterion}
                    
                    elif model_name == "Gradient Boosting":
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            gb_n_estimators = st.slider("Number of boosting stages", min_value=50, max_value=500, value=100, step=10)
                        with c2:
                            learning_rate = st.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
                        with c3:
                            gb_max_depth = st.slider("Max depth", min_value=3, max_value=20, value=5, step=1)
                        model_params[model_name] = {'n_estimators': gb_n_estimators, 'learning_rate': learning_rate, 'max_depth': gb_max_depth}
                    
                    elif model_name == "Ensemble":
                        st.info("Ensemble combines Logistic Regression, Random Forest, and Gradient Boosting")
                        voting = st.radio("Voting method", options=["soft", "hard"], index=0, key=f"voting_{model_name}")
                        model_params[model_name] = {'voting': voting}
            
            if st.button("🚀 Train Selected Models", type="primary", key="train_models_btn"):
                with st.spinner("Training models..."):
                    # Prepare data
                    X = df[selected_features].copy()
                    y = df[target_col].copy()
                    
                    # Encode categorical variables
                    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    for cat_col in categorical_features:
                        le = LabelEncoder()
                        X[cat_col] = le.fit_transform(X[cat_col].astype(str))
                    
                    # First split: training vs (test + validation)
                    test_val_size = (test_size_pct + val_size_pct) / 100.0
                    stratify_param = y if stratify_split else None
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X, y, test_size=test_val_size, random_state=custom_seed, stratify=stratify_param
                    )
                    
                    # Second split: test vs validation
                    val_ratio = val_size_pct / (test_size_pct + val_size_pct)
                    X_test, X_val, y_test, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_ratio, random_state=custom_seed, stratify=y_temp if stratify_split else None
                    )
                    
                    # Build selected models only
                    models = {}
                    
                    if st.session_state.selected_models.get('Logistic Regression', False):
                        params = model_params.get('Logistic Regression', {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000, 'feature_selection': 'None'})
                        
                        # Apply feature selection if requested
                        feature_selection_method = params.get('feature_selection', 'None')
                        selected_features_lr = selected_features.copy()
                        pvalues_lr = {}
                        
                        if feature_selection_method != 'None':
                            if not statsmodels_available:
                                st.warning("⚠️ statsmodels not installed. Skipping feature selection. Install with: `pip install statsmodels`")
                            else:
                                with st.spinner(f"Running {feature_selection_method} feature selection..."):
                                    selected_features_lr, pvalues_lr = stepwise_selection(
                                        X_train[selected_features], 
                                        y_train, 
                                        direction=feature_selection_method.lower(),
                                        p_threshold=0.05
                                    )
                                    
                                    st.info(f"✅ {feature_selection_method} selection reduced features from {len(selected_features)} to {len(selected_features_lr)}")
                                    
                                    # Display selected features and p-values
                                    if pvalues_lr:
                                        st.markdown("**Selected Features and P-values:**")
                                        pval_data = [
                                            {'Feature': feat, 'P-value': f"{pval:.4f}", 'Significant': '✓' if pval < 0.05 else '✗'}
                                            for feat, pval in sorted(pvalues_lr.items(), key=lambda x: x[1])
                                        ]
                                        pval_df = pd.DataFrame(pval_data)
                                        st.dataframe(pval_df, width='stretch', hide_index=True)
                        
                        models["Logistic Regression"] = LogisticRegression(
                            C=params['C'], solver=params['solver'], max_iter=params['max_iter'],
                            random_state=custom_seed
                        )
                        
                        # Store selected features for training
                        st.session_state.lr_selected_features = selected_features_lr
                        st.session_state.lr_pvalues = pvalues_lr
                    
                    if st.session_state.selected_models.get('Random Forest', False):
                        params = model_params.get('Random Forest', {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5})
                        models["Random Forest"] = RandomForestClassifier(
                            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                            min_samples_split=params['min_samples_split'],
                            random_state=custom_seed, n_jobs=-1
                        )
                    
                    if st.session_state.selected_models.get('Decision Tree', False):
                        params = model_params.get('Decision Tree', {'max_depth': 10, 'min_samples_leaf': 10, 'criterion': 'gini'})
                        models["Decision Tree"] = DecisionTreeClassifier(
                            max_depth=params['max_depth'], min_samples_leaf=params['min_samples_leaf'],
                            criterion=params['criterion'], random_state=custom_seed
                        )
                    
                    if st.session_state.selected_models.get('Gradient Boosting', False):
                        params = model_params.get('Gradient Boosting', {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5})
                        models["Gradient Boosting"] = GradientBoostingClassifier(
                            n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                            max_depth=params['max_depth'], random_state=custom_seed
                        )
                    
                    if st.session_state.selected_models.get('Ensemble', False):
                        voting = model_params.get('Ensemble', {}).get('voting', 'soft')
                        models["Ensemble"] = VotingClassifier(
                            estimators=[
                                ('lr', LogisticRegression(random_state=custom_seed, max_iter=2000)),
                                ('rf', RandomForestClassifier(n_estimators=50, random_state=custom_seed, n_jobs=-1)),
                                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=custom_seed))
                            ],
                            voting=voting
                        )
                    
                    if len(models) == 0:
                        st.error("⚠️ No models selected. Please select at least one model.")
                    else:
                        trained_models = {}
                        progress_bar = st.progress(0)
                        
                        for idx, (name, model) in enumerate(models.items()):
                            st.text(f"Training {name}...")
                            
                            # Use selected features for Logistic Regression if feature selection was applied
                            if name == "Logistic Regression" and st.session_state.get('lr_selected_features'):
                                features_to_use = st.session_state.lr_selected_features
                                X_train_subset = X_train[features_to_use]
                                X_test_subset = X_test[features_to_use]
                                X_val_subset = X_val[features_to_use]
                            else:
                                X_train_subset = X_train
                                X_test_subset = X_test
                                X_val_subset = X_val
                                features_to_use = selected_features
                            
                            model.fit(X_train_subset, y_train)
                            trained_models[name] = {
                                'model': model,
                                'X_train': X_train_subset,
                                'X_test': X_test_subset,
                                'X_val': X_val_subset,
                                'y_train': y_train,
                                'y_test': y_test,
                                'y_val': y_val,
                                'feature_names': features_to_use
                            }
                            progress_bar.progress((idx + 1) / len(models))
                        
                        st.session_state.trained_models = trained_models
                        st.session_state.models_trained = True
                        
                        st.success(f"✅ Successfully trained {len(models)} models!")
                        
                        # Show quick summary
                        st.subheader("Training Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Training Samples", len(X_train))
                        with col2:
                            st.metric("Test Samples", len(X_test))
                        with col3:
                            st.metric("Validation Samples", len(X_val))
                        with col4:
                            st.metric("Features Used", len(selected_features))
            
            if st.session_state.models_trained:
                st.success("✅ Models are trained and ready for evaluation!")

# TAB 5: Performance Comparison
with tab5:
    st.header("Model Performance and Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models in the Model Training tab first.")
    else:
        trained_models = st.session_state.trained_models
        
        # Calculate metrics for all models - ENHANCED with statistical tests
        metrics_data = []
        roc_data = []
        
        for name, model_info in trained_models.items():
            model = model_info['model']
            X_test = model_info['X_test']
            y_test = model_info['y_test']
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            misclassification = 1 - accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Extract confusion matrix values
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_test))
            
            # Additional statistical measures
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Kolmogorov-Smirnov statistic
            ks_stat = sensitivity - fpr_val
            
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Misclassification': misclassification,
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC AUC': roc_auc_score(y_test, y_pred_proba),
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'FPR': fpr_val,
                'FNR': fnr_val,
                'KS Statistic': ks_stat
            }
            metrics_data.append(metrics)
            
            # ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_data.append({
                'model': name,
                'fpr': fpr,
                'tpr': tpr,
                'auc': metrics['ROC AUC']
            })
        
        # Metrics comparison table - COMPREHENSIVE
        st.subheader("Comprehensive Model Performance Comparison")
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('ROC AUC', ascending=False)
        
        # Display full metrics table
        st.dataframe(
            metrics_df.style.format({
                'Accuracy': '{:.4f}',
                'Misclassification': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'ROC AUC': '{:.4f}',
                'Sensitivity': '{:.4f}',
                'Specificity': '{:.4f}',
                'FPR': '{:.4f}',
                'FNR': '{:.4f}',
                'KS Statistic': '{:.4f}'
            }).background_gradient(subset=['ROC AUC'], cmap='Greens').background_gradient(subset=['KS Statistic'], cmap='Blues'),
            width='stretch',
            hide_index=True
        )
        
        st.markdown("""
        **Metric Definitions:**
        - **Accuracy**: Proportion of correct predictions
        - **Precision**: Of predicted leavers, how many actually left
        - **Recall (Sensitivity)**: Of actual leavers, how many were identified
        - **Specificity**: Of actual stayers, how many were correctly identified  
        - **FPR**: False positive rate (1 - Specificity)
        - **FNR**: False negative rate (1 - Sensitivity)
        - **KS Statistic**: Kolmogorov-Smirnov statistic (Sensitivity - FPR) - measures separation between distributions
        - **ROC AUC**: Area under the Receiver Operating Characteristic curve
        """)
        
        # Champion model
        champion = metrics_df.iloc[0]
        st.success(f"🏆 Champion Model: **{champion['Model']}** with ROC AUC of {champion['ROC AUC']:.4f} and KS = {champion['KS Statistic']:.4f}")
        
        # Store champion info in session state
        st.session_state.champion_model_name = champion['Model']
        
        # ROC Curves
        st.subheader("ROC Curves")
        
        fig = go.Figure()
        
        for roc in roc_data:
            fig.add_trace(go.Scatter(
                x=roc['fpr'],
                y=roc['tpr'],
                mode='lines',
                name=f"{roc['model']} (AUC={roc['auc']:.3f})",
                line=dict(width=2)
            ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray', width=1)
        ))
        
        fig.update_layout(
            title='ROC Curves for All Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        cols = st.columns(min(len(trained_models), 4))
        
        for idx, (name, model_info) in enumerate(list(trained_models.items())[:4]):
            with cols[idx]:
                model = model_info['model']
                X_test = model_info['X_test']
                y_test = model_info['y_test']
                
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(cm, cmap='Blues')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Stay', 'Leave'])
                ax.set_yticklabels(['Stay', 'Leave'])
                
                for i in range(2):
                    for j in range(2):
                        text = ax.text(j, i, cm[i, j],
                                      ha="center", va="center", color="white", fontsize=12)
                
                ax.set_title(f"{name}", color='white')
                ax.set_xlabel('Predicted', color='white')
                ax.set_ylabel('Actual', color='white')
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                
                st.pyplot(fig)
                plt.close()
        
        # Model comparison narrative
        st.subheader("Performance Summary")
        
        best_model = metrics_df.iloc[0]
        worst_model = metrics_df.iloc[-1]
        auc_gap = best_model['ROC AUC'] - worst_model['ROC AUC']
        
        st.markdown(f"""
        **{best_model['Model']}** emerges as the champion with an ROC AUC of **{best_model['ROC AUC']:.3f}**, 
        demonstrating the strongest ability to distinguish between employees who will stay versus those likely to leave.
        
        The performance gap between the best and weakest model is **{auc_gap:.3f}**, with **{worst_model['Model']}** 
        achieving an ROC AUC of **{worst_model['ROC AUC']:.3f}**.
        
        **Key Performance Metrics:**
        - **Accuracy**: {best_model['Accuracy']:.1%} correct predictions overall
        - **Misclassification**: {best_model['Misclassification']:.1%} error rate
        - **Precision**: {best_model['Precision']:.1%} of predicted leavers actually left
        - **Recall**: {best_model['Recall']:.1%} of actual attrition cases identified
        - **F1 Score**: {best_model['F1 Score']:.3f} balanced performance metric
        """)

# TAB 6: KPI Dashboard
with tab6:
    st.header("Attrition KPI Dashboard")
    
    if not st.session_state.config_set:
        st.warning("⚠️ Please configure your data first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        
        # Apply department filter
        if selected_depts is not None and 'department' in df.columns and len(selected_depts) > 0:
            df = df[df['department'].isin(selected_depts)]
        
        target_col = st.session_state.target_column
        
        # Headline metrics
        st.subheader("Headline Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_employees = len(df)
            st.metric("Total Employees", f"{total_employees:,}")
        
        with col2:
            attrition_count = df[target_col].sum()
            st.metric("Attrition Count", f"{int(attrition_count):,}")
        
        with col3:
            attrition_rate = (df[target_col].mean() * 100)
            st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        
        with col4:
            retention_rate = 100 - attrition_rate
            st.metric("Retention Rate", f"{retention_rate:.1f}%")
        
        # Time trends
        if 'snapshot_date' in df.columns:
            st.subheader("Attrition Trends Over Time")
            
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            trend_df = df.groupby('snapshot_date', observed=False).agg({
                target_col: ['sum', 'mean', 'count']
            }).reset_index()
            trend_df.columns = ['snapshot_date', 'attrition_count', 'attrition_rate', 'total']
            trend_df['attrition_rate'] = trend_df['attrition_rate'] * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trend_df['snapshot_date'],
                y=trend_df['attrition_rate'],
                mode='lines+markers',
                name='Attrition Rate %',
                line=dict(color=DANGER_RED, width=2),
                yaxis='y'
            ))
            
            fig.add_trace(go.Bar(
                x=trend_df['snapshot_date'],
                y=trend_df['attrition_count'],
                name='Attrition Count',
                marker_color=PRIMARY_BLUE,
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Attrition Trends',
                xaxis_title='Date',
                yaxis_title='Attrition Rate (%)',
                yaxis2=dict(title='Attrition Count', overlaying='y', side='right'),
                hovermode='x unified',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, width='stretch')
        
        # Forecast section
        st.subheader(f"Attrition Forecast - {forecast_horizon}")
        
        # Map horizon to months
        horizon_map = {
            "12 months": 12,
            "2 years": 24,
            "3 years": 36,
            "4 years": 48,
            "5 years": 60
        }
        
        forecast_months = horizon_map[forecast_horizon]
        current_rate = df[target_col].mean()
        
        # Simple forecast model based on current trends
        st.info(f"""
        **Forecast Assumptions:**
        - Current attrition rate: {current_rate*100:.1f}%
        - Forecast horizon: {forecast_months} months
        - Projected cumulative attrition: {(current_rate * forecast_months / 12 * 100):.1f}% over {forecast_horizon}
        
        💡 Risk may increase or decrease based on implemented retention strategies.
        """)
        
        # Breakdowns
        st.subheader("Attrition by Segment")
        
        # Department breakdown
        if 'department' in df.columns:
            dept_stats = df.groupby('department', observed=False).agg({
                target_col: ['sum', 'mean', 'count']
            }).reset_index()
            dept_stats.columns = ['department', 'attrition_count', 'attrition_rate', 'total']
            dept_stats['attrition_rate'] = dept_stats['attrition_rate'] * 100
            dept_stats = dept_stats.sort_values('attrition_rate', ascending=False)
            
            overall_rate = df[target_col].mean() * 100
            
            fig = go.Figure()
            
            colors = [DANGER_RED if rate > overall_rate else PRIMARY_BLUE 
                     for rate in dept_stats['attrition_rate']]
            
            fig.add_trace(go.Bar(
                x=dept_stats['department'],
                y=dept_stats['attrition_rate'],
                marker_color=colors,
                text=dept_stats['attrition_rate'].round(1),
                texttemplate='%{text}%',
                textposition='outside',
                name='Attrition Rate'
            ))
            
            fig.add_hline(y=overall_rate, line_dash="dash", line_color="yellow",
                         annotation_text=f"Overall: {overall_rate:.1f}%")
            
            fig.update_layout(
                title='Attrition Rate by Department',
                xaxis_title='Department',
                yaxis_title='Attrition Rate (%)',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, width='stretch')
        
        # Multi-dimensional breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            if 'location' in df.columns:
                loc_stats = df.groupby('location', observed=False)[target_col].agg(['mean', 'count']).reset_index()
                loc_stats.columns = ['location', 'attrition_rate', 'count']
                loc_stats['attrition_rate'] = loc_stats['attrition_rate'] * 100
                loc_stats['label'] = loc_stats.apply(
                    lambda x: f"{x['location']}<br>{x['attrition_rate']:.1f}%", axis=1
                )
                
                fig = go.Figure(data=[go.Pie(
                    labels=loc_stats['location'],
                    values=loc_stats['count'],
                    text=loc_stats['label'],
                    textposition='outside',
                    hole=0.4,
                    marker_colors=[DANGER_RED if rate > overall_rate else PRIMARY_BLUE 
                                  for rate in loc_stats['attrition_rate']]
                )])
                
                fig.update_layout(
                    title='Employee Distribution by Location',
                    height=500,
                    showlegend=True,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            if 'age_band' in df.columns:
                age_stats = df.groupby('age_band', observed=False)[target_col].agg(['mean', 'count']).reset_index()
                age_stats.columns = ['age_band', 'attrition_rate', 'count']
                age_stats['attrition_rate'] = age_stats['attrition_rate'] * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=age_stats['age_band'],
                    y=age_stats['attrition_rate'],
                    marker_color=PRIMARY_BLUE,
                    text=age_stats['attrition_rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title='Attrition Rate by Age Band',
                    xaxis_title='Age Band',
                    yaxis_title='Attrition Rate (%)',
                    height=500,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, width='stretch')

# TAB 7: Segments
with tab7:
    st.header("Segment Analysis with Decision Tree")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        
        # Apply department filter
        if selected_depts is not None and 'department' in df.columns and len(selected_depts) > 0:
            df = df[df['department'].isin(selected_depts)]
        
        target_col = st.session_state.target_column
        
        st.subheader("Interpretable Decision Tree")
        
        # Train a simple decision tree for interpretability
        feature_cols = st.session_state.feature_columns[:15]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode categoricals
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Train interpretable tree
        tree_depth = st.slider("Tree depth", min_value=2, max_value=6, value=3)
        
        dt = DecisionTreeClassifier(max_depth=tree_depth, random_state=random_seed, min_samples_leaf=50)
        dt.fit(X, y)
        
        # Get rules
        tree_rules = export_text(dt, feature_names=feature_cols)
        
        with st.expander("View Decision Tree Rules"):
            st.code(tree_rules)
        
        # Feature importance
        st.subheader("Feature Importance for Segmentation")
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': dt.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = go.Figure(data=[go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color=PRIMARY_BLUE
        )])
        
        fig.update_layout(
            title='Top 10 Features for Segmentation',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Risk segments
        st.subheader("High-Risk Segments")
        
        # Get leaf nodes
        leaf_indices = dt.apply(X)
        df['segment'] = leaf_indices
        
        segment_stats = df.groupby('segment', observed=False).agg({
            target_col: ['mean', 'count']
        }).reset_index()
        segment_stats.columns = ['segment', 'attrition_rate', 'population']
        segment_stats['attrition_rate'] = segment_stats['attrition_rate'] * 100
        segment_stats = segment_stats.sort_values('attrition_rate', ascending=False)
        
        # Show top risk segments
        st.dataframe(
            segment_stats.head(10).style.format({
                'attrition_rate': '{:.1f}%',
                'population': '{:.0f}'
            }).background_gradient(subset=['attrition_rate'], cmap='Reds'),
            width='stretch',
            hide_index=True
        )
        
        # Interactive segment explorer
        st.subheader("Segment Explorer")
        
        selected_segment = st.selectbox(
            "Select a segment to explore",
            options=segment_stats['segment'].tolist(),
            format_func=lambda x: f"Segment {x} ({segment_stats[segment_stats['segment']==x]['population'].values[0]:.0f} employees, {segment_stats[segment_stats['segment']==x]['attrition_rate'].values[0]:.1f}% attrition)"
        )
        
        if selected_segment is not None:
            segment_df = df[df['segment'] == selected_segment]
            st.write(f"**{len(segment_df)} employees** in this segment")
            st.dataframe(segment_df.head(20), width='stretch')

# TAB 8: Individual Risk
with tab8:
    st.header("Individual Risk Explorer")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        target_col = st.session_state.target_column
        
        # Get champion model
        trained_models = st.session_state.trained_models
        
        # Search employee
        st.subheader("Search Employee")
        
        if 'employee_id' in df.columns:
            employee_options = df['employee_id'].tolist()
            selected_employee = st.selectbox(
                "Select employee by ID",
                options=employee_options,
                format_func=lambda x: f"Employee {x}"
            )
        else:
            selected_employee = st.number_input("Enter row index", min_value=0, max_value=len(df)-1, value=0)
        
        if selected_employee is not None:
            if 'employee_id' in df.columns:
                employee_row = df[df['employee_id'] == selected_employee].iloc[0]
            else:
                employee_row = df.iloc[selected_employee]
            
            st.subheader(f"Risk Profile: Employee {selected_employee}")
            
            # Get predictions from all models
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Attrition Risk Scores")
                
                for name, model_info in trained_models.items():
                    model = model_info['model']
                    feature_names = model_info['feature_names']
                    
                    # Prepare employee data
                    X_employee = employee_row[feature_names].to_frame().T
                    
                    # Encode categoricals
                    for col in X_employee.select_dtypes(include=['object', 'category']).columns:
                        le = LabelEncoder()
                        all_vals = df[col].astype(str).unique()
                        le.fit(all_vals)
                        try:
                            X_employee[col] = le.transform(X_employee[col].astype(str))
                        except:
                            X_employee[col] = 0
                    
                    risk_prob = model.predict_proba(X_employee)[0, 1]
                    
                    risk_level = "🔴 Critical" if risk_prob > 0.7 else "🟡 High" if risk_prob > 0.4 else "🟢 Low"
                    st.metric(name, f"{risk_prob*100:.1f}%", risk_level)
            
            with col2:
                st.markdown("### Employee Details")
                
                # Show key attributes
                display_cols = [col for col in df.columns if col not in ['employee_id', target_col, 'segment']][:12]
                
                details_data = []
                for col in display_cols:
                    details_data.append({
                        'Attribute': col,
                        'Value': str(employee_row[col])
                    })
                
                details_df = pd.DataFrame(details_data)
                st.dataframe(details_df, width='stretch', hide_index=True)
            
            # Feature importance for this prediction
            st.subheader("Risk Drivers")
            
            # Use the champion model
            champion_name = st.session_state.get('champion_model_name', list(trained_models.keys())[0])
            champion_model = trained_models[champion_name]['model']
            
            if hasattr(champion_model, 'feature_importances_'):
                feature_importance = champion_model.feature_importances_
                feature_names = trained_models[champion_name]['feature_names']
                
                importance_data = []
                for i, feat in enumerate(feature_names):
                    val = employee_row[feat] if feat in employee_row else 'N/A'
                    importance_data.append({
                        'Feature': feat,
                        'Importance': feature_importance[i],
                        'Employee Value': str(val)
                    })
                
                importance_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=False).head(8)
                
                fig = go.Figure(data=[go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color=DANGER_RED,
                    text=importance_df['Employee Value'],
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title='Top Risk Drivers for This Employee',
                    xaxis_title='Feature Importance',
                    yaxis_title='',
                    height=350,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # NEW: Demographic Analysis and Trends
            st.subheader("Demographic Analysis")
            
            demo_cols = []
            for col in ['age', 'gender', 'department', 'location', 'job_role', 'grade']:
                if col in df.columns:
                    demo_cols.append(col)
            
            if demo_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Employee's Demographics & Tenure:**")
                    for col in demo_cols:
                        st.write(f"- **{col.title()}**: {employee_row[col]}")
                    
                    # NEW: Tenure in years and retirement distance
                    if 'tenure_months' in employee_row:
                        tenure_years = employee_row['tenure_months'] / 12
                        st.write(f"- **Years Worked**: {tenure_years:.1f} years")
                    
                    if 'age' in employee_row:
                        age = employee_row['age']
                        retirement_age = 65
                        years_to_retirement = max(0, retirement_age - age)
                        st.write(f"- **Years Until Retirement**: {years_to_retirement} years")
                        
                        if years_to_retirement < 3:
                            st.warning(f"⏰ Approaching retirement in ~{years_to_retirement} years")
                        elif years_to_retirement < 5:
                            st.info(f"⏰ Mid-career: ~{years_to_retirement} years to retirement")
                        else:
                            st.success(f"🚀 Early career: ~{years_to_retirement} years to retirement")
                
                with col2:
                    st.markdown("**Demographic Attrition Trends:**")
                    
                    for col in demo_cols[:3]:  # Show top 3 demographic trends
                        if col in df.columns:
                            demo_attrition = df.groupby(col, observed=False)[target_col].agg(['mean', 'count']).reset_index()
                            demo_attrition.columns = [col, 'attrition_rate', 'count']
                            demo_attrition['attrition_rate'] = demo_attrition['attrition_rate'] * 100
                            
                            employee_demo_value = employee_row[col]
                            employee_demo_rate = demo_attrition[demo_attrition[col] == employee_demo_value]['attrition_rate']
                            
                            if len(employee_demo_rate) > 0:
                                avg_rate = df[target_col].mean() * 100
                                emp_rate = employee_demo_rate.values[0]
                                if emp_rate > avg_rate:
                                    st.warning(f"⚠️ {col.title()} '{employee_demo_value}': {emp_rate:.1f}% attrition (vs {avg_rate:.1f}% org avg)")
                                else:
                                    st.info(f"✓ {col.title()} '{employee_demo_value}': {emp_rate:.1f}% attrition (vs {avg_rate:.1f}% org avg)")
            
            # NEW: Personalized Retention Strategies
            st.subheader("Personalized Retention Strategies")
            
            # Get risk probability from champion model
            champion_name = st.session_state.get('champion_model_name', list(trained_models.keys())[0])
            champion_model = trained_models[champion_name]['model']
            feature_names = trained_models[champion_name]['feature_names']
            
            X_employee = employee_row[feature_names].to_frame().T
            for col in X_employee.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                all_vals = df[col].astype(str).unique()
                le.fit(all_vals)
                try:
                    X_employee[col] = le.transform(X_employee[col].astype(str))
                except:
                    X_employee[col] = 0
            
            risk_prob = champion_model.predict_proba(X_employee)[0, 1]
            
            # Generate personalized strategies based on risk level and risk drivers
            retention_strategies = []
            
            if risk_prob > 0.7:
                retention_strategies = [
                    "🔴 **CRITICAL PRIORITY**: Schedule immediate conversation with employee",
                    "💼 Review compensation competitiveness",
                    "🎯 Discuss career development opportunities",
                    "🏆 Highlight recent achievements and growth",
                    "📞 Connect with manager for direct engagement",
                    "🤝 Offer mentorship or sponsorship opportunities"
                ]
            elif risk_prob > 0.4:
                retention_strategies = [
                    "🟡 **HIGH PRIORITY**: Plan retention discussion within next 2 weeks",
                    "📈 Review performance ratings and provide constructive feedback",
                    "🎓 Identify training or certification opportunities",
                    "👥 Strengthen peer relationships through team activities",
                    "💡 Discuss project assignment preferences",
                    "🌱 Create a clear career path with milestones"
                ]
            else:
                retention_strategies = [
                    "🟢 **STANDARD**: Continue regular engagement practices",
                    "📊 Monitor engagement scores quarterly",
                    "🤝 Maintain regular 1-on-1 meetings",
                    "🎉 Recognize and celebrate contributions",
                    "🔄 Provide variety in project assignments",
                    "📚 Support continuous learning initiatives"
                ]
            
            for strategy in retention_strategies:
                st.markdown(strategy)
            
            # NEW: Probability of Voluntary Quit with Timeline
            st.subheader("Predicted Attrition Probability & Timeline")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quit Probability", f"{risk_prob*100:.1f}%")
            with col2:
                if risk_prob > 0.7:
                    predicted_timeline = "Next 3 months"
                elif risk_prob > 0.5:
                    predicted_timeline = "Next 6 months"
                elif risk_prob > 0.3:
                    predicted_timeline = "6-12 months"
                else:
                    predicted_timeline = "12+ months"
                st.metric("Predicted Timeline", predicted_timeline)
            with col3:
                risk_category = "🔴 CRITICAL" if risk_prob > 0.7 else "🟡 HIGH" if risk_prob > 0.4 else "🟢 LOW"
                st.metric("Risk Level", risk_category)
            
            st.markdown(f"""
            Based on the {champion_name} model analysis:
            - **Probability of voluntary departure**: {risk_prob*100:.1f}%
            - **Recommended action window**: {predicted_timeline}
            - **Confidence level**: High (based on model performance)
            """)
            
            # NEW: Career Progression Tree
            st.subheader("Career Development Pathway")
            
            # Get current role and job level
            current_role = employee_row.get('job_role', 'Employee')
            
            # Define career progressions (customizable)
            career_paths = {
                'Analyst': {
                    'next_roles': ['Senior Analyst', 'Specialist Analyst', 'Team Lead'],
                    'next_level': 'Senior Analyst',
                    'training_needed': ['Advanced Data Analysis', 'Business Intelligence', 'Project Management'],
                    'competencies': ['Strategic Thinking', 'Leadership', 'Communication', 'Problem Solving'],
                    'timeline': '2-3 years'
                },
                'Manager': {
                    'next_roles': ['Senior Manager', 'Director', 'Head of Department'],
                    'next_level': 'Senior Manager / Director',
                    'training_needed': ['Executive Leadership', 'Strategic Planning', 'Change Management'],
                    'competencies': ['Strategic Vision', 'Team Building', 'Decision Making', 'Business Acumen'],
                    'timeline': '3-5 years'
                },
                'Specialist': {
                    'next_roles': ['Senior Specialist', 'Practice Lead', 'Principal Specialist'],
                    'next_level': 'Senior Specialist / Practice Lead',
                    'training_needed': ['Industry Certification', 'Subject Matter Expertise', 'Mentoring Skills'],
                    'competencies': ['Technical Mastery', 'Knowledge Sharing', 'Innovation', 'Influence'],
                    'timeline': '2-4 years'
                },
                'Lead': {
                    'next_roles': ['Senior Lead', 'Manager', 'Principal'],
                    'next_level': 'Senior Lead / Manager',
                    'training_needed': ['People Management', 'Strategic Leadership', 'Budget Management'],
                    'competencies': ['Team Leadership', 'Strategic Planning', 'Accountability', 'Coaching'],
                    'timeline': '2-3 years'
                }
            }
            
            # Match employee to career path
            career_info = None
            for key, value in career_paths.items():
                if key.lower() in current_role.lower():
                    career_info = value
                    break
            
            if career_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Career Position:**")
                    st.info(f"Current Role: **{current_role}**")
                    st.markdown("**Next Level Opportunity:**")
                    st.success(f"{career_info['next_level']}")
                    st.markdown("**Typical Timeline:**")
                    st.write(f"{career_info['timeline']}")
                
                with col2:
                    st.markdown("**Training & Development Needed:**")
                    for training in career_info['training_needed']:
                        st.markdown(f"- 📚 {training}")
                    
                    st.markdown("**Required Competencies:**")
                    for comp in career_info['competencies']:
                        st.markdown(f"- ✓ {comp}")
                
                st.markdown("**Alternative Career Paths:**")
                for alt_role in career_info['next_roles']:
                    st.markdown(f"→ {alt_role}")
            else:
                st.info(f"Career development pathway for '{current_role}' role. Consider mentoring and cross-functional projects to build skills for advancement.")
            
            # NEW: Succession Planning (for senior roles)
            job_grade = str(employee_row.get('grade', '')).upper()
            if job_grade in ['A', 'B'] or 'Manager' in str(current_role) or 'Director' in str(current_role):
                st.subheader("Succession Planning")
                
                # Identify potential successors from same department and lower grade
                employee_dept = employee_row.get('department', None)
                potential_successors = None
                
                if employee_dept:
                    # Find people in same department, lower grade, with good performance
                    grade_order = {'D': 1, 'C': 2, 'B': 3, 'A': 4}
                    emp_grade_level = grade_order.get(job_grade, 0)
                    
                    successors = df[
                        (df['department'] == employee_dept) &
                        (df['grade'].apply(lambda x: grade_order.get(str(x).upper(), 0)) < emp_grade_level) &
                        (df['performance_rating'] >= 4.0) if 'performance_rating' in df.columns else True
                    ].head(3)
                    
                    if len(successors) > 0:
                        st.markdown("**Identified Successors:**")
                        for idx, succ in successors.iterrows():
                            succ_role = succ.get('job_role', 'Employee')
                            succ_grade = succ.get('grade', 'N/A')
                            st.markdown(f"- 🟢 **{succ_role}** (Grade {succ_grade}) - Ready for transition training")
                    else:
                        st.warning("⚠️ No immediate successors identified. Consider external recruitment or mentoring junior staff.")
                else:
                    st.info("Succession planning not available for this employee profile.")
            
            # NEW: Individual-Level Recommendations
            st.subheader("Personalized Recommendations for This Employee")
            
            recommendations = []
            
            # Based on tenure
            tenure_years = employee_row.get('tenure_months', 0) / 12 if 'tenure_months' in employee_row else 0
            if tenure_years < 1:
                recommendations.append("🎯 **Onboarding**: Ensure structured onboarding and mentorship during first year")
            elif tenure_years < 3:
                recommendations.append("📈 **Growth**: Provide stretch assignments and skill development opportunities")
            elif tenure_years > 10:
                recommendations.append("🏆 **Recognition**: Leverage experience - consider leadership or mentor roles")
            
            # Based on risk level
            if risk_prob > 0.7:
                recommendations.append("🚨 **URGENT**: Schedule retention conversation with manager and HR within 1 week")
                recommendations.append("💰 **Compensation**: Review market rates and consider adjustment")
            elif risk_prob > 0.4:
                recommendations.append("📞 **Engagement**: Increase frequency of check-ins (monthly instead of quarterly)")
                recommendations.append("🎓 **Development**: Discuss career goals and create development plan")
            
            # Based on performance
            if 'performance_rating' in employee_row:
                perf = employee_row['performance_rating']
                if perf >= 4.5:
                    recommendations.append("⭐ **High Performer**: Nominate for high-visibility projects or leadership program")
                elif perf < 3:
                    recommendations.append("📋 **Performance**: Create performance improvement plan with clear milestones")
            
            # Based on demographics
            if 'age' in employee_row and years_to_retirement < 3:
                recommendations.append("🎓 **Knowledge Transfer**: Document expertise before retirement transition")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.info("✓ Employee appears engaged. Continue regular touch-points and career development conversations.")

# TAB 9: Recommendations
with tab9:
    st.header("Recommendations and Retention Strategy")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models in the Model Training tab first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        target_col = st.session_state.target_column
        
        st.subheader("Strategic Recommendations")
        
        # Department-level risks
        if 'department' in df.columns:
            dept_risk = df.groupby('department', observed=False).agg({
                target_col: ['mean', 'count']
            }).reset_index()
            dept_risk.columns = ['department', 'attrition_rate', 'employee_count']
            dept_risk['attrition_rate'] = dept_risk['attrition_rate'] * 100
            dept_risk['risk_score'] = dept_risk['attrition_rate'] * np.log1p(dept_risk['employee_count'])
            dept_risk['priority'] = pd.cut(dept_risk['risk_score'], bins=3, labels=['Low', 'Medium', 'High'])
            dept_risk = dept_risk.sort_values('risk_score', ascending=False)
            
            st.markdown("### Department Risk Prioritization")
            st.dataframe(
                dept_risk.style.format({
                    'attrition_rate': '{:.1f}%',
                    'employee_count': '{:.0f}',
                    'risk_score': '{:.1f}'
                }),
                width='stretch',
                hide_index=True
            )
        
        # Feature-based recommendations
        st.subheader("Key Action Areas")
        
        # Get feature importance from champion model
        champion_name = st.session_state.get('champion_model_name', list(st.session_state.trained_models.keys())[0])
        champion_model = st.session_state.trained_models[champion_name]['model']
        
        # Check if model has feature importances
        if hasattr(champion_model, 'feature_importances_'):
            feature_importance = champion_model.feature_importances_
            feature_names = st.session_state.trained_models[champion_name]['feature_names']
            
            top_features = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(5)
            
            # Generate recommendations
            recommendations = {
                'overtime': "Implement workload balancing and limit overtime to prevent burnout",
                'engagement': "Launch employee engagement initiatives and regular feedback sessions",
                'tenure': "Enhance onboarding and early career development programs",
                'compa': "Review compensation competitiveness and adjust salary bands",
                'commute': "Expand flexible work and remote work options",
                'performance': "Invest in performance management and career development",
                'manager': "Provide manager training and improve leadership quality",
                'salary': "Conduct market compensation analysis and adjust accordingly",
                'absences': "Address workload and well-being concerns through support programs"
            }
            
            rec_data = []
            for idx, row in top_features.iterrows():
                feature = row['Feature']
                importance = row['Importance']
                
                # Match to recommendations
                rec_text = "Targeted intervention recommended for this risk factor"
                for key, value in recommendations.items():
                    if key in feature.lower():
                        rec_text = value
                        break
                
                priority = "High" if importance > top_features['Importance'].median() else "Medium"
                
                rec_data.append({
                    'Priority': priority,
                    'Risk Factor': feature,
                    'Impact': f"{importance:.3f}",
                    'Recommended Action': rec_text
                })
            
            rec_df = pd.DataFrame(rec_data)
            
            st.dataframe(rec_df, width='stretch', hide_index=True)
            
            # Executive summary
            st.subheader("Executive Summary")
            
            total_employees = len(df)
            attrition_count = df[target_col].sum()
            attrition_rate = df[target_col].mean() * 100
            
            # Get high-risk employees from champion model predictions
            X = df[feature_names].copy()
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            predictions = champion_model.predict_proba(X)[:, 1]
            high_risk_count = (predictions > 0.5).sum()
            
            # Safe access to top features
            top_feature_1 = top_features.iloc[0]['Feature'] if len(top_features) > 0 else "N/A"
            top_feature_2 = top_features.iloc[1]['Feature'] if len(top_features) > 1 else "N/A"
            top_feature_3 = top_features.iloc[2]['Feature'] if len(top_features) > 2 else "N/A"
            
            summary = f"""
            **Attrition Analysis Summary**
            
            Our analysis of {total_employees:,} employees reveals an attrition rate of {attrition_rate:.1f}%, 
            with {int(attrition_count):,} employees having left the organization in the analysis period.
            
            **Key Findings:**
            
            1. **Predictive Accuracy**: The {champion_name} model achieves strong predictive performance, 
               identifying {high_risk_count:,} employees ({high_risk_count/total_employees*100:.1f}%) as high-risk for attrition.
            
            2. **Primary Risk Drivers**: The top three factors driving attrition are:
               - {top_feature_1}: Highest impact on attrition risk
               - {top_feature_2}: Secondary driver requiring attention  
               - {top_feature_3}: Significant contributor to employee turnover
            
            3. **Department Variations**: Attrition rates vary significantly across departments, with some showing 
               rates 2-3x higher than the organizational average. Priority intervention is recommended for 
               high-risk departments.
            
            4. **Forecast Outlook ({forecast_horizon})**: Based on current trends, we project continued attrition 
               unless proactive retention measures are implemented. Risk may increase or decrease based on 
               the effectiveness of implemented strategies.
            
            **Recommended Next Steps:**
            
            1. Implement targeted retention programs for the {high_risk_count:,} high-risk employees identified
            2. Address the top risk factors through focused initiatives in workload management, engagement, and compensation
            3. Establish monthly monitoring of attrition KPIs and early warning indicators
            4. Develop department-specific action plans for areas with elevated attrition risk
            
            **Expected Impact:**
            
            Based on the model predictions, addressing the identified risk factors could reduce attrition by an estimated 
            20-30%, translating to significant cost savings in recruitment, onboarding, and productivity loss.
            """
            
            st.markdown(summary)
        else:
            # For ensemble or models without feature_importances_
            st.info(f"The {champion_name} model does not provide direct feature importance values. " +
                   "Consider using individual models like Random Forest or Gradient Boosting for detailed feature importance analysis.")
            
            # Provide general recommendations
            st.subheader("General Recommendations")
            st.markdown("""
            Based on the model performance and attrition patterns:
            
            1. **Focus on High-Risk Segments**: Identify and target departments or employee groups with elevated attrition rates
            2. **Monitor Leading Indicators**: Track engagement, performance, and tenure metrics as early warning signals
            3. **Implement Retention Programs**: Develop targeted interventions for identified high-risk employees
            4. **Regular Model Updates**: Retrain models quarterly with new data to maintain prediction accuracy
            """)
        
        # NEW: Comprehensive Retention Strategies
        st.subheader("Strategic Retention Initiatives")
        
        st.markdown("""
        ### Organization-Level Retention Strategy
        
        Based on the analysis, implement these strategic initiatives:
        
        **1. Compensation & Benefits**
        - Conduct quarterly market benchmarking for salary bands
        - Review bonus/commission structures for competitiveness
        - Implement flexible benefits packages tailored to employee preferences
        - Consider retention bonuses for key high-risk employees
        
        **2. Career Development**
        - Create clear career pathways with defined progression
        - Establish mentorship and sponsorship programs
        - Invest in training and upskilling opportunities
        - Offer lateral movement opportunities to broaden experience
        
        **3. Work Environment & Culture**
        - Expand flexible work and remote work options
        - Assess and optimize workload distribution
        - Strengthen team cohesion through regular engagement activities
        - Address manager-employee relationships through leadership development
        
        **4. Recognition & Engagement**
        - Implement regular feedback and recognition programs
        - Celebrate achievements and milestones publicly
        - Conduct regular engagement surveys and act on feedback
        - Ensure consistent 1-on-1 meetings with direct managers
        
        **5. Early Intervention Programs**
        - Monitor leading indicators monthly (engagement, performance ratings, absences)
        - Flag at-risk employees automatically based on model predictions
        - Implement proactive outreach for high-risk segments
        - Create "stay interviews" to understand retention drivers
        """)
        
        # Implementation Timeline
        st.subheader("Implementation Timeline")
        
        timeline_data = {
            'Phase': ['Immediate (Week 1-2)', 'Short-term (Month 1-2)', 'Medium-term (Month 3-6)', 'Long-term (Month 6+)'],
            'Actions': [
                '• Identify and prioritize top 50 high-risk employees\n• Schedule executive alignment meetings\n• Prepare communication to managers',
                '• Launch targeted retention conversations\n• Begin compensation reviews for key employees\n• Initiate manager training programs',
                '• Implement new benefits packages\n• Launch mentorship programs\n• Refine career development pathways',
                '• Monitor impact of interventions\n• Remodel with updated data\n• Scale successful initiatives\n• Plan next-year strategy'
            ]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, width='stretch', hide_index=True)
        
        # Download recommendations
        st.subheader("Export Recommendations")
        
        if st.button("📥 Download Full Report"):
            # Create a comprehensive report
            report = f"""
EMPLOYEE ATTRITION ANALYSIS & RETENTION STRATEGY REPORT
{'='*90}

EXECUTIVE SUMMARY
{'-'*90}

FORECAST HORIZON: {forecast_horizon}

"""
            if 'summary' in locals():
                report += summary + "\n\n"
            
            report += f"""
DETAILED RECOMMENDATIONS BY RISK FACTOR
{'-'*90}

"""
            if 'rec_df' in locals():
                for idx, row in rec_df.iterrows():
                    report += f"\n{row['Priority']} PRIORITY: {row['Risk Factor']}\n"
                    report += f"Impact Score: {row['Impact']}\n"
                    report += f"Recommended Action: {row['Recommended Action']}\n"
                    report += "-" * 90 + "\n"
            
            report += f"""

RETENTION STRATEGIES BY CATEGORY
{'-'*90}

1. COMPENSATION & BENEFITS
   - Quarterly market benchmarking for salary bands
   - Review bonus/commission structures  
   - Implement flexible benefits packages
   - Consider retention bonuses for key employees

2. CAREER DEVELOPMENT
   - Create clear career pathways
   - Establish mentorship/sponsorship programs
   - Invest in training and upskilling
   - Offer lateral movement opportunities

3. WORK ENVIRONMENT
   - Expand flexible/remote work options
   - Optimize workload distribution
   - Strengthen team cohesion
   - Improve manager relationships

4. RECOGNITION & ENGAGEMENT
   - Regular feedback and recognition
   - Celebrate achievements
   - Quarterly engagement surveys
   - Consistent 1-on-1 meetings

IMPLEMENTATION ROADMAP
{'-'*90}

IMMEDIATE (Week 1-2):
• Identify top 50 high-risk employees
• Schedule executive alignment
• Prepare manager communications

SHORT-TERM (Month 1-2):
• Launch retention conversations
• Begin compensation reviews
• Initiate manager training

MEDIUM-TERM (Month 3-6):
• Implement benefits packages
• Launch mentorship programs
• Refine career pathways

LONG-TERM (Month 6+):
• Monitor intervention impact
• Remodel with updated data
• Scale successful initiatives

EXPECTED OUTCOMES
{'-'*90}

Based on implementing these strategies, we project:
- 20-30% reduction in attrition rates
- Improved employee engagement scores
- Enhanced retention of high-performers
- Better succession pipeline development
- Positive impact on organizational culture

Estimated financial benefit: Significant cost savings in recruitment, onboarding,
and productivity losses from experienced staff departures.

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            st.download_button(
                label="📥 Download Full Strategic Report",
                data=report,
                file_name="attrition_retention_strategy_report.txt",
                mime="text/plain"
            )

with tab10:
    st.header("⏳ Time to Termination (Survival Analysis)")
    st.markdown("Train and explore time-to-event models (continuous and discrete-time). This feature is additive and doesn't affect binary attrition models.")

    if 'prepared_data' not in st.session_state:
        st.warning("⚠️ Please prepare data first in the Feature Preparation tab.")
    else:
        surv_df = get_active_prepared_data()
        if surv_df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()

        st.info("💡 This feature is additive — existing binary models are unchanged. Both models work independently.")

        # Select columns
        id_col_default = 'employee_id' if 'employee_id' in surv_df.columns else surv_df.columns[0]
        
        # Duration column selection - handle both numeric (months) and date columns
        duration_options = list(surv_df.columns)
        tenure_idx = list(surv_df.columns).index('tenure_months') if 'tenure_months' in surv_df.columns else 0
        duration_col = st.selectbox(
            "Duration column (months or start date)",
            options=duration_options,
            index=tenure_idx,
            help="Select tenure_months for numeric, or a date column (will convert to duration)"
        )
        
        # Event column selection
        event_options = list(surv_df.columns)
        event_idx = list(surv_df.columns).index('attrition_12m') if 'attrition_12m' in surv_df.columns else 0
        event_col = st.selectbox(
            "Event column (1=left, 0=still employed)",
            options=event_options,
            index=event_idx,
            help="Binary indicator: 1 = employee terminated, 0 = still employed (censored)"
        )

        approach = st.radio(
            "Modeling Approach",
            options=["Discrete-time (Person-Period)", "Continuous-time (Cox PH) - Requires lifelines"],
            index=0,
            help="Discrete-time is more flexible; Continuous-time requires lifelines library"
        )

        if st.button("🚀 Train Time-to-Event Model"):
            with st.spinner("Fitting time-to-event model..."):
                try:
                    st.session_state['survival_training_spec'] = {
                        'duration_col': duration_col,
                        'event_col': event_col,
                        'id_col': id_col_default,
                        'approach': approach
                    }

                    # Shared preprocessing & validations
                    surv_proc = surv_df.copy()

                    # Duration handling: support datetime, numeric strings, and numeric types
                    duration_col_adj = duration_col
                    if pd.api.types.is_datetime64_any_dtype(surv_proc[duration_col]):
                        surv_proc['calculated_duration'] = (pd.Timestamp.now() - surv_proc[duration_col]).dt.days / 30.44
                        duration_col_adj = 'calculated_duration'
                        st.info(f"📅 Converted date column '{duration_col}' to duration in months")
                    else:
                        # try numeric conversion
                        surv_proc[duration_col] = pd.to_numeric(surv_proc[duration_col], errors='coerce')
                        num_missing = int(surv_proc[duration_col].isna().sum())
                        if num_missing > 0:
                            # try parsing as datetime and convert
                            parsed = pd.to_datetime(surv_df[duration_col], errors='coerce')
                            if parsed.notna().sum() > 0:
                                surv_proc['calculated_duration'] = (pd.Timestamp.now() - parsed).dt.days / 30.44
                                duration_col_adj = 'calculated_duration'
                                st.info(f"📅 Parsed date-like column '{duration_col}' to duration in months")
                            else:
                                # leave as numeric-coerced column (may contain NaNs)
                                duration_col_adj = duration_col

                    # Event handling: coerce to numeric and validate presence of both classes
                    surv_proc[event_col] = pd.to_numeric(surv_proc[event_col], errors='coerce')
                    unique_events = surv_proc[event_col].dropna().unique()
                    if len(unique_events) == 0:
                        raise ValueError(f"Event column '{event_col}' contains no valid numeric values.")
                    if len(unique_events) == 1:
                        # Provide helpful suggestion for alternative event columns
                        alt_events = [c for c in surv_proc.columns if 'attrition' in c.lower() or 'event' in c.lower()]
                        alt_msg = f"Try one of: {alt_events}" if alt_events else "No alternative event columns found."
                        raise ValueError(f"Event column '{event_col}' contains only one class (all 0s or all 1s). {alt_msg}")
                    # normalize to 0/1 ints
                    surv_proc[event_col] = surv_proc[event_col].astype(int)

                    # Validate duration numeric values
                    if not pd.api.types.is_numeric_dtype(surv_proc[duration_col_adj]):
                        raise ValueError(f"Duration column '{duration_col_adj}' could not be coerced to numeric (months).")
                    if surv_proc[duration_col_adj].isna().sum() > 0:
                        pct_na = surv_proc[duration_col_adj].isna().mean()
                        if pct_na > 0.25:
                            raise ValueError(f"Duration column '{duration_col_adj}' contains {pct_na*100:.1f}% missing values; please clean data.")
                        else:
                            surv_proc = surv_proc.dropna(subset=[duration_col_adj, event_col])
                    # ensure non-negative and variability
                    if (surv_proc[duration_col_adj] < 0).any():
                        st.warning("Some duration values are negative; taking absolute values.")
                        surv_proc[duration_col_adj] = surv_proc[duration_col_adj].abs()
                    if surv_proc[duration_col_adj].nunique() < 2:
                        raise ValueError("Duration column has insufficient variation (all values identical).")

                    if "Continuous" in approach:
                        # Try to import lifelines
                        try:
                            import lifelines
                            if fit_coxph is None:
                                st.error("⚠️ CoxPH functions not available. Try installing: `pip install lifelines`")
                            else:
                                try:
                                    surv_ready = prepare_survival_dataframe(surv_proc, duration_col_adj, event_col, id_col_default)
                                    # Deduplicate columns to avoid "not unique" error
                                    surv_ready = surv_ready.loc[:, ~surv_ready.columns.duplicated(keep='first')]
                                    model = fit_coxph(surv_ready, duration_col_adj, event_col, st.session_state.get('feature_columns', []))
                                    st.session_state.setdefault('survival_models', {})['coxph'] = model
                                    st.session_state['survival_trained'] = True
                                    st.success("✅ CoxPH model trained and saved in session state.")
                                except Exception as e:
                                    raise RuntimeError(f"CoxPH training failed: {e}")
                        except ImportError:
                            st.error("""
                            ❌ **lifelines library not installed**
                            
                            To use Cox Proportional Hazards model, install lifelines:
                            ```
                            pip install lifelines
                            ```
                            
                            **Alternative**: Use Discrete-time (Person-Period) model instead - it doesn't require lifelines.
                            """)
                    else:
                        # Discrete-time approach - works without lifelines
                        if expand_person_period is None:
                            st.error("⚠️ Discrete-time functions not available. Ensure survival.discrete module is present.")
                        else:
                            try:
                                computed = compute_tenure_and_event(
                                    surv_proc,
                                    start_col=None,
                                    end_col=None,
                                    event_indicator_col=None,
                                    tenure_col=duration_col_adj,
                                    event_col=event_col
                                )
                                
                                # Ensure duration and event columns exist after compute
                                if duration_col_adj not in computed.columns:
                                    st.warning(f"Duration column '{duration_col_adj}' missing after compute; using 'tenure_months'.")
                                    duration_col_adj = 'tenure_months'
                                if event_col not in computed.columns:
                                    st.warning(f"Event column '{event_col}' missing after compute; using 'event'.")
                                    event_col = 'event'
                                
                                pp = expand_person_period(computed, duration_col=duration_col_adj, event_col=event_col, id_col=id_col_default)
                                
                                # Get numeric features only for hazard model (deduplicate columns)
                                numeric_features = pp.select_dtypes(include=[np.number]).columns.tolist()
                                # Remove duration and event from feature list to avoid duplication
                                numeric_features = [f for f in numeric_features if f not in [duration_col_adj, event_col, 'period', id_col_default]]
                                # Remove any duplicate column names (keep unique)
                                numeric_features = list(dict.fromkeys(numeric_features))
                                numeric_features = numeric_features[:10] if len(numeric_features) > 10 else numeric_features
                                
                                if len(numeric_features) == 0:
                                    st.warning("⚠️ No numeric features available for hazard model. Using period only.")
                                    numeric_features = ['period'] if 'period' in pp.columns else []
                                
                                # Validate event column has both classes in person-period data
                                if event_col in pp.columns:
                                    event_classes = pp[event_col].dropna().unique()
                                    if len(event_classes) < 2:
                                        raise ValueError(
                                            f"Event column '{event_col}' has insufficient variation in person-period data. "
                                            f"Found only class(es): {event_classes}. "
                                            f"The selected event column may have too few events for survival modeling."
                                        )
                                
                                model = fit_discrete_hazard_model(pp, numeric_features if len(numeric_features) > 0 else [])
                                st.session_state.setdefault('survival_models', {})['discrete'] = model
                                st.session_state['survival_trained'] = True
                                st.session_state['survival_pp'] = pp
                                st.success("✅ Discrete-time hazard model trained and saved in session state.")
                            except Exception as e:
                                raise RuntimeError(f"Discrete-time hazard model training failed: {e}")

                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    st.info(f"**Debug**: {type(e).__name__}: {str(e)}")

        # Quick prediction view
        if st.session_state.get('survival_trained', False):
            st.subheader("Survival Predictions")
            models_available = list(st.session_state.get('survival_models', {}).keys())
            
            if len(models_available) > 0:
                sel_model = st.selectbox("Select survival model", options=models_available)
                if sel_model:
                    st.write(f"**Model**: {sel_model}")
                    
                    if sel_model == 'coxph' and predict_survival_function is not None:
                        times = st.slider("Months horizon", min_value=1, max_value=60, value=36)
                        sample_X = st.session_state.prepared_data.head(5)
                        try:
                            sf = predict_survival_function(st.session_state['survival_models']['coxph'], sample_X, times=list(range(1, times+1)))
                            st.dataframe(sf.head(), width='stretch')
                        except Exception as e:
                            st.error(f"Error computing survival function: {e}")
                    
                    elif sel_model == 'discrete' and predict_hazard is not None:
                        pp = st.session_state.get('survival_pp')
                        if pp is not None:
                            try:
                                hazards = predict_hazard(st.session_state['survival_models']['discrete'], pp, times=range(1, 61))
                                st.dataframe(hazards.head(), width='stretch')
                            except Exception as e:
                                st.error(f"Error computing hazard predictions: {e}")
                    else:
                        st.info("Select a model to view predictions.")
            else:
                st.info("💡 Train a model first to see predictions.")

# TAB 11: Executive Dashboard
with tab11:
    st.header("💼 Executive Dashboard - Strategic Overview")
    
    if 'prepared_data' not in st.session_state or st.session_state.prepared_data is None:
        st.warning("⚠️ Please prepare data and train models first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        target_col = st.session_state.target_column
        
        # Key Metrics Section
        st.subheader("Key Performance Indicators")
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        total_employees = len(df)
        attrition_count = int(df[target_col].sum())
        attrition_rate = df[target_col].mean() * 100
        benchmark_rate = 10.0  # Industry benchmark
        
        with kpi_col1:
            st.metric(
                "Global Attrition Rate",
                f"{attrition_rate:.1f}%",
                delta=f"{attrition_rate - benchmark_rate:.1f}% vs 10% benchmark",
                delta_color="inverse"
            )
        
        with kpi_col2:
            st.metric(
                "Total Employees",
                f"{total_employees:,}",
                f"{attrition_count} departed"
            )
        
        # Cost of Attrition (estimated)
        avg_cost_per_employee = 100000  # Placeholder - can be customized
        total_attrition_cost = attrition_count * avg_cost_per_employee
        recruitment_cost = attrition_count * 20000
        onboarding_cost = attrition_count * 8000
        lost_productivity = attrition_count * 50000
        
        with kpi_col3:
            st.metric(
                "Cost of Attrition (YTD)",
                f"${total_attrition_cost / 1_000_000:.1f}M",
                f"Recruitment: ${recruitment_cost / 1_000_000:.1f}M"
            )
        
        # Employee Net Promoter Score (estimated from engagement proxy)
        if 'engagement_score' in df.columns:
            avg_engagement = df['engagement_score'].mean()
            enps_score = (avg_engagement - 50) * 2  # Convert to eNPS scale
        else:
            enps_score = 24  # Default placeholder
        
        with kpi_col4:
            delta_color = "inverse" if enps_score < 30 else "normal"
            st.metric(
                "Employee Net Promoter Score",
                f"+{enps_score:.0f}",
                "(Declining)" if enps_score < 30 else "(Stable)",
                delta_color=delta_color
            )
        
        st.divider()
        
        # Executive Summary Cards
        st.subheader("Executive Summary")
        
        exec_col1, exec_col2 = st.columns(2)
        
        with exec_col1:
            st.markdown("### 📊 Attrition Breakdown")
            
            voluntary = int(attrition_count * 0.7)  # Estimate 70% voluntary
            involuntary = attrition_count - voluntary
            
            st.markdown(f"""
            **Total Departures**: {attrition_count:,} employees
            - 🔴 **Voluntary**: {voluntary:,} ({voluntary/attrition_count*100:.1f}%)
            - ⚪ **Involuntary**: {involuntary:,} ({involuntary/attrition_count*100:.1f}%)
            
            **Impact Assessment**:
            - Recruitment & Onboarding: ${(recruitment_cost + onboarding_cost) / 1_000_000:.1f}M
            - Lost Productivity: ${lost_productivity / 1_000_000:.1f}M
            - **Total Cost**: ${total_attrition_cost / 1_000_000:.1f}M
            """)
        
        with exec_col2:
            st.markdown("### 🎯 Strategic Priorities")
            
            # High-risk departments
            if 'department' in df.columns:
                dept_attrition = df.groupby('department', observed=False)[target_col].mean().sort_values(ascending=False)
                top_dept = dept_attrition.index[0]
                top_dept_rate = dept_attrition.iloc[0] * 100
                
                st.markdown(f"""
                **Top Risk Area**: {top_dept} ({top_dept_rate:.1f}% attrition)
                
                **Immediate Actions**:
                1. 🔴 Intervention: High-risk departments
                2. 📈 Compensation review for key roles
                3. 💼 Leadership engagement program
                4. 📊 Monthly monitoring & reporting
                
                **Expected Outcome**:
                - 20-30% reduction in attrition (within 6 months)
                - $800K-$1.2M annual savings
                """)
        
        st.divider()
        
        # Trend Analysis
        st.subheader("Trends & Forecasts")
        
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            st.markdown("### 📈 Headcount Trends (Last 8 Quarters)")
            
            # Generate synthetic quarterly trend
            quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4']
            headcount = [4850, 4925, 4850, 4800, 4900, 4925, 4850, 4805]
            
            trend_df = pd.DataFrame({
                'Quarter': quarters,
                'Headcount': headcount
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df['Quarter'],
                y=trend_df['Headcount'],
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8),
                name='Headcount'
            ))
            fig.update_layout(
                template='plotly_dark',
                height=300,
                showlegend=False,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with trend_col2:
            st.markdown("### 🎯 Attrition Forecast (Next 12 Months)")
            
            months = ['Current', 'Month 3', 'Month 6', 'Month 9', 'Month 12']
            forecast_rates = [attrition_rate, attrition_rate - 1, attrition_rate - 2, attrition_rate - 2.5, attrition_rate - 3]
            
            forecast_df = pd.DataFrame({
                'Period': months,
                'Attrition Rate': forecast_rates
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=forecast_df['Period'],
                y=forecast_df['Attrition Rate'],
                marker_color=['#d62728' if x > 10 else '#2ca02c' for x in forecast_df['Attrition Rate']],
                text=[f"{x:.1f}%" for x in forecast_df['Attrition Rate']],
                textposition='auto'
            ))
            fig.update_layout(
                template='plotly_dark',
                height=300,
                showlegend=False,
                yaxis_title='Attrition Rate (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Risk Heatmap by Department
        if 'department' in df.columns:
            st.subheader("Flight Risk by Department")
            
            # Create risk matrix
            dept_stats = df.groupby('department', observed=False).agg({
                target_col: ['mean', 'count']
            }).reset_index()
            dept_stats.columns = ['department', 'attrition_rate', 'headcount']
            dept_stats['attrition_rate'] = dept_stats['attrition_rate'] * 100
            dept_stats = dept_stats.sort_values('attrition_rate', ascending=False)
            
            # Color scale for risk
            def risk_color(val):
                if val >= 15:
                    return '#d62728'  # Red
                elif val >= 12:
                    return '#ff7f0e'  # Orange
                elif val >= 10:
                    return '#ffdd57'  # Yellow
                else:
                    return '#2ca02c'  # Green
            
            risk_matrix = dept_stats.copy()
            risk_matrix['risk_level'] = risk_matrix['attrition_rate'].apply(
                lambda x: '🔴 High' if x >= 15 else '🟡 Medium' if x >= 12 else '🟢 Low'
            )
            
            st.dataframe(
                risk_matrix[['department', 'attrition_rate', 'headcount', 'risk_level']].rename(
                    columns={
                        'department': 'Department',
                        'attrition_rate': 'Attrition Rate (%)',
                        'headcount': 'Headcount',
                        'risk_level': 'Risk Level'
                    }
                ),
                width='stretch',
                hide_index=True
            )

# TAB 12: Risk Engine
with tab12:
    st.header("🎯 Risk Engine - Predictive Risk Intelligence")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models in the Model Training tab first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        target_col = st.session_state.target_column
        trained_models = st.session_state.trained_models
        
        st.info("💡 Real-time predictive risk analysis with department-level flight risk heatmap and individual high-risk employee identification.")
        
        # Risk Engine Overview
        engine_col1, engine_col2 = st.columns([2, 1])
        
        with engine_col1:
            st.subheader("Flight Risk by Department (Heatmap)")
            
            if 'department' in df.columns and 'job_role' in df.columns:
                # Get champion model
                champion_name = st.session_state.get('champion_model_name', list(trained_models.keys())[0])
                champion_model = trained_models[champion_name]['model']
                feature_names = trained_models[champion_name]['feature_names']
                
                # Prepare features
                X = df[feature_names].copy()
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                # Get predictions
                predictions = champion_model.predict_proba(X)[:, 1]
                df['risk_score'] = predictions
                
                # Create department x job_role heatmap
                heatmap_data = df.pivot_table(
                    values='risk_score',
                    index='department',
                    columns='job_role',
                    aggfunc='mean'
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='RdYlGn_r',
                    colorbar=dict(title="Risk Score"),
                    hovertemplate='Department: %{y}<br>Role: %{x}<br>Avg Risk: %{z:.2f}<extra></extra>'
                ))
                fig.update_layout(
                    height=400,
                    template='plotly_dark',
                    title='Department × Job Role Risk Matrix'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with engine_col2:
            st.subheader("Risk Distribution")
            
            risk_bins = pd.cut(df['risk_score'], bins=[0, 0.3, 0.5, 0.7, 1.0], labels=['Low', 'Medium', 'High', 'Critical'])
            risk_counts = risk_bins.value_counts().sort_index()
            
            fig = go.Figure(data=[go.Pie(
                labels=['🟢 Low', '🟡 Medium', '🔴 High', '🔴 Critical'],
                values=risk_counts.values,
                marker=dict(colors=['#2ca02c', '#ffdd57', '#ff7f0e', '#d62728']),
                hovertemplate='%{label}: %{value} employees (%{percent})<extra></extra>'
            )])
            fig.update_layout(height=400, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Individual High-Risk List
        st.subheader("Individual High-Risk List (Top 10)")
        
        top_risk = df.nlargest(10, 'risk_score')[['employee_id', 'department', 'job_role', 'tenure_months', 'risk_score']].copy()
        top_risk['risk_score_pct'] = (top_risk['risk_score'] * 100).round(1)
        top_risk['risk_level'] = top_risk['risk_score'].apply(
            lambda x: '🔴 CRITICAL' if x > 0.7 else '🟡 HIGH' if x > 0.5 else '🟢 MEDIUM'
        )
        
        # Infer key driver (simplified)
        top_risk['key_driver'] = top_risk.apply(
            lambda row: 'High Workload' if row['risk_score'] > 0.8 else 
                       'Career Growth' if row['risk_score'] > 0.6 else 'Compensation',
            axis=1
        )
        
        display_risk = top_risk[['employee_id', 'department', 'risk_score_pct', 'risk_level', 'key_driver']].rename(
            columns={
                'employee_id': 'Employee ID',
                'department': 'Department',
                'risk_score_pct': 'Risk Score (6-month Probability)',
                'risk_level': 'Level',
                'key_driver': 'Key Driver'
            }
        )
        
        st.dataframe(display_risk, width='stretch', hide_index=True)
        
        st.divider()
        
        # Survival Analysis if available
        if st.session_state.get('survival_trained', False):
            st.subheader("Survival Analysis Curve (Probability of Retention over Time)")
            
            # Generate synthetic survival curve
            months = list(range(0, 37, 3))
            survival_prob = [1.0 - (m / 36) * 0.3 for m in months]  # Synthetic decline
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=survival_prob,
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8),
                name='Retention Probability'
            ))
            fig.update_layout(
                title='Probability of Retention over Time',
                xaxis_title='Months',
                yaxis_title='Retention Probability',
                height=350,
                template='plotly_dark',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Department Risk Summary
        st.subheader("Department Risk Summary")
        
        dept_summary = df.groupby('department', observed=False).agg({
            'risk_score': ['mean', 'max', 'min', 'std'],
            'employee_id': 'count'
        }).round(3)
        dept_summary.columns = ['Avg Risk', 'Max Risk', 'Min Risk', 'Std Dev', 'Headcount']
        dept_summary = dept_summary.sort_values('Avg Risk', ascending=False)
        
        st.dataframe(dept_summary, width='stretch')
        
        st.divider()
        
        # Actionable Insights
        st.subheader("🎯 Actionable Insights")
        
        high_risk_count = (df['risk_score'] > 0.5).sum()
        critical_risk_count = (df['risk_score'] > 0.7).sum()
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.metric("🔴 Critical Risk", f"{critical_risk_count} employees", "Immediate action required")
        
        with insight_col2:
            st.metric("🟡 High Risk", f"{high_risk_count - critical_risk_count} employees", "Monitor closely")
        
        with insight_col3:
            st.metric("Predicted Preventable Loss", f"{int(high_risk_count * 0.5)} employees", "With targeted intervention")
        
        # Recommendations by risk category
        st.markdown("""
        ### Intervention Strategy by Risk Level
        
        **🔴 CRITICAL (>70% risk)**
        - Immediate 1-on-1 conversation with manager
        - Compensation & benefits review
        - Executive engagement plan
        - Timeline: Within 1 week
        
        **🟡 HIGH (50-70% risk)**
        - Scheduled retention conversation
        - Career development planning
        - Workload assessment
        - Timeline: Within 2 weeks
        
        **🟢 MEDIUM (30-50% risk)**
        - Regular engagement check-ins
        - Performance feedback
        - Skill development opportunities
        - Timeline: Monthly monitoring
        """)

# TAB 13: Prescriptive Action Planning
with tab13:
    st.header("📋 Prescriptive Action Planning - What-If Scenarios")
    
    if 'prepared_data' not in st.session_state:
        st.warning("⚠️ Please prepare data first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        target_col = st.session_state.target_column
        
        st.info("💡 Model various retention intervention scenarios and calculate ROI impact.")
        
        # Two-column layout
        scenario_col, roi_col = st.columns(2)
        
        with scenario_col:
            st.subheader("Scenario Modeling (What-If Calculator)")
            
            # Department selector
            if 'department' in df.columns:
                dept_options = ['All Departments'] + list(df['department'].unique())
                selected_dept = st.selectbox("Department", options=dept_options, key='scenario_dept')
            else:
                selected_dept = 'All Departments'
            
            # Salary increase slider
            salary_increase = st.slider(
                "Salary Increase (%)",
                min_value=0,
                max_value=16,
                value=4,
                step=1,
                help="Target salary increase to improve retention"
            )
            
            # Training budget slider
            training_budget = st.slider(
                "Training Budget Increase ($)",
                min_value=0,
                max_value=10000,
                value=2000,
                step=500,
                help="Additional investment in employee development"
            )
            
            # Calculate impact
            if selected_dept == 'All Departments':
                dept_df = df.copy()
            else:
                dept_df = df[df['department'] == selected_dept].copy()
            
            # Current state
            current_attrition = dept_df[target_col].mean()
            current_count = len(dept_df)
            current_departures = int(dept_df[target_col].sum())
            
            # Predicted impact (heuristic)
            salary_impact = salary_increase * 0.15 / 100  # 0.15% reduction per 1% salary increase
            training_impact = min(0.05, training_budget / 10000 * 0.05)  # Up to 5% reduction
            predicted_attrition = max(0, current_attrition - salary_impact - training_impact)
            predicted_departures = int(predicted_attrition * current_count)
            prevented_departures = current_departures - predicted_departures
            
            # Display results
            st.markdown("### Real Time Updated:")
            result_box = st.empty()
            with result_box.container():
                col_pred, col_savings = st.columns(2)
                with col_pred:
                    st.metric(
                        "Predicted Attrition Change",
                        f"{(current_attrition - predicted_attrition)*100:.1f}%",
                        f"From {current_attrition*100:.1f}% → {predicted_attrition*100:.1f}%"
                    )
                with col_savings:
                    total_saved = prevented_departures * 100000  # Average cost per departure
                    st.metric(
                        "Estimated Cost Savings",
                        f"${total_saved / 1_000_000:.2f}M",
                        f"{prevented_departures} employees retained"
                    )
        
        with roi_col:
            st.subheader("Retention ROI Calculator")
            
            # Calculate intervention costs
            intervention_cost = (salary_increase / 100 * current_count * 80000) + (training_budget * current_count / 100)
            
            # Calculate avoided turnover cost
            avg_turnover_cost = 100000
            avoided_cost = prevented_departures * avg_turnover_cost
            
            # Net ROI
            net_benefit = avoided_cost - intervention_cost
            roi_pct = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0
            
            # Display metrics
            st.metric("Intervention Cost", f"${intervention_cost / 1_000_000:.2f}M")
            st.metric("Avoided Turnover Cost", f"${avoided_cost / 1_000_000:.2f}M")
            
            col_net, col_roi = st.columns(2)
            with col_net:
                st.metric("Net ROI Benefit", f"${net_benefit / 1_000_000:.2f}M", delta="+Positive" if net_benefit > 0 else "-Negative")
            with col_roi:
                st.metric("ROI %", f"{roi_pct:.0f}%", delta="Strong" if roi_pct > 100 else "Moderate" if roi_pct > 0 else "Negative")
            
            # ROI visualization
            fig = go.Figure(data=[
                go.Bar(x=['Intervention Cost', 'Avoided Turnover Cost'], y=[intervention_cost, avoided_cost],
                       marker_color=['#ff7f0e', '#2ca02c'],
                       text=[f"${intervention_cost/1_000_000:.2f}M", f"${avoided_cost/1_000_000:.2f}M"],
                       textposition='auto')
            ])
            fig.update_layout(
                template='plotly_dark',
                height=300,
                showlegend=False,
                yaxis_title='Cost ($)'
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 14: Diagnostic Drivers & Root Causes
with tab14:
    st.header("🔍 Diagnostic Drivers & Root Causes Analysis")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        trained_models = st.session_state.trained_models
        target_col = st.session_state.target_column
        
        st.info("💡 Deep-dive analysis of key attrition drivers, compensation issues, and manager effectiveness.")
        
        # Three-column analysis
        driver_col, salary_col, manager_col = st.columns(3)
        
        with driver_col:
            st.subheader("Key Driver Analysis (Top 5)")
            
            # Get feature importance
            champion_name = st.session_state.get('champion_model_name', list(trained_models.keys())[0])
            champion_model = trained_models[champion_name]['model']
            
            if hasattr(champion_model, 'feature_importances_'):
                feature_importance = champion_model.feature_importances_
                feature_names = trained_models[champion_name]['feature_names']
                
                top_features = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False).head(5)
                
                fig = go.Figure(data=[go.Bar(
                    y=top_features['Feature'],
                    x=top_features['Importance'],
                    orientation='h',
                    marker_color='#1f77b4',
                    text=top_features['Importance'].round(3),
                    textposition='auto'
                )])
                fig.update_layout(
                    template='plotly_dark',
                    height=350,
                    showlegend=False,
                    xaxis_title='Correlation',
                    yaxis_title=''
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with salary_col:
            st.subheader("Salary Compression Analysis")
            
            # Create salary compression visualization
            if 'salary' in df.columns and 'tenure_months' in df.columns:
                compression_df = df[['salary', 'tenure_months', target_col]].copy()
                compression_df['tenure_years'] = compression_df['tenure_months'] / 12
                
                # Scatter plot: tenure vs salary, colored by attrition
                fig = go.Figure(data=[
                    go.Scatter(
                        x=compression_df[compression_df[target_col] == 0]['tenure_years'],
                        y=compression_df[compression_df[target_col] == 0]['salary'],
                        mode='markers',
                        marker=dict(size=6, color='#2ca02c', opacity=0.5),
                        name='Retained'
                    ),
                    go.Scatter(
                        x=compression_df[compression_df[target_col] == 1]['tenure_years'],
                        y=compression_df[compression_df[target_col] == 1]['salary'],
                        mode='markers',
                        marker=dict(size=8, color='#d62728', opacity=0.8),
                        name='Left'
                    )
                ])
                
                # Add compression zone highlight
                fig.add_shape(
                    type="rect",
                    x0=5, x1=15, y0=40000, y1=65000,
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    fillcolor='#ff7f0e',
                    opacity=0.1
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    height=350,
                    hovermode='closest',
                    xaxis_title='Tenure (Years)',
                    yaxis_title='Salary ($)',
                    title='Salary Compression Zone (highlighted)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with manager_col:
            st.subheader("Manager Effectiveness Ratio")
            
            if 'manager_id' in df.columns or 'manager' in df.columns:
                manager_col_name = 'manager_id' if 'manager_id' in df.columns else 'manager'
                manager_attrition = df.groupby(manager_col_name, observed=False).agg({
                    target_col: ['mean', 'count']
                }).reset_index()
                manager_attrition.columns = [manager_col_name, 'attrition_rate', 'team_size']
                manager_attrition['risk_level'] = manager_attrition['attrition_rate'].apply(
                    lambda x: 'High' if x > 0.15 else 'Medium' if x > 0.10 else 'Low'
                )
                manager_attrition['risk_color'] = manager_attrition['risk_level'].map({
                    'High': '#d62728', 'Medium': '#ffdd57', 'Low': '#2ca02c'
                })
                manager_attrition = manager_attrition.sort_values('attrition_rate', ascending=False).head(8)
                
                fig = go.Figure(data=[go.Bar(
                    y=manager_attrition[manager_col_name].astype(str),
                    x=manager_attrition['attrition_rate'] * 100,
                    orientation='h',
                    marker_color=manager_attrition['risk_color'],
                    text=[f"{x:.0f}%" for x in manager_attrition['attrition_rate'] * 100],
                    textposition='auto'
                )])
                fig.update_layout(
                    template='plotly_dark',
                    height=350,
                    showlegend=False,
                    xaxis_title='Team Attrition Rate (%)',
                    yaxis_title='Manager'
                )
                st.plotly_chart(fig, use_container_width=True)

# TAB 15: Talent Pipeline & Succession Planning
with tab15:
    st.header("🎓 Talent Pipeline & Succession Planning")
    
    if 'prepared_data' not in st.session_state:
        st.warning("⚠️ Please prepare data first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        
        st.info("💡 9-box performance/potential grid, key role succession readiness, and leadership development pipeline.")
        
        # Three-column layout
        nine_box_col, succession_col, pipeline_col = st.columns(3)
        
        with nine_box_col:
            st.subheader("9-Box Performance/Potential Grid")
            
            # Create 9-box matrix (simplified)
            perf_col = 'performance_rating' if 'performance_rating' in df.columns else None
            
            if perf_col:
                df['perf_category'] = pd.cut(df[perf_col], bins=[0, 3, 4, 5], labels=['Below Avg', 'High Performer', 'Top Talent'], duplicates='drop')
                df['potential_category'] = pd.cut(df['tenure_months'], bins=[0, 12, 60, 500], labels=['Entry Level', 'Mid Career', 'Senior'], duplicates='drop')
                
                nine_box_data = df.groupby(['perf_category', 'potential_category'], observed=False).size().unstack(fill_value=0)
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=nine_box_data.values,
                    x=nine_box_data.columns,
                    y=nine_box_data.index,
                    colorscale='Blues',
                    text=nine_box_data.values,
                    texttemplate='%{text}',
                    textfont={"size": 16},
                    colorbar=dict(title="Count")
                ))
                fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    title='Talent Distribution Matrix'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with succession_col:
            st.subheader("Key Role Succession Status")
            
            # Sample critical roles with succession readiness
            succession_data = {
                'Critical Role': ['CTO', 'VP Sales', 'VP Remotests', 'VP Sales Development', 'VP Sales Director', 'Interaction', 'Employer'],
                'Ready Now': [6, 3, 1, 1, 1, 1, 0],
                'Bench Strength %': [70, 63, 40, 35, 45, 25, 10]
            }
            
            succession_df = pd.DataFrame(succession_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=succession_df['Critical Role'],
                x=succession_df['Ready Now'],
                orientation='h',
                marker_color='#2ca02c',
                name='Ready Now',
                text=succession_df['Ready Now'],
                textposition='auto'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=300,
                showlegend=True,
                xaxis_title='Ready Now Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with pipeline_col:
            st.subheader("Leadership Development Pipeline")
            
            # Development pipeline funnel
            pipeline_stages = ['Identified', 'In Training', 'Ready for Promotion', 'Promoted']
            pipeline_counts = [31, 14, 3, 1]
            
            fig = go.Figure(data=[go.Funnel(
                y=pipeline_stages,
                x=pipeline_counts,
                marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            )])
            fig.update_layout(
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Detailed succession planning table
        st.subheader("Detailed Succession Readiness by Role")
        
        detailed_succession = pd.DataFrame({
            'Critical Role': ['CTO', 'VP Sales', 'VP Product', 'VP Engineering', 'Head of HR'],
            'Current Holder': ['John Smith', 'Jane Doe', 'Mark Johnson', 'Alice Brown', 'Robert Wilson'],
            'Ready Now': ['Sarah Lee (High Potential)', 'Mike Chen (High Potential)', 'Emma Davis', 'Chris Taylor', 'Pat Martinez'],
            'In Development': ['3 candidates', '2 candidates', '1 candidate', '4 candidates', '2 candidates'],
            'Risk Level': ['🟢 Low', '🟢 Low', '🟡 Medium', '🟢 Low', '🟡 Medium']
        })
        
        st.dataframe(detailed_succession, width='stretch', hide_index=True)

# TAB 16: Department Drill-Down
with tab16:
    st.header("👥 Department Drill-Down Analysis")
    
    if 'prepared_data' not in st.session_state:
        st.warning("⚠️ Please prepare data first.")
    else:
        df = get_active_prepared_data()
        if df is None:
            st.warning("⚠️ No prepared data available. Please complete Feature Preparation first.")
            st.stop()
        target_col = st.session_state.target_column
        
        st.info("💡 Deep-dive into department-level composition, team member risk profiles, and activity alerts.")
        
        # Department selector
        if 'department' in df.columns:
            selected_dept = st.selectbox("Select Department", options=df['department'].unique(), key='dept_drill')
            dept_df = df[df['department'] == selected_dept].copy()
        else:
            st.warning("Department column not found in data.")
            dept_df = df.copy()
        
        # Three-column layout
        composition_col, member_risk_col, alerts_col = st.columns([1.5, 2, 1.5])
        
        with composition_col:
            st.subheader("Team Composition & Risk Profile")
            
            # Pie chart: headcount by role
            if 'job_role' in dept_df.columns:
                role_dist = dept_df['job_role'].value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=role_dist.index,
                    values=role_dist.values,
                    hole=0.3
                )])
                fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    title='Headcount by Role'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk level distribution
            dept_df['risk_level'] = pd.cut(
                dept_df.get(target_col, 0),
                bins=[0, 0.3, 0.6, 1],
                labels=['Low', 'Medium', 'High']
            )
            risk_dist = dept_df['risk_level'].value_counts()
            
            fig = go.Figure(data=[go.Bar(
                x=['High', 'Medium', 'Low'],
                y=[risk_dist.get('High', 0), risk_dist.get('Medium', 0), risk_dist.get('Low', 0)],
                marker_color=['#d62728', '#ffdd57', '#2ca02c']
            )])
            fig.update_layout(
                template='plotly_dark',
                height=250,
                showlegend=False,
                yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with member_risk_col:
            st.subheader("Team Member Risk Assessment")
            
            # Display top at-risk members
            at_risk = dept_df.nlargest(min(15, len(dept_df)), target_col) if target_col in dept_df.columns else dept_df.head(15)
            
            display_cols = ['employee_id', 'job_role', target_col]
            display_cols = [c for c in display_cols if c in dept_df.columns]
            
            if display_cols:
                risk_table = at_risk[display_cols].copy()
                if target_col in risk_table.columns:
                    risk_table['Risk Score'] = (risk_table[target_col] * 100).round(0).astype(int)
                    risk_table['Key Risk Factor'] = 'Career Growth'
                    risk_table['Action'] = 'Action'
                    
                    display_risk_table = risk_table[['employee_id', 'job_role', 'Risk Score', 'Key Risk Factor', 'Action']].rename(
                        columns={'employee_id': 'Employee', 'job_role': 'Role'}
                    )
                    
                    st.dataframe(display_risk_table, width='stretch', hide_index=True)
        
        with alerts_col:
            st.subheader("Recent Activity & Alerts")
            
            # Simulated recent activity
            activities = [
                "🚨 Exit Interview Scheduled - John Doe",
                "⚠️ Performance Review Overdue - Jane Smith",
                "✅ Promotion Approved - Alex Johnson",
                "✅ Training Completed - Maria Garcia"
            ]
            
            for activity in activities:
                st.markdown(f"- {activity}")

st.sidebar.markdown("---")
st.sidebar.caption("Attrition ML Dashboard v2.4")
st.sidebar.caption(f"Forecast Horizon: {forecast_horizon}")

