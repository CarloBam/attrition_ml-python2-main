from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix
)
import pandas as pd
import numpy as np
import streamlit as st
from generate_sample import make_synthetic

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

# Sidebar controls
st.sidebar.header("Global Controls")

random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

# Data Upload Tab
st.title("🎯 Employee Attrition Prediction Workbench")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Data Upload",
    "📤 Bulk Training Data",
    "🔧 Feature Preparation",
    "🤖 Model Training",
    "📈 Performance",
    "📉 KPI Dashboard",
    "🌳 Segments",
    "👤 Individual Risk",
    "💡 Recommendations"
])

# TAB 1: Data Upload
with tab1:
    st.header("Data Source and Configuration")
    
    data_source = st.radio(
        "Select Data Source",
        options=["Upload CSV", "Use Sample Data", "Generate Synthetic Data"],
        horizontal=True
    )
    
    raw_df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload employee data CSV", type=['csv'])
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(raw_df)} rows from uploaded file")
    
    elif data_source == "Use Sample Data":
        try:
            raw_df = pd.read_csv("data/sample_attrition_data.csv")
            st.success(f"Loaded {len(raw_df)} rows from sample data")
        except FileNotFoundError:
            st.error("Sample data file not found. Please upload a CSV or generate synthetic data.")
    
    elif data_source == "Generate Synthetic Data":
        synth_size = st.slider("Number of employees", min_value=500, max_value=30000, value=2000, step=500)
        if st.button("Generate Data"):
            raw_df = make_synthetic(n=synth_size, seed=random_seed)
            st.success(f"Generated {len(raw_df)} synthetic employee records")
    
    if raw_df is not None:
        st.subheader("Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        # Store raw data in session state
        st.session_state.raw_data = raw_df.copy()
        st.session_state.data_loaded = True
        
        # Column configuration
        st.subheader("Column Configuration")
        # Additional column configuration logic can be added here

# Additional tabs for bulk training, feature preparation, model training, performance, KPI dashboard, segments, individual risk, and recommendations can be implemented similarly.