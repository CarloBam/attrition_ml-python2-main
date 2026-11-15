# Quick Start Guide - New Features

## 📁 Column Configuration (Now Persistent!)

### What Changed?
- Column include/exclude settings now **stay** when you navigate between tabs
- Data type selections are **remembered** for the session
- No more re-configuring after clicking "Set Configuration"

### How to Use:
1. Go to **Tab 1: Data Upload**
2. Upload your CSV or generate synthetic data
3. Expand each column you want to configure
4. Check/uncheck "Include in analysis" ✓ **This now persists!**
5. Change "Data Type" as needed ✓ **This now persists!**
6. View 5 sample values (increased from 3) ✓ **Better visibility!**
7. Click "✅ Set Configuration" 
8. Your settings are saved for the entire session

---

## ☁️ Load Data from Cloud (BigQuery, S3, Azure)

### Option 1: Google BigQuery

**Setup**:
```powershell
# Install
pip install google-cloud-bigquery

# Authenticate
gcloud auth application-default login
```

**In App**:
1. Go to **Tab 2: Bulk Training Data**
2. Select **"☁️ Google BigQuery"**
3. Enter:
   - **GCP Project ID**: `my-project-id`
   - **BigQuery Dataset**: `my_dataset`
   - **Table Name**: `employee_data`
4. Click **"Load from BigQuery"**
5. Watch progress bar: Connecting → Executing → Loaded ✓

**Example Query Generated**:
```sql
SELECT * FROM `my-project-id.my_dataset.employee_data` LIMIT 1000000
```

---

### Option 2: Amazon S3

**Setup**:
```powershell
# Install
pip install boto3

# Configure AWS credentials
aws configure
# Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

**In App**:
1. Go to **Tab 2: Bulk Training Data**
2. Select **"📦 Amazon S3"**
3. Enter:
   - **S3 Bucket**: `my-data-bucket`
   - **Object Key**: `hr-data/employees.csv`
4. Click **"Load from S3"**
5. Progress: Connecting → Downloading → Loaded ✓

**Supported Formats**: CSV, Excel, Parquet

---

### Option 3: Azure Blob Storage

**Setup**:
```powershell
# Install
pip install azure-storage-blob
```

**In App**:
1. Go to **Tab 2: Bulk Training Data**
2. Select **"☁️ Azure Blob"**
3. Enter:
   - **Connection String**: (from Azure Portal)
   - **Container**: `employee-data`
   - **Blob Name**: `attrition_data.csv`
4. Click **"Load from Azure"**
5. Progress: Connecting → Downloading → Loaded ✓

---

## 📊 Export Reports (New Multi-Format Support!)

### What Changed?
Previously: Only text export available
Now: 4 export formats! ✨

### Tab 9: Recommendations → Export Recommendations

#### 📄 Text Report
- Lightweight format
- Copy-paste friendly
- **No dependencies needed**
- Best for: Quick reference, email

#### 📊 Excel Report (.xlsx)
- Professional spreadsheet
- Multiple sheets:
  - Summary (key metrics)
  - Recommendations (action items)
- Requires: `pip install openpyxl`
- Best for: Data teams, analysis, sharing

#### 📄 Word Report (.docx)
- Formatted document
- Professional appearance
- Tables and structured content
- Requires: `pip install python-docx`
- Best for: Sharing with executives, email distribution

#### 📊 PDF Report
- Print-ready format
- Professional styling
- Easy to archive
- Requires: `pip install reportlab`
- Best for: Formal presentations, compliance

### Export Workflow:
1. Train models (Tab 4)
2. Evaluate performance (Tab 5)
3. Go to **Tab 9: Recommendations**
4. Click desired export button:
   - **📥 Text** → Downloads .txt
   - **📥 Excel** → Downloads .xlsx
   - **📥 Word** → Downloads .docx
   - **📥 PDF** → Downloads .pdf
5. File saves to your Downloads folder

### Exported Report Contains:
- Forecast horizon
- Total employee count
- Attrition rate percentage
- High-risk employee count
- Detailed recommendations by priority
- Risk factors and impact scores
- Actionable next steps

---

## 🎯 Training/Test Split (Now More Flexible!)

### What Changed?
- **Old Range**: 50% training → 90% training (50% test → 10% test)
- **New Range**: 10% training → 70% training (90% test → 30% test)

### How to Use:
In **Tab 4: Model Training**:
1. Select your features
2. Adjust "Training set size" slider:
   - **Small models** (testing quickly): 10-20% training
   - **Balanced approach** (default): 70% training, 30% test
   - **Data-rich scenario** (large dataset): 50-70% training
3. Choose "Stratify by target" if classes are imbalanced
4. Click "🚀 Train Models"

**Example Configurations**:
| Use Case | Train Size | Test Size | Reason |
|----------|-----------|-----------|--------|
| Quick test | 10% | 90% | Fast iteration |
| Production | 70% | 30% | Robust models |
| Data-rich | 60% | 40% | More validation |
| Few samples | 50% | 50% | Maximum generalization |

---

## 🔧 Installation Quick Reference

### All Features (Recommended):
```powershell
# From project directory
pip install -e .

# Or manually:
pip install streamlit pandas numpy scikit-learn
pip install plotly altair matplotlib
pip install lifelines
pip install openpyxl python-docx reportlab
pip install google-cloud-bigquery boto3 azure-storage-blob
```

### Minimal (Local files only):
```powershell
pip install streamlit pandas numpy scikit-learn plotly altair matplotlib
```

### Add Later (Optional):
```powershell
# When you need cloud access
pip install google-cloud-bigquery      # For BigQuery
pip install boto3                      # For S3
pip install azure-storage-blob         # For Azure

# When you need export formats
pip install openpyxl                   # For Excel
pip install python-docx                # For Word
pip install reportlab                  # For PDF
```

---

## 🚀 Run & Test

```powershell
# Navigate to project
cd <project-folder>

# Run app
streamlit run app.py

# Open in browser (usually auto-opens)
# http://localhost:8501
```

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] **Tab 1**: Column config persists when switching tabs
- [ ] **Tab 1**: Sample values show 5 rows (not 3)
- [ ] **Tab 2**: Can select BigQuery, S3, or Azure options
- [ ] **Tab 2**: Progress bar shows during file load
- [ ] **Tab 4**: Train/test slider goes from 10% to 70%
- [ ] **Tab 9**: Four export buttons visible (Text, Excel, Word, PDF)
- [ ] **Tab 9**: Export produces correct format files

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "openpyxl not installed" | `pip install openpyxl` |
| "python-docx not installed" | `pip install python-docx` |
| "reportlab not installed" | `pip install reportlab` |
| BigQuery connection fails | Run `gcloud auth application-default login` |
| S3 connection fails | Run `aws configure` |
| Column config not saving | Make sure to click "Set Configuration" button |
| Sample values show only 3 | Update to latest app.py (should show 5) |

---

## 📚 Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Pandas Documentation**: https://pandas.pydata.org/docs
- **Scikit-learn**: https://scikit-learn.org/stable
- **BigQuery**: https://cloud.google.com/bigquery/docs
- **AWS S3**: https://docs.aws.amazon.com/s3/
- **Azure Blob**: https://docs.microsoft.com/azure/storage/blobs

---

**Last Updated**: November 26, 2025
**Version**: 2.1.0
