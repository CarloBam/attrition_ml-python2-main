from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def plot_roc_curve(y_true: List[int], y_scores: List[float], model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    
    fig.update_layout(title=f'ROC Curve for {model_name}',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      template='plotly_dark')
    
    fig.show()

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, 
                                      x=['Predicted Stay', 'Predicted Leave'],
                                      y=['Actual Stay', 'Actual Leave'],
                                      colorscale='Blues'))
    
    fig.update_layout(title=f'Confusion Matrix for {model_name}',
                      xaxis_title='Predicted',
                      yaxis_title='Actual',
                      template='plotly_dark')
    
    fig.show()

def kpi_dashboard(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    total_employees = len(df)
    attrition_count = df[target_col].sum()
    attrition_rate = (attrition_count / total_employees) * 100
    retention_rate = 100 - attrition_rate
    
    kpi_metrics = {
        "Total Employees": total_employees,
        "Attrition Count": attrition_count,
        "Attrition Rate (%)": attrition_rate,
        "Retention Rate (%)": retention_rate
    }
    
    return kpi_metrics

def plot_attrition_trends(df: pd.DataFrame, target_col: str, date_col: str) -> None:
    df[date_col] = pd.to_datetime(df[date_col])
    trend_df = df.groupby(date_col).agg({target_col: ['sum', 'mean', 'count']}).reset_index()
    trend_df.columns = [date_col, 'attrition_count', 'attrition_rate', 'total']
    trend_df['attrition_rate'] = trend_df['attrition_rate'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df[date_col], y=trend_df['attrition_rate'], mode='lines+markers', name='Attrition Rate %'))
    fig.add_trace(go.Bar(x=trend_df[date_col], y=trend_df['attrition_count'], name='Attrition Count'))
    
    fig.update_layout(title='Attrition Trends Over Time',
                      xaxis_title='Date',
                      yaxis_title='Attrition Rate (%)',
                      template='plotly_dark')
    
    fig.show()