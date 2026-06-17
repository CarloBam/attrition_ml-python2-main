from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def train_logistic_regression(X, y, random_seed=42):
    model = LogisticRegression(random_state=random_seed, max_iter=2000)
    model.fit(X, y)
    return model

def train_random_forest(X, y, random_seed=42):
    model = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
    model.fit(X, y)
    return model

def train_gradient_boosting(X, y, random_seed=42):
    model = GradientBoostingClassifier(n_estimators=100, random_state=random_seed)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics