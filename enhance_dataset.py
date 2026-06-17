#!/usr/bin/env python
"""
Script to enhance the sample attrition dataset with additional features.
Run this once to add new columns to sample_attrition_data.csv
"""

import pandas as pd
import numpy as np

# Load the existing dataset
df = pd.read_csv('sample_attrition_data.csv')

print(f"📊 Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# === NEW COLUMNS TO ADD ===

# 1. Performance Category (based on performance_rating)
def categorize_performance(rating):
    if pd.isna(rating):
        return 'Unknown'
    if rating >= 4.5:
        return 'Top Performer'
    elif rating >= 3.5:
        return 'High Performer'
    elif rating >= 2.5:
        return 'Meets Expectations'
    else:
        return 'Below Expectations'

df['performance_category'] = df['performance_rating'].apply(categorize_performance)

# 2. Engagement Level (based on engagement_score)
def categorize_engagement(score):
    if pd.isna(score):
        return 'Unknown'
    if score >= 75:
        return 'Highly Engaged'
    elif score >= 60:
        return 'Engaged'
    elif score >= 40:
        return 'Neutral'
    else:
        return 'Disengaged'

df['engagement_level'] = df['engagement_score'].apply(categorize_engagement)

# 3. Risk Category (synthetic but realistic)
def calculate_risk(row):
    risk_score = 0
    
    # Low engagement increases risk
    if row.get('engagement_score', 50) < 40:
        risk_score += 3
    elif row.get('engagement_score', 50) < 60:
        risk_score += 1
    
    # Low performance increases risk
    if row.get('performance_rating', 3) < 3:
        risk_score += 2
    
    # High overtime/stress increases risk
    if row.get('overtime_hours_month', 0) > 30:
        risk_score += 1
    
    # Low compensation relative to market increases risk
    if row.get('compa_ratio', 1.0) < 0.8:
        risk_score += 2
    
    # Long tenure actually reduces risk (stability)
    if row.get('tenure_months', 50) > 100:
        risk_score -= 1
    
    # Recent manager change increases risk
    if row.get('manager_change_6m', 'No') == 'Yes':
        risk_score += 1
    
    # No promotions in recent years increases risk
    if row.get('num_promotions', 0) == 0 and row.get('tenure_months', 0) > 36:
        risk_score += 1
    
    risk_score = max(0, min(risk_score, 10))  # Clamp to 0-10
    
    if risk_score >= 7:
        return 'Critical'
    elif risk_score >= 5:
        return 'High'
    elif risk_score >= 3:
        return 'Medium'
    else:
        return 'Low'

df['attrition_risk'] = df.apply(calculate_risk, axis=1)

# 4. Career Stage (based on tenure and grade)
def categorize_career_stage(tenure, grade):
    if pd.isna(tenure) or pd.isna(grade):
        return 'Unknown'
    
    if tenure < 24:
        return 'Early Career (0-2 yrs)'
    elif tenure < 60:
        return 'Mid Career (2-5 yrs)'
    elif tenure < 120:
        return 'Senior (5-10 yrs)'
    else:
        return 'Veteran (10+ yrs)'

df['career_stage'] = df.apply(lambda row: categorize_career_stage(row['tenure_months'], row['grade']), axis=1)

# 5. Compensation Band (based on salary quartiles)
salary_quartiles = df['salary'].quantile([0.25, 0.5, 0.75])
def categorize_salary(sal):
    if pd.isna(sal):
        return 'Unknown'
    if sal <= salary_quartiles[0.25]:
        return 'Q1 (Entry)'
    elif sal <= salary_quartiles[0.5]:
        return 'Q2 (Mid-level)'
    elif sal <= salary_quartiles[0.75]:
        return 'Q3 (Senior)'
    else:
        return 'Q4 (Executive)'

df['compensation_band'] = df['salary'].apply(categorize_salary)

# 6. Work-Life Balance Risk (based on overtime, remote_ratio, leave_balance)
def assess_work_life_balance(row):
    score = 0
    
    # High overtime is bad
    if row.get('overtime_hours_month', 0) > 25:
        score += 2
    elif row.get('overtime_hours_month', 0) > 10:
        score += 1
    
    # Low remote ratio combined with high commute is bad
    if row.get('remote_ratio', 0) < 25 and row.get('commute_km', 0) > 40:
        score += 2
    
    # Low leave balance indicates possible burnout
    if row.get('leave_balance_days', 15) < 5:
        score += 2
    elif row.get('leave_balance_days', 15) < 10:
        score += 1
    
    if score >= 4:
        return 'Poor'
    elif score >= 2:
        return 'Fair'
    else:
        return 'Good'

df['work_life_balance'] = df.apply(assess_work_life_balance, axis=1)

# 7. Promotion Likelihood (based on performance, tenure, num_promotions)
def promotion_likelihood(row):
    likelihood = 0
    
    # High performers more likely to be promoted
    if row.get('performance_rating', 3) >= 4.5:
        likelihood += 3
    elif row.get('performance_rating', 3) >= 3.5:
        likelihood += 1
    
    # Mid-career stage is sweet spot for promotions
    tenure = row.get('tenure_months', 50)
    if 24 <= tenure <= 120:
        likelihood += 2
    
    # Already promoted employees more likely to be promoted again
    if row.get('num_promotions', 0) > 0:
        likelihood += 1
    
    # Grade A more likely to be promoted
    if row.get('grade', 'B') == 'A':
        likelihood += 1
    
    if likelihood >= 5:
        return 'Very High'
    elif likelihood >= 3:
        return 'High'
    elif likelihood >= 1:
        return 'Medium'
    else:
        return 'Low'

df['promotion_likelihood'] = df.apply(promotion_likelihood, axis=1)

# 8. Learning Investment (based on training_hours_last_yr)
def categorize_learning(hours):
    if pd.isna(hours):
        return 'Unknown'
    if hours >= 50:
        return 'High Investment'
    elif hours >= 25:
        return 'Moderate Investment'
    elif hours >= 10:
        return 'Low Investment'
    else:
        return 'Minimal Investment'

df['learning_investment'] = df['training_hours_last_yr'].apply(categorize_learning)

print(f"\n✅ Enhanced dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n📋 New columns added:")
print(f"   - performance_category")
print(f"   - engagement_level")
print(f"   - attrition_risk")
print(f"   - career_stage")
print(f"   - compensation_band")
print(f"   - work_life_balance")
print(f"   - promotion_likelihood")
print(f"   - learning_investment")

# Display sample of new columns
print("\n📊 Sample of new columns:")
print(df[['employee_id', 'performance_category', 'engagement_level', 'attrition_risk', 'career_stage', 'compensation_band']].head(10))

# Save enhanced dataset
df.to_csv('sample_attrition_data.csv', index=False)
print(f"\n✅ Enhanced dataset saved to sample_attrition_data.csv")

# Display summary statistics for new columns
print("\n📈 Distribution of new categorical features:")
for col in ['performance_category', 'engagement_level', 'attrition_risk', 'work_life_balance', 'promotion_likelihood']:
    print(f"\n{col}:")
    print(df[col].value_counts().to_string())
