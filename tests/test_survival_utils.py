import pandas as pd
import numpy as np
from survival.discrete import compute_tenure_and_event, expand_person_period, hazard_to_survival


def test_compute_tenure_and_event_from_tenure_and_flag():
    df = pd.DataFrame({
        'employee_id': [1,2],
        'tenure_months': [3,5],
        'attrition_12m': [1,0]
    })

    out = compute_tenure_and_event(df, tenure_col='tenure_months', event_col='event', event_indicator_col='attrition_12m')
    assert 'tenure_months' in out.columns
    assert 'event' in out.columns
    assert out.loc[0,'tenure_months'] == 3
    assert out.loc[0,'event'] == 1
    assert out.loc[1,'event'] == 0


def test_expand_person_period_basic():
    df = pd.DataFrame({
        'employee_id': [1,2],
        'tenure_months': [3,5],
        'event': [1,0],
        'age': [30,40]
    })

    pp = expand_person_period(df, duration_col='tenure_months', event_col='event', id_col='employee_id', max_periods=12)
    # employee 1 should have 3 rows, employee 2 should have 5 rows
    assert len(pp[pp['employee_id']==1]) == 3
    assert len(pp[pp['employee_id']==2]) == 5
    # event flag should be 1 only on last period for employee 1
    assert pp[(pp['employee_id']==1) & (pp['period']==3)]['event'].iloc[0] == 1
    assert pp[(pp['employee_id']==1) & (pp['period']<3)]['event'].sum() == 0


def test_hazard_to_survival():
    hazards = np.array([0.1, 0.2, 0.3])
    surv = hazard_to_survival(hazards)
    # S1 = 0.9, S2 = 0.9*0.8=0.72, S3=0.72*0.7=0.504
    assert np.isclose(surv[0], 0.9)
    assert np.isclose(surv[1], 0.72)
    assert np.isclose(surv[2], 0.504)
