# Employee Attrition Prediction Workbench

A Streamlit app for exploring why employees leave and who is at risk, built on a synthetic dataset of 2,500 employees.

## Why I built this

I work in people analytics, where attrition is a recurring question. I wanted one place to try the full range of approaches on the same data: classic classification models that ask *whether* someone will leave, and survival analysis that asks *when*.

## What it does

- **Data:** use the bundled sample, generate a fresh synthetic dataset, or upload your own CSV. Optional connectors for BigQuery, Amazon S3 and Azure Blob are included but disabled in the hosted demo.
- **Preparation:** missing-value treatment, encoding and target selection.
- **Models:** logistic regression (with optional p-value based feature selection), decision tree, random forest, gradient boosting, and a soft or hard voting ensemble, compared side by side.
- **Survival analysis:** a Cox proportional hazards model and a discrete-time hazard model, which estimate how risk changes with tenure rather than giving a single yes or no.
- **Explaining results:** a decision tree that segments the workforce into risk groups, an individual risk explorer, driver analysis, department drill-downs and what-if scenarios.

## The data

All data is synthetic. The generator creates realistic HR fields such as tenure, salary, compa-ratio, overtime, performance, engagement and commute distance, with South African locations. No real employee data is used anywhere in this project.

## Limitations

- Because the data is synthetic, model accuracy here says nothing about performance on a real workforce.
- Most of the logic lives in one large `app.py`. I am rebuilding it with a cleaner, modular structure.
- The `attrition-ml-streamlit/` folder is an earlier, smaller prototype kept for reference. Development notes are in `docs/`.

## Running it locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Copyright

© 2026 Carlo Bam. All rights reserved. This repository is shared for portfolio review only.
