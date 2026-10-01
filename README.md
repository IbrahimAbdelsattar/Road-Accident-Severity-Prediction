# Road Accident Severity Prediction

A Streamlit interface that predicts a four-level accident severity label using a saved XGBoost bundle.

**Technology:** Python · XGBoost · scikit-learn · pandas · Streamlit

## Features

- Build numeric, categorical, and Boolean inputs from the stored training schema.
- Collect geographic, road, weather, and daylight conditions.
- Run the saved model and decode its predicted severity through the bundle's label encoder.

## Repository guide

| Path | Purpose |
|---|---|
| [app.py](app.py) | Feature form and severity prediction. |
| [severity_xgb_bundle.pkl](severity_xgb_bundle.pkl) | Model, label encoder, and feature schema. |
| [requirements.txt](requirements.txt) | Inference dependencies. |

## Requirements and current limitations

The bundle must provide `model`, `label_encoder`, `numeric_cols`, `categorical_cols`, and `bool_cols`. Run from the root and preserve its expected feature order and category handling. This repository contains the inference app and bundle; it does not include a standalone training notebook or source training dataset.

## Getting started

```bash
git clone https://github.com/IbrahimAbdelsattar/Road-Accident-Severity-Prediction.git
cd Road-Accident-Severity-Prediction
```

Use a Python virtual environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
