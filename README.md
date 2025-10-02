# Household Power Forecasting Project

Project structure scaffold created. Fill in details as you progress through tasks.

## Folders
- dataset/: Original provided dataset file(s)
- data/raw: Copy of raw immutable data
- data/processed: Cleaned & feature engineered datasets
- notebooks/: Jupyter notebooks for EDA, modeling prototypes
- src/: Modular production-ready code
  - src/data: Loading & cleaning scripts
  - src/features: Feature engineering utilities
  - src/models: Model definitions & training routines
  - src/evaluation: Metrics & comparison code
  - src/utils: Shared helpers
- models/: Saved trained model artifacts (serialized)
- experiments/: Experiment tracking artifacts (configs, logs, results)
- reports/: Final report, figures, tables
- config/: YAML/JSON configuration files for reproducibility

## Tasks Mapping
1. Dataset Justification & Literature Review -> `reports/` (literature.md) & `notebooks/01_dataset_overview.ipynb`
2. Exploratory Analysis & Preprocessing -> `notebooks/02_eda_preprocessing.ipynb`, code in `src/data`, `src/features`
3. Baseline/Class Models (Prophet, Chronos/Forecast, LSTM) -> `notebooks/03_classical_models.ipynb` + code in `src/models`
4. Advanced Model (e.g., TFT / N-BEATS / SARIMAX) -> `notebooks/04_advanced_model.ipynb`
5. Comparison & Error Analysis -> `notebooks/05_model_comparison.ipynb` & `src/evaluation`
6. Critical Reflection -> `reports/final_report.md`

## Getting Started
Create a virtual environment, install dependencies once `requirements.txt` is added.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add a `requirements.txt` soon including (proposed):
```
pandas
numpy
matplotlib
seaborn
scikit-learn
prophet
statsmodels
torch
pytorch-lightning
chronos-forecasting  # if available / or amazon-braket libs depending on env
tensorflow           # optional (if choosing TF-based models)
plotly
pydantic
pyyaml
```
Adjust as needed.
