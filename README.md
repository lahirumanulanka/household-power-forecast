# Household Power Forecast

End-to-end time series forecasting project for daily household power consumption. It includes data preparation, feature engineering, classical and advanced models, evaluations, and reports.

## Folder Structure

```
household-power-forecast/
├── config/
│   └── project.yaml                 # Project configuration (target, seeds, etc.)
├── data/
│   ├── processed/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   ├── test.parquet
│   │   └── daily_features.parquet
│   └── ...
├── dataset/
│   └── household_power_consumption.txt
├── docs/
│   ├── 01_dataset_overview.md
│   ├── 02_eda_preprocessing.md
│   ├── 03_classical_models.md
│   ├── 04_advanced_model.md
│   ├── 05_model_comparison.md
│   └── notebooks/                  # Auto-generated notebook summaries
├── models/                          # Saved models and metadata
│   ├── classical_YYYYMMDD_HHMMSS/
│   │   ├── prophet_model.json
│   │   ├── lstm_state_dict.pt
│   │   ├── lstm_scaler.joblib
│   │   ├── sarima_fit.joblib
│   │   ├── sarima_fit_full.joblib
│   │   └── best_model.json
│   ├── advanced_YYYYMMDD_HHMMSS/
│   │   ├── xgboost_model.json
│   │   ├── lightgbm_model.txt
│   │   ├── random_forest.joblib
│   │   ├── sarimax_fit.joblib
│   │   ├── sarimax_full.joblib
│   │   ├── nbeats_val/ (if available)
│   │   ├── nbeats_test/ (if available)
│   │   └── best_model.json
│   ├── best_classical.json         # Pointer to latest best classical run
│   └── best_advanced.json          # Pointer to latest best advanced run
├── notebooks/
│   ├── 01_dataset_overview.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_advanced_model.ipynb
│   └── 05_model_comparison.ipynb
├── reports/
│   ├── classical_models_comparison.csv
│   ├── advanced_models_comparison.csv
│   ├── advanced_models_comparison_ranked.csv
│   ├── comprehensive_comparison.png
│   ├── prophet_forecast.png
│   ├── lstm_training_history.png
│   ├── lstm_forecast.png
│   ├── xgboost_feature_importance.png
│   └── xgboost_forecast.png
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── convert_to_csv.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── features/
│   │   └── build_features.py
│   └── utils/
│       └── seed.py
├── requirements.txt
├── PROJECT_SUMMARY.md
└── README.md
```

## Running the Notebooks

1. Open notebooks in the listed order:
	- 01_dataset_overview.ipynb
	- 02_eda_preprocessing.ipynb
	- 03_classical_models.ipynb
	- 04_advanced_model.ipynb
	- 05_model_comparison.ipynb

2. Execute each notebook top-to-bottom. Model artifacts and metrics are saved automatically at the end of the training notebooks into `models/` with timestamped folders and `best_model.json` summaries. A convenient pointer file is also updated: `models/best_classical.json` and `models/best_advanced.json`.

## Best Models

- Classical best (by Test RMSE): LSTM (≈ 0.2614)
- Advanced best (by Test RMSE): XGBoost (≈ 0.0125)
- Overall recommendation: XGBoost for deployment due to best Test performance; consider Prophet for uncertainty intervals and ensembling with Random Forest.

## Reports and Documentation

- Notebook write-ups: see `docs/*.md` and `docs/notebooks/*.md`.
- Comparison tables and figures are stored under `reports/`.

## Reproducibility

- Random seed is set via `config/project.yaml`.
- Metrics implementation (`src/evaluation/metrics.py`) uses robust RMSE and MAPE to avoid division-by-zero issues.

