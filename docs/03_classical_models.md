# 03 - Classical Models

This notebook establishes strong, interpretable baselines and simple deep-learning references before introducing more advanced models.

Models implemented and what they do
- Naive baseline: uses the last observed value as the forecast; sanity check baseline.
- Moving Average: smooths short-term noise; forecasts with recent averaged values.
- Prophet: decomposable model (trend + seasonality + holiday); robust to missing data and outliers.
- LSTM: sequence model capturing non-linear temporal dependencies; simple univariate setup here.
- SARIMA: seasonal ARIMA model capturing autoregression, differencing, and moving average with seasonality.

Data used
- `data/processed/train.parquet`, `val.parquet`, `test.parquet`
- Target: `Global_active_power`

Outputs (under reports/)
- classical_models_comparison.csv (Val/Test MAE, RMSE, MAPE, R²)
- classical_models_comparison.png (visual summary)
- prophet_forecast.png
- lstm_forecast.png
- lstm_training_history.png

Evaluation metrics
- MAE: average absolute error (lower is better)
- RMSE: penalizes larger errors (lower is better)
- MAPE: relative error in percent (lower is better; be mindful near zero)
- R²: proportion of variance explained (higher is better)

How to run
1) Open `notebooks/03_classical_models.ipynb`
2) Run all cells; the notebook installs Prophet on demand if missing

Troubleshooting
- Moving Average: Ensure you reference the Series directly (bug fixed to avoid Series[name] confusion).
- Prophet: First run may download/build CmdStan; allow time and keep internet on. If build fails, re-run install cell.
- LSTM: If GPU isn’t available, training will run on CPU and may be slower; reduce epochs for quick tests.
