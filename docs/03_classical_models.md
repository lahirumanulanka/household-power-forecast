# 03 - Classical Forecasting Models

This notebook implements and evaluates classical models for forecasting household power consumption.

## Contents
- Naive (Persistence)
- Moving Average
- Prophet
- LSTM
- SARIMA

## Data
- Train shape: 1292 x 46
- Validation shape: 60 x 46
- Test shape: 60 x 46
- Target: Global_active_power

## Results (Validation/Test)
- Naive: MAE=0.1418 / 0.2611, RMSE=0.2219 / 0.3631, MAPE=16.75% / 23.37%, R²=0.4040 / -0.8545
- Moving Average: MAE=0.1468 / 0.2234, RMSE=0.2066 / 0.3005, MAPE=20.46% / 20.36%, R²=0.4830 / -0.2700
- Prophet: MAE=0.1892 / 0.2206, RMSE=0.2447 / 0.3019, MAPE=29.77% / 20.75%, R²=0.2752 / -0.2822
- LSTM: MAE=0.1455 / 0.1910, RMSE=0.1860 / 0.2614, MAPE=17.43% / 16.75%, R²=-0.2059 / -0.2431
- SARIMA: MAE=0.3744 / 0.2832, RMSE=0.4444 / 0.3468, MAPE=47.35% / 22.62%, R²=-1.3914 / -0.6919

## Best Classical Model
- Based on Test RMSE (lower is better): LSTM with Test RMSE=0.2614

## Notes
- Metrics computed with robust MAPE and RMSE (sklearn-compatible implementation).
- Figures saved to `reports/prophet_forecast.png`, `reports/lstm_forecast.png`, and `reports/lstm_training_history.png`.
- Serialized artifacts and best-model metadata saved under `models/classical_*` after running the save cell.
