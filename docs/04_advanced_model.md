# 04 - Advanced Forecasting Models

Advanced models trained on engineered features with train/val/test splits.

## Models
- XGBoost
- LightGBM
- Random Forest
- SARIMAX (exogenous features – safe subset)
- N-BEATS (via darts; saved if available)

## Key Implementation Details
- Safe exogenous feature filtering to avoid leakage (drops lag/rolling/diff/shift patterns and target-related names).
- All exogenous regressors coerced to numeric; missing handled by ffill/bfill.
- SARIMAX uses order=(1,1,1), seasonal_order=(0,1,1,7) as a baseline.
- Tree models trained with early stopping/validation where supported.

## Top Results (Validation/Test)
From `reports/advanced_models_comparison.csv`:

- Random Forest: Val RMSE=0.0106, Test RMSE=0.0136
- XGBoost: Val RMSE=0.0116, Test RMSE=0.0125
- SARIMAX: Val RMSE=0.0110, Test RMSE=0.0177
- LightGBM: Val RMSE=0.0223, Test RMSE=0.0191
- N-BEATS: Val RMSE=0.2691, Test RMSE=0.5797

## Best Advanced Model
- Based on Test RMSE (lower is better): XGBoost with Test RMSE=0.0125 (Rank 1 on Test)
- Random Forest has slightly better Val RMSE but ranks 2 on Test. Overall Avg Rank in the ranked table favors Random Forest (1.5) vs XGBoost (2.0). For deployment, prefer Test performance; hence choose XGBoost.

## Artifacts
- Feature importance figure: `reports/xgboost_feature_importance.png`
- Forecast figures: `reports/xgboost_forecast.png`
- Ranked comparison: `reports/advanced_models_comparison_ranked.csv`
- Models saved to `models/advanced_*` with `best_model.json` and pointer `models/best_advanced.json`
