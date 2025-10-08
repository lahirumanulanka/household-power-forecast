# 05 - Comprehensive Model Comparison and Error Analysis

This document consolidates results from classical and advanced models and performs global comparisons.

## Inputs
- Classical results: `reports/classical_models_comparison.csv`
- Advanced results: `reports/advanced_models_comparison.csv`
- Optional novel models: SARIMAX_EXOG, TFT, N-HiTS from `reports/novel_*.csv` if present

## Visual Comparison
- Aggregated plots saved to `reports/comprehensive_comparison.png`.

## Best Overall Model
Considering Test RMSE across all available models:
- Classical best: LSTM (Test RMSE=0.2614)
- Advanced best: XGBoost (Test RMSE=0.0125)

Therefore, the best overall model is XGBoost.

## Recommendations
- Prefer XGBoost for deployment due to best Test RMSE and robust tree-based performance.
- Use Prophet alongside for uncertainty intervals, and consider ensembling with Random Forest.

## Notes
- Full rankings per metric are generated inside the notebook.
- Future work: add prediction export and full error breakdown by calendar attributes.
