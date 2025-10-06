# 04 - Advanced Forecasting Models

This notebook implements strong, scalable tree-based models well-suited for tabular time series with engineered lags and calendar features.

Models implemented and how they work
- XGBoost: gradient-boosted trees with robust regularization; excels at non-linear feature interactions.
- LightGBM: efficient gradient boosting with histogram-based splits; fast training and good accuracy.
- Random Forest: bagging ensemble of decision trees; robust baseline with fewer hyperparameters.

Features used
- All engineered features from the preprocessing notebook except the target and any explicit flags like `is_outlier`.
- Missing values imputed via forward/backward fill in the notebook before training.

Outputs (under reports/)
- advanced_models_comparison.csv (Val/Test metrics)
- xgboost_feature_importance.png (top features)
- xgboost_forecast.png (val/test overlays)

How to run
1) Open `notebooks/04_advanced_model.ipynb`
2) Run all cells; the notebook installs `xgboost` and `lightgbm` if needed

Early stopping and logging
- XGBoost: validation set provided for stopping; some versions support early_stopping_rounds directly.
- LightGBM: uses callbacks (early_stopping and log_evaluation) for compatibility across versions.

Tips
- If training is slow, reduce `n_estimators` or `max_depth`.
- Use feature importance to prune low-value features and iterate.
