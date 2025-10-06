# 04 - Advanced Forecasting Models

Implement advanced deep learning models not covered in class:
1. N-BEATS (Neural Basis Expansion Analysis for Time Series)
2. XGBoost (Tree-based approach)
3. Model comparison and analysis

```python
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import yaml

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)

from src.evaluation.metrics import regression_metrics
from src.utils.seed import set_seed

with open('../config/project.yaml') as f:
    config = yaml.safe_load(f)

set_seed(config['random_seed'])
print('Setup complete')
```

## 1. Load Data

```python
data_dir = Path('../data/processed')
train_df = pd.read_parquet(data_dir / 'train.parquet')
val_df = pd.read_parquet(data_dir / 'val.parquet')
test_df = pd.read_parquet(data_dir / 'test.parquet')

target = config['project']['target_variable']
print(f'Data loaded. Target: {target}')
print(f'Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}')
```

## 2. XGBoost Model

Gradient boosted trees with lag and rolling features.

```python
# Install required packages
%pip install xgboost lightgbm
```

```python
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Prepare features
feature_cols = [col for col in train_df.columns if col not in [target, 'is_outlier']]
print(f'Using {len(feature_cols)} features')

# Handle NaN values
X_train = train_df[feature_cols].fillna(method='ffill').fillna(method='bfill')
y_train = train_df[target]

X_val = val_df[feature_cols].fillna(method='ffill').fillna(method='bfill')
y_val = val_df[target]

X_test = test_df[feature_cols].fillna(method='ffill').fillna(method='bfill')
y_test = test_df[target]

print(f'Training XGBoost...')
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=config['random_seed'],
    n_jobs=-1,
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print('XGBoost training complete')
```

```python
# Predictions
xgb_pred_val = xgb_model.predict(X_val)
xgb_pred_test = xgb_model.predict(X_test)

# Metrics
xgb_metrics_val = regression_metrics(y_val, xgb_pred_val)
xgb_metrics_test = regression_metrics(y_test, xgb_pred_test)

print('XGBoost - Validation:')
print(xgb_metrics_val)
print('\nXGBoost - Test:')
print(xgb_metrics_test)

results = {'xgboost': {'val': xgb_metrics_val, 'test': xgb_metrics_test}}
```

```python
# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(importance['feature'], importance['importance'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Importance')
ax.set_title('XGBoost Top 20 Feature Importances', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../reports/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
# Visualize predictions
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

axes[0].plot(val_df.index, y_val, label='Actual', linewidth=2)
axes[0].plot(val_df.index, xgb_pred_val, label='XGBoost Forecast', linewidth=2, linestyle='--')
axes[0].set_title('XGBoost Forecast - Validation Set', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(test_df.index, y_test, label='Actual', linewidth=2)
axes[1].plot(test_df.index, xgb_pred_test, label='XGBoost Forecast', linewidth=2, linestyle='--')
axes[1].set_title('XGBoost Forecast - Test Set', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Power (kW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/xgboost_forecast.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 3. LightGBM Model

Another gradient boosting variant for comparison.

```python
import lightgbm as lgb

print('Training LightGBM...')
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=config['random_seed'],
    n_jobs=-1,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

print('LightGBM training complete')
```

```python
# Predictions
lgb_pred_val = lgb_model.predict(X_val)
lgb_pred_test = lgb_model.predict(X_test)

# Metrics
lgb_metrics_val = regression_metrics(y_val, lgb_pred_val)
lgb_metrics_test = regression_metrics(y_test, lgb_pred_test)

print('LightGBM - Validation:')
print(lgb_metrics_val)
print('\nLightGBM - Test:')
print(lgb_metrics_test)

results['lightgbm'] = {'val': lgb_metrics_val, 'test': lgb_metrics_test}
```

## 4. Random Forest

Ensemble of decision trees.

```python
from sklearn.ensemble import RandomForestRegressor

print('Training Random Forest...')
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=config['random_seed'],
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print('Random Forest training complete')

# Predictions
rf_pred_val = rf_model.predict(X_val)
rf_pred_test = rf_model.predict(X_test)

# Metrics
rf_metrics_val = regression_metrics(y_val, rf_pred_val)
rf_metrics_test = regression_metrics(y_test, rf_pred_test)

print('Random Forest - Validation:')
print(rf_metrics_val)
print('\nRandom Forest - Test:')
print(rf_metrics_test)

results['random_forest'] = {'val': rf_metrics_val, 'test': rf_metrics_test}
```

## 5. SARIMAX (with Exogenous Features)

Statistical seasonal ARIMA with exogenous regressors taken from the engineered feature set. Trains on train, validates on val, then retrains on train+val to forecast test.

```python
# SARIMAX with exogenous features
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Use a SAFE subset of exogenous regressors to avoid leakage:
# - keep calendar/time features (e.g., hour, day_of_week, month, is_weekend, etc.)
# - drop any features likely derived from the target (lag/rolling/diff/ewm/shifts)
# - also drop anything containing the target name just in case
_leak_patterns = (
    'lag', 'rolling', 'expanding', 'ewm', 'diff', 'shift', 'ma', 'ewma', 'cumsum'
)
exog_cols = []
for c in feature_cols:
    cname = c.lower()
    if (target.lower() in cname) or any(p in cname for p in _leak_patterns):
        continue
    exog_cols.append(c)

if len(exog_cols) == 0:
    print('No safe exogenous columns detected; proceeding without exogenous regressors.')

# Ensure all exogenous regressors are numeric and aligned across splits
# Convert any non-numeric to NaN, then fill and cast to float
def to_numeric_exog(df, cols):
    if not cols:
        return None
    exog = df[cols].apply(pd.to_numeric, errors='coerce')
    exog = exog.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    return exog.astype(float)

X_train_exog = to_numeric_exog(X_train, exog_cols)
X_val_exog = to_numeric_exog(X_val, exog_cols)
X_test_exog = to_numeric_exog(X_test, exog_cols)

# Basic SARIMAX configuration; adjust if you maintain config in YAML
order = (1, 1, 1)
seasonal_order = (0, 1, 1, 7)

print('Training SARIMAX (safe exogenous)...')
sarimax_model = SARIMAX(
    y_train.astype(float),
    exog=X_train_exog,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarimax_fit = sarimax_model.fit(disp=False)

sarimax_val_pred = sarimax_fit.get_forecast(steps=len(y_val), exog=X_val_exog).predicted_mean.values

# Retrain on train+val for test forecast
y_train_val = pd.concat([y_train, y_val])
X_train_val_exog = None if X_train_exog is None else pd.concat([X_train_exog, X_val_exog])

sarimax_full = SARIMAX(
    y_train_val.astype(float),
    exog=X_train_val_exog,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sarimax_test_pred = sarimax_full.get_forecast(steps=len(y_test), exog=X_test_exog).predicted_mean.values

sarimax_metrics_val = regression_metrics(y_val.values, sarimax_val_pred)
sarimax_metrics_test = regression_metrics(y_test.values, sarimax_test_pred)

print('SARIMAX - Validation:', sarimax_metrics_val)
print('SARIMAX - Test:', sarimax_metrics_test)

results['sarimax'] = {'val': sarimax_metrics_val, 'test': sarimax_metrics_test}
```

## 6. N-BEATS (Neural Basis Expansion Analysis for Time Series)

A deep learning model designed specifically for univariate time series forecasting using backward and forward basis expansions.

```python
# N-BEATS via darts with sklearn compatibility patches; skip gracefully if still unavailable.
import sys, subprocess, warnings, importlib

# Ensure darts is installed (without importing it yet to avoid early import errors)
if importlib.util.find_spec("darts") is None:
    try:
        print("Installing u8darts ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "u8darts", "--quiet"]) 
    except Exception as _install_err:
        print(f"Skipping N-BEATS: could not install darts (u8darts). Error: {_install_err}")

# Ensure deep learning deps for darts are installed
try:
    import pytorch_lightning  # noqa: F401
except Exception:
    try:
        print("Installing pytorch-lightning and torchmetrics ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-lightning", "torchmetrics", "--quiet"]) 
    except Exception as _pl_err:
        print(f"Warning: could not install pytorch-lightning deps: {_pl_err}")

# Patch sklearn for darts compatibility
try:
    import sklearn.utils as _sku
    # get_tags fallback on sklearn.utils
    if not hasattr(_sku, "get_tags"):
        warnings.warn("Patching sklearn.utils.get_tags for darts compatibility")
        def _get_tags(estimator):
            if hasattr(estimator, "get_tags"):
                return estimator.get_tags()
            return {}
        _sku.get_tags = _get_tags  # type: ignore[attr-defined]

    # validate_data fallback (import location used by some darts modules)
    import sklearn.utils.validation as _skv
    if not hasattr(_skv, "validate_data"):
        warnings.warn("Patching sklearn.utils.validation.validate_data for darts compatibility")
        def _validate_data(X=None, *args, **kwargs):
            # Minimal passthrough; darts should not rely on strict checks for N-BEATS
            return X
        _skv.validate_data = _validate_data  # type: ignore[attr-defined]

    # get_tags fallback on sklearn.utils._tags
    try:
        import sklearn.utils._tags as _sktags  # type: ignore
        if not hasattr(_sktags, "get_tags"):
            warnings.warn("Patching sklearn.utils._tags.get_tags for darts compatibility")
            def _get_tags_internal(estimator):
                if hasattr(estimator, "get_tags"):
                    return estimator.get_tags()
                return {}
            _sktags.get_tags = _get_tags_internal  # type: ignore[attr-defined]
    except Exception as _e:
        print(f"Warning: could not patch sklearn.utils._tags: {_e}")
except Exception as _patch_err:
    print(f"Warning: could not fully patch sklearn compatibility: {_patch_err}")

try:
    import numpy as np
    import torch
    from darts import TimeSeries
    from darts.models import NBEATSModel
    from darts.utils.missing_values import fill_missing_values

    # Force float32 and CPU to avoid macOS MPS float64 limitations
    torch.set_default_dtype(torch.float32)
    pl_kwargs = {"accelerator": "cpu"}

    # Build univariate series from the target, casting to float32
    concat_df = pd.concat([train_df, val_df, test_df]).copy()
    concat_df[target] = pd.to_numeric(concat_df[target], errors='coerce').astype(np.float32)

    series_full = TimeSeries.from_dataframe(
        concat_df,
        time_col=None,  # use index
        value_cols=target,
    )
    try:
        series_full = series_full.astype(np.float32)
    except Exception:
        # some versions may not have astype; already float32 from df
        pass
    series_full = fill_missing_values(series_full)

    # Split into train/val/test by length
    train_len = len(train_df)
    val_len = len(val_df)

    train_series = series_full[:train_len]
    val_series = series_full[train_len:train_len+val_len]
    test_series = series_full[train_len+val_len:]

    print('Training N-BEATS...')
    nbeats = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=val_len,
        n_epochs=50,
        random_state=config['random_seed'],
        pl_trainer_kwargs=pl_kwargs
    )

    nbeats.fit(train_series, verbose=False)
    val_forecast = nbeats.predict(n=val_len)

    nbeats_val_pred = val_forecast.values().flatten().astype(np.float32)
    nbeats_val_true = val_series.values().flatten().astype(np.float32)

    nbeats_metrics_val = regression_metrics(nbeats_val_true, nbeats_val_pred)
    print('N-BEATS - Validation:', nbeats_metrics_val)

    # Retrain on train+val, predict test
    nbeats_test = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=len(test_df),
        n_epochs=50,
        random_state=config['random_seed'],
        pl_trainer_kwargs=pl_kwargs
    )

    nbeats_test.fit(series_full[:-len(test_df)], verbose=False)
    test_forecast = nbeats_test.predict(n=len(test_df))

    nbeats_test_pred = test_forecast.values().flatten().astype(np.float32)
    nbeats_test_true = test_series.values().flatten().astype(np.float32)

    nbeats_metrics_test = regression_metrics(nbeats_test_true, nbeats_test_pred)
    print('N-BEATS - Test:', nbeats_metrics_test)

    results['n_beats'] = {'val': nbeats_metrics_val, 'test': nbeats_metrics_test}
except Exception as e:
    print(f"Skipping N-BEATS due to error: {e}")
```

## 5. Model Comparison

We aggregate validation and test metrics across all advanced models, rank them (lower RMSE is better), and save:
- reports/advanced_models_comparison.csv (raw table)
- reports/advanced_models_comparison_ranked.csv (with ranks and average rank)
- reports/figures/advanced_models_comparison.png (bar chart of RMSE by model)

This helps quickly identify the best-performing model(s) and compare validation vs. test performance.

```python
# Create comparison table
comparison_data = []
# Friendly display names for models
_display_name = {
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM',
    'random_forest': 'Random Forest',
    'sarimax': 'SARIMAX',
    'n_beats': 'N-BEATS',
}
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': _display_name.get(model_name, model_name.replace('_', '-').upper()),
        'Val MAE': metrics['val']['mae'],
        'Val RMSE': metrics['val']['rmse'],
        'Val MAPE': metrics['val']['mape'],
        'Val R²': metrics['val']['r2'],
        'Test MAE': metrics['test']['mae'],
        'Test RMSE': metrics['test']['rmse'],
        'Test MAPE': metrics['test']['mape'],
        'Test R²': metrics['test']['r2']
    })

comparison_df = pd.DataFrame(comparison_data).round(6)

# Ranking (lower RMSE is better)
comparison_df['Val Rank'] = comparison_df['Val RMSE'].rank(method='min')
comparison_df['Test Rank'] = comparison_df['Test RMSE'].rank(method='min')
comparison_df['Avg Rank'] = (comparison_df['Val Rank'] + comparison_df['Test Rank']) / 2
comparison_df = comparison_df.sort_values(['Avg Rank', 'Test RMSE']).reset_index(drop=True)

print('\nAdvanced Model Comparison (ranked by RMSE):')
print(comparison_df.to_string(index=False))

# Save tables
reports_dir = Path('../reports')
fig_dir = reports_dir / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

comparison_df.to_csv(reports_dir / 'advanced_models_comparison_ranked.csv', index=False)
comparison_df.round(4).to_csv(reports_dir / 'advanced_models_comparison.csv', index=False)

# Plot RMSEs
plt.figure(figsize=(12, 5))
x = np.arange(len(comparison_df))
width = 0.35
plt.bar(x - width/2, comparison_df['Val RMSE'], width=width, label='Val RMSE')
plt.bar(x + width/2, comparison_df['Test RMSE'], width=width, label='Test RMSE')
plt.xticks(x, comparison_df['Model'], rotation=20)
plt.ylabel('RMSE')
plt.title('Advanced Models: RMSE by Model')
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / 'advanced_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print('\nResults saved to:')
print(f"- {reports_dir / 'advanced_models_comparison.csv'}")
print(f"- {reports_dir / 'advanced_models_comparison_ranked.csv'}")
print(f"- {fig_dir / 'advanced_models_comparison.png'}")
```

## Summary

### Advanced Models Implemented:
1. ✅ XGBoost (Gradient Boosted Trees)
2. ✅ LightGBM (Efficient Gradient Boosting)
3. ✅ Random Forest (Ensemble)

### Justification:
**XGBoost and LightGBM** were chosen as advanced models because:
- Excel at capturing non-linear relationships in time series
- Handle missing values naturally
- Provide feature importance for interpretability
- Proven track record in forecasting competitions
- Efficient computation compared to deep learning

### Key Advantages:
- **Feature Importance**: Unlike black-box models, tree-based methods show which features drive predictions
- **Robustness**: Less sensitive to outliers and missing values
- **Speed**: Faster training than deep neural networks
- **Interpretability**: Decision paths can be traced

### Expected Performance:
- Should outperform classical statistical methods (SARIMA)
- Competitive with or better than LSTM for tabular time series
- Excellent for capturing complex lag interactions

Proceed to notebook 05_model_comparison.ipynb for comprehensive comparison.



