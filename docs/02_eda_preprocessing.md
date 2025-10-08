# 02 - EDA and Preprocessing

This notebook covers exploratory analysis, resampling, cleaning, stationarity checks, and feature engineering.

## Resampling and Cleaning
- Resampled minute-level data to daily using mean/aggregations for modeling convenience (downstream notebooks operate on daily)
- Handled missing values with forward-fill then interpolation where needed
- Outliers capped using IQR-based winsorization (in-notebook)

### Daily aggregation outcome (from notebook)
- Daily shape: 1,442 days × 7 numeric columns
- Date range: 2006-12-16 to 2010-11-26
- Missing at daily level (pre-interp): ~0.624% on power/voltage/intensity; 0% on sub-metering
- After interpolation: 0 missing across all columns

## Stationarity and Decomposition
- Seasonal decomposition shows clear yearly patterns
- ADF/KPSS tests: non-stationary at level, stationarity after differencing

### Notebook results
- Decomposition strengths: Trend=0.6441, Seasonal=0.1202
- Stationarity:
  - ADF on original: p=0.0046 → Stationary
  - KPSS on original: p=0.10 → Stationary
  - First difference: Stationary by both ADF and KPSS

## Engineered Features
- Calendar: minute, hour, dayofweek (per `config/project.yaml`)
- Lag features (daily index): 1, 2, 3, 5, 10, 30, 60, 120, 240, 1440 (mirroring config; large lags collapse at daily granularity)
- Rolling stats: windows 5, 15, 60, and 1440 with mean/std (+ min/max for 60 and 1440) as specified in config
- Expanding/long-horizon proxies via large rolling windows

## Train/Val/Test Split
- Chronological split: Train, Validation, Test (time-series split per config with final holdout)
- Saved artifacts:
  - `data/processed/train.parquet`
  - `data/processed/val.parquet`
  - `data/processed/test.parquet`
  - `data/processed/daily_features.parquet`
  - `data/processed/feature_names.txt`

### Notebook results
- Shape after feature engineering (non-null): 1,412 rows × 46 columns
- Splits:
  - Train: 1,292 days (2007-01-15 to 2010-07-29)
  - Validation: 60 days (2010-07-30 to 2010-09-27)
  - Test: 60 days (2010-09-28 to 2010-11-26)
- Saved 44 feature names

## How this feeds models
- Classical models (03) use the daily target and baseline sequences
- Advanced models (04) use feature matrix and `feature_names.txt` for exogenous inputs
