# 02 - EDA and Preprocessing

Purpose: Explore the data and build cleaned, feature-rich time series for modeling.

What this notebook covers:
# 02 - EDA and Preprocessing

This notebook transforms the raw series into analysis-ready datasets with consistent types, engineered features, and chronological splits for fair evaluation.

What happens step by step:
1) Load and clean
	- Read raw TXT (semicolon-delimited; `?` as missing)
	- Combine `Date` + `Time` into a DateTime index; sort
	- Convert all measurement columns to numeric using coercion; drop rows with missing target
2) Exploratory checks
	- Visualize distribution of `Global_active_power`
	- Inspect missingness patterns and simple seasonality (daily/weekly)
3) Feature engineering
	- Calendar/time features: hour, day, dayofweek, month, is_weekend
	- Lag features: e.g., t-1, t-24, t-168 (depending on your config or notebook defaults)
	- Rolling windows: mean and std over windows like 24, 72, 168 to capture local trends/volatility
4) Chronological splitting
	- Split into Train, Validation, and Test by time (no shuffling)
	- Persist each split as Parquet for fast IO and stable dtypes

Inputs
- dataset/household_power_consumption.txt

Outputs (under data/processed/)
- train.parquet
- val.parquet
- test.parquet

Why this matters
- Prevents target leakage by using only past information (lags/rolls) for each timestamp.
- Standardizes the pipeline so all models consume the same features and splits.

How to run
1) Open `notebooks/02_eda_preprocessing.ipynb`
2) Run all cells; outputs will be written to `data/processed/`

Edge cases and tips
- Feature engineering mirrors what’s used by classical and advanced models
