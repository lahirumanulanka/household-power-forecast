# 01 - Dataset Overview

This notebook ingests the original Household Power Consumption dataset and builds a clear picture of its structure, time coverage, and basic statistics. It sets up consistent parsing for timestamps and highlights any data quality issues early.

What happens step by step:
1) Load raw data from `dataset/household_power_consumption.txt` using a semicolon delimiter and treating `?` as missing values.
2) Combine the `Date` and `Time` columns into a single pandas DateTime index, sort chronologically, and ensure it is timezone-naive for consistency.
3) Cast all power/energy-related columns to numeric, coercing invalid strings to NaN.
4) Compute dataset-level summaries: row/column counts, start/end timestamps, missing value counts per column, and simple descriptive statistics (mean, std, min/max).
5) Persist a compact JSON summary to speed up later notebooks that only need high-level metadata.

Inputs
- dataset/household_power_consumption.txt (semicolon-separated; `?` indicates missing)

Outputs (under reports/)
- dataset_summary.json
	- Example schema:
		- rows, cols
		- start, end (ISO timestamps)
		- missing: { column -> count }
		- describe: { column -> {mean, std, min, 25%, 50%, 75%, max} }

Why this matters
- Ensures consistent and reproducible timestamp parsing for all downstream work.
- Provides an at-a-glance view to decide resampling, cleaning rules, and feature engineering choices.

How to run
1) Open `notebooks/01_dataset_overview.ipynb`
2) Run all cells (no special dependencies beyond pandas/numpy/matplotlib)

Troubleshooting
- If you see parsing warnings, confirm the file delimiter is `;` and that `Date`/`Time` formats match the raw dataset.
- If memory becomes an issue on first load, consider reading in chunks or sampling; later notebooks work from processed parquet files.
