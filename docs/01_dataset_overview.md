# 01 - Dataset Overview

This notebook introduces the Household Power Consumption dataset and provides a high-level overview.

## Dataset
- Source: `dataset/household_power_consumption.txt`
- Original frequency: 1-minute
- Project target variable: `Global_active_power` (from config)

### Shape and Range (from notebook run)
- Raw shape: 2,075,259 rows × 8 columns
- Date range: 2006-12-16 17:24:00 to 2010-11-26 21:02:00
- Duration: 1,441 days

### Columns
- Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3

## Summary (from reports/dataset_summary.json)
- Total observations: 2,075,259
- Date range: 2006-12-16 17:24:00 to 2010-11-26 21:02:00
- Duration: 1,441 days
- Missing percentage: 1.2518%
- Target mean/std: 1.0916 / 1.0573
- Target min/max: 0.076 / 11.122
- Outlier percentage: 4.6312%

## Visuals (in notebook)
- Time range inspection
- Basic distributions and missingness checks

## Notes
- Subsequent processing aggregates the series to daily granularity for model training and evaluation.
- See `02_eda_preprocessing.md` for detailed preprocessing and feature engineering steps.
