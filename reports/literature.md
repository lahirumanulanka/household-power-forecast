# Literature Review (Draft)

(Fill with summaries of at least 5 related studies. Suggested structure below.)

## Dataset Description
- Source: UCI Individual household electric power consumption dataset.
- Frequency: 1-minute samples over ~4 years (Dec 2006 - Nov 2010) for one household.
- Variables: Global_active_power (kW), Global_reactive_power, Voltage, Global_intensity, Sub_metering_1/2/3, etc.
- Target: Forecast future Global_active_power over horizon (define: e.g., next 1 day or multi-day).

## Selected Forecast Horizon
Justify horizon based on use-case (e.g., daily scheduling, demand response). Provide reasoning for single-step vs multi-step.

## Prior Work Summaries
1. Study A (Year): Methods (ARIMA, ETS, etc.), key findings, performance metrics.
2. Study B (Year): Use of LSTM/CNN hybrid, improvements vs classical baselines.
3. Study C (Year): Probabilistic forecasting approach (e.g., quantile regression, DeepAR).
4. Study D (Year): Transformer-based sequence modeling for energy consumption.
5. Study E (Year): Feature engineering (weather, calendar) impact analysis.

(Replace placeholders with actual citations, include DOIs/URLs.)

## Gaps Identified
- Limited exploration of attention-based architectures on fine-grained minute-level horizon.
- Under-utilization of hierarchical sub-metering information.
- Need for comparison under consistent evaluation splits & unified metrics dashboard.

## Proposed Contribution
- Standardized feature pipeline + cross-model comparison (Prophet, Chronos/Forecast, LSTM, Advanced Model e.g., TFT/N-BEATS).
- Robust error analysis (temporal segmentation: peak vs off-peak, weekdays vs weekends).
- Reproducible configuration-driven workflow.
