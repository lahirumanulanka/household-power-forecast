# 05 - Comprehensive Model Comparison

This notebook aggregates performance from all modeling steps into a single view so you can quickly see which approaches work best and why.

What happens step by step
1) Load results
  - Reads `reports/classical_models_comparison.csv` and `reports/advanced_models_comparison.csv`.
  - Optionally reads novel models’ metrics if present:
    - `reports/novel_sarimax_exog_metrics.csv` → SARIMAX_EXOG
    - `reports/novel_tft_metrics.csv` → TFT
    - `reports/novel_nhits_metrics.csv` → N-HiTS
  - Concatenates all into a single comparison table.
2) Visualize
  - Creates side-by-side bar charts for Test MAE, RMSE, MAPE, and R².
  - Orders bars to highlight best/worst per metric.
3) Rank models
  - Prints the best and worst model for each Test metric to guide selections.
4) Key findings
  - Summarizes insights and writes them to `reports/key_findings.txt` for reporting.

Outputs (under reports/)
- comprehensive_comparison.png
- key_findings.txt

How to run
1) Ensure you have first run the classical, advanced, and (optionally) novel models notebooks to generate the CSVs.
2) Open `notebooks/05_model_comparison.ipynb` and run all cells.

Notes and tips
- If a novel model CSV is missing, it’s simply skipped; re-run this notebook after you generate it.
- Use this notebook to justify model selection in your report; include both accuracy and interpretability considerations.
