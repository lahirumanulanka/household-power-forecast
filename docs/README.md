# Project Notebooks Documentation

This directory contains comprehensive documentation for all project notebooks. Each `.md` file provides a detailed explanation of the corresponding Jupyter notebook, including code explanations, methodology, and key findings.

## 📚 Notebook Documentation

### 1. Dataset Overview (`01_dataset_overview.md`)
**Notebook**: `notebooks/01_dataset_overview.ipynb`

**Purpose**: Comprehensive exploration and initial analysis of the UCI Individual Household Electric Power Consumption dataset.

**Key Topics**:
- Dataset loading and structure examination
- Data types and column descriptions
- Basic statistical analysis
- Missing values analysis and patterns
- Temporal patterns visualization (daily, weekly, monthly, yearly)
- Outlier detection and analysis
- Sub-metering analysis
- Data quality assessment

**Key Findings**:
- 2,075,259 observations spanning ~4 years (Dec 2006 - Nov 2010)
- 8 features including power consumption, voltage, intensity, and sub-metering
- ~1.25% missing values requiring imputation
- Strong daily and weekly seasonality patterns
- Multiple outliers that need consideration in modeling

---

### 2. EDA and Preprocessing (`02_eda_preprocessing.md`)
**Notebook**: `notebooks/02_eda_preprocessing.ipynb`

**Purpose**: In-depth exploratory data analysis and feature engineering for time series forecasting.

**Key Topics**:
- Time series decomposition (trend, seasonality, residuals)
- Missing value handling strategies (forward-fill, interpolation)
- Outlier treatment approaches
- Feature engineering:
  - Lag features (1, 7, 14, 30 days)
  - Rolling statistics (mean, std, min, max)
  - Temporal features (hour, day, month, season)
  - Cyclical encodings (sin/cos transformations)
- Data normalization and scaling
- Train/Validation/Test split (temporal ordering preserved)

**Key Outputs**:
- 60+ engineered features
- Processed datasets saved to `data/processed/`
- Feature importance analysis
- Correlation analysis

**Recommendations**:
- Daily aggregation reduces noise and improves model stability
- Lag features are crucial for capturing temporal dependencies
- Seasonal decomposition reveals multiple periodicities

---

### 3. Classical Models (`03_classical_models.md`)
**Notebook**: `notebooks/03_classical_models.ipynb`

**Purpose**: Implementation and evaluation of class-discussed forecasting models.

**Models Implemented**:

1. **Naive Baseline (Persistence Model)**
   - Uses previous day's value as prediction
   - Serves as performance baseline
   - Test MAE: ~0.220 kW

2. **Moving Average**
   - 7-day rolling average
   - Smooths short-term fluctuations
   - Test MAE: ~0.180 kW

3. **Facebook Prophet**
   - Additive model with trend and seasonality
   - Handles multiple seasonalities (daily, weekly, yearly)
   - Uncertainty quantification with confidence intervals
   - Test MAE: ~0.130 kW, R²: ~0.810

4. **SARIMA (Seasonal ARIMA)**
   - Statistical time series model
   - Captures autocorrelation and seasonality
   - Test MAE: ~0.150 kW, R²: ~0.780

5. **LSTM (Long Short-Term Memory)**
   - 2-layer deep learning architecture
   - 30-day sequence input
   - Captures long-term dependencies
   - Test MAE: ~0.110 kW, R²: ~0.850

**Key Insights**:
- Deep learning (LSTM) outperforms statistical methods
- Prophet provides excellent interpretability with uncertainty
- Baseline models establish minimum performance threshold

---

### 4. Advanced Models (`04_advanced_model.md`)
**Notebook**: `notebooks/04_advanced_model.ipynb`

**Purpose**: Implementation of advanced tree-based and ensemble models.

**Models Implemented**:

1. **XGBoost (Extreme Gradient Boosting)** ⭐ **BEST PERFORMER**
   - 500 estimators, max_depth=5
   - Early stopping on validation set
   - Feature importance analysis
   - **Test MAE: 0.090 kW, R²: 0.900**
   - **60% improvement over naive baseline**

2. **LightGBM (Light Gradient Boosting Machine)**
   - Efficient gradient boosting framework
   - Leaf-wise tree growth strategy
   - Fast training and prediction
   - Test MAE: 0.092 kW, R²: 0.895

3. **Random Forest**
   - Ensemble of 200 decision trees
   - Robust to overfitting
   - Parallel training capability
   - Test MAE: 0.105 kW, R²: 0.860

**Why These Models?**
- Tree-based models excel at capturing non-linear feature interactions
- Handle missing values naturally
- Provide feature importance for interpretability
- Require minimal feature scaling/normalization
- Strong performance on tabular data with engineered features

**Key Insights**:
- XGBoost achieves state-of-the-art results
- Lag features and rolling statistics are most important
- Ensemble methods are highly effective for this dataset

---

### 5. Model Comparison (`05_model_comparison.md`)
**Notebook**: `notebooks/05_model_comparison.ipynb`

**Purpose**: Comprehensive comparison and error analysis of all implemented models.

**Analysis Includes**:
- Unified metrics comparison (MAE, RMSE, MAPE, R²)
- Visual comparison charts and tables
- Forecast vs actual plots for each model
- Error distribution analysis
- Temporal error patterns
- Residual analysis
- Feature importance comparison
- Model strengths and weaknesses

**Key Findings**:
- Tree-based models (XGBoost, LightGBM) dominate performance
- Deep learning (LSTM) competitive but requires more tuning
- Prophet excels in interpretability and uncertainty quantification
- Statistical models (SARIMA) underperform on this dataset
- Feature engineering provides 30-40% improvement

**Final Recommendations**:
- **Production Use**: XGBoost for accuracy, Prophet for interpretability
- **Ensemble Approach**: Combine XGBoost + LSTM for robustness
- **Real-time**: LightGBM for speed with minimal accuracy loss

---

## 🚀 Usage Guide

### Sequential Execution
Run notebooks in order to ensure proper data flow:

```bash
1. notebooks/01_dataset_overview.ipynb    # Explore dataset
2. notebooks/02_eda_preprocessing.ipynb   # Create features
3. notebooks/03_classical_models.ipynb    # Train classical models
4. notebooks/04_advanced_model.ipynb      # Train advanced models
5. notebooks/05_model_comparison.ipynb    # Compare all models
```

### Reading Documentation
- Start with `01_dataset_overview.md` to understand the data
- Follow with `02_eda_preprocessing.md` for feature engineering insights
- Review model-specific docs based on your interests
- Finish with `05_model_comparison.md` for overall insights

---

## 📊 Quick Performance Summary

| Model | MAE (kW) | RMSE (kW) | MAPE (%) | R² | Training Time |
|-------|----------|-----------|----------|-----|---------------|
| XGBoost | **0.090** | **0.125** | **8.5** | **0.900** | ~2 min |
| LightGBM | 0.092 | 0.127 | 8.7 | 0.895 | ~1 min |
| LSTM | 0.110 | 0.148 | 10.2 | 0.850 | ~15 min |
| Random Forest | 0.105 | 0.142 | 9.8 | 0.860 | ~3 min |
| Prophet | 0.130 | 0.172 | 12.1 | 0.810 | ~5 min |
| SARIMA | 0.150 | 0.195 | 13.8 | 0.780 | ~10 min |
| Moving Avg | 0.180 | 0.230 | 16.5 | 0.650 | <1 min |
| Naive | 0.220 | 0.285 | 20.1 | 0.550 | <1 min |

---

## 💡 Tips

- **Generate model outputs first**: Run classical and advanced model notebooks before comparison
- **Novel models**: If implementing new models (SARIMAX_EXOG, TFT, N-HiTS), re-run comparison notebook
- **Reproducibility**: All notebooks use seed=42 for consistent results
- **Resources**: LSTM training benefits from GPU acceleration
- **Customization**: Modify `config/project.yaml` for hyperparameter tuning

---

## 📖 Additional Resources

- **Literature Review**: See `../reports/literature.md` for academic background
- **Final Report**: See `../reports/final_report.md` for comprehensive analysis
- **Source Code**: See `../src/` for modular, reusable code components

---

**Last Updated**: 2024  
**Status**: ✅ Complete documentation for all 5 notebooks