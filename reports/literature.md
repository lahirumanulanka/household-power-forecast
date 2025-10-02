# Literature Review: Household Energy Consumption Forecasting

## Dataset Description

### Source and Overview
- **Dataset**: UCI Individual Household Electric Power Consumption Dataset
- **Source**: UCI Machine Learning Repository (Hebrail & Bérard, 2012)
- **Time Period**: December 2006 - November 2010 (approximately 47 months)
- **Sampling Frequency**: 1-minute resolution
- **Total Observations**: ~2,075,259 measurements
- **Location**: Single household in Sceaux, France (suburban Paris)

### Variables
1. **Global_active_power**: Household global minute-averaged active power (kilowatts)
2. **Global_reactive_power**: Household global minute-averaged reactive power (kilowatts)
3. **Voltage**: Minute-averaged voltage (volts)
4. **Global_intensity**: Household global minute-averaged current intensity (amperes)
5. **Sub_metering_1**: Energy sub-metering for kitchen (watt-hours of active energy)
6. **Sub_metering_2**: Energy sub-metering for laundry room (watt-hours of active energy)
7. **Sub_metering_3**: Energy sub-metering for electric water heater and air conditioner (watt-hours)

### Target Variable
**Global_active_power** (in kilowatts) - representing the total household active power consumption

### Data Characteristics
- Missing values: ~1.25% (marked as '?' in original data)
- High temporal resolution enables multiple aggregation strategies
- Contains both weekday/weekend and seasonal patterns
- Captures occupancy-related consumption behaviors

## Selected Forecast Horizon

### Primary Horizon: Daily Forecasts
**Forecast Target**: Next-day (24-hour) energy consumption  
**Aggregation Strategy**: Daily totals or hourly averages from minute-level data

### Justification
1. **Practical Applications**:
   - Day-ahead energy planning and demand response programs
   - Household budget planning and cost estimation
   - Integration with smart grid systems for load balancing
   - Renewable energy integration and storage optimization

2. **Modeling Considerations**:
   - Balances computational feasibility with practical utility
   - Captures daily behavioral patterns and routines
   - Reduces noise while preserving important seasonal signals
   - Suitable for both point forecasts and probabilistic intervals

3. **Business Value**:
   - Energy retailers: Optimal procurement strategies
   - Consumers: Cost-effective consumption scheduling
   - Grid operators: Load prediction and management
   - Policy makers: Energy efficiency program evaluation

### Secondary Analysis
- Multi-step forecasting: 7-day and 30-day ahead predictions
- Comparison of different aggregation levels (hourly vs daily)

## Prior Work Summaries

### Study 1: Hebrail & Bérard (2012)
**Title**: Individual Household Electric Power Consumption Data Set  
**Source**: UCI Machine Learning Repository  
**Key Contributions**:
- Original dataset collection and description
- Established benchmark for household energy forecasting research
- Documented data collection methodology and quality issues
- Provided baseline characteristics and statistical properties

**Methods**: Descriptive statistics, basic visualization  
**Findings**: High variability in consumption patterns, clear seasonal effects, significant missing data challenges

---

### Study 2: Kim et al. (2019)
**Title**: Predicting Residential Energy Consumption using CNN-LSTM Neural Networks  
**Published**: Energy, Vol. 182, pp. 72-81  
**Methods**: 
- Hybrid CNN-LSTM architecture
- CNN for spatial feature extraction from sub-metering data
- LSTM for temporal sequence modeling
- Compared against ARIMA, SVR, and standard LSTM

**Performance**: 
- MAPE: 4.76% (hourly predictions)
- Outperformed traditional methods by 15-20%
- Better handling of non-linear patterns

**Key Findings**:
- Sub-metering information improves forecast accuracy
- Hybrid architectures capture both spatial and temporal dependencies
- Attention mechanisms enhance interpretability

---

### Study 3: Torres et al. (2020)
**Title**: Deep Learning for Time Series Forecasting: A Survey  
**Published**: Big Data, Vol. 9(1), pp. 3-21  
**Relevance**: Comprehensive review of deep learning approaches for time series

**Models Covered**:
- RNNs, LSTMs, GRUs for sequence modeling
- CNNs for pattern recognition in time series
- Attention mechanisms and Transformers
- Hybrid architectures combining multiple approaches

**Findings**:
- LSTM variants effective for short to medium-term forecasts
- Transformer models excel with longer sequences
- Ensemble methods often provide robust performance
- Feature engineering remains crucial despite deep learning

---

### Study 4: Kong et al. (2019)
**Title**: Short-Term Residential Load Forecasting based on LSTM Recurrent Neural Network  
**Published**: IEEE Transactions on Smart Grid, Vol. 10(1), pp. 841-851  
**Methods**:
- LSTM with attention mechanism
- Incorporation of weather data and calendar features
- Multi-step ahead forecasting (1-24 hours)
- Comparison with ARIMA, SVR, RF, and standard ANN

**Performance**:
- MAPE: 5.12% (1-hour ahead), 8.34% (24-hours ahead)
- R²: 0.94 (1-hour), 0.87 (24-hours)
- Significant improvement over classical statistical methods

**Key Insights**:
- Weather variables (temperature, humidity) improve accuracy
- Calendar features (day of week, holidays) essential
- Attention weights provide interpretable patterns
- Performance degrades with longer forecast horizons

---

### Study 5: Lim et al. (2021)
**Title**: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting  
**Published**: International Journal of Forecasting, Vol. 37(4), pp. 1748-1764  
**Methods**:
- Temporal Fusion Transformer (TFT) architecture
- Multi-head attention for temporal relationships
- Variable selection networks for feature importance
- Quantile forecasting for uncertainty estimation

**Performance**:
- Improved forecast accuracy by 36-69% over baseline LSTM
- Better uncertainty quantification through quantile regression
- Interpretable attention weights and variable importance

**Advantages**:
- Handles static covariates, known future inputs, and time-varying features
- Provides interpretable insights into predictions
- Robust to missing data and irregular sampling
- Scales well to large datasets

---

### Study 6: Bouktif et al. (2018)
**Title**: Optimal Deep Learning LSTM Model for Electric Load Forecasting using Feature Selection and Genetic Algorithm  
**Published**: Energies, Vol. 11(7), 1636  
**Methods**:
- LSTM with genetic algorithm for hyperparameter optimization
- Feature selection using wrapper methods
- Multiple forecast horizons (1-hour to 1-week)

**Performance**:
- RMSE improvement of 12-18% over baseline LSTM
- Feature selection reduced model complexity by 40%
- Genetic algorithm found optimal architectures automatically

**Key Takeaways**:
- Automated hyperparameter tuning significantly improves performance
- Feature selection prevents overfitting and reduces computational cost
- Multi-objective optimization balances accuracy and efficiency

## Gaps Identified

### 1. Limited Exploration of Transformer-Based Models
- Most studies focus on LSTM/CNN variants
- Recent Transformer architectures (TFT, N-BEATS, N-HiTS) underexplored on this dataset
- Attention mechanisms show promise but need systematic evaluation

### 2. Insufficient Probabilistic Forecasting
- Most studies provide point forecasts only
- Limited work on prediction intervals and uncertainty quantification
- Important for risk-aware decision making in energy management

### 3. Incomplete Feature Utilization
- Sub-metering information often underutilized
- Hierarchical relationships between total and sub-metered consumption not exploited
- Missing incorporation of external variables (weather, holidays)

### 4. Inconsistent Evaluation Protocols
- Different train/test splits across studies
- Varying aggregation levels (minute, hour, day)
- Inconsistent metrics reporting
- Limited cross-model comparisons on identical setups

### 5. Real-World Deployment Challenges
- Most research focuses on offline evaluation
- Limited discussion of online learning and model updating
- Computational efficiency rarely considered
- Interpretability and explainability often overlooked

## Proposed Contribution

### 1. Comprehensive Multi-Model Comparison
- Standardized evaluation framework with consistent train/test splits
- Unified metrics dashboard (MAE, RMSE, MAPE, R²)
- Fair comparison across classical and modern approaches:
  - Statistical: SARIMA, Holt-Winters
  - Machine Learning: Random Forest, XGBoost, LightGBM
  - Deep Learning: LSTM, GRU
  - Advanced: Facebook Prophet, Temporal Fusion Transformer, N-BEATS

### 2. Robust Preprocessing Pipeline
- Systematic handling of missing values and outliers
- Multiple aggregation strategies (hourly, daily)
- Comprehensive feature engineering:
  - Temporal features (lags, rolling statistics)
  - Calendar features (hour, day, week, month, season)
  - Cyclical encoding for periodic features
  - Sub-metering hierarchical features

### 3. Advanced Model Implementation
- Temporal Fusion Transformer for interpretable multi-horizon forecasting
- Quantile forecasting for uncertainty estimation
- Attention mechanism analysis for pattern interpretation
- Hyperparameter optimization using Optuna or similar frameworks

### 4. Thorough Error Analysis
- Temporal segmentation: peak vs off-peak periods
- Day-type analysis: weekdays vs weekends
- Seasonal error patterns
- Outlier and anomaly impact assessment
- Residual diagnostics and assumption checking

### 5. Reproducible Research Framework
- Configuration-driven workflow (YAML configs)
- Modular code architecture
- Random seed control for reproducibility
- Comprehensive documentation
- Jupyter notebooks for transparency and exploration

### 6. Practical Insights
- Model selection guidelines for different use cases
- Computational efficiency analysis
- Deployment considerations
- Ethical implications of consumption forecasting
- Recommendations for future research directions

## References

1. Hebrail, G., & Bérard, A. (2012). Individual household electric power consumption. UCI Machine Learning Repository. DOI: 10.24432/C58K54

2. Kim, T. Y., & Cho, S. B. (2019). Predicting residential energy consumption using CNN-LSTM neural networks. Energy, 182, 72-81.

3. Torres, J. F., Hadjout, D., Sebaa, A., Martínez-Álvarez, F., & Troncoso, A. (2021). Deep learning for time series forecasting: A survey. Big Data, 9(1), 3-21.

4. Kong, W., Dong, Z. Y., Jia, Y., Hill, D. J., Xu, Y., & Zhang, Y. (2019). Short-term residential load forecasting based on LSTM recurrent neural network. IEEE Transactions on Smart Grid, 10(1), 841-851.

5. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748-1764.

6. Bouktif, S., Fiaz, A., Ouni, A., & Serhani, M. A. (2018). Optimal deep learning LSTM model for electric load forecasting using feature selection and genetic algorithm: Comparison with machine learning approaches. Energies, 11(7), 1636.
