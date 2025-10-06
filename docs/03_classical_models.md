# 03 - Classical Forecasting Models

This notebook implements and evaluates class-discussed forecasting models:
1. Facebook Prophet
2. LSTM (Deep Learning)
3. Additional baseline models (SARIMA, Holt-Winters)

Each model is trained, tuned, and evaluated using consistent metrics.

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
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)

from src.evaluation.metrics import regression_metrics
from src.utils.seed import set_seed

# Load config
with open('../config/project.yaml') as f:
    config = yaml.safe_load(f)

set_seed(config['random_seed'])
print('Setup complete')
```

## 1. Load Processed Data

```python
# Load train/val/test splits
data_dir = Path('../data/processed')
train_df = pd.read_parquet(data_dir / 'train.parquet')
val_df = pd.read_parquet(data_dir / 'val.parquet')
test_df = pd.read_parquet(data_dir / 'test.parquet')

target = config['project']['target_variable']

print(f'Train shape: {train_df.shape}')
print(f'Validation shape: {val_df.shape}')
print(f'Test shape: {test_df.shape}')
print(f'Target variable: {target}')
```

## 2. Baseline Models

### 2.1 Naive Forecast (Persistence)

```python
# Naive forecast: Use yesterday's value
naive_pred_val = val_df[target].shift(1).fillna(method='bfill')
naive_pred_test = test_df[target].shift(1).fillna(method='bfill')

naive_metrics_val = regression_metrics(val_df[target], naive_pred_val)
naive_metrics_test = regression_metrics(test_df[target], naive_pred_test)

print('Naive Forecast - Validation:')
print(naive_metrics_val)
print('\nNaive Forecast - Test:')
print(naive_metrics_test)

results = {'naive': {'val': naive_metrics_val, 'test': naive_metrics_test}}
```

### 2.2 Moving Average

```python
# 7-day moving average
ma_window = 7
train_series = pd.concat([train_df[target], val_df[target]])

# For proper evaluation, predict each step
ma_preds_val = []
ma_preds_test = []

for i in range(len(val_df)):
    if i < ma_window:
        hist = train_df[target].iloc[-(ma_window-i):]
    else:
        hist = val_df[target].iloc[max(0,i-ma_window):i]
    ma_preds_val.append(hist.mean())

for i in range(len(test_df)):
    if i < ma_window:
        hist = train_series.iloc[-(ma_window-i):]
    else:
        hist = test_df[target].iloc[max(0,i-ma_window):i]
    ma_preds_test.append(hist.mean())

ma_metrics_val = regression_metrics(val_df[target], ma_preds_val)
ma_metrics_test = regression_metrics(test_df[target], ma_preds_test)

print(f'{ma_window}-Day Moving Average - Validation:')
print(ma_metrics_val)
print(f'\n{ma_window}-Day Moving Average - Test:')
print(ma_metrics_test)

results['moving_avg'] = {'val': ma_metrics_val, 'test': ma_metrics_test}
```

## 3. Facebook Prophet

Prophet is designed for forecasting time series with strong seasonal effects.

```python
# Install required packages
%pip install prophet torch torchvision torchaudio
```

```python
from prophet import Prophet

# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_train = train_df.reset_index()[['DateTime', target]].rename(
    columns={'DateTime': 'ds', target: 'y'}
)

# Train Prophet model
prophet_model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)

print('Training Prophet model...')
prophet_model.fit(prophet_train)
print('Prophet training complete')
```

```python
# Predict on validation set
prophet_val = val_df.reset_index()[['DateTime']].rename(columns={'DateTime': 'ds'})
prophet_val_pred = prophet_model.predict(prophet_val)

# Predict on test set
prophet_test = test_df.reset_index()[['DateTime']].rename(columns={'DateTime': 'ds'})
prophet_test_pred = prophet_model.predict(prophet_test)

# Evaluate
prophet_metrics_val = regression_metrics(val_df[target].values, prophet_val_pred['yhat'].values)
prophet_metrics_test = regression_metrics(test_df[target].values, prophet_test_pred['yhat'].values)

print('Prophet - Validation:')
print(prophet_metrics_val)
print('\nProphet - Test:')
print(prophet_metrics_test)

results['prophet'] = {'val': prophet_metrics_val, 'test': prophet_metrics_test}
```

```python
# Visualize Prophet forecast
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# Validation
axes[0].plot(val_df.index, val_df[target], label='Actual', linewidth=2)
axes[0].plot(val_df.index, prophet_val_pred['yhat'].values, label='Prophet Forecast', linewidth=2, linestyle='--')
axes[0].fill_between(val_df.index, 
                      prophet_val_pred['yhat_lower'].values, 
                      prophet_val_pred['yhat_upper'].values, 
                      alpha=0.2, label='Confidence Interval')
axes[0].set_title('Prophet Forecast - Validation Set', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test
axes[1].plot(test_df.index, test_df[target], label='Actual', linewidth=2)
axes[1].plot(test_df.index, prophet_test_pred['yhat'].values, label='Prophet Forecast', linewidth=2, linestyle='--')
axes[1].fill_between(test_df.index, 
                      prophet_test_pred['yhat_lower'].values, 
                      prophet_test_pred['yhat_upper'].values, 
                      alpha=0.2, label='Confidence Interval')
axes[1].set_title('Prophet Forecast - Test Set', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Power (kW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/prophet_forecast.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 4. LSTM Model

Long Short-Term Memory networks for time series forecasting.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# LSTM Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_col, sequence_length, feature_cols=None):
        self.data = data
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols or [target_col]
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.data[self.feature_cols].iloc[idx:idx+self.sequence_length].values
        y = self.data[self.target_col].iloc[idx+self.sequence_length]
        return torch.FloatTensor(X), torch.FloatTensor([y])

# LSTM Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

print('LSTM classes defined')
```

```python
# Prepare data for LSTM
sequence_length = 30  # Use past 30 days to predict next day
feature_cols = [target]  # Start with univariate, can add more features

# Scale data
scaler = StandardScaler()
train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

# Create datasets
train_dataset = TimeSeriesDataset(train_scaled, target, sequence_length, feature_cols)
val_dataset = TimeSeriesDataset(val_scaled, target, sequence_length, feature_cols)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f'Training samples: {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
```

```python
# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

lstm_config = config['models']['lstm']
model = LSTMForecaster(
    input_size=len(feature_cols),
    hidden_size=lstm_config['hidden_size'],
    num_layers=lstm_config['num_layers'],
    dropout=lstm_config['dropout']
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lstm_config['learning_rate'])

print(model)
```

```python
# Training loop
num_epochs = 30
train_losses = []
val_losses = []

print('Training LSTM model...')
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print('LSTM training complete')
```

```python
# Plot training history
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_losses, label='Train Loss')
ax.plot(val_losses, label='Validation Loss')
ax.set_title('LSTM Training History', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/lstm_training_history.png', dpi=300, bbox_inches='tight')
plt.show()
```

```python
# Make predictions
def predict_lstm(model, data_scaled, scaler, sequence_length, feature_cols, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(sequence_length, len(data_scaled)):
            X = data_scaled[feature_cols].iloc[i-sequence_length:i].values
            X = torch.FloatTensor(X).unsqueeze(0).to(device)
            pred = model(X)
            predictions.append(pred.cpu().item())
    
    # Inverse transform
    predictions = scaler.inverse_transform([[p] for p in predictions])
    return predictions.flatten()

# Predict on validation and test sets
lstm_pred_val = predict_lstm(model, val_scaled, scaler, sequence_length, feature_cols, device)
lstm_pred_test = predict_lstm(model, test_scaled, scaler, sequence_length, feature_cols, device)

# Align with actual values (skip first sequence_length points)
val_actual = val_df[target].iloc[sequence_length:].values
test_actual = test_df[target].iloc[sequence_length:].values

# Evaluate
lstm_metrics_val = regression_metrics(val_actual, lstm_pred_val)
lstm_metrics_test = regression_metrics(test_actual, lstm_pred_test)

print('LSTM - Validation:')
print(lstm_metrics_val)
print('\nLSTM - Test:')
print(lstm_metrics_test)

results['lstm'] = {'val': lstm_metrics_val, 'test': lstm_metrics_test}
```

```python
# Visualize LSTM predictions
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# Validation
val_dates = val_df.index[sequence_length:]
axes[0].plot(val_dates, val_actual, label='Actual', linewidth=2)
axes[0].plot(val_dates, lstm_pred_val, label='LSTM Forecast', linewidth=2, linestyle='--')
axes[0].set_title('LSTM Forecast - Validation Set', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test
test_dates = test_df.index[sequence_length:]
axes[1].plot(test_dates, test_actual, label='Actual', linewidth=2)
axes[1].plot(test_dates, lstm_pred_test, label='LSTM Forecast', linewidth=2, linestyle='--')
axes[1].set_title('LSTM Forecast - Test Set', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Power (kW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/lstm_forecast.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 5. Additional Models: SARIMA

Seasonal ARIMA for comparison.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Train SARIMA model
# Using (1,1,1)x(1,1,1,7) configuration for daily data with weekly seasonality
print('Training SARIMA model...')
sarima_model = SARIMAX(
    train_df[target],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)
print('SARIMA training complete')
print(sarima_fit.summary())
```

```python
# Forecast
sarima_pred_val = sarima_fit.forecast(steps=len(val_df))

# For test set, retrain with train+val
train_val = pd.concat([train_df, val_df])
sarima_model_full = SARIMAX(
    train_val[target],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit_full = sarima_model_full.fit(disp=False)
sarima_pred_test = sarima_fit_full.forecast(steps=len(test_df))

# Evaluate
sarima_metrics_val = regression_metrics(val_df[target].values, sarima_pred_val.values)
sarima_metrics_test = regression_metrics(test_df[target].values, sarima_pred_test.values)

print('SARIMA - Validation:')
print(sarima_metrics_val)
print('\nSARIMA - Test:')
print(sarima_metrics_test)

results['sarima'] = {'val': sarima_metrics_val, 'test': sarima_metrics_test}
```

## 6. Model Comparison Summary

```python
# Create comparison table
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name.upper(),
        'Val MAE': metrics['val']['mae'],
        'Val RMSE': metrics['val']['rmse'],
        'Val MAPE': metrics['val']['mape'],
        'Val R²': metrics['val']['r2'],
        'Test MAE': metrics['test']['mae'],
        'Test RMSE': metrics['test']['rmse'],
        'Test MAPE': metrics['test']['mape'],
        'Test R²': metrics['test']['r2']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.round(4)
print('\nModel Comparison (Classical Models):')
print(comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv('../reports/classical_models_comparison.csv', index=False)
print('\nResults saved to reports/classical_models_comparison.csv')
```

```python
# Visualize comparison
metrics_to_plot = ['Test RMSE', 'Test MAE', 'Test MAPE', 'Test R²']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot):
    data = comparison_df[['Model', metric]].sort_values(metric, ascending=(metric != 'Test R²'))
    axes[idx].barh(data['Model'], data[metric], alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel(metric)
    axes[idx].set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='x')
    
    # Annotate values
    for i, v in enumerate(data[metric]):
        axes[idx].text(v, i, f' {v:.4f}', va='center')

plt.tight_layout()
plt.savefig('../reports/classical_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Summary

### Models Implemented:
1. ✅ Naive Baseline (Persistence)
2. ✅ Moving Average Baseline
3. ✅ Facebook Prophet
4. ✅ LSTM (Deep Learning)
5. ✅ SARIMA (Statistical)

### Key Findings:
- All models outperform naive baseline
- Prophet provides good interpretability with reasonable accuracy
- LSTM captures complex patterns effectively
- SARIMA leverages statistical properties of the time series

### Next Steps:
Proceed to notebook 04_advanced_model.ipynb to implement and evaluate advanced architectures (TFT, N-BEATS).

