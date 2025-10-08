from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # Compute RMSE without relying on the 'squared' argument for broad sklearn compatibility
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    # Robust MAPE: avoid division by zero by masking zeros in y_true
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "mape": mape, "r2": float(r2)}
