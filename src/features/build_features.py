import pandas as pd
from pathlib import Path
from typing import Sequence


def add_lags(df: pd.DataFrame, target: str, lags: Sequence[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling(
    df: pd.DataFrame, target: str, windows_config: list[dict]
) -> pd.DataFrame:
    for entry in windows_config:
        window = entry["window"]
        stats = entry["stats"]
        roll = df[target].rolling(window)
        if "mean" in stats:
            df[f"{target}_rollmean_{window}"] = roll.mean()
        if "std" in stats:
            df[f"{target}_rollstd_{window}"] = roll.std()
        if "min" in stats:
            df[f"{target}_rollmin_{window}"] = roll.min()
        if "max" in stats:
            df[f"{target}_rollmax_{window}"] = roll.max()
    return df


def add_calendar(df: pd.DataFrame, calendar_fields: list[str]) -> pd.DataFrame:
    dt = df["DateTime"]
    if "minute" in calendar_fields:
        df["minute"] = dt.dt.minute
    if "hour" in calendar_fields:
        df["hour"] = dt.dt.hour
    if "dayofweek" in calendar_fields:
        df["dayofweek"] = dt.dt.dayofweek
    return df


def build_feature_matrix(df: pd.DataFrame, cfg: dict, target: str) -> pd.DataFrame:
    df = df.copy()
    df = add_lags(df, target, cfg["feature_engineering"]["lags"])
    df = add_rolling(df, target, cfg["feature_engineering"]["rolling"])
    df = add_calendar(df, cfg["feature_engineering"]["calendar"])
    return df


if __name__ == "__main__":
    import yaml

    with open("config/project.yaml") as f:
        cfg = yaml.safe_load(f)
    processed_path = cfg["paths"]["processed_data"]
    features_path = Path(cfg["paths"]["features"])
    features_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_path)
    target = cfg["project"]["target_variable"]
    feat_df = build_feature_matrix(df, cfg, target)
    feat_df.to_parquet(features_path, index=False)
    print(f"Saved features to {features_path}")
