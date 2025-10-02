import pandas as pd
from pathlib import Path


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw household power consumption dataset.

    Assumes semicolon separated values and '?' for missing.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=";",
        parse_dates={"DateTime": ["Date", "Time"]},
        infer_datetime_format=True,
        na_values="?",
        low_memory=False,
    )
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: enforce numeric types and drop rows with all NaNs in key cols."""
    numeric_cols = [c for c in df.columns if c != "DateTime"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows where target is missing (can later impute instead)
    df = df.dropna(subset=["Global_active_power"])
    return df


if __name__ == "__main__":
    import yaml

    with open("config/project.yaml") as f:
        cfg = yaml.safe_load(f)
    raw_path = cfg["paths"]["raw_data"]
    out_path = Path(cfg["paths"]["processed_data"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_raw(raw_path)
    data = basic_clean(data)
    data.to_parquet(out_path, index=False)
    print(f"Saved processed data to {out_path}")
