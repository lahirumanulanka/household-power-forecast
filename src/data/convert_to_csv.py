"""Utility script to convert the original semicolon-separated
household power consumption text file into a standard comma-separated CSV.

Usage (default raw -> CSV):
    python src/data/convert_to_csv.py

Optional cleaned version (numeric coercion + drop missing target):
    python src/data/convert_to_csv.py --clean

Arguments:
    --input  Path to original .txt file (default: dataset/household_power_consumption.txt)
    --output Path to output CSV (default: data/raw/household_power_consumption.csv)
    --clean  Apply basic cleaning (types + drop NA target) and save an additional
             cleaned file alongside with suffix _cleaned.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Allow running as standalone script without installed package
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # If used as a package module
    from .load_data import load_raw, basic_clean  # type: ignore
except ImportError:  # pragma: no cover
    # Fallback to absolute import path when executed directly
    from src.data.load_data import load_raw, basic_clean  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw power consumption data to CSV")
    p.add_argument(
        "--input",
        default="dataset/household_power_consumption.txt",
        help="Input semicolon TXT path",
    )
    p.add_argument(
        "--output",
        default="data/raw/household_power_consumption.csv",
        help="Output CSV path (raw)",
    )
    p.add_argument("--clean", action="store_true", help="Also produce a cleaned CSV")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Loading raw data from {in_path} ...")
    df = load_raw(in_path)
    print(f"Loaded {len(df):,} rows. Saving raw CSV -> {out_path}")
    df.to_csv(out_path, index=False)

    if args.clean:
        print("Applying basic cleaning ...")
        cleaned = basic_clean(df.copy())
        cleaned_path = out_path.with_name(out_path.stem + "_cleaned.csv")
        cleaned.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned CSV -> {cleaned_path}")

    print("Done.")


if __name__ == "__main__":
    main()
