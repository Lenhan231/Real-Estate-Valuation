from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_cleaning import clean_for_modeling_with_report

DEFAULT_INPUT = ROOT / "data" / "processed" / "alonhadat_features.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "real_estate_cleaned_2.csv"


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and impute engineered real-estate data for modeling.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input engineered feature CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output cleaned CSV.")
    return parser.parse_args()


def main() -> None:
    configure_console_encoding()

    args = parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    df = pd.read_csv(input_path)
    cleaned, iqr_report = clean_for_modeling_with_report(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df)} -> {len(cleaned)}")
    print(f"Columns: {df.shape[1]} -> {cleaned.shape[1]}")
    print(f"Missing values remaining: {int(cleaned.isna().sum().sum())}")
    if iqr_report:
        print("IQR outlier filtering by property_type on price_per_m2:")
        for row in iqr_report:
            print(
                "  "
                f"{row['property_type']} ({row['property_type_label']}): "
                f"{row['rows_before']} -> {row['rows_after']} "
                f"removed={row['rows_removed']} "
                f"bounds=[{row['lower_bound']:.2f}, {row['upper_bound']:.2f}]"
            )


if __name__ == "__main__":
    main()
