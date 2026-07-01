from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "data" / "processed" / "model_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prediction-level real-estate model errors.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS, help="Prediction CSV to analyze.")
    parser.add_argument("--top", type=int, default=12, help="Number of worst groups to print.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["price_vnd", "price_per_m2", "area_m2", "road_width_m"]:
        if column not in df.columns or df[column].nunique(dropna=True) < 4:
            continue
        bucket_col = f"{column}_bucket"
        df[bucket_col] = pd.qcut(df[column], q=4, duplicates="drop").astype(str)
    return df


def summarize_group(group: pd.DataFrame) -> pd.Series:
    r2 = np.nan
    if len(group) >= 2 and group["actual"].nunique() > 1:
        r2 = r2_score(group["actual"], group["prediction"])

    return pd.Series(
        {
            "rows": len(group),
            "mape_percent": group["absolute_percentage_error"].mean(),
            "mae": group["absolute_error"].mean(),
            "bias": (group["prediction"] - group["actual"]).mean(),
            "r2": r2,
        }
    )


def print_group_summary(df: pd.DataFrame, group_col: str, top: int) -> None:
    if group_col not in df.columns:
        return

    summary = (
        df.groupby(["training_scope", "target", "model", group_col], dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )
    summary = summary[summary["rows"] >= 10].sort_values("mape_percent", ascending=False).head(top)
    if summary.empty:
        return

    print()
    print(f"Worst groups by {group_col}:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mape_percent": "{:.2f}".format,
                "mae": "{:.2f}".format,
                "bias": "{:.2f}".format,
                "r2": "{:.3f}".format,
            },
        )
    )


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions)
    df = pd.read_csv(predictions_path)
    df = add_bucket_columns(df)

    print(f"Predictions: {predictions_path}")
    print(f"Rows: {len(df)}")

    overall = (
        df.groupby(["training_scope", "target", "model", "split"], dropna=False)
        .apply(summarize_group, include_groups=False)
        .reset_index()
        .sort_values(["target", "split", "mape_percent"])
    )
    print()
    print("Overall prediction error:")
    print(
        overall.to_string(
            index=False,
            formatters={
                "mape_percent": "{:.2f}".format,
                "mae": "{:.2f}".format,
                "bias": "{:.2f}".format,
                "r2": "{:.3f}".format,
            },
        )
    )

    for group_col in [
        "property_type",
        "locality",
        "price_vnd_bucket",
        "price_per_m2_bucket",
        "area_m2_bucket",
        "road_width_m_bucket",
    ]:
        print_group_summary(df, group_col, args.top)


if __name__ == "__main__":
    main()
