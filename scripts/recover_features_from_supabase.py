from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "alonhadat_features.csv"
DEFAULT_RAW_OUTPUT = ROOT / "data" / "processed" / "alonhadat_features_supabase_raw.csv"


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover feature rows from Supabase.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--table", default=None)
    parser.add_argument("--page-size", type=int, default=1000)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def get_supabase_client():
    load_dotenv(ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not service_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
    return create_client(url, service_key)


def fetch_all_rows(table: str, page_size: int) -> pd.DataFrame:
    client = get_supabase_client()
    rows: list[dict] = []
    offset = 0

    while True:
        response = (
            client.table(table)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        print(f"Fetched {len(rows)} rows...")
        if len(batch) < page_size:
            break
        offset += page_size

    return pd.DataFrame(rows)


def dedupe_features(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    before = len(df)
    df = df.copy()

    if "post_day" in df.columns:
        df["_post_day_sort"] = pd.to_datetime(df["post_day"], errors="coerce")
        df = df.sort_values("_post_day_sort", na_position="first", kind="stable")

    if "listing_id" in df.columns:
        keyed = df[df["listing_id"].notna()].drop_duplicates(subset=["listing_id"], keep="last")
        missing_key = df[df["listing_id"].isna()]
        if "link" in missing_key.columns and not missing_key.empty:
            missing_key = missing_key.drop_duplicates(subset=["link"], keep="last")
        df = pd.concat([keyed, missing_key], ignore_index=True)
    elif "link" in df.columns:
        df = df.drop_duplicates(subset=["link"], keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    df = df.drop(columns=["_post_day_sort"], errors="ignore")
    return df.reset_index(drop=True), before - len(df)


def main() -> None:
    configure_console_encoding()
    args = parse_args()
    table = args.table or os.getenv("SUPABASE_TABLE", "Raw_Features")
    output_path = resolve_path(args.output)
    raw_output_path = resolve_path(args.raw_output)

    raw_df = fetch_all_rows(table, args.page_size)
    if raw_df.empty:
        raise SystemExit(f"No rows found in Supabase table {table!r}")

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_output_path, index=False)

    deduped_df, removed = dedupe_features(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    deduped_df.to_csv(temp_output_path, index=False)
    temp_output_path.replace(output_path)

    print(f"Supabase table: {table}")
    print(f"Raw backup: {raw_output_path}")
    print(f"Recovered output: {output_path}")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Deduped rows: {len(deduped_df)}")
    print(f"Removed duplicates: {removed}")
    if "listing_id" in deduped_df.columns:
        print(
            "Duplicate listing_id rows after:",
            int(deduped_df.duplicated(subset=["listing_id"], keep=False).sum()),
        )
    if "link" in deduped_df.columns:
        print(
            "Duplicate link rows after:",
            int(deduped_df.duplicated(subset=["link"], keep=False).sum()),
        )


if __name__ == "__main__":
    main()
