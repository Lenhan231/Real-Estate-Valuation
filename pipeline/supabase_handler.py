"""Supabase integration for pushing data to database."""
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "Raw_Features")


def get_supabase_client() -> Client:
    """Initialize and return Supabase client."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_existing_listing_ids():
    client = get_supabase_client()

    all_ids = set()
    offset = 0
    limit = 1000

    while True:
        response = (
            client.table(SUPABASE_TABLE)
            .select("listing_id")
            .range(offset, offset + limit - 1)
            .execute()
        )

        data = response.data
        if not data:
            break

        for row in data:
            if row.get("listing_id"):
                all_ids.add(row["listing_id"])

        offset += limit

    print(f"Found {len(all_ids)} unique listing_ids")
    return all_ids


def fetch_csv_from_supabase(table_name: str = None) -> pd.DataFrame:
    """
    Fetch all data from Supabase table and return as DataFrame.

    Args:
        table_name: Table name to fetch from (defaults to SUPABASE_TABLE)

    Returns:
        DataFrame with all rows from table
    """
    table_name = table_name or SUPABASE_TABLE

    try:
        print(f"[Supabase] Fetching data from table: {table_name}")
        client = get_supabase_client()

        # Fetch all data
        all_data = []
        offset = 0
        limit = 1000

        while True:
            response = (
                client.table(table_name)
                .select("*")
                .range(offset, offset + limit - 1)
                .execute()
            )

            data = response.data
            if not data:
                break

            all_data.extend(data)
            offset += limit

        df = pd.DataFrame(all_data)
        print(f"[Supabase] Fetched {len(df)} rows from {table_name}\n")
        return df

    except Exception as e:
        print(f"[Supabase Error] fetching from Supabase: {e}")
        return pd.DataFrame()


def push_csv_to_supabase(csv_path: str | Path) -> bool:
    """
    Read CSV file and insert all rows into Supabase Raw_Features table.

    Args:
        csv_path: Path to CSV file to push

    Returns:
        True if successful, False otherwise
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False

    try:
        print(f"📖 Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Found {len(df)} total rows")

        if len(df) == 0:
            print("   ⚠️  CSV is empty, nothing to push")
            return True

        # Get existing listing_ids from Supabase
        print("🔗 Checking existing data in Supabase...")
        existing_ids = get_existing_listing_ids()

        # Filter only NEW rows (not in Supabase)
        df_new = df[~df['listing_id'].isin(existing_ids)].copy()
        print(f"   → {len(df_new)} new rows to push, {len(df) - len(df_new)} already exist")

        if len(df_new) == 0:
            print("   ✓ All data already in Supabase, skipping push")
            return True

        # Clean: Replace NaN and infinity with None for JSON compliance
        print("   Cleaning new data...")

        # Replace infinity
        df_new = df_new.replace({float('inf'): None, float('-inf'): None})

        # Convert NaN to None properly using astype(object)
        df_new = df_new.astype(object).where(pd.notna(df_new), None)

        print(f"🔗 Pushing {len(df_new)} new rows to Supabase...")
        client = get_supabase_client()

        # Convert dataframe to list of dicts
        records = df_new.to_dict(orient='records')

        # Insert only NEW rows
        BATCH_SIZE = 100
        total_inserted = 0

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

            try:
                response = client.table(SUPABASE_TABLE).insert(batch).execute()
                total_inserted += len(batch)
                print(f"   ✓ Batch {batch_num}/{total_batches}: Inserted {len(batch)} new rows")
            except Exception as e:
                print(f"   ❌ Batch {batch_num} failed: {e}")
                return False

        print(f"\n✅ Successfully pushed {total_inserted} new rows to {SUPABASE_TABLE}")
        return True

    except Exception as e:
        print(f"❌ Error pushing to Supabase: {e}")
        return False

if __name__ == "__main__":
    push_csv_to_supabase(r"data\processed\alonhadat_features.csv")