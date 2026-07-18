"""Supabase-based address and POI feature caching."""
import pandas as pd
from pipeline.supabase_handler import get_supabase_client
from datetime import datetime


def get_cached_features(street: str, locality: str, region: str) -> dict | None:
    """
    Get cached address features from Supabase.

    Args:
        street: Street name
        locality: Locality/district
        region: Region/city

    Returns:
        Dict with cached features or None if not cached
    """
    try:
        client = get_supabase_client()

        response = client.table('address_cache') \
            .select("*") \
            .eq('street', street) \
            .eq('locality', locality) \
            .eq('region', region) \
            .single() \
            .execute()

        if response.data:
            # Update last_used timestamp
            client.table('address_cache') \
                .update({'last_used': datetime.now().isoformat()}) \
                .eq('id', response.data['id']) \
                .execute()

            print(f"  ✓ Cache hit: {street}, {locality}")
            return response.data

        return None

    except Exception as e:
        print(f"  ⚠️  Cache lookup failed: {e}")
        return None


def cache_address_features(street: str, locality: str, region: str,
                           old_address: str, lat: float, lon: float,
                           features: dict) -> bool:
    """
    Store address and POI features in Supabase cache.

    Args:
        street: Street name
        locality: Locality/district
        region: Region/city
        old_address: Full address string
        lat: Latitude
        lon: Longitude
        features: Dict with POI features:
                 {
                     'nearest_school_km': float,
                     'school_count_3km': int,
                     'nearest_hospital_km': float,
                     ...
                 }

    Returns:
        True if successful
    """
    try:
        client = get_supabase_client()

        data = {
            'street': street,
            'locality': locality,
            'region': region,
            'old_address': old_address,
            'lat': lat,
            'lon': lon,
            'cached_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat(),
            **features  # Merge POI features
        }

        # Upsert (insert or update if exists)
        client.table('address_cache').upsert(data).execute()

        print(f"  💾 Cached: {street}, {locality}")
        return True

    except Exception as e:
        print(f"  ❌ Cache store failed: {e}")
        return False


def load_csv_to_supabase(csv_path: str) -> bool:
    """
    Migrate existing localities.csv to Supabase.
    Handles NaN, infinity, and invalid float values.

    Args:
        csv_path: Path to localities.csv file

    Returns:
        True if successful
    """
    print(f"📖 Loading {csv_path} to Supabase...")

    try:
        df = pd.read_csv(csv_path)
        print(f"   Found {len(df)} rows")

        # Remove rows with NULL street, locality, or region (required for cache key)
        print("   Filtering invalid rows (NULL street/locality/region)...")
        rows_before = len(df)
        df = df.dropna(subset=['street', 'locality', 'region'])
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            print(f"   Removed {rows_removed} invalid rows")

        # Remove duplicate cache keys (street, locality, region)
        print("   Removing duplicate entries...")
        dupes_before = len(df)
        df = df.drop_duplicates(subset=['street', 'locality', 'region'], keep='first')
        dupes_removed = dupes_before - len(df)
        if dupes_removed > 0:
            print(f"   Removed {dupes_removed} duplicate rows")

        # Clean invalid float values (NaN, inf, -inf → None)
        print("   Cleaning invalid float values...")
        df = df.replace({float('inf'): None, float('-inf'): None})

        # Convert count columns from float to int FIRST (before replacing NaN with None)
        print("   Converting count columns to integers...")
        count_columns = [col for col in df.columns if 'count' in col.lower()]
        for col in count_columns:
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)

        # Then replace NaN with None for JSON compliance
        df = df.astype(object).where(pd.notna(df), None)

        print(f"   Found {len(df)} rows\n")

        client = get_supabase_client()

        # Convert to records for batch insert
        records = df.to_dict(orient='records')

        # Ensure count columns are int, not float strings
        count_columns = [col for col in df.columns if 'count' in col.lower()]
        for record in records:
            for col in count_columns:
                if col in record and record[col] is not None:
                    record[col] = int(record[col]) if isinstance(record[col], (int, float)) else record[col]

        # Insert in batches
        BATCH_SIZE = 100
        total_inserted = 0
        failed_batches = []

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

            try:
                # Upsert to handle duplicates
                client.table('address_cache').upsert(batch).execute()
                total_inserted += len(batch)
                print(f"   ✓ Batch {batch_num}/{total_batches}: {len(batch)} rows")
            except Exception as e:
                print(f"   ❌ Batch {batch_num} failed: {e}")
                failed_batches.append((batch_num, str(e)))
                # Continue with next batch instead of failing completely
                continue

        if failed_batches:
            print(f"\n⚠️  {len(failed_batches)} batches failed (see details above)")
            if total_inserted > 0:
                print(f"✅ Partially migrated {total_inserted} rows to Supabase\n")
                return True
            else:
                print(f"❌ No rows inserted\n")
                return False
        else:
            print(f"\n✅ Migrated {total_inserted} rows to Supabase\n")
            return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


def get_cache_stats() -> dict:
    """Get cache statistics."""
    try:
        client = get_supabase_client()

        response = client.table('address_cache') \
            .select('id', 'cached_at') \
            .execute()

        total = len(response.data)

        return {
            'total_cached': total,
            'status': '✅ Cache ready' if total > 0 else '⚠️  Cache empty'
        }

    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    # Migrate CSV to Supabase
    load_csv_to_supabase("data/cache/localities.csv")

    # Check stats
    stats = get_cache_stats()
    print(f"Cache stats: {stats}")
