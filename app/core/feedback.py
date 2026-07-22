"""Feedback collection and management for model improvement."""
import os
import pandas as pd
from datetime import datetime


def save_feedback_to_supabase(feedback_data: dict, row_dict: dict = None) -> bool:
    """Save user feedback to Supabase for model retraining.

    Args:
        feedback_data: Dict with prediction, actual price, rating, property details
        row_dict: Full feature dict (80 features) for model retraining

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            print("❌ Supabase credentials not set")
            return False

        client = create_client(url, key)

        # Prepare feedback record
        record = {
            "predicted_price_vnd": feedback_data.get("predicted_price_vnd"),
            "actual_price_vnd": feedback_data.get("actual_price_vnd"),
            "rating": feedback_data.get("rating"),
            "street": feedback_data.get("street"),
            "locality": feedback_data.get("locality"),
            "area_m2": feedback_data.get("area_m2"),
            "bucket": feedback_data.get("bucket"),
            "confidence": feedback_data.get("confidence"),
            "timestamp": feedback_data.get("timestamp", datetime.now().isoformat()),
            "features_json": row_dict if row_dict else {},
        }

        # Insert into Supabase
        response = client.table("feedback").insert(record).execute()

        if response.data:
            print(f"✅ Feedback saved to Supabase")
            return True
        else:
            print(f"❌ Failed to save feedback: {response}")
            return False

    except Exception as e:
        print(f"❌ Error saving feedback: {e}")
        return False


def get_feedback_for_retraining(bucket=None):
    """Extract feedback data for model retraining.

    Args:
        bucket: Filter by price tier ('low', 'mid', 'high') or None for all

    Returns:
        DataFrame with features + actual_price_vnd for retraining, or None if no data
    """
    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            return None

        client = create_client(url, key)

        # Fetch feedback with features
        query = client.table("feedback").select("*").neq("features_json", "{}")
        if bucket:
            query = query.eq("bucket", bucket)

        response = query.execute()

        if not response.data:
            return None

        # Reconstruct training dataframe
        records = []
        for row in response.data:
            if row.get("features_json") and row.get("actual_price_vnd"):
                # Combine features with actual price as target
                feature_row = row["features_json"]
                feature_row["price_vnd"] = row["actual_price_vnd"]
                records.append(feature_row)

        if not records:
            return None

        df = pd.DataFrame(records)
        return df

    except Exception as e:
        print(f"Error getting feedback for retraining: {e}")
        return None


def get_feedback_stats():
    """Get feedback statistics for dashboard."""
    try:
        from supabase import create_client
        import numpy as np

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            return None

        client = create_client(url, key)

        # Fetch all feedback
        response = client.table("feedback").select("*").execute()

        if not response.data:
            return None

        df = pd.DataFrame(response.data)
        df = df.dropna(subset=["predicted_price_vnd", "actual_price_vnd"])

        if df.empty:
            return None

        # Calculate error metrics
        df["error_vnd"] = df["actual_price_vnd"] - df["predicted_price_vnd"]
        df["error_pct"] = (df["error_vnd"] / df["actual_price_vnd"]) * 100
        df["abs_error_pct"] = df["error_pct"].abs()

        stats = {
            "total_feedback": len(response.data),
            "feedback_with_prices": len(df),
            "mean_error_pct": float(df["error_pct"].mean()),
            "mae_pct": float(df["abs_error_pct"].mean()),  # Mean Absolute Error %
            "mape_pct": float(df["abs_error_pct"].mean()),  # MAPE
            "worst_predictions": df.nlargest(5, "abs_error_pct")[
                ["street", "locality", "predicted_price_vnd", "actual_price_vnd", "abs_error_pct"]
            ].to_dict("records"),
        }

        return stats

    except Exception as e:
        print(f"Error getting feedback stats: {e}")
        return None
