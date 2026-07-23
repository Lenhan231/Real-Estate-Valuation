"""Test feedback feature end-to-end."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import requests
from datetime import datetime

# Load env
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

print("=" * 80)
print("FEEDBACK FEATURE TEST")
print("=" * 80)

# Test 1: Check environment variables
print("\n[1/4] Checking environment variables...")
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

if url and key:
    print(f"  ✓ SUPABASE_URL: {url[:45]}...")
    print(f"  ✓ SUPABASE_SERVICE_KEY: {key[:20]}...")
else:
    print("  ✗ Missing Supabase credentials")
    sys.exit(1)

# Test 2: Check Supabase connection
print("\n[2/4] Testing Supabase connection...")
try:
    from supabase import create_client
    client = create_client(url, key)
    response = client.table("feedback").select("*").limit(1).execute()
    print(f"  ✓ Connected to Supabase")
    print(f"  ✓ Feedback table accessible ({len(response.data)} existing records)")
except Exception as e:
    print(f"  ✗ Supabase connection failed: {e}")
    sys.exit(1)

# Test 3: Check API
print("\n[3/4] Testing API feedback endpoint...")
try:
    # Create test payload
    payload = {
        "predicted_price_vnd": 6.5e9,
        "actual_price_vnd": 7.0e9,
        "rating": "👍 Accurate",
        "bucket": "mid",
        "confidence": 87.3,
        "feature_version": 1,
        "features_json": {
            "street": "Đường Test",
            "locality": "Phường Bình Thạnh",
            "area_m2": 100.0
        },
        "timestamp": datetime.now().isoformat()
    }

    response = requests.post(
        "http://localhost:8000/api/feedback",
        json=payload,
        timeout=10
    )

    print(f"  ✓ API responded with status {response.status_code}")
    result = response.json()
    print(f"  ✓ Response: {result['message']}")

    if response.status_code != 200:
        print(f"  ✗ Expected 200, got {response.status_code}")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ API test failed: {e}")
    sys.exit(1)

# Test 4: Verify data in Supabase
print("\n[4/4] Verifying data saved to Supabase...")
try:
    response = client.table("feedback").select("*").order("id", desc=True).limit(1).execute()
    if response.data:
        record = response.data[0]
        print(f"  ✓ Latest record ID: {record['id']}")
        print(f"  ✓ Predicted price: {record['predicted_price_vnd'] / 1e9:.1f}B VND")
        print(f"  ✓ Actual price: {record.get('actual_price_vnd', 'N/A')}")
        print(f"  ✓ Rating: {record['rating']}")
        print(f"  ✓ Created: {record['created_at']}")
    else:
        print("  ✗ No records found")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Verification failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - Feedback feature is working correctly!")
print("=" * 80)
