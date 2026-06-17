"""
Verify checkpoint is working: show input before/after pipeline
"""
import pandas as pd
from pathlib import Path
import subprocess
import sys

INPUT_FILE = Path("data/raw/alonhadat_details.csv")
OUTPUT_FILE = Path("data/processed/alonhadat_features.csv")

def main():
    print("=" * 70)
    print("CHECKPOINT VERIFICATION")
    print("=" * 70)

    # Before
    if INPUT_FILE.exists():
        df_before = pd.read_csv(INPUT_FILE)
        input_before = len(df_before)
        print(f"\n[BEFORE] Input file: {input_before} records")
    else:
        input_before = 0
        print(f"\n[BEFORE] Input file: NOT FOUND")

    if OUTPUT_FILE.exists():
        df_output_before = pd.read_csv(OUTPUT_FILE)
        output_before = len(df_output_before)
        print(f"[BEFORE] Output file: {output_before} records")
    else:
        output_before = 0
        print(f"[BEFORE] Output file: NOT FOUND")

    # Run pipeline
    print(f"\n{'='*70}")
    print("RUNNING PIPELINE...")
    print(f"{'='*70}\n")
    result = subprocess.run([sys.executable, "main.py"], capture_output=False)

    # After
    print(f"\n{'='*70}")
    print("CHECKPOINT VERIFICATION RESULT")
    print(f"{'='*70}\n")

    if INPUT_FILE.exists():
        df_after = pd.read_csv(INPUT_FILE)
        input_after = len(df_after)
        print(f"[AFTER] Input file: {input_after} records")
    else:
        input_after = 0
        print(f"[AFTER] Input file: EMPTY (header only)")

    if OUTPUT_FILE.exists():
        df_output_after = pd.read_csv(OUTPUT_FILE)
        output_after = len(df_output_after)
        print(f"[AFTER] Output file: {output_after} records")
    else:
        output_after = 0
        print(f"[AFTER] Output file: NOT FOUND")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Input:  {input_before} → {input_after} ({input_before - input_after} removed)")
    print(f"Output: {output_before} → {output_after} ({output_after - output_before} added)")

    if input_before > input_after:
        print(f"\n✓ CHECKPOINT WORKING: {input_before - input_after} rows removed from input")
    else:
        print(f"\n✗ CHECKPOINT NOT WORKING: Input still has {input_after} records")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
