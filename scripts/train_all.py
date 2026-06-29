"""
Train All Models — Per Property Type
======================================
Runs XGBoost and TabPFN on every dataset split in sequence and logs all
results to the same W&B project for easy comparison.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --models xgboost          # XGBoost only
    python scripts/train_all.py --models tabpfn           # TabPFN only
    python scripts/train_all.py --splits full             # Full dataset only
    python scripts/train_all.py --models xgboost tabpfn --splits full nha_mat_tien nha_trong_hem
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Map of split name -> csv filename (auto-discovered below)
ALL_SPLITS = {
    "full":          "alonhadat_features_cleaned.csv",
    "nha_mat_tien":  "split_nha_mat_tien.csv",
    "nha_trong_hem": "split_nha_trong_hem.csv",
}

ALL_MODELS = ["xgboost", "tabpfn"]

SCRIPT_MAP = {
    "xgboost": PROJECT_ROOT / "scripts" / "train_xgboost.py",
    "tabpfn":  PROJECT_ROOT / "scripts" / "train_tabpfn.py",
}


def run(cmd: list[str]) -> int:
    print(f"\n{'='*70}")
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train all models on all dataset splits.")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Which model(s) to train (default: both).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(ALL_SPLITS.keys()),
        choices=list(ALL_SPLITS.keys()),
        help="Which split(s) to use (default: all).",
    )
    parser.add_argument(
        "--token", default=None,
        help="TabPFN API token. Falls back to TABPFN_TOKEN env variable.",
    )
    args = parser.parse_args()

    # Verify all requested split files exist
    missing = []
    for split in args.splits:
        csv = DATA_DIR / ALL_SPLITS[split]
        if not csv.exists():
            missing.append(str(csv))
    if missing:
        print("\n[ERROR] The following split files are missing. Run clean_features.py first:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    failures = []

    for split in args.splits:
        csv_filename = ALL_SPLITS[split]
        dataset_arg = csv_filename  # training scripts resolve this from data/processed/

        for model in args.models:
            script = SCRIPT_MAP[model]
            cmd = [sys.executable, str(script), "--dataset", dataset_arg]

            if model == "tabpfn" and args.token:
                cmd += ["--token", args.token]

            rc = run(cmd)
            if rc != 0:
                failures.append(f"{model} on {split} (exit code {rc})")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  DONE — Trained {len(args.models)} model(s) × {len(args.splits)} split(s)")
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    ✗ {f}")
        sys.exit(1)
    else:
        print("  All runs completed successfully [OK]")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
