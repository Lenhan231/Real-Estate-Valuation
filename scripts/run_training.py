from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_BRANCH = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the data folder from the main repository and run selected model trainers."
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Fetch the remote repository and update only the local data/ folder.",
    )
    parser.add_argument(
        "--clean-data",
        action="store_true",
        help="Run scripts/clean_model_data.py before training.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Do not clean data after refreshing it.",
    )
    parser.add_argument("--remote", default="origin", help="Git remote to fetch from.")
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Remote branch to read data/ from. Defaults to main.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to train, or all. Example: --models xgboost random_forest",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only refresh data; do not run training.",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to scripts/train_regression_models.py. Prefix with --.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def refresh_data(remote: str, branch: str) -> None:
    run(["git", "fetch", remote, f"{branch}:refs/remotes/{remote}/{branch}"])
    run(["git", "restore", "--source", f"{remote}/{branch}", "--worktree", "--", "data"])


def clean_data() -> None:
    run([sys.executable, str(SCRIPTS_DIR / "clean_model_data.py")])


def normalize_training_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def train_models(models: list[str], training_args: list[str]) -> None:
    command = [sys.executable, str(SCRIPTS_DIR / "train_regression_models.py")]
    if models != ["all"]:
        command.extend(["--models", *models])
    command.extend(normalize_training_args(training_args))
    run(command)


def main() -> None:
    args = parse_args()

    if args.refresh_data:
        refresh_data(args.remote, args.branch)

    if (args.refresh_data or args.clean_data) and not args.skip_clean:
        clean_data()

    if not args.skip_training:
        train_models(args.models, args.training_args)


if __name__ == "__main__":
    main()
