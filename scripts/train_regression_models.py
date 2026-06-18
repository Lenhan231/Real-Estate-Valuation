from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_cleaning import clean_for_modeling

DEFAULT_INPUT = ROOT / "data" / "processed" / "real_estate_cleaned.csv"
DEFAULT_RAW_INPUT = ROOT / "data" / "processed" / "alonhadat_features.csv"
DEFAULT_RESULTS = ROOT / "data" / "processed" / "model_results.csv"
DEFAULT_MODEL_DIR = ROOT / "models"
TARGETS = ["price_vnd", "price_per_m2"]
LEAKAGE_COLS = ["price_vnd", "price_per_m2"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline/comparison/final regressors for property price targets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input cleaned modeling CSV. Defaults to data/processed/real_estate_cleaned.csv.",
    )
    parser.add_argument(
        "--clean-input",
        action="store_true",
        help=(
            "Apply scripts/model_cleaning.py before training. Use this only when --input is "
            "the raw engineered file, such as data/processed/alonhadat_features.csv."
        ),
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Output CSV for model metrics.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory for fitted model files.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Train and report metrics without writing .joblib model files.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics and artifacts to Weights & Biases.")
    parser.add_argument("--wandb-project", default="real-estate-valuation", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team name.")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode. Use offline if you are not logged in or have no network.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def import_wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. Install it with: python -m pip install wandb"
        ) from exc
    return wandb


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_models(random_state: int) -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def load_modeling_data(input_path: Path, clean_input: bool) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if clean_input:
        df = clean_for_modeling(df)

    missing_targets = [target for target in TARGETS if target not in df.columns]
    if missing_targets:
        raise ValueError(
            f"Missing target column(s): {missing_targets}. "
            f"Use --input {DEFAULT_RAW_INPUT.relative_to(ROOT)} --clean-input for raw engineered data."
        )

    return df.reset_index(drop=True)


def make_feature_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    drop_cols = [col for col in LEAKAGE_COLS if col in df.columns]
    x = df.drop(columns=drop_cols)
    y = df[target]
    return x, y


def positive_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, 1, None)


def train_one_target(
    df: pd.DataFrame,
    target: str,
    model_dir: Path,
    save_models: bool,
    test_size: float,
    random_state: int,
) -> list[dict[str, float | int | str]]:
    x, y = make_feature_target(df, target)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    results: list[dict[str, float | int | str]] = []
    preprocessor = build_preprocessor(x_train)

    for model_name, estimator in build_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        model = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        model.fit(x_train, y_train)
        predictions = positive_predictions(model.predict(x_test))

        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "target": target,
                "model": model_name,
                "mape_percent": mape,
                "mae": mae,
                "r2": r2,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "feature_count_before_encoding": x_train.shape[1],
            }
        )

        if save_models:
            model_path = model_dir / f"{target}_{model_name}.joblib"
            joblib.dump(model, model_path)

    return results


def log_to_wandb(
    args: argparse.Namespace,
    results_df: pd.DataFrame,
    input_path: Path,
    results_path: Path,
    model_dir: Path,
    row_count: int,
) -> None:
    if not args.wandb:
        return

    wandb = import_wandb()
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config={
            "input": str(input_path),
            "clean_input": args.clean_input,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "targets": TARGETS,
            "models": list(build_models(args.random_state).keys()),
            "row_count": row_count,
            "leakage_columns_dropped_from_features": LEAKAGE_COLS,
        },
    )

    wandb.log({"results": wandb.Table(dataframe=results_df)})
    for _, row in results_df.iterrows():
        prefix = f"{row['target']}/{row['model']}"
        wandb.log(
            {
                f"{prefix}/mape_percent": row["mape_percent"],
                f"{prefix}/mae": row["mae"],
                f"{prefix}/r2": row["r2"],
            }
        )

    best_by_target = results_df.loc[results_df.groupby("target")["mape_percent"].idxmin()]
    for _, row in best_by_target.iterrows():
        wandb.summary[f"best_{row['target']}_model"] = row["model"]
        wandb.summary[f"best_{row['target']}_mape_percent"] = row["mape_percent"]

    results_artifact = wandb.Artifact("model-results", type="results")
    results_artifact.add_file(str(results_path))
    run.log_artifact(results_artifact)

    if not args.no_save_models and model_dir.exists():
        model_artifact = wandb.Artifact("trained-models", type="model")
        for model_path in model_dir.glob("*.joblib"):
            model_artifact.add_file(str(model_path))
        run.log_artifact(model_artifact)

    run.finish()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    results_path = resolve_path(args.results)
    model_dir = resolve_path(args.model_dir)

    df = load_modeling_data(input_path, args.clean_input)

    if not args.no_save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, float | int | str]] = []
    for target in TARGETS:
        all_results.extend(
            train_one_target(
                df=df,
                target=target,
                model_dir=model_dir,
                save_models=not args.no_save_models,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        )

    results_df = pd.DataFrame(all_results).sort_values(["target", "mape_percent"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    log_to_wandb(args, results_df, input_path, results_path, model_dir, len(df))

    print(f"Input: {input_path}")
    print(f"Rows used: {len(df)}")
    print(f"Results: {results_path}")
    if not args.no_save_models:
        print(f"Models: {model_dir}")
    print()
    print(
        results_df.to_string(
            index=False,
            formatters={
                "mape_percent": "{:.2f}".format,
                "mae": "{:.2f}".format,
                "r2": "{:.3f}".format,
            },
        )
    )

    best_by_target = results_df.loc[results_df.groupby("target")["mape_percent"].idxmin()]
    print()
    print("Best model by target:")
    print(
        best_by_target[["target", "model", "mape_percent"]].to_string(
            index=False,
            formatters={"mape_percent": "{:.2f}".format},
        )
    )


if __name__ == "__main__":
    main()
