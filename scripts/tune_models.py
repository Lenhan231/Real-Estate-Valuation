from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_regression_models import (
    DEFAULT_INPUT,
    build_training_model,
    get_property_type_strata,
    make_feature_target,
    score_predictions,
)
from training_models.params import PARAMS_PATH

SUPPORTED_MODELS = ["lightgbm", "catboost", "xgboost"]

SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "lightgbm": {
        "n_estimators": [600, 900, 1200, 1600],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 20, 40, 80],
        "subsample": [0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.75, 0.85, 0.95, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 3.0, 5.0, 10.0],
    },
    "catboost": {
        "iterations": [700, 1000, 1300, 1600],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "depth": [4, 5, 6, 7, 8],
        "l2_leaf_reg": [1, 3, 5, 8, 10],
        "bagging_temperature": [0, 0.5, 1, 2],
        "random_strength": [0.5, 1, 2],
    },
    "xgboost": {
        "n_estimators": [600, 900, 1200, 1600],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 5, 8, 10],
        "subsample": [0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.75, 0.85, 0.95, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [1, 3, 5, 10],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search tuning for tabular real-estate models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Cleaned modeling CSV.")
    parser.add_argument("--target", default="price_vnd", choices=["price_vnd", "price_per_m2"])
    parser.add_argument("--models", nargs="+", default=["lightgbm", "catboost", "xgboost"], choices=SUPPORTED_MODELS)
    parser.add_argument("--trials", type=int, default=30, help="Random trials per model.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PARAMS_PATH, help="JSON file for best tuned params.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sample_params(model_name: str, rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in SEARCH_SPACES[model_name].items()}


def build_estimator(model_name: str, params: dict[str, Any], random_state: int) -> object:
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**params, random_state=random_state, n_jobs=-1, verbose=-1)
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            **params,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_params(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: dict[str, Any],
    cv_folds: int,
    random_state: int,
) -> dict[str, float]:
    strata = get_property_type_strata(x)
    if strata is not None and strata.value_counts(dropna=False).min() >= cv_folds:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(x, strata)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(x)

    scores: list[dict[str, float]] = []
    for train_index, test_index in split_iterator:
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        estimator = build_estimator(model_name, params, random_state)
        model = build_training_model(x_train, estimator)
        model.fit(x_train, y_train)
        scores.append(score_predictions(y_test, model.predict(x_test)))

    return {
        "cv_mape_percent_mean": float(np.mean([score["mape_percent"] for score in scores])),
        "cv_r2_mean": float(np.mean([score["r2"] for score in scores])),
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    df = pd.read_csv(input_path)
    x, y = make_feature_target(df, args.target)
    rng = random.Random(args.random_state)

    existing_params: dict[str, Any] = {}
    if output_path.exists():
        existing_params = json.loads(output_path.read_text(encoding="utf-8"))

    for model_name in args.models:
        best_params: dict[str, Any] | None = None
        best_score: dict[str, float] | None = None
        print(f"Tuning {model_name} on {args.target}...")

        for trial in range(1, args.trials + 1):
            params = sample_params(model_name, rng)
            score = evaluate_params(x, y, model_name, params, args.cv_folds, args.random_state)
            if best_score is None or score["cv_mape_percent_mean"] < best_score["cv_mape_percent_mean"]:
                best_score = score
                best_params = params
            print(
                f"  trial {trial:03d}: mape={score['cv_mape_percent_mean']:.2f} "
                f"r2={score['cv_r2_mean']:.3f}"
            )

        existing_params[model_name] = best_params
        print(f"Best {model_name}: {best_score} params={best_params}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(existing_params, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote tuned params: {output_path}")


if __name__ == "__main__":
    main()
