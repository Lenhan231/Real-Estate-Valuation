from __future__ import annotations

import argparse
import locale
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_cleaning import clean_for_modeling
from training_models import available_models, build_models

DEFAULT_INPUT = ROOT / "data" / "processed" / "real_estate_cleaned_2.csv"
DEFAULT_RAW_INPUT = ROOT / "data" / "processed" / "alonhadat_features.csv"
DEFAULT_RESULTS = ROOT / "data" / "processed" / "model_results.csv"
DEFAULT_IMPORTANCE_RESULTS = ROOT / "data" / "processed" / "feature_importance.csv"
DEFAULT_IMPORTANCE_PLOT_DIR = ROOT / "data" / "processed" / "feature_importance_plots"
DEFAULT_WANDB_DIR = Path(tempfile.gettempdir()) / "real_estate_valuation_wandb"
DEFAULT_MODEL_DIR = ROOT / "models"
TARGETS = ["price_vnd", "price_per_m2"]
LEAKAGE_COLS = ["price_vnd", "price_per_m2"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_FEATURES_TO_PRINT = 15


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def can_encode_for_console(value: Path) -> bool:
    encoding = locale.getpreferredencoding(False)
    try:
        str(value).encode(encoding)
    except UnicodeEncodeError:
        return False
    return True


def parse_args(default_models: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline/comparison/final regressors for property price targets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input cleaned modeling CSV. Defaults to data/processed/real_estate_cleaned_2.csv.",
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
    parser.add_argument(
        "--importance-results",
        type=Path,
        default=DEFAULT_IMPORTANCE_RESULTS,
        help="Output CSV for fitted model feature importance.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=TOP_FEATURES_TO_PRINT,
        help="Number of top features to print and plot per target/model.",
    )
    parser.add_argument(
        "--importance-plot-dir",
        type=Path,
        default=DEFAULT_IMPORTANCE_PLOT_DIR,
        help="Output directory for feature importance PNG charts.",
    )
    parser.add_argument(
        "--no-importance-plots",
        action="store_true",
        help="Skip writing feature importance PNG charts.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory for fitted model files.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        choices=available_models(),
        help=(
            "One or more models to train. Defaults to all registered models, or the model "
            "selected by a per-model script."
        ),
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="Number of K-fold validation folds. Use 1 to skip cross-validation.",
    )
    parser.add_argument(
        "--separate-property-models",
        action="store_true",
        help="Also train/evaluate separate models for each property_type value.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Train and report metrics without writing .joblib model files.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics and artifacts to Weights & Biases.")
    parser.add_argument("--wandb-project", default="real-estate-valuation", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team name.")
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=None,
        help=(
            "Local W&B run directory. Defaults to ./wandb when the project path is console-safe, "
            "otherwise a temp directory with an ASCII path."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode. Use offline if you are not logged in or have no network.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def path_for_wandb_config(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_wandb_dir(path: Path | None) -> Path | None:
    if path is not None:
        return resolve_path(path)
    if can_encode_for_console(ROOT):
        return None
    return DEFAULT_WANDB_DIR


def configure_wandb_environment(wandb_dir: Path | None) -> None:
    if wandb_dir is None:
        return

    env_dirs = {
        "WANDB_DIR": wandb_dir,
        "WANDB_DATA_DIR": wandb_dir / "data",
        "WANDB_CACHE_DIR": wandb_dir / "cache",
        "WANDB_CONFIG_DIR": wandb_dir / "config",
        "WANDB_ARTIFACT_DIR": wandb_dir / "artifacts",
    }
    for name, path in env_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))


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


def build_training_model(x: pd.DataFrame, estimator: object) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(x)),
            ("model", estimator),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def score_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    predictions = positive_predictions(predictions)
    return {
        "mape_percent": mean_absolute_percentage_error(y_true, predictions) * 100,
        "mae": mean_absolute_error(y_true, predictions),
        "r2": r2_score(y_true, predictions),
    }


def get_property_type_strata(x: pd.DataFrame) -> pd.Series | None:
    if "property_type" not in x.columns:
        return None

    strata = x["property_type"]
    if strata.isna().any() or strata.nunique(dropna=True) < 2:
        return None

    counts = strata.value_counts(dropna=False)
    if counts.min() < 2:
        return None

    return strata


def get_preprocessed_feature_sources(preprocessor: ColumnTransformer) -> list[str]:
    feature_sources: list[str] = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or transformer_name == "remainder":
            continue

        column_names = list(columns)
        if not column_names:
            continue

        if transformer_name == "numeric":
            feature_sources.extend(column_names)
        elif transformer_name == "categorical":
            onehot = transformer.named_steps["onehot"]
            for column_name, categories in zip(column_names, onehot.categories_):
                feature_sources.extend([column_name] * len(categories))

    return feature_sources


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = preprocessor.get_feature_names_out()
    return [name.split("__", maxsplit=1)[-1] for name in names]


def extract_feature_importance(
    fitted_model: TransformedTargetRegressor,
    target: str,
    model_name: str,
    training_scope: str,
) -> list[dict[str, float | int | str]]:
    pipeline = fitted_model.regressor_
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    encoded_feature_names = get_preprocessed_feature_names(preprocessor)
    feature_sources = get_preprocessed_feature_sources(preprocessor)

    signed_effects: np.ndarray | None = None
    if hasattr(estimator, "feature_importances_"):
        raw_importances = np.asarray(estimator.feature_importances_, dtype=float)
        importance_type = "model_feature_importance"
    elif hasattr(estimator, "coef_"):
        signed_effects = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        raw_importances = np.abs(signed_effects)
        importance_type = "absolute_coefficient"
    else:
        return []

    if len(raw_importances) != len(encoded_feature_names) or len(raw_importances) != len(feature_sources):
        return []

    encoded_importance = pd.DataFrame(
        {
            "source_feature": feature_sources,
            "encoded_feature": encoded_feature_names,
            "importance": raw_importances,
            "signed_effect": signed_effects if signed_effects is not None else np.nan,
        }
    )
    total_importance = encoded_importance["importance"].sum()
    grouped = (
        encoded_importance.groupby("source_feature", as_index=False)
        .agg(
            importance=("importance", "sum"),
            signed_effect=("signed_effect", lambda values: values.iloc[0] if len(values) == 1 else np.nan),
            encoded_feature_count=("encoded_feature", "count"),
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["importance_normalized"] = (
        grouped["importance"] / total_importance if total_importance > 0 else 0
    )
    grouped["target"] = target
    grouped["model"] = model_name
    grouped["training_scope"] = training_scope
    grouped["importance_type"] = importance_type

    columns = [
        "training_scope",
        "target",
        "model",
        "rank",
        "source_feature",
        "importance",
        "importance_normalized",
        "signed_effect",
        "encoded_feature_count",
        "importance_type",
    ]
    return grouped[columns].to_dict("records")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    plot_dir: Path,
    top_n: int,
) -> list[Path]:
    if feature_importance_df.empty or top_n <= 0:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    model_colors = {
        "catboost": "#8B5CF6",
        "ensemble": "#6B7280",
        "lightgbm": "#06B6D4",
        "linear_regression": "#4C78A8",
        "random_forest": "#F58518",
        "xgboost": "#54A24B",
    }

    group_cols = ["target", "model"]
    if "training_scope" in feature_importance_df.columns:
        group_cols = ["training_scope", *group_cols]

    for group_key, group in feature_importance_df.groupby(group_cols, sort=True):
        if len(group_cols) == 3:
            training_scope, target, model_name = group_key
        else:
            target, model_name = group_key
            training_scope = "global"
        top_features = group.nsmallest(top_n, "rank").sort_values("importance_normalized", ascending=True)
        if top_features.empty:
            continue

        height = max(4.5, 0.38 * len(top_features) + 1.8)
        fig, ax = plt.subplots(figsize=(10, height))
        values = top_features["importance_normalized"] * 100
        labels = top_features["source_feature"]
        color = model_colors.get(str(model_name), "#6B7280")

        ax.barh(labels, values, color=color, alpha=0.88)
        ax.set_xlabel("Normalized importance (%)")
        ax.set_ylabel("")
        title_scope = f"{training_scope} - " if training_scope != "global" else ""
        ax.set_title(f"Top {len(top_features)} features for {title_scope}{target} - {model_name}")
        ax.grid(axis="x", linestyle="--", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        max_value = values.max()
        for index, value in enumerate(values):
            ax.text(
                value + max_value * 0.012,
                index,
                f"{value:.1f}%",
                va="center",
                fontsize=9,
            )

        ax.set_xlim(0, max_value * 1.16 if max_value > 0 else 1)
        fig.tight_layout()

        filename = (
            f"{slugify(str(training_scope))}_{slugify(str(target))}_{slugify(str(model_name))}"
            f"_top_{len(top_features)}_feature_importance.png"
        )
        plot_path = plot_dir / filename
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def train_one_target(
    df: pd.DataFrame,
    target: str,
    training_scope: str,
    model_names: list[str] | None,
    model_dir: Path,
    save_models: bool,
    test_size: float,
    random_state: int,
    cv_folds: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    x, y = make_feature_target(df, target)
    strata = get_property_type_strata(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )

    results: list[dict[str, float | int | str]] = []
    feature_importances: list[dict[str, float | int | str]] = []

    for model_name, estimator in build_models(random_state, model_names).items():
        model = build_training_model(x_train, estimator)
        model.fit(x_train, y_train)
        holdout_scores = score_predictions(y_test, model.predict(x_test))
        cv_scores = cross_validate_model(x, y, model_name, random_state, cv_folds, strata)

        result = {
            "training_scope": training_scope,
            "target": target,
            "model": model_name,
            "mape_percent": holdout_scores["mape_percent"],
            "mae": holdout_scores["mae"],
            "r2": holdout_scores["r2"],
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "feature_count_before_encoding": x_train.shape[1],
        }
        result.update(cv_scores)
        results.append(result)

        if save_models:
            model_prefix = "" if training_scope == "global" else f"{training_scope}_"
            model_path = model_dir / f"{model_prefix}{target}_{model_name}.joblib"
            joblib.dump(model, model_path)

        feature_importances.extend(extract_feature_importance(model, target, model_name, training_scope))

    return results, feature_importances


def cross_validate_model(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    random_state: int,
    cv_folds: int,
    strata: pd.Series | None,
) -> dict[str, float | int | str]:
    if cv_folds <= 1:
        return {"cv_folds": 1}
    if cv_folds > len(x):
        raise ValueError(f"--cv-folds must be <= row count ({len(x)}). Got {cv_folds}.")

    fold_scores: list[dict[str, float]] = []
    if strata is not None and strata.value_counts(dropna=False).min() >= cv_folds:
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(x, strata)
        split_strategy = "property_type_stratified"
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(x)
        split_strategy = "kfold"

    for train_index, test_index in split_iterator:
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        estimator = build_models(random_state, [model_name])[model_name]
        model = build_training_model(x_train, estimator)
        model.fit(x_train, y_train)
        fold_scores.append(score_predictions(y_test, model.predict(x_test)))

    return {
        "cv_folds": cv_folds,
        "cv_split_strategy": split_strategy,
        "cv_mape_percent_mean": float(np.mean([score["mape_percent"] for score in fold_scores])),
        "cv_mape_percent_std": float(np.std([score["mape_percent"] for score in fold_scores], ddof=1)),
        "cv_mae_mean": float(np.mean([score["mae"] for score in fold_scores])),
        "cv_mae_std": float(np.std([score["mae"] for score in fold_scores], ddof=1)),
        "cv_r2_mean": float(np.mean([score["r2"] for score in fold_scores])),
        "cv_r2_std": float(np.std([score["r2"] for score in fold_scores], ddof=1)),
    }


def build_training_scopes(
    df: pd.DataFrame,
    separate_property_models: bool,
) -> list[tuple[str, pd.DataFrame]]:
    scopes = [("global", df)]
    if not separate_property_models:
        return scopes

    if "property_type" not in df.columns:
        raise ValueError("--separate-property-models requires a property_type column.")

    property_types = sorted(df["property_type"].dropna().unique().tolist())
    if not property_types:
        raise ValueError("--separate-property-models found no property_type values.")

    for property_type in property_types:
        scope = f"property_type_{property_type:g}"
        subset = df[df["property_type"] == property_type].reset_index(drop=True)
        scopes.append((scope, subset))

    return scopes


def log_to_wandb(
    args: argparse.Namespace,
    results_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    input_path: Path,
    results_path: Path,
    importance_path: Path,
    importance_plot_paths: list[Path],
    model_dir: Path,
    row_count: int,
) -> None:
    if not args.wandb or args.wandb_mode == "disabled":
        return

    wandb_dir = resolve_wandb_dir(args.wandb_dir)
    if wandb_dir is not None:
        wandb_dir.mkdir(parents=True, exist_ok=True)
    configure_wandb_environment(wandb_dir)
    wandb = import_wandb()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        dir=str(wandb_dir) if wandb_dir is not None else None,
        config={
            "input": path_for_wandb_config(input_path),
            "clean_input": args.clean_input,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "cv_folds": args.cv_folds,
            "separate_property_models": args.separate_property_models,
            "targets": TARGETS,
            "models": list(build_models(args.random_state, args.models).keys()),
            "row_count": row_count,
            "leakage_columns_dropped_from_features": LEAKAGE_COLS,
            "top_features": args.top_features,
        },
    )

    wandb.log({"results": wandb.Table(dataframe=results_df)})
    if not feature_importance_df.empty:
        wandb.log({"feature_importance": wandb.Table(dataframe=feature_importance_df)})
    for plot_path in importance_plot_paths:
        wandb.log({f"feature_importance_plots/{plot_path.stem}": wandb.Image(str(plot_path))})

    for _, row in results_df.iterrows():
        prefix = f"{row.get('training_scope', 'global')}/{row['target']}/{row['model']}"
        wandb.log(
            {
                f"{prefix}/mape_percent": row["mape_percent"],
                f"{prefix}/mae": row["mae"],
                f"{prefix}/r2": row["r2"],
            }
        )
        if row.get("cv_folds", 1) and row.get("cv_folds", 1) > 1:
            wandb.log(
                {
                    f"{prefix}/cv_mape_percent_mean": row["cv_mape_percent_mean"],
                    f"{prefix}/cv_mape_percent_std": row["cv_mape_percent_std"],
                    f"{prefix}/cv_mae_mean": row["cv_mae_mean"],
                    f"{prefix}/cv_mae_std": row["cv_mae_std"],
                    f"{prefix}/cv_r2_mean": row["cv_r2_mean"],
                    f"{prefix}/cv_r2_std": row["cv_r2_std"],
                }
            )

    best_by_target = results_df.loc[results_df.groupby("target")["mape_percent"].idxmin()]
    for _, row in best_by_target.iterrows():
        wandb.summary[f"best_{row['target']}_model"] = row["model"]
        wandb.summary[f"best_{row['target']}_training_scope"] = row.get("training_scope", "global")
        wandb.summary[f"best_{row['target']}_mape_percent"] = row["mape_percent"]

    results_artifact = wandb.Artifact("model-results", type="results")
    results_artifact.add_file(str(results_path))
    if importance_path.exists():
        results_artifact.add_file(str(importance_path))
    for plot_path in importance_plot_paths:
        results_artifact.add_file(str(plot_path))
    run.log_artifact(results_artifact)

    if not args.no_save_models and model_dir.exists():
        model_artifact = wandb.Artifact("trained-models", type="model")
        for model_path in model_dir.glob("*.joblib"):
            model_artifact.add_file(str(model_path))
        run.log_artifact(model_artifact)

    run.finish()


def main(default_models: list[str] | None = None) -> None:
    configure_console_encoding()

    args = parse_args(default_models)
    input_path = resolve_path(args.input)
    results_path = resolve_path(args.results)
    importance_path = resolve_path(args.importance_results)
    importance_plot_dir = resolve_path(args.importance_plot_dir)
    model_dir = resolve_path(args.model_dir)

    df = load_modeling_data(input_path, args.clean_input)

    if not args.no_save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, float | int | str]] = []
    all_feature_importances: list[dict[str, float | int | str]] = []
    training_scopes = build_training_scopes(df, args.separate_property_models)
    for training_scope, scope_df in training_scopes:
        for target in TARGETS:
            target_results, target_feature_importances = train_one_target(
                df=scope_df,
                target=target,
                training_scope=training_scope,
                model_names=args.models,
                model_dir=model_dir,
                save_models=not args.no_save_models,
                test_size=args.test_size,
                random_state=args.random_state,
                cv_folds=args.cv_folds,
            )
            all_results.extend(target_results)
            all_feature_importances.extend(target_feature_importances)

    results_df = pd.DataFrame(all_results).sort_values(["target", "training_scope", "mape_percent"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    feature_importance_df = pd.DataFrame(all_feature_importances)
    if not feature_importance_df.empty:
        feature_importance_df = feature_importance_df.sort_values(["training_scope", "target", "model", "rank"])
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance_df.to_csv(importance_path, index=False)
    importance_plot_paths: list[Path] = []
    if not args.no_importance_plots:
        importance_plot_paths = plot_feature_importance(
            feature_importance_df,
            importance_plot_dir,
            args.top_features,
        )

    log_to_wandb(
        args,
        results_df,
        feature_importance_df,
        input_path,
        results_path,
        importance_path,
        importance_plot_paths,
        model_dir,
        len(df),
    )

    print(f"Input: {input_path}")
    print(f"Rows used: {len(df)}")
    if args.separate_property_models:
        print("Training scopes:")
        for training_scope, scope_df in training_scopes:
            print(f"  {training_scope}: {len(scope_df)} rows")
    print(f"Results: {results_path}")
    print(f"Feature importance: {importance_path}")
    if importance_plot_paths:
        print(f"Feature importance plots: {importance_plot_dir}")
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
                "cv_mape_percent_mean": "{:.2f}".format,
                "cv_mape_percent_std": "{:.2f}".format,
                "cv_mae_mean": "{:.2f}".format,
                "cv_mae_std": "{:.2f}".format,
                "cv_r2_mean": "{:.3f}".format,
                "cv_r2_std": "{:.3f}".format,
            },
        )
    )

    best_by_target = results_df.loc[results_df.groupby("target")["mape_percent"].idxmin()]
    print()
    print("Best model by target:")
    print(
        best_by_target[["target", "training_scope", "model", "mape_percent"]].to_string(
            index=False,
            formatters={"mape_percent": "{:.2f}".format},
        )
    )

    if not feature_importance_df.empty and args.top_features > 0:
        top_features = feature_importance_df[feature_importance_df["rank"] <= args.top_features]
        print()
        print(f"Top {args.top_features} feature importances by target/model:")
        print(
            top_features[
                [
                    "target",
                    "training_scope",
                    "model",
                    "rank",
                    "source_feature",
                    "importance_normalized",
                    "importance_type",
                ]
            ].to_string(
                index=False,
                formatters={"importance_normalized": "{:.4f}".format},
            )
        )


if __name__ == "__main__":
    main()
