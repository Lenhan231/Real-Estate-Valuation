from __future__ import annotations

from .params import load_tuned_params

MODEL_NAME = "lightgbm"


def build_model(random_state: int | None = None) -> object:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Install modeling dependencies with: "
            "python -m pip install -r scripts/requirements_modeling.txt"
        ) from exc

    params = {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(load_tuned_params(MODEL_NAME))
    return LGBMRegressor(**params)
