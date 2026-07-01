from __future__ import annotations

from .params import load_tuned_params

MODEL_NAME = "catboost"


def build_model(random_state: int | None = None) -> object:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "CatBoost is not installed. Install modeling dependencies with: "
            "python -m pip install -r scripts/requirements_modeling.txt"
        ) from exc

    params = {
        "iterations": 1000,
        "learning_rate": 0.025,
        "depth": 6,
        "l2_leaf_reg": 5,
        "bagging_temperature": 0.5,
        "random_strength": 1,
        "loss_function": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(load_tuned_params(MODEL_NAME))
    return CatBoostRegressor(**params)
