from __future__ import annotations

from xgboost import XGBRegressor

from .params import load_tuned_params

MODEL_NAME = "xgboost"


def build_model(random_state: int | None = None) -> XGBRegressor:
    params = {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(load_tuned_params(MODEL_NAME))
    return XGBRegressor(**params)
