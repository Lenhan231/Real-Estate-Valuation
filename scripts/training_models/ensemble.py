from __future__ import annotations

from sklearn.ensemble import VotingRegressor

from .catboost import build_model as build_catboost
from .lightgbm import build_model as build_lightgbm
from .xgboost import build_model as build_xgboost

MODEL_NAME = "ensemble"


def build_model(random_state: int | None = None) -> VotingRegressor:
    return VotingRegressor(
        estimators=[
            ("lightgbm", build_lightgbm(random_state)),
            ("catboost", build_catboost(random_state)),
            ("xgboost", build_xgboost(random_state)),
        ],
        weights=[2, 2, 1],
        n_jobs=-1,
    )
