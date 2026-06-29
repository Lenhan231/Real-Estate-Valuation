from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

MODEL_NAME = "random_forest"


def build_model(random_state: int | None = None) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
