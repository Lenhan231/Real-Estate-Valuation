from __future__ import annotations

from xgboost import XGBRegressor

MODEL_NAME = "xgboost"


def build_model(random_state: int | None = None) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
