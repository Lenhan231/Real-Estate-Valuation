from __future__ import annotations

MODEL_NAME = "catboost"


def build_model(random_state: int | None = None) -> object:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "CatBoost is not installed. Install modeling dependencies with: "
            "python -m pip install -r scripts/requirements_modeling.txt"
        ) from exc

    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=5,
        bagging_temperature=0.5,
        random_strength=1,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
