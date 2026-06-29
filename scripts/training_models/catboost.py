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
        iterations=700,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
