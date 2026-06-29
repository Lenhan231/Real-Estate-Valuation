from __future__ import annotations

MODEL_NAME = "lightgbm"


def build_model(random_state: int | None = None) -> object:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Install modeling dependencies with: "
            "python -m pip install -r scripts/requirements_modeling.txt"
        ) from exc

    return LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
