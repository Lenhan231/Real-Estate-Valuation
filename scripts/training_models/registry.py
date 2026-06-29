from __future__ import annotations

from collections.abc import Iterable

from . import catboost, ensemble, lightgbm, linear_regression, random_forest, xgboost

MODEL_BUILDERS = {
    catboost.MODEL_NAME: catboost.build_model,
    ensemble.MODEL_NAME: ensemble.build_model,
    lightgbm.MODEL_NAME: lightgbm.build_model,
    linear_regression.MODEL_NAME: linear_regression.build_model,
    random_forest.MODEL_NAME: random_forest.build_model,
    xgboost.MODEL_NAME: xgboost.build_model,
}


def available_models() -> list[str]:
    return sorted(MODEL_BUILDERS)


def build_model(model_name: str, random_state: int) -> object:
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as exc:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {choices}") from exc
    return builder(random_state)


def build_models(random_state: int, model_names: Iterable[str] | None = None) -> dict[str, object]:
    selected_model_names = list(model_names) if model_names is not None else available_models()
    return {model_name: build_model(model_name, random_state) for model_name in selected_model_names}
