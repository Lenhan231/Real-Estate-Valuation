from __future__ import annotations

from sklearn.linear_model import LinearRegression

MODEL_NAME = "linear_regression"


def build_model(random_state: int | None = None) -> LinearRegression:
    return LinearRegression()
