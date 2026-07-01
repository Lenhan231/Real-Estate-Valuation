from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PARAMS_PATH = Path(__file__).resolve().parent / "tuned_params.json"


def load_tuned_params(model_name: str) -> dict[str, Any]:
    if not PARAMS_PATH.exists():
        return {}

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        all_params = json.load(file)

    params = all_params.get(model_name, {})
    if not isinstance(params, dict):
        return {}
    return params
