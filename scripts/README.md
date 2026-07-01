# Modeling Handoff Notes

This note summarizes the modeling/training refactor and current context so another agent can continue quickly.

## Current Training Structure

- `scripts/train_regression_models.py` is the shared training pipeline.
  - Loads cleaned modeling data.
  - Splits train/test.
  - Builds preprocessing.
  - Applies `log1p` target transform with `TransformedTargetRegressor`.
  - Trains selected registered models.
  - Uses property-type stratified train/test and CV splits when `property_type` is available.
  - Supports `--separate-property-models` to append specialist evaluations for each property type.
  - Supports `--routed-property-models` to evaluate a property-type router over specialist models.
  - Supports `--save-predictions` for row-level error diagnostics.
  - Supports `--derive-price-vnd-from-price-per-m2` to test deriving total price from predicted unit price.
  - Writes metrics, feature importance, plots, saved models, and optional W&B logs.
- Individual model definitions live in `scripts/training_models/`.
- Shortcut launchers exist for single-model training:
  - `scripts/trainers/train_linear_regression.py`
  - `scripts/trainers/train_random_forest.py`
  - `scripts/trainers/train_xgboost.py`
  - `scripts/trainers/train_lightgbm.py`
  - `scripts/trainers/train_catboost.py`
  - `scripts/trainers/train_ensemble.py`
- `scripts/run_training.py` is the workflow runner.
  - `--refresh-data` fetches `origin/main`, restores only `data/`, cleans data, then trains.
  - `--clean-data` cleans current local modeling data before training.
  - Extra trainer args go after `--`, for example `-- --cv-folds 5 --wandb`.

## Registered Models

Registered in `scripts/training_models/registry.py`:

- `linear_regression`
- `random_forest`
- `xgboost`
- `lightgbm`
- `catboost`
- `ensemble`

The current `ensemble` is a `VotingRegressor` over LightGBM, CatBoost, and XGBoost.

To add a model:

1. Create `scripts/training_models/<model_name>.py`.
2. Define `MODEL_NAME = "<model_name>"`.
3. Define `build_model(random_state)`.
4. Import/register it in `scripts/training_models/registry.py`.

Do not edit `train_regression_models.py` for normal model additions. Edit it only when changing shared training logic, metrics, preprocessing, saving, W&B logging, or CV behavior.

## Data Cleaning Changes

`scripts/model_cleaning.py` was updated for modeling:

- Keeps `locality` instead of dropping it.
- Adds `locality` to `MODEL_TEXT_COLS`, so the trainer one-hot encodes it automatically.
- Recovers useful `post_day` features:
  - `post_month`
  - `post_day_of_month`
  - `post_weekday`
  - `post_is_weekend`
  - `post_age_days`
- Does not add `post_year` because the user considered same-year signal less useful.
- Adds `nearest_school_km` to median imputation.
- Drops price-per-m2 outliers with a per-property-type IQR filter.
- Adds unsupervised lat/lon KMeans cluster features: `location_cluster_20`, `location_cluster_50`, `location_cluster_100`.
- Adds `locality_listing_count`.
- Adds property-type interaction features because `nha_mat_tien` and `nha_trong_hem` have very different price distributions:
  - `property_type_area_m2`
  - `property_type_road_width_m`
  - `property_type_width_m`
  - `property_type_length_m`
  - `property_type_num_floors`
  - `property_type_distance_to_center_km`
  - `property_type_locality_population_density`

After latest cleaning, the modeling file had:

- Raw rows: `3780`
- Cleaned rows: `3295`
- Cleaned columns: `48`
- Missing values: `0`

Latest IQR filtering report:

- `property_type=0` (`nha_trong_hem`): `1792 -> 1751`, removed `41`.
- `property_type=1` (`nha_mat_tien`): `1912 -> 1836`, removed `76`.

Default cleaned input:

```bash
data/processed/real_estate_cleaned_2.csv
```

## Commands

Clean current local data:

```bash
.venv/bin/python scripts/clean_model_data.py
```

Refresh only `data/` from `origin/main`, clean it, and train:

```bash
.venv/bin/python scripts/run_training.py --refresh-data
```

Refresh and clean only:

```bash
.venv/bin/python scripts/run_training.py --refresh-data --skip-training
```

Train all registered models:

```bash
.venv/bin/python scripts/train_regression_models.py
```

Train selected models:

```bash
.venv/bin/python scripts/train_regression_models.py --models lightgbm catboost ensemble
```

Train with 5-fold validation:

```bash
.venv/bin/python scripts/train_regression_models.py --cv-folds 5
```

Train and log W&B:

```bash
.venv/bin/python scripts/train_regression_models.py --cv-folds 5 --wandb
```

Compare global models against property-type specialist models:

```bash
.venv/bin/python scripts/train_regression_models.py --models lightgbm catboost ensemble --cv-folds 5 --separate-property-models --wandb
```

Save prediction-level diagnostics, evaluate routed specialists, and test derived total price:

```bash
.venv/bin/python scripts/train_regression_models.py --models lightgbm catboost ensemble --cv-folds 5 --separate-property-models --routed-property-models --derive-price-vnd-from-price-per-m2 --save-predictions --wandb
```

Analyze saved prediction errors:

```bash
.venv/bin/python scripts/analyze_model_errors.py --predictions data/processed/model_predictions.csv
```

Run a small random-search tuning job:

```bash
.venv/bin/python scripts/tune_models.py --models lightgbm catboost xgboost --target price_vnd --trials 50 --cv-folds 5
```

Using the runner with W&B:

```bash
.venv/bin/python scripts/run_training.py --models catboost lightgbm ensemble -- --cv-folds 5 --separate-property-models --wandb
```

## Dependency Notes

Modeling dependencies are in `scripts/requirements_modeling.txt`.

LightGBM was installed, but one terminal output showed a missing system library:

```text
OSError: libgomp.so.1: cannot open shared object file: No such file or directory
```

Fix on Ubuntu/WSL:

```bash
sudo apt update
sudo apt install libgomp1
```

Then rerun training from `.venv/bin/python`.

## Current Results Snapshot

Latest tuned full benchmark was written on `2026-07-01 23:51 +0700` to:

- `data/processed/model_results.csv`
- `data/processed/model_predictions.csv`

Run shape:

- `33` result rows.
- `3295` modeling rows.
- `5` CV folds.
- Models: `lightgbm`, `catboost`, `ensemble`.
- Scopes: `global`, `property_type_0`, `property_type_1`, `routed_property`.
- Targets: `price_vnd`, `price_per_m2`, and derived `price_vnd_from_price_per_m2`.
- Tuned params loaded from `scripts/training_models/tuned_params.json`.

Use `cv_mape_percent_mean` and `cv_r2_mean` as the official comparison metrics.

### `price_per_m2`

- Best global CV MAPE: `ensemble`, `24.00%`.
- Best global CV R2: `lightgbm`, `0.728`.
- Best specialist CV MAPE: `property_type_0 / ensemble`, `19.57%`.
- Hardest specialist: `property_type_1 / ensemble`, `29.42%` CV MAPE.

### `price_vnd`

- Best global CV MAPE: `ensemble`, `23.72%`.
- Best global CV R2: `ensemble`, `0.777`.
- Best global holdout MAPE: `catboost`, `24.92%`.
- Best global holdout R2: `lightgbm`, `0.754`.
- Best specialist CV MAPE: `property_type_0 / ensemble`, `19.66%`.
- Best `property_type_1` specialist by CV MAPE: `catboost`, `28.73%` CV MAPE and `0.718` CV R2.
- Best `property_type_1` specialist by CV R2: `ensemble`, `28.92%` CV MAPE and `0.726` CV R2.

### Derived `price_vnd_from_price_per_m2`

- Best global CV MAPE: `ensemble`, `24.00%`.
- Best global holdout R2: `catboost`, `0.820`.
- Best specialist CV MAPE: `property_type_0 / ensemble`, `19.57%`.
- Best specialist CV R2: `property_type_0 / catboost`, `0.791`.
- Derived total price still does not beat direct `price_vnd` globally by CV MAPE, but it remains useful as a comparison target.

### Routed Specialist Models

`routed_property` trains specialist models by `property_type` and routes each prediction to the matching specialist.

Latest results:

- `routed_property / price_vnd / ensemble`: `24.28%` CV MAPE, `0.773` CV R2.
- `global / price_vnd / ensemble`: `23.72%` CV MAPE, `0.777` CV R2.

The routed approach is not the winner yet. Keep it as an experiment, not the default production choice.

Current practical ranking:

1. `ensemble`
2. `catboost`
3. `lightgbm`
4. `xgboost`
5. `random_forest`
6. `linear_regression`

`linear_regression` is only a baseline. It is unstable for `price_vnd`, including negative R2 and explosive CV metrics.

## Latest Tuning Changes

- Tuned LightGBM defaults toward lower learning rate, more trees, subsampling, and regularization.
- Tuned CatBoost defaults toward more iterations, lower learning rate, stronger L2 regularization, and bagging.
- Tuned XGBoost defaults toward more trees, lower learning rate, stronger regularization, and conservative child weight.
- Updated `ensemble` to combine LightGBM, CatBoost, and XGBoost with weights `[2, 2, 1]`.
- Added property-type stratification for holdout and K-fold validation.
- Added property-type interaction features. In a CatBoost smoke test, `property_type_area_m2` and `property_type_road_width_m` appeared among the strongest features, supporting the user's hypothesis.
- Added `--separate-property-models` comparison mode. It appends `global`, `property_type_0`, and `property_type_1` rows to `model_results.csv`; feature importance gets the same `training_scope` column.
- A CatBoost 2-fold smoke test completed successfully after IQR filtering. Specialist models used normal K-fold, while global used property-type stratified CV.
- Added CV-safe locality target encoding inside the sklearn pipeline. It is fit only on each train fold and appears as `locality_cv_target_median` / `locality_cv_target_count`.
- Added `--save-predictions`, `--routed-property-models`, and `--derive-price-vnd-from-price-per-m2`.
- Added `scripts/analyze_model_errors.py` for grouped error diagnostics.
- Added `scripts/tune_models.py`; it writes best random-search params to `scripts/training_models/tuned_params.json`, and model builders load that file when present.
- Current tuned params file contains entries for `lightgbm`, `catboost`, and `xgboost`; `ensemble` uses those tuned base models through their builders.
- Tuning improved the best global `price_vnd` CV MAPE from about `24.09%` to `23.72%`, but global CV R2 moved from about `0.782` to `0.777`. This is a small MAPE win, not a breakthrough.

Latest full-run diagnostics show the hardest segment is still `property_type=1` (`nha_mat_tien`), especially:

- Low `price_per_m2` street-front homes.
- Large-area homes, especially `area_m2 > 150`.
- Sparse or noisy localities such as `phường tăng nhơn phú`.
- Some road-width buckets, especially small road width inside `property_type=1`.

The best `property_type_0` (`nha_trong_hem`) specialist is already around the `<20%` CV MAPE milestone. The global score is mostly being held back by `property_type_1`.

Training output now suppresses the harmless LightGBM feature-name warning and prints progress lines such as:

```text
[train] global | price_vnd | lightgbm: fit holdout model
[cv] global | price_vnd | lightgbm: fold 1/5 using property_type_stratified
[done] global | price_vnd | lightgbm: holdout MAPE=26.16% CV MAPE=24.85%
```

## Modeling Interpretation

- The target is right-skewed. The trainer already uses `log1p` on targets and converts predictions back with `expm1`.
- Current best global model is clustered around `23.7-24.0%` CV MAPE and `0.77-0.78` CV R2.
- `property_type_0` can reach about `19.5-20%` CV MAPE depending on target strategy.
- `property_type_1` remains around `28.5-30%` CV MAPE and is the main performance bottleneck.
- Next useful work is focused `property_type_1` cleanup/feature work: better street/locality normalization, manual review of low-unit-price street-front rows, better large-area handling, and possibly more precise location features. Adding many more model families is less likely to move the score than fixing this segment.
