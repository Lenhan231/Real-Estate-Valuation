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
- Cleaned columns: `44`
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

The current `data/processed/model_results.csv` may be from the latest smoke test. The last full 5-fold global run showed:

### `price_per_m2`

- Best holdout MAPE: `ensemble`, about `25.77%`.
- Best CV MAPE: `lightgbm`, about `25.45%`.
- Best holdout R2: `catboost`, about `0.675`.
- Best CV R2: `ensemble`, about `0.693`.

### `price_vnd`

- Best holdout MAPE: `catboost`, about `26.90%`.
- Best CV MAPE: `lightgbm`, about `25.79%`.
- Best holdout R2: `ensemble`, about `0.723`.
- Best CV R2: `lightgbm`, about `0.725`.

Current practical ranking:

1. `lightgbm`
2. `catboost`
3. `ensemble`
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

## Modeling Interpretation

- The target is right-skewed. The trainer already uses `log1p` on targets and converts predictions back with `expm1`.
- Good models are clustered around 25-27% MAPE and about 0.69-0.72 R2.
- CV and holdout metrics are close for LightGBM/CatBoost/XGBoost/ensemble, which suggests the results are not just one lucky split.
- Next useful work is probably tuning LightGBM/CatBoost and improving location features, not adding many more model families.
