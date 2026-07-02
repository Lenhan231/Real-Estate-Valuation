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

- Supabase raw backup rows: `8561` in `data/processed/alonhadat_features_supabase_raw.csv`.
- Deduped feature rows: `8541` in `data/processed/alonhadat_features.csv`.
- Duplicate feature rows removed: `20`.
- Duplicate `listing_id` after recovery: `0`.
- Duplicate `link` after recovery: `0`.
- Cleaned modeling rows: `7546`.
- Cleaned columns: `48`.
- Missing values: `10`.

Latest IQR filtering report:

- `property_type=0` (`nha_trong_hem`): `4160 -> 4010`, removed `150`, bounds `[-27290482.95, 283040956.44]`.
- `property_type=1` (`nha_mat_tien`): `4208 -> 3992`, removed `216`, bounds `[-184709119.50, 696737421.38]`.

Cleaned property-type distribution:

- `property_type=0`: `3753` rows.
- `property_type=1`: `3793` rows.

Default cleaned input:

```bash
data/processed/real_estate_cleaned_2.csv
```

## Commands

Clean current local data:

```bash
.venv/bin/python scripts/clean_model_data.py
```

Recover feature rows from Supabase, save a raw backup, and write the deduped feature file:

```bash
.venv/bin/python scripts/recover_features_from_supabase.py
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

Latest tuned full benchmark was reviewed on `2026-07-02 21:13 +0700` and uses the recovered and deduped Supabase feature data. Outputs were written to:

- `data/processed/model_results.csv`
- `data/processed/model_predictions.csv`

Run shape:

- `33` result rows.
- `7546` modeling rows.
- `5` CV folds.
- Models: `lightgbm`, `catboost`, `ensemble`.
- Scopes: `global`, `property_type_0`, `property_type_1`, `routed_property`.
- Targets: `price_vnd`, `price_per_m2`, and derived `price_vnd_from_price_per_m2`.
- Tuned params loaded from `scripts/training_models/tuned_params.json`.

Use `cv_mape_percent_mean` and `cv_r2_mean` as the official comparison metrics.

### `price_per_m2`

- Best global CV MAPE: `ensemble`, `20.31%`.
- Best global CV R2: `ensemble`, `0.784`.
- Best global holdout MAPE: `ensemble`, `19.21%`.
- Best specialist CV MAPE: `property_type_0 / ensemble`, `16.88%`.
- Hardest specialist: `property_type_1 / ensemble`, `24.19%` CV MAPE.

### `price_vnd`

- Best global CV MAPE: `catboost`, `20.50%`.
- Best global CV R2: `ensemble`, `0.773`.
- Best global holdout MAPE: `catboost`, `19.34%`.
- Best global holdout R2: `ensemble`, `0.778`.
- Best specialist CV MAPE: `property_type_0 / ensemble`, `16.86%`.
- Hardest specialist: `property_type_1 / ensemble`, `24.34%` CV MAPE and `0.746` CV R2.

### Derived `price_vnd_from_price_per_m2`

- Best global CV MAPE: `ensemble`, `20.31%`.
- Best global CV R2: `catboost`, `0.846`.
- Best global holdout R2: `catboost`, `0.877`.
- Best specialist CV MAPE: `property_type_0 / ensemble`, `16.88%`.
- Best specialist CV R2: `property_type_0 / ensemble`, `0.859`.
- Derived total price is close to direct `price_vnd` by CV MAPE and stronger by CV R2, so keep it as a serious comparison target.

### Routed Specialist Models

`routed_property` trains specialist models by `property_type` and routes each prediction to the matching specialist.

Latest results:

- `routed_property / price_vnd / ensemble`: `20.58%` CV MAPE, `0.773` CV R2.
- `global / price_vnd / ensemble`: `20.53%` CV MAPE, `0.773` CV R2.
- `global / price_vnd / catboost`: `20.50%` CV MAPE, `0.765` CV R2.

The routed approach is still not a clear winner. Keep it as an experiment, not the default production choice.

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
- The recovered and deduped Supabase dataset improved the best global `price_vnd` CV MAPE to about `20.50%`. The remaining gap is now mostly segment quality, not basic model choice.

Latest full-run diagnostics show the hardest segment is still `property_type=1` (`nha_mat_tien`), especially:

- Low `price_per_m2` street-front homes.
- Large-area homes, especially `area_m2 > 150`.
- Sparse or noisy localities such as `phường tăng nhơn phú`.
- Some road-width buckets, especially small road width inside `property_type=1`.

The best `property_type_0` (`nha_trong_hem`) specialist is now around `16.9%` CV MAPE. The global score is still mostly being held back by `property_type_1`, but that segment improved to roughly `24.2-24.4%` CV MAPE with the recovered dataset.

Training output now suppresses the harmless LightGBM feature-name warning and prints progress lines such as:

```text
[train] global | price_vnd | lightgbm: fit holdout model
[cv] global | price_vnd | lightgbm: fold 1/5 using property_type_stratified
[done] global | price_vnd | lightgbm: holdout MAPE=20.48% CV MAPE=21.57%
```

## Modeling Interpretation

- The target is right-skewed. The trainer already uses `log1p` on targets and converts predictions back with `expm1`.
- Current best global model is clustered around `20.3-20.5%` CV MAPE and `0.76-0.85` CV R2 depending on target strategy.
- `property_type_0` can reach about `16.9%` CV MAPE.
- `property_type_1` remains around `24.2-24.4%` CV MAPE and is the main performance bottleneck.
- Next useful work is focused `property_type_1` cleanup/feature work: better street/locality normalization, manual review of low-unit-price street-front rows, better large-area handling, and possibly more precise location features. Adding many more model families is less likely to move the score than fixing this segment.
