# Model Training Instructions

This document provides instructions on how to use the training scripts to experiment with different datasets and test different model combinations for the real estate valuation project.

## 1. Available Scripts

There are two primary scripts for training ensemble models in the `Models/Scripts/` directory:

*   `train_ensemble_xgb_lgbm.py`: Trains a 6-bucket ensemble regressor using a combination of **LightGBM** and **XGBoost**.
*   `train_ensemble.py`: Trains a 6-bucket ensemble regressor using a combination of **LightGBM** and **CatBoost**.

Both scripts train separate models for different price buckets (low, mid, high) and property types (nha_mat_tien, nha_trong_hem) to improve overall accuracy.

## 2. Using Different Datasets

Both scripts accept a `--dataset` argument to specify which data file to load. The scripts will look for a CSV file named `alonhadat_features_<dataset_label>.csv` in the `Models/data/` directory.

### Example: Training with the default "cleaned" dataset
```bash
python Models/Scripts/train_ensemble.py --dataset cleaned
```
*(This loads `Models/data/alonhadat_features_cleaned.csv`)*

### Example: Training with a custom dataset named "v2"
If you have generated a new feature set and saved it as `alonhadat_features_v2.csv`, you can train on it by running:
```bash
python Models/Scripts/train_ensemble.py --dataset v2
```

## 3. Using Different Data Sources

You can choose to load data locally from the CSV file, or fetch it directly from Supabase using the `--data-source` argument. The options are `local` (default) or `supabase`.

### Example: Fetching data from Supabase
```bash
python Models/Scripts/train_ensemble.py --data-source supabase
```

*(Note: If the Supabase fetch fails, the script will automatically fall back to the local CSV file.)*

## 4. Testing Different Model Combinations

You can easily compare different ensemble combinations by running the respective scripts. Both scripts automatically log their performance metrics (RMSE, MAE, R², MAPE), feature importance, and Prediction vs. Actual plots to Weights & Biases (W&B) for easy comparison.

### Combination 1: LightGBM + CatBoost (Recommended for categorical data)
Run the standard ensemble script:
```bash
python Models/Scripts/train_ensemble.py --dataset cleaned
```

### Combination 2: LightGBM + XGBoost
Run the XGBoost variant script:
```bash
python Models/Scripts/train_ensemble_xgb_lgbm.py --dataset cleaned
```

### Comparing Results
After running both scripts, you can compare the results in several ways:
1.  **Terminal Output**: Both scripts print global metrics and per-segment MAPE breakdowns to the console at the end of the run.
2.  **Local Plots**: Check the `Models/plots/` directory for generated visualizations.
3.  **W&B Dashboard**: Go to your W&B project dashboard (`real-estate-valuation`). You will see separate runs logged (e.g., `ensemble-6bucket-cleaned-local` vs. `ensemble-xgb-lgbm-cleaned-local`). You can overlay their charts to see which combo performed better!

## 5. Summary of Command Line Arguments

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--dataset` | Suffix for the dataset file to load. | `cleaned` | Any string (e.g., `cleaned`, `raw`, `v2`) |
| `--data-source` | Where to load the data from. | `local` | `local`, `supabase` |
| `-h`, `--help` | Show the help message and exit. | N/A | N/A |
