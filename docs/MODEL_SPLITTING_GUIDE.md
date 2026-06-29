# Per-Property-Type Model Splitting Guide

This document explains the recent architectural changes to split our real estate valuation dataset by **property type** (`nha_mat_tien` vs `nha_trong_hem`) and how to clean, split, train, and compare models.

---

## 1. Why Split the Dataset?

Analyzing our raw data showed that Vietnam's street-facing houses and alley houses represent **two very different markets**:
- **Frontage Houses (`nha_mat_tien`)**: Median price **~32.0B VND**, wider price spread, higher luxury ceiling.
- **Alley Houses (`nha_trong_hem`)**: Median price **~9.5B VND**, more compact range, narrower price spread.

By training a single global model, the model had to make huge compromises across both segments, hurting overall performance. Splitting them into dedicated training subsets allows each model (XGBoost/TabPFN) to learn segment-specific features.

---

## 2. Advanced Outlier Filtering (IQR Tuning)

Applying the same global filter to both splits wasn't optimal. We customized our Interquartile Range (IQR) outlier multipliers per segment:

- **Alley Houses (`nha_trong_hem`)**: Kept standard **IQR x3.0** filtering. The distribution is naturally compact, and this keeps 1,595 valid listings.
- **Frontage Houses (`nha_mat_tien`)**: Applied a tighter **IQR x1.5** fence. This was a **massive breakthrough**. It caps the frontage dataset at **125B VND / 427 m2**, removing the long ultra-luxury tail (125B to 1100B VND) that was completely confusing the model.

---

## 3. Results & Leaderboard (W&B)

All metrics are tracked and comparison charts are logged to our unified Weights & Biases project (`real-estate-valuation`).

Here is our current leaderboard:

| Model | Dataset Split | IQR Fence | R2 Score | MAPE | RMSE (log) |
|---|---|---|---|---|---|
| **TabPFN** | `nha_trong_hem` (Alley) | IQR x3.0 | **0.8419** | **18.69%** | **0.2456** |
| **XGBoost** | `nha_trong_hem` (Alley) | IQR x3.0 | 0.8149 | 21.16% | 0.2719 |
| **XGBoost** | Full (Baseline) | IQR x3.0 | 0.8131 | 24.29% | 0.3185 |
| **TabPFN** | Full (Baseline) | IQR x3.0 | 0.7926 | 25.18% | 0.3302 |
| **TabPFN** | `nha_mat_tien` (Frontage) | IQR x1.5 | **0.7058** | **26.63%** | **0.3442** |
| **XGBoost** | `nha_mat_tien` (Frontage) | IQR x1.5 | 0.6789 | 27.88% | 0.3679 |

> [!NOTE]
> Previously, the frontage split (`nha_mat_tien`) without the IQR x1.5 fence had a **107% MAPE** and was completely unusable. Applying the tighter outlier fence brought it down to **26.63% MAPE** and **0.7058 R2**.

---

## 4. How to Reproduce

### Step 1: Clean and Generate Splits
Running the cleaning script performs standard deduplication, imputes dimension lengths, and outputs both the global clean file and the per-segment splits:
```powershell
python scripts/clean_features.py
```
This produces:
- `data/processed/alonhadat_features_cleaned.csv` (Full dataset)
- `data/processed/split_nha_mat_tien.csv` (IQR x1.5 filtered)
- `data/processed/split_nha_trong_hem.csv` (IQR x3.0 filtered)

### Step 2: Train All Models
We added a convenience runner script to automate training both model classes (XGBoost & TabPFN) across all splits sequentially:
```powershell
python scripts/train_all.py
```

To run selectively (e.g., only TabPFN on the alley dataset):
```powershell
python scripts/train_all.py --models tabpfn --splits nha_trong_hem
```

---

## 5. Other Key Changes
- **No more local file-log clutter:** Removed code generating duplicate `.log` and `.json` run reports under `/reports` and `/wandb` locally. We now rely strictly on console printouts and direct sync to your W&B dashboard.
- **Robust Path Resolution:** Both `train_xgboost.py` and `train_tabpfn.py` resolve data files dynamically. You can pass raw names (e.g., `--dataset alonhadat_features.csv`) and they will search both `data/processed/` and `data/raw/` folders while safely handling extension suffixes.
