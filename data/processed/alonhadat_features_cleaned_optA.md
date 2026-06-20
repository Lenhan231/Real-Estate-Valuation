# Processed Data Documentation: `alonhadat_features_cleaned_optA.csv`

This document details the cleaning, imputation, and enrichment pipeline for the real estate valuation datasets located in the `data/processed/` directory.

---

## 1. Overview and Files Used

The processing pipeline takes partially cleaned features and prepares them for machine learning models (e.g., XGBoost, TabPFN) by removing duplicate listings, resolving missing values (imputation), and extracting missing features from textual fields (enrichment).

### Files and Directories
* **Source Input File**: `data/processed/alonhadat_features (1).csv` (760 rows, 44 columns)
* **Option A Output (Unenriched)**: [alonhadat_features_cleaned_optA.csv](file:///c:/Users/Admin/Desktop/SU2026/DSP/Real-Estate-Valuation/data/processed/alonhadat_features_cleaned_optA.csv) (753 rows, 42 columns)
* **Option A Output (Enriched)**: [alonhadat_features_cleaned_optA_enriched.csv](file:///c:/Users/Admin/Desktop/SU2026/DSP/Real-Estate-Valuation/data/processed/alonhadat_features_cleaned_optA_enriched.csv) (753 rows, 42 columns)
* **Option B Output (Unenriched)**: [alonhadat_features_cleaned_optB.csv](file:///c:/Users/Admin/Desktop/SU2026/DSP/Real-Estate-Valuation/data/processed/alonhadat_features_cleaned_optB.csv) (753 rows, 44 columns)
* **Option B Output (Enriched)**: [alonhadat_features_cleaned_optB_enriched.csv](file:///c:/Users/Admin/Desktop/SU2026/DSP/Real-Estate-Valuation/data/processed/alonhadat_features_cleaned_optB_enriched.csv) (753 rows, 44 columns)

---

## 2. Preprocessing & Cleaning Pipeline

The preprocessing is implemented in the script [clean_features.py](file:///c:/Users/Admin/Desktop/SU2026/DSP/Real-Estate-Valuation/scripts/clean_features.py). The pipeline performs the following steps:

### Step 1: Semantic Duplicate Removal
To prevent model bias and leaking data, semantic duplicates are dropped if they have identical coordinates (`lat`, `lon`), price (`price_vnd`), and area (`area_m2`).
* **Original Count**: 760 rows
* **Cleaned Count**: 753 rows (7 duplicates removed)

### Step 2: Imputation of Structural Features
Missing values in key house features are imputed using their respective column-wise medians:
* `num_floors`: Filled missing values (22 nulls) with the median (`4.0`).
* `num_bedrooms`: Filled missing values (29 nulls) with the median (`4.0`).
* `road_width_m`: Filled missing values (2 nulls) with the median (`4.0`).

### Step 3: Imputation of Point of Interest (POI) Distances
Missing distances to POIs (which occur when no POI was found within search buffers) are imputed with a fallback distance of **5.0 km**:
* `nearest_hospital_km`, `nearest_marketplace_km`, `nearest_supermarket_km`, `nearest_mall_km`, `nearest_bus_stop_km`, `nearest_metro_km`, `nearest_school_km`.

---

## 3. Option A vs. Option B Strategies

The primary difference between the Option A and Option B datasets lies in how missing values for physical dimensions (`width_m` and `length_m`) are handled:

### Option A (Drop Dimensions)
* **Strategy**: `width_m` and `length_m` are completely dropped (reducing the columns from 44 to 42).
* **Rationale**: The house area (`area_m2`) is already present. Dimensions have significant missing rates (130 for width, 136 for length), and dropping them eliminates these high-null columns without losing the core spatial size feature (`area_m2`).

### Option B (Dimension Reconstruction)
* **Strategy**: Retain `width_m` and `length_m` and impute missing values using house area:
  * **If both width and length are missing**: Assume a standard aspect ratio of $1:3$.
    $$\text{width\_m} = \sqrt{\frac{\text{area\_m2}}{3.0}}$$
    $$\text{length\_m} = \frac{\text{area\_m2}}{\text{width\_m}}$$
  * **If only width is missing**:
    $$\text{width\_m} = \frac{\text{area\_m2}}{\text{length\_m}}$$
  * **If only length is missing**:
    $$\text{length\_m} = \frac{\text{area\_m2}}{\text{width\_m}}$$

---

## 4. Feature Enrichment: House Direction

The script [extract_direction.py](file:///c:/Users/Admin/Desktop/SU2026/DSP/Real-Estate-Valuation/scripts/extract_direction.py) performs text-mining on listing titles using Vietnamese keyword regex patterns to extract the house direction (`direction`), aiming to recover values for rows where `direction == 'unknown'`.

### Regex Patterns Used:
* **Compound Directions**: `dong_nam` (Đông Nam), `dong_bac` (Đông Bắc), `tay_nam` (Tây Nam), `tay_bac` (Tây Bắc).
* **Cardinal Directions**: `dong` (Đông), `tay` (Tây), `nam` (Nam), `bac` (Bắc).

### Extraction Results:
* Pre-enrichment: **683 / 753 (90.7%)** rows had an `unknown` direction.
* Post-enrichment: **681 / 753 (90.4%)** rows have an `unknown` direction (2 rows recovered from title keywords).
* *Insight*: The raw listing title rarely mentions direction, which validates the decision to use models that handle missing values robustly (e.g. TabPFN, XGBoost with native missing support).

---

## 5. Dataset Schema (Option A)

| Column Name | Data Type | Missing Count (Original) | Imputation Method / Strategy | Description |
| :--- | :---: | :---: | :--- | :--- |
| `link` | object | 0 | None | Listing URL |
| `title` | object | 0 | None | Raw title of the listing |
| `post_day` | object | 0 | None | Date the listing was posted |
| `street` | object | 0 | None | Street address |
| `old_address` | object | 0 | None | Unparsed original address |
| `locality` | object | 0 | None | District / Locality |
| `region` | object | 0 | None | Province / City |
| `listing_id` | int64 | 0 | None | Unique listing ID |
| `direction` | object | 0 (683 unknown) | Regex extraction from titles | House direction |
| `listing_type` | object | 0 | None | Sale / Rent category |
| `property_type` | object | 0 | None | House, Villa, Apartment, etc. |
| `legal_status` | object | 0 | None | Red book, pink book, etc. |
| `num_floors` | float64 | 22 | Median (`4.0`) | Number of floors |
| `num_bedrooms` | float64 | 29 | Median (`4.0`) | Number of bedrooms |
| `road_width_m` | float64 | 2 | Median (`4.0`) | Road width in front of the house |
| `price_vnd` | float64 | 0 | None (Target variable) | Listed price in VND |
| `area_m2` | float64 | 0 | None | Total house/land area |
| `dining_room_bin` | int64 | 0 | None | Binary (1 if dining room exists) |
| `kitchen_bin` | int64 | 0 | None | Binary (1 if kitchen exists) |
| `terrace_bin` | int64 | 0 | None | Binary (1 if terrace exists) |
| `car_parking_bin` | int64 | 0 | None | Binary (1 if parking space exists) |
| `owner_listing_bin` | int64 | 0 | None | Binary (1 if posted by owner) |
| `locality_square` | object | 0 | None | District area in sq km |
| `locality_population_density` | float64 | 0 | None | Population density of district |
| `lat` | float64 | 0 | None | Latitude coordinate |
| `lon` | float64 | 0 | None | Longitude coordinate |
| `matched_address` | object | 0 | None | Normalized geocoded address |
| `distance_to_center_km` | float64 | 0 | None | Distance to city center |
| `nearest_school_km` | float64 | 0 | Filled with 5.0 | Distance to nearest school |
| `school_count_3km` | float64 | 0 | None | Count of schools within 3km |
| `nearest_hospital_km` | float64 | 4 | Filled with 5.0 | Distance to nearest hospital |
| `hospital_count_5km` | float64 | 0 | None | Count of hospitals within 5km |
| `nearest_marketplace_km` | float64 | 5 | Filled with 5.0 | Distance to nearest marketplace |
| `marketplace_count_3km` | float64 | 0 | None | Count of marketplaces within 3km |
| `nearest_supermarket_km` | float64 | 36 | Filled with 5.0 | Distance to nearest supermarket |
| `supermarket_count_3km` | float64 | 0 | None | Count of supermarkets within 3km |
| `nearest_mall_km` | float64 | 46 | Filled with 5.0 | Distance to nearest mall |
| `mall_count_3km` | float64 | 0 | None | Count of malls within 3km |
| `nearest_bus_stop_km` | float64 | 9 | Filled with 5.0 | Distance to nearest bus stop |
| `bus_stop_count_1km` | float64 | 0 | None | Count of bus stops within 1km |
| `nearest_metro_km` | float64 | 281 | Filled with 5.0 | Distance to nearest metro station |
| `metro_count_5km` | float64 | 0 | None | Count of metro stations within 5km |
