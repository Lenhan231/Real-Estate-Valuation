# Exploratory Data Analysis Report

This report presents an Exploratory Data Analysis (EDA) on the raw dataset retrieved from Supabase and the processed model training data from the local codebase.

## Raw Data (Supabase)

**Dataset Shape:** 12832 rows, 46 columns

### Missing Values

| Feature | Missing Count | Missing Percentage |
|---|---|---|
| description | 8640 | 67.33% |
| nearest_metro_km | 5613 | 43.74% |
| length_m | 1689 | 13.16% |
| width_m | 1283 | 10.00% |
| nearest_mall_km | 611 | 4.76% |
| num_floors | 288 | 2.24% |
| num_bedrooms | 229 | 1.78% |
| nearest_supermarket_km | 189 | 1.47% |
| road_width_m | 61 | 0.48% |
| locality_population_density | 48 | 0.37% |
| locality_square | 48 | 0.37% |
| link | 38 | 0.30% |
| title | 38 | 0.30% |
| old_address | 38 | 0.30% |
| street | 38 | 0.30% |
| post_day | 38 | 0.30% |
| property_type | 38 | 0.30% |
| legal_status | 38 | 0.30% |
| locality | 38 | 0.30% |
| listing_id | 38 | 0.30% |
| region | 38 | 0.30% |
| kitchen_bin | 38 | 0.30% |
| dining_room_bin | 38 | 0.30% |
| area_m2 | 38 | 0.30% |
| price_vnd | 38 | 0.30% |
| direction | 38 | 0.30% |
| listing_type | 38 | 0.30% |
| car_parking_bin | 38 | 0.30% |
| terrace_bin | 38 | 0.30% |
| matched_address | 38 | 0.30% |
| lon | 38 | 0.30% |
| lat | 38 | 0.30% |
| owner_listing_bin | 38 | 0.30% |
| distance_to_center_km | 38 | 0.30% |
| nearest_hospital_km | 15 | 0.12% |
| nearest_marketplace_km | 11 | 0.09% |
| nearest_bus_stop_km | 2 | 0.02% |

### Data Types

| Data Type   |   Count |
|:------------|--------:|
| float64     |      19 |
| str         |      15 |
| int64       |       7 |
| object      |       5 |

### Descriptive Statistics (Numeric Features)

|                             |   count |          mean |           std |           min |           25% |           50% |           75% |             max |
|:----------------------------|--------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|----------------:|
| listing_id                  |   12794 |   1.77993e+07 |   1.56533e+06 |   2.55009e+06 |   1.74969e+07 |   1.84357e+07 |   1.87707e+07 |     1.89303e+07 |
| num_floors                  |   12544 |   3.54        |   2.09        |   1           |   2           |   3           |   5           |    52           |
| num_bedrooms                |   12603 |   6.55        |  10.12        |   1           |   3           |   4           |   6           |   220           |
| road_width_m                |   12771 |  13.62        |  12.92        |   1           |   5           |  10           |  20           |   335           |
| width_m                     |   11549 |   6.69        |   6.91        |   1           |   4           |   5           |   7.5         |   425           |
| length_m                    |   11143 |  19.99        |  30.14        |   1           |  14           |  18           |  22           |  2525           |
| price_vnd                   |   12794 |   3.70191e+10 |   8.23828e+10 |  48           |   7.2925e+09  |   1.45e+10    |   3.55e+10    |     2.1e+12     |
| area_m2                     |   12794 | 167.07        | 414.44        |   6           |  60           |  88           | 150           | 14142           |
| locality_population_density |   12784 |  32.28        |  18.38        |   1.17        |  18           |  34.55        |  42.16        |   799           |
| lat                         |   12794 |  10.8         |   0.06        |  10.34        |  10.77        |  10.79        |  10.82        |    11.17        |
| lon                         |   12794 | 106.68        |   0.05        | 106.43        | 106.65        | 106.68        | 106.7         |   107.17        |
| distance_to_center_km       |   12794 |   6.77        |   5.36        |   0.17        |   3.43        |   5.52        |   9.4         |    63.58        |
| nearest_school_km           |   12832 |   0.31        |   0.3         |   0           |   0.14        |   0.24        |   0.41        |     4.91        |
| school_count_3km            |   12832 | 104.47        |  59.36        |   0           |  58           |  96           | 153           |   226           |
| nearest_hospital_km         |   12817 |   1.01        |   0.86        |   0           |   0.38        |   0.73        |   1.41        |     5           |
| hospital_count_5km          |   12832 |  43.37        |  30.61        |   0           |  12           |  40           |  76           |    92           |
| nearest_marketplace_km      |   12821 |   0.65        |   0.53        |   0           |   0.33        |   0.5         |   0.83        |     4.49        |
| marketplace_count_3km       |   12832 |  20.6         |  15.77        |   0           |   8           |  16           |  33           |    61           |
| nearest_supermarket_km      |   12643 |   0.66        |   0.6         |   0           |   0.24        |   0.48        |   0.84        |     4.93        |
| supermarket_count_3km       |   12832 |  39.49        |  33.06        |   0           |  12           |  28           |  61           |   117           |
| nearest_mall_km             |   12221 |   1.73        |   1.09        |   0.03        |   0.81        |   1.54        |   2.31        |     5.12        |
| mall_count_3km              |   12832 |   6.05        |   5.9         |   0           |   1           |   4           |   9           |    21           |
| nearest_bus_stop_km         |   12830 |   0.19        |   0.19        |   0           |   0.09        |   0.14        |   0.22        |     3.27        |
| bus_stop_count_1km          |   12832 |  39.78        |  22.66        |   0           |  24           |  36           |  48           |   118           |
| nearest_metro_km            |    7219 |   2.63        |   1.31        |   0.08        |   1.59        |   2.79        |   3.6         |     5.12        |
| metro_count_5km             |   12832 |   2.51        |   2.65        |   0           |   0           |   2           |   5           |    10           |

### Correlation Matrix

![Correlation Matrix](eda_figures/raw_correlation.png)

### Feature Distributions (Sample)

![Distributions](eda_figures/raw_distributions.png)

## Processed Data (Model Training)

**Dataset Shape:** 10421 rows, 79 columns

### Missing Values

No missing values found.

### Data Types

| Data Type   |   Count |
|:------------|--------:|
| float64     |      45 |
| int64       |      34 |

### Descriptive Statistics (Numeric Features)

|                             |   count |            mean |             std |       min |        25% |         50% |         75% |        max |
|:----------------------------|--------:|----------------:|----------------:|----------:|-----------:|------------:|------------:|-----------:|
| num_floors                  |   10421 |     3.29        |     1.59        |   1       |    2       |    3        |     4       |     33     |
| num_bedrooms                |   10421 |     5.41        |     5.76        |   1       |    3       |    4        |     6       |     85     |
| road_width_m                |   10421 |    11.26        |    10.8         |   1       |    5       |    8        |    15       |    335     |
| width_m                     |   10421 |     5.51        |     5.41        |   1       |    4       |    4.5      |     6       |    425     |
| length_m                    |   10421 |    17.7         |     7.64        |   1       |   13       |   17        |    20       |    202     |
| area_m2                     |   10421 |    95.73        |    62.23        |  15       |   57       |   80        |   111       |    500     |
| locality_square             |   10421 |     6.39        |     7.33        |   0.98    |    2.22    |    3.37     |     8.54    |     63.33  |
| locality_population_density |   10421 |    31.74        |    16.82        |   1.17    |   17.42    |   34.52     |    42.16    |     88.32  |
| distance_to_center_km       |   10421 |     6.91        |     4.43        |   0.17    |    3.76    |    6.05     |     9.59    |     60.17  |
| nearest_school_km           |   10421 |     0.31        |     0.29        |   0       |    0.13    |    0.24     |     0.41    |      4.91  |
| school_count_3km            |   10421 |   100.57        |    58.27        |   0       |   57       |   89        |   147       |    226     |
| nearest_hospital_km         |   10421 |     1.06        |     0.88        |   0       |    0.42    |    0.82     |     1.47    |     10     |
| hospital_count_5km          |   10421 |    40.66        |    29.96        |   0       |   11       |   36        |    73       |     92     |
| nearest_marketplace_km      |   10421 |     0.67        |     0.56        |   0       |    0.34    |    0.51     |     0.85    |      9.49  |
| marketplace_count_3km       |   10421 |    19.48        |    15.45        |   0       |    7       |   15        |    29       |     61     |
| nearest_supermarket_km      |   10421 |     0.77        |     1.06        |   0       |    0.26    |    0.52     |     0.9     |      9.93  |
| supermarket_count_3km       |   10421 |    35.39        |    30.4         |   0       |   10       |   26        |    54       |    117     |
| nearest_mall_km             |   10421 |     2.21        |     2.09        |   0.04    |    0.92    |    1.65     |     2.68    |     10.12  |
| mall_count_3km              |   10421 |     5.35        |     5.47        |   0       |    1       |    4        |     8       |     21     |
| nearest_bus_stop_km         |   10421 |     0.2         |     0.21        |   0       |    0.09    |    0.15     |     0.24    |      7.93  |
| bus_stop_count_1km          |   10421 |    37.27        |    20.4         |   0       |   24       |   35        |    47       |    112     |
| nearest_metro_km            |   10421 |     6.33        |     3.78        |   0.08    |    2.92    |    4.8      |    10.12    |     10.12  |
| metro_count_5km             |   10421 |     2.23        |     2.56        |   0       |    0       |    1        |     4       |     10     |
| post_day_month              |   10421 |     6.51        |     0.5         |   5       |    6       |    7        |     7       |      7     |
| post_day_day                |   10421 |    13.05        |    10.01        |   1       |    7       |    9        |    20       |     30     |
| road_width_m_missing        |   10421 |     0           |     0.05        |   0       |    0       |    0        |     0       |      1     |
| perimeter_m                 |   10421 |    46.39        |    19.3         |   4       |   37.2     |   44        |    52       |    920     |
| shape_ratio                 |   10421 |     0.35        |     0.3         |   0.02    |    0.24    |    0.29     |     0.37    |     12.11  |
| shape_ratio_missing         |   10421 |     0.12        |     0.33        |   0       |    0       |    0        |     0       |      1     |
| width_m_missing             |   10421 |     0.08        |     0.27        |   0       |    0       |    0        |     0       |      1     |
| nearby_amenities            |   10421 |   240.95        |   148.36        |   2       |  113       |  213        |   359       |    568     |
| nearby_amenities_log        |   10421 |     5.24        |     0.78        |   1.1     |    4.74    |    5.37     |     5.89    |      6.34  |
| nearest_metro_km_missing    |   10421 |     0.48        |     0.5         |   0       |    0       |    0        |     1       |      1     |
| amenity_density             |   10421 |     3.42        |     3.2         |   0.01    |    1.25    |    2.41     |     4.64    |     31     |
| is_hem_xe_hoi               |   10421 |     0.22        |     0.42        |   0       |    0       |    0        |     0       |      1     |
| is_mat_tien                 |   10421 |     0.31        |     0.46        |   0       |    0       |    0        |     1       |      1     |
| is_no_hau                   |   10421 |     0.03        |     0.17        |   0       |    0       |    0        |     0       |      1     |
| has_noi_that                |   10421 |     0.04        |     0.18        |   0       |    0       |    0        |     0       |      1     |
| is_gap                      |   10421 |     0.08        |     0.28        |   0       |    0       |    0        |     0       |      1     |
| is_kinh_doanh               |   10421 |     0.07        |     0.26        |   0       |    0       |    0        |     0       |      1     |
| area_x_floors               |   10421 |   320.03        |   297.44        |  18       |  140       |  240        |   396       |   3552     |
| area_x_bedrooms             |   10421 |   641.56        |  1459.75        |  18       |  189       |  315        |   588       |  34000     |
| area_per_bedroom            |   10421 |    26.16        |    32.55        |   1       |   12.5     |   18        |    27.5     |    500     |
| distance_vs_area            |   10421 |     0.09        |     0.08        |   0       |    0.04    |    0.07     |     0.13    |      1.28  |
| log_area                    |   10421 |     4.42        |     0.53        |   2.77    |    4.06    |    4.39     |     4.72    |      6.22  |
| log_distance_to_center      |   10421 |     1.94        |     0.51        |   0.16    |    1.56    |    1.95     |     2.36    |      4.11  |
| log_population_density      |   10421 |     3.31        |     0.66        |   0.77    |    2.91    |    3.57     |     3.76    |      4.49  |
| frontage_ratio              |   10421 |     0.74        |     0.67        |   0.01    |    0.37    |    0.66     |     0.98    |     23.48  |
| depth_ratio                 |   10421 |     3.49        |     1.6         |   0.08    |    2.62    |    3.41     |     4.15    |     51.01  |
| road_area_ratio             |   10421 |     1.18        |     1.11        |   0.05    |    0.59    |    0.83     |     1.42    |     40.09  |
| area_m2_squared             |   10421 | 13035.6         | 22564.2         | 225       | 3249       | 6400        | 12321       | 250000     |
| area_m2_sqrt                |   10421 |     9.4         |     2.73        |   3.89    |    7.56    |    8.95     |    10.54    |     22.36  |
| distance_squared            |   10421 |    67.4         |   129.14        |   0.03    |   14.16    |   36.66     |    91.88    |   3621.01  |
| road_width_squared          |   10421 |   243.3         |  1636.27        |   1       |   25       |   64        |   225       | 112225     |
| bedrooms_squared            |   10421 |    62.43        |   283.78        |   1       |    9       |   16        |    36       |   7225     |
| floors_squared              |   10421 |    13.34        |    17.09        |   1       |    4       |    9        |    16       |   1089     |
| area_x_distance             |   10421 |   676.17        |   719.4         |  13.77    |  270.92    |  493.65     |   803.48    |  13053.7   |
| area_per_distance           |   10421 |    19.25        |    18.86        |   0.76    |    7.69    |   13.37     |    24.07    |    299.08  |
| bedrooms_x_distance         |   10421 |    35.66        |    49.3         |   0.17    |   14.64    |   24.58     |    39.8     |   1472.86  |
| floors_x_distance           |   10421 |    21.64        |    17.1         |   0.17    |   10.22    |   17.51     |    28.05    |    240.7   |
| area_x_road_width           |   10421 |  1253.9         |  1749.44        |  27       |  315       |  640        |  1520       |  36516     |
| width_x_length              |   10421 |   102.65        |   190.06        |   1       |   58       |   79.5      |   115.5     |  14875     |
| density_x_area              |   10421 |  2981.49        |  2561.48        |  77.4     | 1270.4     | 2312.1      |  3817.2     |  44162     |
| locality_sq_x_area          |   10421 |   621.95        |   972.04        |  18.45    |  167       |  314.4      |   701.04    |  17099.1   |
| direction_dong_bac          |   10421 |     0.01        |     0.08        |   0       |    0       |    0        |     0       |      1     |
| direction_dong_nam          |   10421 |     0.01        |     0.12        |   0       |    0       |    0        |     0       |      1     |
| direction_nam               |   10421 |     0.01        |     0.08        |   0       |    0       |    0        |     0       |      1     |
| direction_unknown           |   10421 |     0.93        |     0.25        |   0       |    1       |    1        |     1       |      1     |
| property_type_nha_mat_tien  |   10421 |     0.41        |     0.49        |   0       |    0       |    0        |     1       |      1     |
| property_type_nha_trong_hem |   10421 |     0.59        |     0.49        |   0       |    0       |    1        |     1       |      1     |
| legal_status_giay_to_hop_le |   10421 |     0.03        |     0.17        |   0       |    0       |    0        |     0       |      1     |
| legal_status_so_hong_so_do  |   10421 |     0.71        |     0.45        |   0       |    0       |    1        |     1       |      1     |
| legal_status_unknown        |   10421 |     0.26        |     0.44        |   0       |    0       |    0        |     1       |      1     |
| dining_room_bin_False       |   10421 |     0.49        |     0.5         |   0       |    0       |    0        |     1       |      1     |
| terrace_bin_False           |   10421 |     0.58        |     0.49        |   0       |    0       |    1        |     1       |      1     |
| terrace_bin_True            |   10421 |     0.42        |     0.49        |   0       |    0       |    0        |     1       |      1     |
| car_parking_bin_False       |   10421 |     0.57        |     0.49        |   0       |    0       |    1        |     1       |      1     |
| car_parking_bin_True        |   10421 |     0.43        |     0.49        |   0       |    0       |    0        |     1       |      1     |
| price_vnd                   |   10421 |     1.60458e+10 |     1.19963e+10 |   2.1e+09 |    6.8e+09 |    1.16e+10 |     2.3e+10 |      5e+10 |

### Correlation Matrix

![Correlation Matrix](eda_figures/processed_correlation.png)

### Feature Distributions (Sample)

![Distributions](eda_figures/processed_distributions.png)

