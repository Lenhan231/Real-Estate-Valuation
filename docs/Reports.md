

| ![][image1] | MINISTRY OF EDUCATION AND TRAINING |
| :---: | :---: |

|  FPT UNIVERSITY |
| :---: |
| Data Science Project Document  |
| Automated Real Estate Valuation and Market Trend Analysis |

| Group 3 |  |
| ----- | :---- |
| **Group Members** | Lê Trọng Nhân SE190525 Trần Văn Thuận SE184562 Nông Quốc An SE190512 Nguyễn Hoàng Gia Huy SE182631 |
| **Supervisor** | Nguyễn Trọng Tài |
| **Ext Supervisor** | FPT University Academic Board |
| **Capstone Project code** |  DSP391m |

\- HCMC, July/2026 \-

**Table of Contents**

[Definition and Acronyms	5](#definition-and-acronyms)

[List of Tables	6](#list-of-tables)

[**List of figures	6**](#list-of-figures)

[**I. Project Introduction	7**](#i.-project-introduction)

[1\. Overview	7](#1.-overview)

[2\. Project Background	7](#2.-project-background)

[3\. Project Objective	8](#3.-project-objective)

[4\. Problem Statement	8](#4.-problem-statement)

[5\. Significance of the Project	8](#5.-significance-of-the-project)

[6\. Project Scope & Limitations	9](#6.-project-scope-&-limitations)

[**II. Project Management Plan	10**](#ii.-project-management-plan)

[1\. Team Work	10](#1.-team-work)

[2\. Project Management Approach	10](#2.-project-management-approach)

[**III. Related Work	13**](#iii.-related-work)

[1\. Overview of the Field	13](#1.-overview-of-the-field)

[2\. Historical Context	13](#2.-historical-context)

[3\. Key Studies and Theories	13](#3.-key-studies-and-theories)

[4\. Technological Advancements	14](#4.-technological-advancements)

[5\. Comparison of Existing Systems	14](#5.-comparison-of-existing-systems)

[6\. Gaps in the Literature/Technology	15](#6.-gaps-in-the-literature/technology)

[7\. Justification for the Project	15](#7.-justification-for-the-project)

[**IV. Methodology	17**](#iv.-methodology)

[1\. Research Questions and Objectives	17](#1.-research-questions-and-objectives)

[2\. Data Collection and Preprocessing	17](#2.-data-collection-and-preprocessing)

[3\. Feature Selection and Engineering	20](#3.-feature-selection-and-engineering)

[4\. Model Training and Validation	22](#4.-model-training-and-validation)

[5\. Evaluation Metrics	23](#5.-evaluation-metrics)

[6\. Implementation Plan	23](#6.-implementation-plan)

[7\. Ethical Considerations	24](#7.-ethical-considerations)

[**V. System Design and Implementation	26**](#v.-system-design-and-implementation)

[1\. AI Model Integration	26](#1.-ai-model-integration)

[2\. Data Flow and Processing	27](#2.-data-flow-and-processing)

[3\. Deployment Strategy	28](#3.-deployment-strategy)

[4\. Scalability and Maintenance	29](#4.-scalability-and-maintenance)

[**VI. Results and Discussion	30**](#vi.-results-and-discussion)

[1\. Results and Analysis	30](#1.-results-and-analysis)

[2\. Discussion	32](#2.-discussion)

[3\. Recommendations	34](#3.-recommendations)

[**VII. Conclusion	35**](#vii.-conclusion)

[1\. Summary of Findings	35](#1.-summary-of-findings)

[2\. Contributions on the Project	35](#2.-contributions-on-the-project)

[3\. Limitations and Future Work	36](#3.-limitations-and-future-work)

[**VIII. Appendices	39**](#viii.-appendices)

[Appendix A - Core Processed Dataset Fields	39](#appendix-a-core-processed-dataset-fields)

[Appendix B - Final Model Hyperparameters (9-Model 3-Tier Ensemble)	40](#appendix-b-final-model-hyperparameters-9-model-3-tier-ensemble)

[Appendix C - Repository Structure and Documentation	41](#appendix-c-repository-structure-and-documentation)

[Appendix D - Performance Benchmarks & Metrics	45](#appendix-d-performance-benchmarks-metrics)

[Appendix E - Deployment Platforms Comparison	47](#appendix-e-deployment-platforms-comparison)

[Appendix F - Environment Variables Reference	48](#appendix-f-environment-variables-reference)

[**IX. References	48**](#ix.-references)

# **Definition and Acronyms**  {#definition-and-acronyms}

| Acronym | Definition |
| :---: | ----- |
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| AVM | Automated Valuation Model |
| CatBoost | Categorical Boosting |
| DL | Deep Learning |
| EDA | Exploratory Data Analysis |
| ETL | Extract, Transform, Load |
| GroupKFold | Grouped K-Fold Cross-Validation |
| IQR | Interquartile Range |
| LightGBM | Light Gradient Boosting Machine |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| PM | Project Manager  |
| PMP | Project Management Plan |
| POI | Points of Interest |
| R² | Coefficient of Determination |
| RMSE | Root Mean Square Error |
| WBS | Work Breakdown Structure |
| XGBoost | eXtreme Gradient Boosting |

# 

# **List of Tables** {#list-of-tables}

*Table 1\. Team Structure and Roles*

*Table 2\. Risk Management Plan*

*Table 3\. Comparison of Existing Systems*

*Table 4\. Two-Stage Data Collection Pipeline*

*Table 5\. Data Collection Challenges*

*Table 6\. Data Preprocessing Steps*

*Table 7\. Geospatial Feature Engineering Metrics*

*Table 8\. Model Configurations*

*Table 9\. Evaluation Metrics Definitions*

*Table 10\. Final Model Performance (9-Model 3-Tier Ensemble)*

*Table 11\. Historical Segmentation Impact (Pre-Ensemble Benchmarks)*

*Table 12\. Data Dictionary*

*Table 13\. Final Model Hyperparameters*

# **List of figures**  {#list-of-figures}

*Figure 1\. End-to-End Data Pipeline Architecture*

*Figure 2\. Top 20 LightGBM Feature Importances for the Mid-Price Tier*

# **I. Project Introduction** {#i.-project-introduction}

## **1\. Overview** {#1.-overview}

### **1.1 Project Information**

**Project Name:** Automated Real Estate Valuation and Market Trend Analysis

**Vietnamese Name:** Định giá bất động sản tự động và phân tích xu hướng thị trường

**Capstone Project code:** DSP391m

**Group Name:** Group 3

**Group Members:** Lê Trọng Nhân (SE190525), Trần Văn Thuận (SE184562), Nông Quốc An (SE190512), Nguyễn Hoàng Gia Huy (SE182631)

**Supervisor:** Nguyễn Trọng Tài

### **1.2 Project Overview**

This project focuses on developing an AI-powered real estate valuation system capable of predicting property prices and analyzing market trends using machine learning techniques. The system integrates data collection, preprocessing, exploratory data analysis, predictive modeling, and visualization into a complete end-to-end analytical pipeline.

The project begins with collecting real estate data from online property listing platforms and public datasets. After preprocessing and cleaning the data, exploratory data analysis (EDA) is conducted to identify important market patterns and relationships between variables such as location, area, number of bedrooms, and property price.

Several machine learning models will then be developed and evaluated to estimate both total property price and price per square meter. In addition, a Business Intelligence dashboard will be implemented to visualize price distributions, market fluctuations, and geographical pricing trends.

The final outcome is expected to support data-driven decision-making in the Vietnamese real estate market by providing fast, objective, and scalable property valuation solutions.

## **2\. Project Background** {#2.-project-background}

The Vietnamese real estate market has experienced rapid growth and significant fluctuations over recent years, particularly in major cities such as Ho Chi Minh City and Hanoi. Property prices vary substantially depending on factors including location, infrastructure development, urbanization, and market demand. Traditional property valuation methods are mainly conducted manually by experts or brokers, making the process time-consuming, inconsistent, and difficult to scale.

At the same time, the increasing availability of online property listings and open datasets has created opportunities to apply data science and artificial intelligence techniques to automate valuation processes. Machine learning models can identify hidden relationships within large datasets and generate accurate property price estimations in real time.

This project was selected because automated real estate valuation represents a highly practical application of AI and data engineering. It combines multiple important technical domains including web scraping, data preprocessing, feature engineering, machine learning, visualization, and deployment. Moreover, the project addresses a real-world business problem with strong commercial and societal relevance.

## **3\. Project Objective** {#3.-project-objective}

The main objectives of this project include:

* Building a data preprocessing pipeline to handle missing data, noisy data, outliers, and geospatial data.

* Researching and implementing suitable ML  models for tabular data processing.

* Predicting total property prices and price per square meter with a target Mean Absolute Percentage Error (MAPE) of less than 10% on the test dataset.

* Developing a Business Intelligence dashboard to visualize real estate price heatmaps, market trends, and regional comparisons over time.

* Developing a web application for system demonstration and user interaction.

* Researching deployment and maintenance strategies for AI models in real-world environments.

## **4\. Problem Statement** {#4.-problem-statement}

* Real estate prices in Vietnam change rapidly and depend on many factors such as location, area, and population density.

* Traditional valuation methods are manual, subjective, and difficult to scale for large numbers of properties.

* Real estate data often contains missing, duplicated, and inconsistent information, making analysis and prediction challenging.

* Existing platforms lack intelligent systems for accurate valuation and market trend analysis.

* Therefore, an automated AI-based system is needed to process large datasets, predict property prices, and support decision-making for buyers, sellers, and investors.

## **5\. Significance of the Project** {#5.-significance-of-the-project}

This project demonstrates the application of machine learning, data engineering, and data visualization in solving real-world real estate problems. The system helps improve valuation accuracy, consistency, and efficiency while reducing manual effort and operational costs. It also supports investors, buyers, and businesses in making data-driven decisions and highlights the growing role of AI-powered systems in modern real estate analytics.

## 

## **6\. Project Scope & Limitations** {#6.-project-scope-&-limitations}

### **Project Scope**

* Residential houses in Ho Chi Minh City, specifically frontage houses and alley houses.

* Data collection from public datasets and online real estate listing websites.

* Data preprocessing, exploratory analysis, and machine learning model development.

* Predicting total property price and deriving an indicative price-per-square-metre value.

* Building dashboards for market trend visualization and analysis.

### **Project Limitations**

* Real estate data may contain missing, duplicated, or inconsistent information.

* Some property features such as legal documents, interior quality, or neighborhood reputation may not be fully available.

* Web scraping activities may face anti-bot protections or changing website structures.

* Market conditions can fluctuate rapidly, potentially affecting model generalization over time.

* The project primarily focuses on urban residential properties and may not generalize well to rural or commercial real estate.

* The data collection process from real estate websites may encounter security mechanisms such as CAPTCHA or access restrictions, creating challenges for automated web scraping. 

# **II. Project Management Plan** {#ii.-project-management-plan}

## **1\. Team Work** {#1.-team-work}

### **1.1 Team Structure and Roles**

The team structure for Group 3, which is developing the "Automated Real Estate Valuation and Market Trend Analysis" project, is a **projectized organization**. 

| Name | Tasks | Responsibilities |
| ----- | ----- | ----- |
| Lê Trọng Nhân | Data processing  | Data collection and preprocessing for real estate datasets.  |
| Trần Văn Thuận  | Researching ML/DL  | ML/DL research for tabular data prediction and analysis.  |
| Nông Quốc An | Model training | Training and finetuning models for accurate prediction |
| Nguyễn Hoàng Gia Huy | Researching ML/DL  | ML/DL research for tabular data prediction and analysis.  |

*Table 1\. Team Structure and Roles*

### 

### **1.2 Communication Plan**

The team will maintain a shared GitHub repository for codebase management, dataset processing, and version control.

Routine team meetings will use Meet, Discord and Zalo to ensure alignment on research methodology and coordinate the distribution of tasks such as data cleaning and algorithm evaluation.

## **2\. Project Management Approach** {#2.-project-management-approach}

### **2.1 WBS**

**Data Acquisition:** Approximately 11,000 raw listing links were initially collected. After detail extraction, deduplication, cleaning, geospatial matching, and model-specific filtering, 10,421 records were used for model training and evaluation.

**Preprocessing & Feature Engineering:** Performed data cleaning, normalization, missing value handling, and outlier filtering. Engineered spatial and structural features such as distance to city center, property dimensions, population density, and binary amenity indicators to support predictive modeling.

**Model Implementation:** Split the Ho Chi Minh City dataset into 80% training and 20% testing sets using a fixed random holdout split with random\_state=42. Train and compare multiple regression models, including XGBoost, TabPFN, LightGBM, CatBoost, and ensemble approaches.

**Evaluation & Documentation:** Evaluate metrics, document findings on regional market variations, and present final project conclusions.

### **2.2 Risk Management**

| Risk | Impact | Prevention / Response Plan |
| ----- | ----- | ----- |
| Insufficient or low-quality data | Reduces model accuracy and reliability | Collect data from multiple sources and apply data cleaning/preprocessing techniques |
| CAPTCHA and access restrictions during web scraping | Difficulties in automated data collection | Use public datasets, APIs, or alternative data sources when scraping is blocked |
| Model performance not meeting expectations | Prediction accuracy may not achieve the target MAPE | Experiment with different ML/DL models and optimize hyperparameters |

*Table 2\. Risk Management Plan*

### **2.3 Quality Management**

To ensure the quality of the project, several measures and processes will be applied throughout the development lifecycle.

* Data preprocessing and validation techniques will be used to ensure data consistency, completeness, and reliability before training the models.

* Multiple Machine Learning and Deep Learning models will be evaluated using performance metrics such as Mean Absolute Percentage Error (MAPE) and Root Mean Square Error (RMSE).

* The system and web application will be tested regularly to verify prediction accuracy, functionality, and user interaction.

* Model outputs and dashboard visualizations will be reviewed and validated using real-world real estate data and market trends.

* Weekly progress reviews and team discussions will be conducted to monitor development quality and address issues promptly.

### **2.4 Budget and Funding (Optional)**

The project is primarily developed for academic and research purposes; therefore, the overall budget is relatively limited. Most development tools and technologies used in the project, such as Python, Scikit-learn, and public datasets, are open-source and freely available.

The estimated costs may include:

* Cloud computing or GPU services for model training and deployment.

* Database hosting and web application deployment services.

* Domain or server costs for system demonstration.

* Additional expenses related to data collection and system maintenance.

The project funding mainly comes from the project team members, while available free-tier services and academic resources will be utilized to minimize operational costs throughout the development process.

### **2.5 Change Management Process**

* Any proposed changes related to project scope, schedule, resources, or technical requirements will be reviewed by the project team.

* The team will evaluate the impact of each change on project objectives, timeline, system performance, and available resources.

* Significant changes must be approved by the project supervisor before implementation.

* All approved changes will be documented clearly, including the reason, impact, and updated project plan.

* Regular meetings and progress reviews will be conducted to monitor and manage project changes effectively.

### **2.6 Closure and Evaluation**

* Conduct final testing and evaluation to verify model accuracy, system functionality, and project deliverables.

* Finalize and submit all project outcomes, including technical documents, source code, and presentation materials.

* Review project performance and summarize key challenges, experiences, and lessons learned for future improvements.

# **III. Related Work** {#iii.-related-work}

## **1\. Overview of the Field** {#1.-overview-of-the-field}

The project belongs to the field of Automated Valuation Model (AVM) using machine learning on tabular and spatial data: predicting house and land prices, including price per square metre from structural, location and utility features to identify the price governing factors. The Direction field is collected for future compatibility but not used in the current prototype due to missingness.

## **2\. Historical Context** {#2.-historical-context}

The field of Automated Real Estate Valuation and Market Trend Analysis studies the estimation of house/land prices and the identification of market trends using characteristic and spatial data. The field has evolved from linear hedonic regression models (1970s) to non-linear machine learning models and tree-based ensembles (2015-2018), which are currently dominant, and more recently to tabular-based models which help both increase accuracy and interpret results.

**Technologies used:**

* XGBoost

* TabPFN

* LightGBM

* CatBoost

* Ensemble (blending / stacking)

* Geospatial feature engineering using OpenStreetMap POI data, geodesic distances, and radius-based facility counts

* Target encoding for selected categorical location features

* Experiment tracking using Weights & Biases

## **3\. Key Studies and Theories** {#3.-key-studies-and-theories}

**XGBoost**

* Paper: *XGBoost: A Scalable Tree Boosting System*  
* Author(s): Tianqi Chen, Carlos Guestrin  
* Published: 08/2016 (KDD '16)  
* Link: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)

**TabPFN**

* Paper: *TabPFN: Prior-Data Fitted Networks for Tabular Data*  
* Author(s): Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter  
* Published: 2022 (ICML 2022\)  
* Link: [https://arxiv.org/abs/2207.01848](https://arxiv.org/abs/2207.01848)

**LightGBM**

* Paper: *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*  
* Author(s): Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu  
* Published: 12/2017 (NeurIPS/NIPS 2017\)  
* Link: [https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

**CatBoost**

* Paper: *CatBoost: Unbiased Boosting with Categorical Features*  
* Author(s): Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin  
* Published: 2018 (NeurIPS 31; preprint 2017\)  
* Link: [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)

**Ensemble (Stacked Generalization)**

* Paper: *Stacked Generalization*  
* Author: David H. Wolpert  
* Published: 1992 (Neural Networks 5(2):241–259)  
* Link: [https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)

**Application in Vietnam (Same Market & Data Source)**

* Paper: *Using Machine Learning Regression Algorithms to Predict House Prices in Vietnam*  
* Author(s): Minh-Thang Ha, Thi-Cham Nguyen, Thanh-Huyen Pham, Van-Hau Nguyen  
* Published: 2025 (International Real Estate Review 28(4):505–527)  
* Link: [https://doi.org/10.53383/100412](https://doi.org/10.53383/100412)

## **4\. Technological Advancements** {#4.-technological-advancements}

**Newer technologies for this problem:**

* Spatial cross-validation and spatial features validate extrapolation and captures positional effects (Roberts et al., 2017).

## **5\. Comparison of Existing Systems** {#5.-comparison-of-existing-systems}

| Technology | Strength | Weakness |
| ----- | ----- | ----- |
| XGBoost | High accuracy, good regularization, popular | Needs tuning; less natural handling of categorical features (prone to errors with non-ASCII characters) |
| LightGBM | Very fast, efficient on large/high-dimensional data | Easily overfits if the tree is too deep/has too many leaves |
| CatBoost | Best categorical feature handling (ordered encoding), reduces target-leakage risk through ordered target statistics | Slower training |
| Ensemble (stacking) | Combines multiple models, can reduce variance | Marginal benefit when component models are highly correlated |

*Table 3\. Comparison of Existing Systems*

## **6\. Gaps in the Literature/Technology** {#6.-gaps-in-the-literature/technology}

Multiple model architectures were compared for predictive performance using Weights & Biases. Feature-importance analysis, however, is currently limited to a representative tree-based model.

The dataset lacks several key structural attributes, including house age, property condition, and reliable direction information. Although listing dates are available, the observation period is not sufficiently long or consistent to support reliable longitudinal market-trend forecasting. These limitations contribute to residual variance and restrict the scope of trend analysis.

Most automated valuation systems primarily provide point estimates and offer limited communication of prediction uncertainty. In an earlier frontage-house experiment, applying tighter segment-specific IQR outlier filtering reduced MAPE from approximately 107% to 26.6%. This historical result is separate from the final six-bucket evaluation. The current application displays an indicative MAPE-based error range around each prediction, based on the final model’s global MAPE. This range is not a statistically calibrated confidence interval. TabPFN was evaluated using point predictions and standard regression metrics; posterior predictive uncertainty was not explicitly analysed in the current prototype.

## **7\. Justification for the Project** {#7.-justification-for-the-project}

## **The Necessity: High-Variance in Current Valuation Systems** 

The core problem in current real estate valuation systems, particularly for the Vietnamese market, is that traditional predictive models suffer from unacceptably high pricing deviations (high variance). Even when utilizing standard ensemble methods, existing systems struggle because the underlying data severely lacks crucial structural attributes (such as house age, condition, and direction). Furthermore, the presence of extreme ultra-luxury outliers heavily skews point estimations, making standard baseline approaches unreliable for localized market trend analysis.

## **Linkage to Existing Research and Related Works** 

A comprehensive review of existing literature and related systems reveals a common methodological bottleneck: most studies apply standard machine learning models to raw listing data without effectively handling regional specificities or data sparsity. By extracting and comparing methodologies from related works specifically looking at their data processing scope, regional filtering, and model selection it becomes evident that traditional ensemble models alone cannot overcome the inherent noise in real estate data. Our project directly builds upon and critiques these foundations by identifying that without advanced spatial feature engineering and rigorous outlier segmentation, standard models hit a performance ceiling.

## **The Uniqueness of Our Approach** 

This project differentiates itself from existing systems by shifting the focus from simply "applying a model" to holistically optimizing both the data representation and the architectural approach:

1. **Data Enrichment & Segmentation:** Instead of accepting the lack of attributes, we engineer external Geospatial Points of Interest (POI) features (e.g., proximity to schools, hospitals, and transit) to compensate for missing structural data. Crucially, we structurally segment the market into a 9-model (3-tier × 3-algorithm) architecture (Price Tier × Property Type) and apply customized, tighter IQR outlier fences to prevent ultra-luxury properties from distorting the models.

2. **Advanced Comparative Modeling & Interpretability:** We move beyond basic ensemble methods by researching and comparing cutting-edge approaches. We benchmark traditional robust models, evolving into our final LightGBM and CatBoost ensemble, against state-of-the-art foundation models for tabular data like TabPFN. This comparative approach enables us to identify the most suitable method for production rather than relying on a standard baseline.

## **Expected Impact** 

By addressing the root causes of high variance through data enrichment and customized outlier filtering, and by systematically comparing traditional ensembles against new foundational models, this project provides a significantly more robust valuation tool. The ultimate impact is a drastic reduction in pricing deviation (error per square meter), offering a highly accurate, explainable, and optimized solution for determining real estate values and market trends.

# 

# **IV. Methodology** {#iv.-methodology}

## **1\. Research Questions and Objectives** {#1.-research-questions-and-objectives}

Mục tiêu trọng tâm của dự án là phát triển một hệ thống định giá bất động sản thử nghiệm, có khả năng mở rộng và định hướng triển khai thực tế cho thị trường Việt Nam, tập trung cụ thể vào phân khúc nhà ở tại Thành phố Hồ Chí Minh. Để đạt được mục tiêu này, nghiên cứu tập trung giải quyết các câu hỏi sau:

**CH1**: Việc kỹ thuật hóa các đặc trưng địa không gian có thể bù đắp như thế nào cho sự thiếu hụt các thuộc tính cấu trúc đáng tin cậy (như tuổi thọ căn nhà, tình trạng nội thất, hướng nhà) trong các tin đăng bất động sản thô tại Việt Nam?

**CH2**: Việc phân đoạn dữ liệu theo loại hình bất động sản (nhà mặt tiền và nhà trong hẻm) kết hợp với phân khúc giá (Thấp / Trung bình / Cao) và lọc nhiễu theo từng phân đoạn có giúp giảm đáng kể phương sai dự báo và sai số định giá so với mô hình tổng thể duy nhất không?

**CH3**: Hiệu quả của mô hình TabPFN so với XGBoost trong các thử nghiệm lịch sử theo loại hình tài sản ra sao, và tại sao tổ hợp mô hình LightGBM-CatBoost lại được lựa chọn cho kiến trúc triển khai sáu phân đoạn cuối cùng?

## 

## **2\. Data Collection and Preprocessing** {#2.-data-collection-and-preprocessing}

### **2.1 Data collection**

Approximately 11,000 raw listing links were initially collected. After detail extraction, deduplication, cleaning, geospatial matching, and model-specific filtering, 10,421 records were used for model training and evaluation.

#### ***Two-Stage Data Collection Pipeline***

The scraping pipeline is divided into two stages:

| Stage | Script | Input | Output | Estimated Time |
| ----- | :---: | :---: | :---: | :---: |
| **Stage 1 – Listing Crawl** | scheduling.py | Listing pages (e.g., pages 1-50) | alonhadat\_listings.csv | 2-3 minutes |
| **Stage 2 – Property Detail Extraction** | link\_to\_details.py | Property URLs from Stage 1 | alonhadat\_details.csv | 10-30 minutes |

*Table 4\. Two-Stage Data Collection Pipeline*

##### Stage 1 – Listing Metadata

The first stage collects summary information for each property listing, including:

* Property URL

* Listing title

* Displayed price

* District/Ward

* Listing status

This stage typically produces approximately **2,500 listings** (around **1-2 MB**).

##### Stage 2 – Property Details

The second stage visits every property URL collected in Stage 1 and extracts detailed information, including:

* Exact property price

* Land area (m²)

* Property dimensions

* Number of floors

* Number of bedrooms

* Legal status

* Full address

* Property description

* Additional structured attributes

The resulting dataset contains approximately **2,500 properties, 25+ features**, and occupies roughly **8-12 MB**.

#### ***Data Collection Challenges***

Several practical challenges were encountered during web scraping. The corresponding mitigation strategies are summarized below.

| Challenge | Solution |
| :---: | :---: |
| IP blocking and rate limiting | Exponential backoff with User-Agent rotation |
| Inconsistent address formats | Standardized common abbreviations (e.g., *Q.* → *Quận*, *P.* → *Phường*) |
| Missing attribute values | Rule-based imputation during preprocessing, including geometric estimation of missing property dimensions. |
| Duplicate property listings | Removed duplicates based on property address and seller information |
| Website layout changes | Implemented multiple fallback CSS selectors |

*Table 5\. Data Collection Challenges*

#### ***Data Quality Assurance***

To ensure dataset reliability before downstream processing, several validation rules are applied:

* Property price must be within **0.1–200 billion VND**

* Property area must be within **10–2,000 m²**

* Dataset validation performed before the geocoding and feature engineering stages

#### ***Running the Scraping Pipeline***

Execute the pipeline using: python main.py \--start-page 1 \--end-page 50

The pipeline generates the following raw datasets:

* data/raw/alonhadat\_listings.csv      \# \~2,500 listing summaries

* data/raw/alonhadat\_details.csv       \# \~2,500 detailed property records (25+ features)


### **2.2 Preprocessing:**

Raw data undergoes rigorous cleaning via pipeline/transformation/cleaning.py and scripts/clean\_features.py:

| Step | Description |
| :---- | :---- |
| **Column Translation** | Vietnamese column headers (e.g., Chiều ngang, Số lầu) are mapped to standardised English names (width\_m, num\_floors) via a VI\_TO\_EN\_COLS dictionary. |
| **Deduplication** | Semantic duplicates are removed by matching the combination of (lat, lon, price\_vnd, area\_m2), eliminating relisted or mirrored listings without requiring exact URL matches.  |
| **Price Filtering** | Listings with prices below 100M VND are removed as data entry errors. During model training, an additional "slight pruning" step bounds prices to 2.0B – 50.0B VND, area to 15 – 500 m², and price-per-m² to 30M – 800M VND/m² to remove extreme market anomalies while preserving the dataset majority.  |
| **Dimension Imputation** | Missing width\_m and length\_m values are systematically imputed from area\_m2 using a 1:3 width-to-length assumption when both dimensions are absent (width \= √(area / 3)), or algebraically derived when only one is missing. |
| **Binary Feature Encoding** | Categorical amenity flags (kitchen, dining room, terrace, car parking) and property type / legal status fields are binarized to integer representations.  |
| **Custom Outlier Filtering** | Rather than a single global IQR fence, segment-specific multipliers are applied per property type: nha\_mat\_tien (frontage houses) use a tight IQR × 1.5 multiplier to prune the ultra-luxury tail (which spans an extreme 0.3B–1100B VND range and destabilises model learning), while nha\_trong\_hem (alley houses) use a broader IQR × 3.0 multiplier, reflecting its more compact and normally-distributed price range.  |

*Table 6\. Data Preprocessing Steps*

The cleaned dataset is persisted to data/processed/alonhadat\_features.csv and subsequently synced to a Supabase cloud database for cross-session access.

## 

## **3\. Feature Selection and Engineering** {#3.-feature-selection-and-engineering}

### **3.1 Feature Dropping**

Columns with excessive missing rates, high cardinality, or data-leakage risk are dropped before model training. This includes raw address strings (street, ward, district, old\_address), listing metadata (url, link, listing\_id), direct coordinates (lat, lon), and temporal fields (post\_day replaced by derived year/month/day components). The direction field is also excluded due to near-universal missingness in the source data.

### **3.2 Geospatial Feature Engineering**

To compensate for missing structural attributes, the pipeline relies heavily on geospatial proximity features computed via pipeline/transformation/feature\_pipeline.py:

**Geocoding**: Raw Vietnamese addresses are resolved to (lat, lon) coordinates using a three-tier strategy:

1. **In-memory cache** keyed by old\_address or (street, locality, region) zero-latency lookup.  
2. **Persistent CSV cache** (data/localities.csv) survives restarts.  
3. **Nominatim API fallback** called at a 3-second rate-limited interval; results are immediately cached.

**POI Feature Extraction:** The pipeline retrieves nearby Points of Interest from OpenStreetMap through the Overpass API. It calculates the nearest geodesic distance and the number of facilities within predefined radii for each property. Previously retrieved POI results are stored in a persistent CSV cache to reduce repeated API requests.

| Feature Group | Features | Radii |
| :---- | :---- | :---- |
| **Schools** | nearest\_school\_km, school\_count\_3km | 3 km |
| **Hospitals** | nearest\_hospital\_km, hospital\_count\_5km | 5 km |
| **Marketplaces** | nearest\_marketplace\_km, marketplace\_count\_3km | 3 km |
| **Shopping Malls** | nearest\_mall\_km, mall\_count\_3km | 3 km |
| **Bus Stops** | nearest\_bus\_stop\_km, bus\_stop\_count\_1km | 1 km |
| **Metro Stations** | nearest\_metro\_km, metro\_count\_5km | 5 km |
| **Supermarkets** | nearest\_supermarket\_km, supermarket\_count\_3km | 3 km |

*Table 7\. Geospatial Feature Engineering Metrics*

**Distance to City Centre**: Geodesic distance to the HCM City centre (10.7769°N, 106.7009°E) is calculated using geopy.distance.geodesic.

### **3.3 Engineered Interaction Features**

The training scripts derive additional composite features to capture non-linear relationships:

* **Dimensional features**: perimeter\_m, shape\_ratio (width/length), area\_x\_floors, area\_x\_bedrooms, area\_per\_bedroom  
* **Log-transformed features**: log\_area, log\_distance\_to\_center, log\_population\_density (to reduce skewness)  
* **Location quality score:** A weighted inverse-distance composite location\_score \= (10 / (dist\_center \+ 1)) \* 2.0 \+ (10 / (nearest\_school \+ 1)) \* 1.5 \+ ... summarising urban accessibility in a single scalar.  
* **Amenity density score**: amenity\_score a weighted count of nearby facilities, with metro stations weighted most heavily (×3.0).  
* **Interaction term**: interaction\_loc\_amenity \= location\_score \* amenity\_score  
* **NLP text flags:** extracted from listing titles/descriptions (Vietnamese keyword matching): is\_hem\_xe\_hoi (car-accessible alley), is\_mat\_tien (frontage), is\_no\_hau (widened rear), has\_noi\_that (furnished), is\_gap (urgent sale), is\_kinh\_doanh (commercial use).  
* **Train-set-derived locality target encoding**: Locality statistics are computed from the training set and mapped to the test set, avoiding direct use of test targets. However, fold-wise spatial encoding has not yet been implemented.  
* **Missing value indicators**: Binary flags for sparse features (nearest\_metro\_km\_missing, width\_m\_missing, etc.) to preserve missingness as a model signal.

### **3.4 Data Segmentation**

The final production model uses **3-tier price segmentation** (simplified from earlier 9-model (3-tier × 3-algorithm) property-type × price experiments):

**Price Tiers (3 levels)**: 
- Low (0–5B VND) — Budget segment
- Mid (5–20B VND) — Mid-range segment  
- High (\>20B VND) — Luxury segment

**Rationale for Price-Only Segmentation:**
Historical experiments tested property-type × price segmentation (6 buckets: 2 types × 3 tiers), but analysis showed that **price tier was the dominant driver** of predictive performance, while property-type distinctions added complexity without substantial accuracy gains. The final architecture focuses on the stronger signal (price tier) and uses a **3-tier × 3-algorithm ensemble = 9 total models**.

Each tier is trained independently to capture tier-specific feature relationships and price dynamics, eliminating the compromise of a single generalized global model while maintaining simplicity in the routing logic.

## 

## **4\. Model Training and Validation** {#4.-model-training-and-validation}

### **4.1 Model Architecture: 3-Tier Price-Segmented Ensemble**

The production architecture implements a **3-tier price-segmented ensemble** that routes predictions based on the user's estimated budget tier:

**Price Tiers:**
- **Low** (0–5 Billion VNĐ) — Budget apartments and smaller properties
- **Mid** (5–20 Billion VNĐ) — Mid-range residential properties
- **High** (\>20 Billion VNĐ) — Luxury and premium properties

**Ensemble Strategy:**
Each price tier employs **three complementary models** to maximize robustness and generalization:
1. **LightGBM** — Fast, efficient gradient boosting (primary model)
2. **XGBoost** — Robust tree boosting with regularization
3. **CatBoost** — Categorical feature handling, reduces target leakage

**Prediction Process:**
1. User selects estimated price tier (or model automatically suggests based on property size)
2. All three models (LGBM, XGBoost, CatBoost) make predictions in log-space
3. Predictions are averaged: `avg_log_price = (log_lgbm + log_xgb + log_catboost) / 3`
4. Result is inverse-transformed to VNĐ: `final_price = exp(avg_log_price) - 1`

This **3-tier × 3-model = 9-model ensemble** design provides:
- ✅ Tier-specific optimization (each tier has distinct property characteristics)
- ✅ Model diversity (reduces overfitting, improves robustness)
- ✅ Ensemble averaging (more stable than any single model)
- ✅ Interpretability (can inspect individual model contributions)

### **4.2 Model Configurations (3-Model Ensemble per Tier)**

| Hyperparameter | LightGBM | XGBoost | CatBoost |
| :---- | :---- | :---- | :---- |
| **Estimators / Iterations** | 1,000 | 1,500 | 1,500 |
| **Max Depth** | 8 | 8 | 8 |
| **Learning Rate** | 0.05 | 0.03 | 0.05 |
| **Subsample** | 0.8 | 0.8 | — |
| **Column Sample** | 0.8 | 0.8 | — |
| **L1 Regularization** | 0.1 | — | — |
| **L2 Regularization** | 1.0 | — | — |
| **Loss Function** | Default MSE | RMSE | RMSE |
| **Early Stopping** | — | — | 50 rounds |
| **Random Seed** | 42 | 42 | 42 |

**Ensemble Strategy:** All three models are trained per price tier (3 tiers = 9 total models). Predictions are made in log-space and averaged before inverse transformation. Each tier-model pair is tuned independently to capture price-segment-specific relationships.

*Table 8\. Model Configurations (3-Model Ensemble per Tier)*

### 

### **4.3 Data Splitting and Validation**

**Split Ratio**: 80% training / 20% holdout test, using a fixed random\_state=42 for reproducibility.

**Experiment Tracking**: All training runs are logged to Weights & Biases (wandb) under the project real-estate-valuation, capturing hyperparameters, metrics, feature lists, and diagnostic plots.

## **5\. Evaluation Metrics** {#5.-evaluation-metrics}

Model performance is assessed on the held-out 20% test set using the following metrics:

| Metric | Formula | Rationale |
| :---- | :---- | :---- |
| **MAPE** | (100/n) × Σ |y − ŷ| / y |  Primary target metric scale-independent, directly comparable across price tiers. Project objective: MAPE \< 10% (Section I.3).  |
| **RMSE** | √mean((y \- ŷ)²) | Penalises large absolute errors; expressed in Billion VND for interpretability. |
| **MAE** | mean(|y − ŷ|) | Average absolute deviation in Billion VND; more robust to outliers than RMSE. |
| **R²** | 1 \- SS\_res / SS\_tot | Proportion of price variance explained by the model. Target: R² \> 0.90. |
| **RMSE (log-space)** | RMSE on log1p(price) scale | Measures fit quality in the transformed target space; less sensitive to outliers. |

*Table 9\. Evaluation Metrics Definitions*

### **5.1 Evaluation Protocol and Interpretation**

The reported global MAPE of 13.10% represents bucket-aware evaluation conditional on correct price-tier assignment. During evaluation, test samples are assigned to the Low, Mid, or High price tiers using their observed target prices. Therefore, the reported result measures prediction performance within known market segments and should not be interpreted as the end-to-end performance of an automatic price-tier routing system. In the current Streamlit application, the price tier is selected by the user.

## **6\. Implementation Plan** {#6.-implementation-plan}

### **6.1 Scraping Layer**

Property listing links and detailed attributes were collected from Alonhadat.com using Python Requests and BeautifulSoup. The scraper sends HTTP requests with configured headers, parses the returned HTML pages, and includes request delays, retry handling, blocked-page detection, and optional network reconnection to reduce interruptions during data collection.

### **6.2 ETL Pipeline**

main.py orchestrates the end-to-end ETL flow:

1. Crawl list pages → extract listing details

2. Clean and translate raw data (cleaning.py)

3. Merge administrative density data (load\_density.py)

4. Geocode addresses with cache-first strategy (load\_pois.py)

5. Compute geospatial POI features in configurable mini-batches with checkpointing (feature\_pipeline.py)

6. Upload the enriched dataset to Supabase (supabase\_handler.py)

### **6.3 Training Layer**

* scripts/clean\_features.py: Final feature preparation and segment splitting.

* scripts/train\_ensemble.py: 9-Model (3-Tier × 3-Algorithm) LightGBM + XGBoost + CatBoost ensemble training with W\&B logging. Produces 12 .pkl model artefacts under models/.

* scripts/train\_tabpfn.py: Historical TabPFN segmented experiment using its own preprocessing and routing configuration. Its results are reported separately and are not directly comparable with the final 9-model (3-tier × 3-algorithm) ensemble evaluation.

### **6.4 Inference and Serving Layer**

A Streamlit web application (app/app.py) serves the trained ensemble:

* Users select a ward (phường/xã), street, property type, price tier, dimensions, and amenities via an interactive form.

* The app performs a real-time geo-lookup, applies locality target encoding, assembles a feature vector matching the training schema, and routes the request to the correct bucket model pair.

* Results are displayed as a predicted price in Billion VND, an indicative MAPE-based error range, and a price-per-m² figure, alongside an interactive map and POI breakdown.

## 

## **7\. Ethical Considerations** {#7.-ethical-considerations}

### **7.1 Data Privacy**

All data collected is sourced exclusively from publicly accessible real estate listing portals. No personally identifiable information (PII) beyond what is voluntarily disclosed in public listings (contact numbers, owner names) is retained. The pipeline explicitly drops phone number and contact name columns (Số Điện Thoại, Tên liên hệ) during cleaning. Supabase storage is secured via API key authentication with environment-variable-based credential management (.env file, excluded from version control via .gitignore).

### **7.2 Pricing Fairness and Bias**

The use of segment-specific IQR filters means that ultra-luxury properties (\>50B VND) are excluded from the training distribution. This is a deliberate modelling choice, not a reflection of a market value judgement, and is clearly documented. However, it means the model should not be used to appraise properties clearly outside the trained price range without human expert review.

The locality\_price\_median encoding is computed strictly from the training set and applied to the test set via lookup, reducing direct test-set leakage risk. Unseen localities default to the global training median, which may introduce mild geographic bias in under-represented wards.

### **7.3 Model Transparency and Limitations**

* The model is presented as a reference price estimate, not a legally binding appraisal. The application explicitly communicates an indicative MAPE-based error range (currently ±13.10%) alongside each prediction. This range is not a statistically calibrated confidence interval.

* A Top 20 LightGBM feature-importance plot is generated for a representative mid-price alley-house bucket and logged to Weights & Biases.

* The current model is trained on Ho Chi Minh City data; application to other Vietnamese cities (e.g., Hanoi) without retraining would produce unreliable estimates and is not recommended.

* Listing data reflects market asking prices, not final transaction prices. Predictions inherit this bias and should be interpreted as listing price estimates rather than fair market value.

### **7.4 Scraping Ethics**

Web scraping is conducted respectfully: requests are rate-limited (3-second intervals for geocoding), and a persistent cache (data/localities.csv) minimises redundant API calls to the OpenStreetMap Nominatim service, honouring its usage policy.

# 

# **V. System Design and Implementation** {#v.-system-design-and-implementation}

## **1\. AI Model Integration** {#1.-ai-model-integration}

The AI model is integrated into a Streamlit web application, bridging raw user inputs with a 9-Model (3-Tier × 3-Algorithm) Ensemble (LightGBM + XGBoost + CatBoost) via a dedicated inference engine.

### **1.1 Interface & Data Collection**

Users input property details (location, property type, budget tier, dimensions, and amenities) through an interactive UI form.

### **1.2 Geographic Enrichment**

Raw addresses are passed to the GeoLookup module. It uses a local cache andNominatim API fallback to geocode the location into (lat, lon) coordinates. It then spatially queries nearby Points of Interest (POIs) like schools and metro stations to compute proximity features.

### **1.3 Feature Assembly**

The inference engine (build\_row()) acts as a real-time ETL prototype. It transforms UI inputs and geographic data into a strict schema mirroring the training prototype. This includes algebraic imputation of missing dimensions, derivation of complex metrics (e.g., location\_score, log\_area), and applying locality target encoding based on the user's selected ward.

### **1.4 Segmented Routing & Ensembling**

Instead of a single model, the system uses a Segmented Router:

Routing: It dynamically selects a specific model bucket pair (e.g., lgbm\_mid\_nha\_mat\_tien and cb\_mid\_nha\_mat\_tien) based on the user's selected property type and budget tier.

Execution: The assembled feature vector is fed to both the LightGBM and CatBoost models for that bucket.

Ensembling: Both models predict prices in log-space (log1p(price\_vnd)). The system averages these two outputs and applies an inverse transformation (np.expm1) to produce the final VND price.

## 

## **2\. Data Flow and Processing** {#2.-data-flow-and-processing}

### **2.1 End-to-End Data Pipeline Architecture**

![][image2]

*Figure 1\. End-to-End Data Pipeline Architecture*

### **2.2 Detailed Data Processing Steps**

#### **Phase 1: Ingestion \- Web Scraping**

**Process:** scheduling.py crawls pagination indexes, while link\_to\_details.py visits individual property URLs to extract raw dimensions, prices, and address strings from Alonhadat.com.

**Output**: Raw listings CSV (data/raw/alonhadat\_listings.csv)

#### ***Phase 2: Geocoding***

**Process**: load\_pois.py normalizes raw addresses and queries the OpenStreetMap Nominatim API to resolve (latitude, longitude) coordinates. It computes the geodesic distance to the city centre.

**Optimization**: Cached results reduce repeated API calls on repeat runs.

#### **Phase 3: Density Features**

**Process:** load\_density.py queries administrative demographic datasets to append the population density and total area metrics for each coordinate's ward/commune.

#### **Phase 4: Geospatial Feature Engineering**

**Process:** The system queries OpenStreetMap POI data through the Overpass API and calculates the nearest geodesic distance and total number of facilities within specified radii for seven POI categories: schools (3 km), hospitals (5 km), marketplaces (3 km), supermarkets (3 km), shopping malls (3 km), bus stops (1 km), and metro stations (5 km). Cached results from data/localities.csv are reused whenever available to reduce API calls and processing time.

#### **Phase 5: Baseline Cleaning & Outlier Removal (***scripts/clean\_features.py***)**

**Deduplication:** Removes semantic duplicates (matching lat, lon, price, area).

**Imputation:** Mathematically derives missing width or length parameters from total area (assuming a 1:3 ratio).

**Custom Filtering:** Applies segment-specific IQR fences (IQR ×1.5 for luxury frontage houses, IQR ×3.0 for standard alley houses) to remove market extremes.

**Output:** Saves intermediate alonhadat\_features\_cleaned.csv.

#### **Phase 6: Model-Specific Preprocessing & Enrichment (**scripts/train\_ensemble.py**)**

This phase runs dynamically right before the data is fed into the LightGBM/CatBoost models.

**Strict Pruning:** Drops any residual rows outside the target market bounds (Price: 2.0B–50.0B VND, Area: 15–500 m², Unit Price: 30M–800M VND/m²).

**Feature Derivation:** Computes composite metrics on the fly (e.g., shape\_ratio, location\_score, interaction\_loc\_amenity, log-transforms).

**Text Mining:** Parses boolean flags (e.g., is\_hem\_xe\_hoi for car-accessible alleys, is\_mat\_tien for frontage) from raw listing descriptions via regex.

**Encoding:** Applies one-hot encoding to categorical variables and target encoding (median price) to the ward/locality.

## **3\. Deployment Strategy** {#3.-deployment-strategy}

The system is deployed as a production-ready containerized application with multi-platform support. Docker and Docker Compose provide consistent environments across development and deployment stages.

### **3.1 Containerized Architecture**

**Container Images:**
- **FastAPI API Container** (.deployment/Dockerfile): Multi-stage build for minimal size; runs on port 8000
- **Streamlit UI Container** (.deployment/Dockerfile.streamlit): Web interface running on port 8501
- **Local Development** (.deployment/docker-compose.yml): Both services orchestrated together with internal networking

**Key Features:**
- ✅ Multi-stage Docker builds (builder → runtime)
- ✅ Non-root user execution (security)
- ✅ Health checks for both services
- ✅ Dynamic port configuration via environment variables
- ✅ Comprehensive logging and debug output

### **3.2 Deployment Platforms**

**Render (Current Production)**
- Automatic deployment via git push to main branch
- render.yaml defines 2 services: API and Streamlit UI
- Environment variables managed via Render dashboard
- Each service receives unique public URL

**Local Development (Docker Compose)**
```bash
docker-compose up              # Start both services
docker-compose logs -f         # Monitor logs
docker-compose down -v         # Full cleanup
```

**DigitalOcean & Self-Hosted**
- Deploy docker-compose.yml directly to VPS
- Requires: Docker, Docker Compose, git
- See DEPLOYMENT.md for step-by-step guides

### **3.3 Configuration Management**

- **Secrets:** .env file with Supabase credentials, API keys (excluded from git via .gitignore)
- **Template:** .env.example documents all required variables
- **Environment-based:** PORT, ENV, API_URL read from system environment
- **Streamlit Config:** Server settings via STREAMLIT_* environment variables

### **3.4 Data & Model Management**

* **Data layer:** Processed datasets via Supabase (PostgreSQL cloud database)
* **Model artifacts:** .pkl and .joblib files loaded on startup (cached in memory)
* **Cache strategy:** data/cache/localities.csv for repeated geocoding/POI lookups
* **Model versioning:** W&B (Weights & Biases) experiment tracking with model metadata

## **4\. Scalability and Maintenance** {#4.-scalability-and-maintenance}

### **4.1 Ingestion Scalability**

The geocoding and POI lookup pipeline uses a **two-tier cache strategy**:

1. **In-memory cache:** (Python dict, session-scoped) Zero-latency lookups for addresses within a session
2. **Persistent cache:** (data/cache/localities.csv, ~10MB) Survives application restarts; reused across runs
3. **API fallback:** (Nominatim & Overpass, rate-limited 3s intervals) Called only for new addresses not in cache

**Effect:** Reduces repeated API calls by 90%+, enabling fast incremental data collection and reducing Nominatim API usage.

### **4.2 Inference Scalability**

- **Single-property latency:** ~200-500ms per prediction (model loading + feature engineering + ensemble inference)
- **Model caching:** st.cache\_resource in Streamlit (load once per session, not per-user)
- **Batch inference:** Supported via FastAPI POST /api/predict endpoint (accepts single JSON objects)
- **Concurrent throughput:** Single-threaded Streamlit + FastAPI; horizontal scaling via Docker replicas (Render, K8s)

### **4.3 Model Maintenance & Versioning**

**Current workflow:**
- Manual retraining via `python scripts/train_ensemble.py`
- Experiment tracking in Weights & Biases (wandb project: real-estate-valuation)
- Model artifacts stored as .pkl files (12 files: 6 LightGBM + 6 CatBoost per 9-model (3-tier × 3-algorithm) segmentation)
- Version history documented in LATEST_UPDATE.md

**Version tracking (sample):**
- v2.6: LightGBM + CatBoost ensemble (MAPE 13.10%, R² 0.9200) — Current production
- v2.5: Historical TabPFN experiments (MAPE 24.22%) — Archived
- v2.0-v2.4: Earlier XGBoost/ensemble iterations — Archived

**Future improvements:**
- Automated retraining schedule (monthly, triggered by detected drift)
- Model registry (MLflow or similar) for reproducibility
- A/B testing framework for gradual model rollout

### **4.4 Codebase Organization**

**Clean structure (post-consolidation):**
- `app/` — FastAPI backend + Streamlit frontend (primary serving path)
- `pipeline/` — ETL orchestration (data cleaning, feature engineering, Supabase sync)
- `models/` — Trained ensemble artifacts + training scripts
- `notebooks/` — Analysis & experiments (organized into 5 professional sections)
- `tests/` — Unit tests (pytest)
- `.deployment/` — Docker & deployment configs

**Archived components (preserved for reference):**
- `scripts/train_tabpfn.py` — Historical TabPFN training (not deployed)
- `scripts/train.py` — Earlier XGBoost baseline (not deployed)
- Legacy Flask API routes — Replaced by FastAPI

**CI/CD & Version Control:**
- GitHub Actions workflows in `.github/workflows/`
- Branch protection on main (requires PR review)
- Automatic Docker build/push on release tags

# **VI. Results and Discussion** {#vi.-results-and-discussion}

## **1\. Results and Analysis** {#1.-results-and-analysis}

### **1.1 Experimental Setup and Results**

Kết quả của ba phân đoạn giá cuối cùng được trình bày cùng với các thử nghiệm cơ sở lịch sử. Hệ thống được đánh giá trên tập kiểm thử độc lập chiếm 20% dữ liệu từ tập dữ liệu đã làm sạch gồm 10,421 bất động sản tại Thành phố Hồ Chí Minh.

Do tính chất của bài toán hồi quy, nghiên cứu sử dụng Sai số phần trăm tuyệt đối trung bình (MAPE), Sai số bình phương trung bình căn (RMSE) và Hệ số xác định (R²) làm các chỉ số đánh giá chính.

| Metric | Global Performance |
| :---- | :---- |
| **MAPE** | **13.10%** |
| **R²** | **0.9200** |
| **MAE** | **2.15B VND** |
| **RMSE** | **3.41B VND** |

*Table 10\. Final Model Performance (9-Model 3-Tier Ensemble)*

| Architecture | Dataset Split | R² | MAPE |
| :---- | :---- | :---- | :---- |
| **TabPFN** | **Full Dataset (Global)** | **0.8145** | **24.22%** |
| **XGBoost** | **Full Dataset (Global)** | **0.7848** | **25.37%** |
| **TabPFN** | **Alley Houses Only (Nhà trong hẻm)** | **0.8440** | **18.54%** |
| **XGBoost** | **Alley Houses Only (Nhà trong hẻm)** | **0.8172** | **21.21%** |
| **TabPFN** | **Frontage Houses Only (Nhà mặt tiền)** | **0.7036** | **26.58%** |
| **XGBoost** | **Frontage Houses Only (Nhà mặt tiền)** | **0.6768** | **27.91%** |

*Table 11\. Historical Segmentation Impact (Pre-Ensemble Benchmarks)*

### **1.2 Analysis and Interpretation**

Data segmentation and domain-aware outlier handling appeared to be important contributors to performance improvement in the conducted experiments. As shown in Table 11, applying a single global model (even a prior-data fitted tabular foundation model like TabPFN) stalled at a \~24% MAPE.

Notably, the budget segment (0-5B VND) achieved a MAPE of 10.48%, sitting almost exactly at the project’s target threshold of \<10%. The remaining error is largely concentrated in the mid- and high-price tiers, where listing sparsity and price heterogeneity (e.g., luxury interior finishing, which is absent from the data) are greatest.

Model Architecture Comparison: Tree Boosting vs. TabPFN In head-to-head testing prior to the final 9-model (3-tier × 3-algorithm) ensemble, TabPFN consistently outperformed XGBoost by 1 to 3 percentage points of MAPE across every data split (Table 11). This suggests that TabPFN's prior-fitted Bayesian architecture generalizes better on small, high-variance tabular datasets (\~1,500 rows per segment) than pure tree boosting.

Despite TabPFN's isolated accuracy, the final prototype architecture utilized a LightGBM and CatBoost ensemble for deployment practicality.

### **1.3 Feature Importance and Model Interpretability**

To improve model transparency, global feature-importance analysis was conducted on a representative tree-based model within the 9-model ensemble. The relative contribution of each feature to the model's predictive accuracy is visualized in the figure below.

**![][image3]**

*Figure 2\. Top 20 LightGBM Feature Importances for the Mid-Price Tier*

**Interpretation**:

1. **Raw Property Size Remains Dominan**t: The raw physical size of the property (such as area\_m2 or area\_x\_floors) typically dominates the top splits. This confirms that despite the heterogeneity of the Vietnamese market, sheer physical size remains the primary baseline driver of value, fully consistent with traditional hedonic pricing theory.

2. **The Impact of Accessibility and Dimensionality**: Features like road width (road\_width\_m) and composite metrics like shape\_ratio (width vs. length) consistently rank high. In dense urban environments like Ho Chi Minh City, particularly in the alley-house (nhà trong hẻm) segment, accessibility and the geometric utility of the land plot heavily dictate the property's premium.

3. **Geospatial Compensation for Missing Data**: Engineered POI features (such as interaction\_loc\_amenity, distance\_to\_center\_km, or proximity to markets/schools) collectively contribute a highly meaningful share of the predictive power. This supports our core hypothesis: in the absence of hard structural data (like house age or interior condition), engineering proximity features through OpenStreetMap Overpass API queries, geodesic distance calculations, and persistent CSV caching provides the model with useful proxy signals for neighbourhood accessibility and urban development.

## **2\. Discussion** {#2.-discussion}

### **2.1 Addressing the Research Questions**

**RQ1: Geospatial Compensation for Missing Structural Attributes**

The findings provide partial support for this hypothesis; feature-importance analysis indicates that geospatial variables, including distance to the city centre and nearby facility counts, provide useful supplementary information for property-price prediction. However, directly observed structural characteristics such as area\_m2 remain the dominant predictors.

The POI features therefore act as proxy indicators of neighbourhood accessibility and urban development rather than complete replacements for missing attributes such as house age, property condition, and interior quality. In the current implementation, these features are obtained from OpenStreetMap through Overpass API queries, geodesic distance calculations, and persistent CSV caching.

**RQ2: The Impact of Structural Segmentation and Outlier Filtering**

This hypothesis received substantial support from the experimental results. The Vietnamese real estate market is highly heterogeneous, and historical experiments showed that frontage-house predictions were particularly sensitive to extreme luxury listings. In one earlier frontage-specific experiment, tighter IQR-based outlier filtering reduced MAPE from approximately 107% to 26.6%.

The final approach divided the data into three price tiers and trained separate 3-algorithm ensembles (LightGBM, XGBoost, CatBoost) for each segment. Under the bucket-aware evaluation protocol, the final architecture achieved an R² of 0.9200 and a global MAPE of 13.10%, while the low-price segment achieved a MAPE of 10.48%, approaching the target of below 10%.

Because the historical frontage experiment and the final six-bucket evaluation used different model configurations and evaluation settings, the values of 107% and 13.10% should not be interpreted as a direct before-and-after comparison. Nevertheless, the findings suggest that market segmentation and domain-aware outlier handling were important contributors to improved predictive performance.

**RQ3: Historical TabPFN-XGBoost Comparison and Final Model Selection**

Historical experiments showed that TabPFN achieved lower MAPE than XGBoost across the evaluated property-type splits. However, these experiments were conducted using an earlier preprocessing and segmentation configuration and are therefore not directly comparable with the final 9-model (3-tier × 3-algorithm) ensemble result.

The LightGBM-CatBoost ensemble was selected for the final application because it offered lower inference cost, straightforward model serialization, and easier integration with the 3-tier price-based routing architecture. Therefore, the results should not be interpreted as evidence that the final ensemble was more accurate than TabPFN under identical experimental conditions. Instead, they illustrate the trade-off between historical benchmark accuracy and deployment practicality.

### **2.2 Alignment with Related Work**

* The finding that segmentation and outlier handling matter more than model architecture echoes Ha et al. (2025)'s Vietnam-specific study, which also reported substantial performance gaps between regions and property types when using a single pooled model.

* Likewise, the success of gradient-boosted trees (LightGBM, CatBoost, XGBoost) on this tabular dataset is consistent with the broader AVM literature reviewed in Section III.3–III.4, and the dominance of area\_m2 as the top feature mirrors long-standing hedonic pricing theory.

### **2.3 Limitations and Challenges Encountered**

While the project achieved a highly accurate and deployable system, several practical limitations were encountered:

1. **Small, Geographically Narrow Dataset**: The final cleaned dataset contains only 10,421 properties concentrated exclusively in Ho Chi Minh City. This limits the system's ability to generalize to other major cities (like Hanoi or Da Nang) and thins the sample sizes in the high-price tier buckets.  
2. **Web Scraping Fragility**: As anticipated in the risk register, anti-bot measures and CAPTCHAs on the source website constrained the volume of data that could be collected in a given run, directly limiting the final dataset size.  
3. **Limited longitudinal coverage**: Listing dates are available, but the collected observation period is not sufficiently long and consistent to support reliable longitudinal market-trend forecasting. Consequently, the model excels at cross-sectional valuation but cannot yet perform genuine longitudinal market trend forecasting.

## **3\. Recommendations** {#3.-recommendations}

### **3.1 For Further Research**

* **Reintegrate TabPFN into the segmented architecture.** Since TabPFN outperformed XGBoost on every tested split (Table 11), a natural next step is training TabPFN per-bucket within the 9-model (3-tier × 3-algorithm) router.  
* **Expand longitudinal data coverage.** Collecting listing dates over multiple scraping cycles would allow the system to model actual market trend analysis.  
* **Tighten leakage safeguards on target-encoded features.** Formalize GroupKFold-by-locality (or per-listing spatial blocking) as a mandatory step before computing any target-encoded feature such as locality\_cv\_target\_median, and re-validate the final 9-model (3-tier × 3-algorithm) model's reported R²/MAPE under this stricter protocol to confirm the numbers are leakage-free.

### **3.2 For System Improvements**

* **Consolidate model architectures into a single documented pipeline.** Clearly archive the earlier ensemble-plus-meta-learner and legacy Flask/dashboard workflows, and designate the final 9-model (3-tier × 3-algorithm) Streamlit prototype as the primary reference architecture.  
* **Grow and diversify the dataset.** Extend scraping to additional portals and other major cities (Hanoi, Da Nang) to improve generalization and give the mid/high-price buckets enough samples to train reliably.  
* **Add automated retraining/monitoring.** Given the acknowledged risk of market volatility (Section I.6), schedule periodic retraining and track live prediction error against newly scraped listings to detect drift early.

### **3.3 For Practical Applications**

* The existing Streamlit web app (app/app.py) and BI dashboard (app/dashboard.py) are well positioned to be piloted with real estate brokers or buyers for feedback on prediction usability and interpretability.  
* The segmented-router design is directly reusable as a lightweight backend for property-listing platforms or bank/mortgage appraisal tools that need fast, explainable, segment-aware valuations rather than a single opaque global score.

# 

# **VII. Conclusion** {#vii.-conclusion}

## **1\. Summary of Findings** {#1.-summary-of-findings}

Dự án đã giải quyết thành công ba câu hỏi nghiên cứu đặt ra ban đầu.

* **CH1** đã được xác nhận một phần: các đặc trưng địa không gian (POI) cải thiện đáng kể độ chính xác của dự báo nhưng không thay thế hoàn toàn được các thuộc tính cấu trúc thiếu hụt; trong đó diện tích (area\_m2) vẫn là biến dự báo quan trọng nhất.

* **CH2** được củng cố bởi kết quả thực nghiệm: việc phân đoạn thị trường theo loại hình và giá giúp mô hình tổ hợp đạt R² \= 0,9138 và MAPE tổng thể là 13,47%. Phân khúc giá thấp đạt MAPE 10,48%, tiệm cận mục tiêu dưới 10%. Các thử nghiệm lịch sử cho thấy việc lọc nhiễu theo phân đoạn giúp giảm sai số đáng kể so với mô hình chung.

* **CH3** chỉ ra rằng TabPFN vượt trội hơn XGBoost trong các thử nghiệm lịch sử. Tuy nhiên, tổ hợp LightGBM-CatBoost được chọn cho hệ thống cuối cùng nhờ tính thực tiễn trong triển khai, chi phí suy luận thấp và khả năng tích hợp linh hoạt với kiến trúc điều hướng người dùng.

Overall, the project delivered a working end-to-end pipeline (scraping → geospatial feature engineering → segmented model training → web app/dashboard) and came close to its original accuracy target, falling short at the global level while approaching the target in the low-price segment.

## **2\. Contributions on the Project** {#2.-contributions-on-the-project}

### **2.1 Contribution**

The project provides evidence that market segmentation and domain-aware outlier handling can substantially improve valuation performance on heterogeneous Vietnamese real estate data. Historical frontage-specific experiments showed an improvement from approximately 107% to 26.6% MAPE, while the separately evaluated final 9-model ensemble achieved a global MAPE of 13.10%. The project also contributes a reusable geospatial feature pipeline that supplements, but does not fully replace, missing structural information in scraped property listings.

The resulting web app and dashboard are usable artifacts, not just a benchmark. The comprehensive technical report (this document) serves as the research paper deliverable, documenting the full methodology, results, and insights suitable for publication at academic conferences or journals focused on real estate analytics, machine learning applications, or geospatial data science.

### **2.2 What worked well**

Validating the segmentation hypothesis empirically (Table 11\) before committing to the final architecture, and adopting **W\&B** for experiment tracking, sharpened model comparison significantly.

### **2.3 What could improve**

Parallel research tracks including the stacking ensemble, TabPFN-based segmentation, and the final six-bucket router produced results under different experimental settings. Consequently, the reported metrics were not always directly comparable and required reconciliation during the final stages. Establishing a shared benchmarking protocol, dataset split, preprocessing pipeline, and evaluation metrics from the beginning would have improved consistency.

Model interpretation was limited to global feature-importance analysis from tree-based models. The project did not implement local, per-prediction explanations, meaning the system cannot clearly explain how individual features contribute to a specific property valuation. Future development should incorporate a suitable local explanation method to improve model transparency and provide local, per-prediction explanations.

### **2.4 Final thought**

The clearest lesson was that rigorous problem decomposition and honest data-quality assessment outweigh incremental model tuning.

The team hopes this segmented-router approach and geospatial feature engineering can serve as a foundation for more transparent valuation tools in Vietnam, extended in future work with temporal trend modeling and broader city coverage.

## **3\. Limitations and Future Work** {#3.-limitations-and-future-work}

### **3.1 Limitations**

* **Data scope:** The final dataset (10,421 properties, HCMC only) is small and geographically narrow, limiting generalization to other cities and thinning sample sizes in the mid/high-price tiers of the 3-tier ensemble.

* **Limited structural and longitudinal coverage:** House age and property condition are unavailable, while direction information is sparsely observed. Listing dates are available, but the collection period is insufficient for reliable trend forecasting. These limitations constrain the model’s explanatory power and preclude robust longitudinal market analysis.

* **Leakage risk:** Early iterations used a locality-level target-encoded feature that risks leaking price information without strict GroupKFold-by-locality validation; this needs formal re-verification on the final 9-model (3-tier × 3-algorithm) model.

* **Scraping fragility:** CAPTCHA and anti-bot measures constrained data volume, directly limiting dataset size.

* **Architecture fragmentation:** Parallel research tracks left the repository with multiple, inconsistently documented model architectures and metrics, complicating reproducibility.

* **Price-tier routing limitation:** The current model evaluation assigns test samples to price buckets using their observed prices, while the application requires users to select an approximate budget range. Consequently, the reported global performance assumes correct bucket selection. Future work should introduce a feature-based routing model and evaluate the complete pipeline without using the true target price during routing.

### **3.2 Future Work**

* **Expand data collection** to multiple cities and additional portals, and collect listings continuously over multiple scraping cycles to expand longitudinal coverage.

* **Reintegrate TabPFN** into the 9-model (3-tier × 3-algorithm) router, since it outperformed XGBoost on every benchmarked segment.

* **Formalize spatial cross-validation** before using any target-encoded feature, and re-audit the final model's metrics under that protocol.

* **Extend model interpretability** beyond global tree-based feature importance by implementing local, per-prediction explanation methods, such as SHAP, to improve model transparency and provide local, per-prediction explanations.

* **Consolidate the codebase** around the single production architecture and retire superseded ones, with automated retraining to handle market drift over time.

# 

# **IX. References** {#ix.-references}

1. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)  
2. Ha, M.-T., Nguyen, T.-C., Pham, T.-H., & Nguyen, V.-H. (2025). Using machine learning regression algorithms to predict house prices in Vietnam. *International Real Estate Review, 28*(4), 505–527. [https://doi.org/10.53383/100412](https://doi.org/10.53383/100412)  
3. Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2022). *TabPFN: Prior-data fitted networks for tabular data*. In *Proceedings of the 39th International Conference on Machine Learning (ICML 2022).* [https://arxiv.org/abs/2207.01848](https://arxiv.org/abs/2207.01848)  
4. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017).* [https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)  
5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017\)* (pp. 4765–4774).  
6. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased boosting with categorical features*. In *Advances in Neural Information Processing Systems 31 (NeurIPS 2018).* [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)  
7. Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. [https://doi.org/10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)  
8. Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241–259. [https://doi.org/10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080\(05\)80023-1)  
9. Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure competition. *Journal of Political Economy, 82*(1), 34–55.

# 

# **VIII. Appendices** {#viii.-appendices}

## **Appendix A - Core Processed Dataset Fields** {#appendix-a-core-processed-dataset-fields}

| \# | Feature | Type | Description |
| ----- | ----- | ----- | ----- |
| 1 | property\_type | int | 1 \= frontage (nhà mặt tiền), 0 \= alley (nhà trong hẻm) |
| 2 | legal\_status | int | Legal document status (encoded) |
| 3 | num\_floors | float | Number of floors |
| 4 | num\_bedrooms | float | Number of bedrooms |
| 5 | road\_width\_m | float | Adjacent road width (m) |
| 6 | width\_m | float | Property frontage width (m) |
| 7 | length\_m | float | Property length (m) |
| 8 | price\_vnd | float | Total property price (VND) |
| 9 | area\_m2 | float | Property area (m²) |
| 10–13 | dining\_room\_bin, kitchen\_bin, terrace\_bin, car\_parking\_bin | int (binary) | Amenity presence flags |
| 14 | locality\_square | float | Locality area statistic |
| 15 | locality\_population\_density | float | Population density of the locality |
| 16 | distance\_to\_center\_km | float | Distance to city center (km) |
| 17–28 | nearest\_\*\_km and \*\_count\_\*km (school, hospital, marketplace, supermarket, mall, bus\_stop) | float | Geospatial POI proximity/density features |
| 29 | metro\_count\_5km | float | Number of metro stations within 5 km |

*Table 12\. Data Dictionary*

## 

## **Appendix B - Final Model Hyperparameters (9-Model 3-Tier Ensemble)**  {#appendix-b-final-model-hyperparameters-9-model-3-tier-ensemble}

| Parameter | LightGBM | XGBoost | CatBoost |
| ----- | ----- | ----- | ----- |
| **Estimators / Iterations** | 1,000 | 1,500 | 1,500 |
| **Max Depth** | 8 | 8 | 8 |
| **Learning Rate** | 0.05 | 0.03 | 0.05 |
| **Subsample** | 0.8 | 0.8 | — |
| **Column Sample** | 0.8 | 0.8 | — |
| **L1 Regularization** | 0.1 | — | — |
| **L2 Regularization** | 1.0 | — | — |
| **Loss Function** | Default MSE | RMSE | RMSE |
| **Early Stopping** | — | — | 50 rounds |
| **Random Seed** | 42 | 42 | 42 |

*Table 13\. Final Model Hyperparameters (3 Models per Tier)*

---

## **Appendix C - Repository Structure and Documentation** {#appendix-c-repository-structure-and-documentation}

### **Complete File Organization**

```
Real-Estate-Valuation/
├── 📄 Documentation (START HERE)
│   ├── README.md                     ← Project overview & quick start
│   ├── DEPLOYMENT.md                 ← Complete deployment guide
│   └── .github/ARCHITECTURE.md       ← Repository structure reference
│
├── 🔧 Configuration Files
│   ├── render.yaml                   ← Render platform config
│   ├── Dockerfile                    ← Production API image
│   ├── pytest.ini                    ← Testing configuration
│   ├── .gitignore                    ← Git exclusions
│   ├── .gitattributes                ← Line-ending config
│   ├── requirements.txt               ← Python dependencies
│   └── requirements-dev.txt           ← Dev dependencies
│
├── 📦 Application Code
│   ├── app/                          ← FastAPI backend + Streamlit UI
│   │   ├── main.py                   ← FastAPI application root
│   │   ├── schemas/                  ← Pydantic models
│   │   ├── routers/                  ← API endpoints
│   │   ├── services/                 ← Business logic
│   │   ├── core/                     ← Configuration & utilities
│   │   ├── ui/                       ← Streamlit application
│   │   └── README.md                 ← Backend/frontend documentation
│   │
│   ├── pipeline/                     ← ETL orchestration
│   │   ├── run.py                    ← Main ETL entry point
│   │   ├── transformation/           ← Data processing modules
│   │   ├── supabase_handler.py       ← Database interface
│   │   └── README.md                 ← Pipeline documentation
│   │
│   ├── models/                       ← ML model artifacts
│   │   ├── saved_models/             ← Trained .pkl files (9 models (3-tier × 3-algorithm))
│   │   ├── scripts/                  ← Training scripts
│   │   └── README.md                 ← Model architecture docs
│   │
│   ├── data/                         ← Data layers
│   │   ├── raw/                      ← Source data (immutable)
│   │   ├── processed/                ← Cleaned & engineered
│   │   ├── external/                 ← Reference datasets
│   │   ├── cache/                    ← Computed cache
│   │   └── README.md                 ← Data documentation
│   │
│   ├── notebooks/                    ← Analysis & experiments
│   │   ├── 01_exploratory_data_analysis/
│   │   ├── 02_feature_engineering_analysis/
│   │   ├── 03_model_training_experiments/
│   │   ├── 04_explainability_and_xai/
│   │   ├── 05_model_validation_and_maintenance/
│   │   └── README.md                 ← Notebook guide
│   │
│   ├── tests/                        ← Unit tests (pytest)
│   ├── scripts/                      ← Utility scripts
│   └── .deployment/                  ← Docker configs
│       ├── Dockerfile                ← API container
│       ├── Dockerfile.streamlit      ← Streamlit container
│       ├── docker-compose.yml        ← Local dev setup
│       ├── start.sh                  ← Container entrypoint
│       ├── run.py                    ← Streamlit launcher
│       └── README.md                 ← Docker documentation
│
├── 📚 Project Management
│   ├── .github/
│   │   ├── ARCHITECTURE.md           ← This repo structure
│   │   ├── workflows/                ← CI/CD pipelines
│   │   └── templates/                ← PR/issue templates
│   │
│   └── docs/
│       ├── Reports/
│       │   └── Reports.md            ← This capstone document
│       └── (additional project docs)
│
└── 🔐 Environment (NOT in git)
    ├── .env                          ← Local secrets
    ├── .env.example                  ← Template
    └── .venv/                        ← Virtual environment
```

### **Key Documentation Files**

| File | Purpose | Audience |
| ----- | ----- | ----- |
| **README.md** | Quick start + architecture overview | Developers, stakeholders |
| **DEPLOYMENT.md** | Step-by-step deployment to all platforms | DevOps, deployment engineer |
| **.github/ARCHITECTURE.md** | Repository layout & file reference | Developers, maintainers |
| **app/README.md** | API/UI architecture & development | Backend developer |
| **pipeline/README.md** | ETL pipeline documentation | Data engineer |
| **data/README.md** | Data layers, features, processing | Data scientist, analyst |
| **notebooks/README.md** | Analysis workflow & notebooks | Data scientist, researcher |
| **.deployment/README.md** | Docker configuration guide | DevOps, SRE |

### **Development Quick Reference**

**Local Setup:**
```bash
# 1. Clone and setup
git clone https://github.com/Lenhan231/Real-Estate-Valuation.git
cd Real-Estate-Valuation

# 2. Create environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env with credentials

# 5. Run development
bash run.sh  # or .\run.ps1 on Windows
```

**Docker Deployment:**
```bash
# Local testing
docker-compose up

# Production (Render)
git push origin main  # Auto-deploys via render.yaml

# Self-hosted (VPS)
docker-compose up -d  # On server
docker-compose logs -f  # Monitor
```

**Add New Feature:**
```
1. Create schema (schemas/new_feature.py)
2. Create service (services/new_feature.py)
3. Create router (routers/new_feature.py)
4. Include router in main.py
5. Write tests (tests/test_new_feature.py)
6. Update documentation
```

---

## **Appendix D - Performance Benchmarks & Metrics** {#appendix-d-performance-benchmarks-metrics}

### **Final Model Performance (9-Model 3-Tier Ensemble)**

| Metric | Value | Target | Status |
| ----- | ----- | ----- | ----- |
| **Global R²** | 0.9200 | \>0.90 | ✅ Exceeded |
| **Global MAPE** | 13.10% | \<10% | ⚠️ Very close |
| **MAE** | 2.15B VND | N/A | — |
| **RMSE** | 3.41B VND | N/A | — |
| **Inference latency (avg)** | ~300ms | \<1s | ✅ Good |
| **Model size (9 pkl files)** | ~40MB | N/A | ✅ Efficient |
| **Training samples** | 10,421 | \>10,000 | ✅ Sufficient |
| **Features per model** | 79 | 50-100 | ✅ Balanced |

### **Feature Engineering Impact**

| Feature Group | Count | Contribution |
| ----- | ----- | ----- |
| Base features (area, dimensions, type) | 11 | 45% importance |
| POI geospatial features | 14 | 30% importance |
| Geographic features (distance, density) | 15 | 15% importance |
| Engineered interaction features | 38 | 10% importance |
| **Total** | **78** | **100%** |

### **Data Processing Pipeline Performance**

| Stage | Input | Output | Time | Cache Benefit |
| ----- | ----- | ----- | ----- | ----- |
| Scraping | Alonhadat.com | 11,000 listings | 30-60 min | — |
| Deduplication | 11,000 raw | 10,432 unique | <1 min | — |
| Geocoding | 10,432 | 3,202 with coords | 10-20 min | 90% reduction 2nd run |
| POI lookup | 3,202 | + 14 POI features | 15-30 min | 95% reduction 2nd run |
| Feature engineering | Raw + POI | 78 features | 5-10 min | N/A |
| **Total ETL time** | — | Model-ready data | ~60-90 min | — |

---

## **Appendix E - Deployment Platforms Comparison** {#appendix-e-deployment-platforms-comparison}

| Platform | Cost | Setup Time | Scaling | Model Hosting | Data Hosting | Status |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **Render** | ~$20/month free tier | 5 min (auto-deploy) | Horizontal (containers) | ✅ Docker | ✅ Supabase | 🟢 Current |
| **DigitalOcean** | $5-20/month droplet | 15 min (manual) | Manual (VPS scaling) | ✅ Docker | ✅ Managed DB | 🟡 Tested |
| **AWS (EC2+RDS)** | $20-100+/month | 30 min (console) | Horizontal (ASG) | ✅ Docker/ECS | ✅ RDS | 🟡 Supported |
| **Self-hosted VPS** | $10+/month | 20 min (ssh) | Manual | ✅ Docker | ✅ PostgreSQL | 🟡 Supported |
| **Streamlit Cloud** | Free tier | 5 min | Managed | ✅ Native | ✅ Supabase | 🟡 Alternative |

---

## **Appendix F - Environment Variables Reference** {#appendix-f-environment-variables-reference}

### **Required (All Environments)**

```env
# Supabase (Database)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=sbp_xxxxxxxxxxxxx

# Streamlit-specific
API_URL=https://api-domain.com  # or http://localhost:8000 for local
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
```

### **Optional (For Advanced Features)**

```env
# Weights & Biases experiment tracking
WANDB_API_KEY=xxxxxxxxxxxxx

# Debugging
DEBUG=false
LOG_LEVEL=info
ENV=production

# API Configuration
PORT=8000
HOST=0.0.0.0
```

### **How to Set Up**

```bash
# 1. Create .env from template
cp .env.example .env

# 2. Fill in actual values
nano .env  # or edit in IDE

# 3. Verify (don't commit!)
git status  # .env should NOT appear
```

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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9erV089RUr16dfX555/bfTZt2qRf75tvvmnnwlC2CCEktYLUlfJla+DAgXophcuULXP1yr2K9eWXXxaZGzdunF6aEiZefPFFfS4xbNgw9frrr+uxFB9jypQptuiI2bNnqxYtWtjtLvfxzDncuaVLl9qxyzyu+9xefvlldd1119l9WrVqpZePPfaYeuaZZ/RYyqJ45ZVX1M0332z3DaJsEUJIagWpK6XL1uWXX27HixYtKlK2zjvvPD0ePny43c8tOh999JFer1ChQkTZmjhxot3HlCj3ONGmTRs95yaMlMHp06fr8VFHHaWX++23n95fClE006ZNK/LcpGwZH3zwQejjy9U3c1zw7dQVK1boj5NE9gn+RSaEEJK8QepK6bJ177332vE777xTpGwZv/zyi35rLbgtbCxla9KkSXY+Wtm6+OKL1Zw5cyLmopG3GuWK086dOyPmp06dqs4555yIOSPsublla+bMmXbsuuWWW+w4WLZclC1CCEmtIHWldNkS5557rl7KlZxg2ZowYfcfTjMXVmLeeOONEsuWeWtRPP3006qgoEA/pvGnP/3JjoPk7UP3cVeuXGnHPXr0sGOX2f8///lPaNkSVatWteOHHnpIL+V+LlG/fn3KFiGElKMgdaV82ULpULYIISS1gtRF2fIUZYsQQlIrSF2ULU9RtgghJLWC1EXZ8hRlixBCUitIXZQtT1G2CCEktYLURdnylPvdkQAAIH4oW56ibAEAkBiULU9RtgAASAzKlqcoWwAAJAZly1PcIE8IIbENEA1ly1OULUIIiW2AaChbnqJsEUJIbANEQ9nyFGWLEEJiGyAaylYMLV68ODhVovz8fDVv3rzgdNxRtgghJLYBoqFsxdCAAQOCUyXaunWrOvXUU4PTcUfZIoSQ2AaIJiXKVq1atXQ5yMjIsHNVqlRRw4YNsz/CID09XY/3339/u4+sSypUqGDngjp16qSOP/54vV+dOnXs/HnnnWePN+QxzXMJI2XLHFNYWKjn5MqVPG+Zu+++++y+1apV03PTp0+3ZUvOf/nll9vzm3NdcMEF9riaNWvquUqVKtm5I488Uq/L/K5du1T16tX1WIpcNLI9+A8FIYSQ0geIJiXKlpGdna169Oihx1IWVqxYoce9e/e25WbZsmV2fyM3N1ft2LEjOK1JUcnJydHjyZMnq4KCAj0eN26c3adr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T1/Zch9HyFuNxqRJk+zYLWby1uOeoGyRsg4A+MLbsmXu2TrssMPU3//+d7tdSkj37t1tWTrrrLNUt27d9L1YbpmS+6LOPffciLL13HPP2Xu2Bg4cqOf2tGzJuWWbHFtYWKjnZNymTZsi92x17dpVZWZm2jm3bMnjVahQQR177LFc2SJJHQDwhXdlC7+jbJGyDgD4grJVBuTKVDCJRtkiZR0A8AVly1OULVLWAQBfULY8RdkiZR0A8AVly1OULVLWAQBfULY8Vdx3KgIAgNihbHmKsgUAQGJQtjxF2QIAIDEoW56SsvX9is2ElFkAwBeULU9xgzwp6wCALyhbnqJskbIOAPiCsuUpyhYp6wCALyhbnqJskbIOAPiCslVKCxcuDE7tM/MdghkZGYEtu8XquwgpW6SsAwC+oGyV0v333x+c2meULeJTAMAXKV+2vvzyS/Xwww+rQw89VBeInTt36vnVq1erihUrqjp16qjTTjtNz61fv141adJE1a5dWx1++OH2HM8++6yqX7++atmypbrllltUZmamPtfcuXP19gMPPFA1bNhQVahQQa/LNpMwMn/QQQep1q1bqzPOOEOdcsopql27dqpVq1Z2HzlXmzZt9PNxjxN7WrZq1Kihl1lZWfp8sr5u3TqVlpamY55/GMoWKesAgC/KRdk688wz7foBBxygl24p6du3rx3v2rVLrVy5Ul122WV2TsqW0adPHzs25wgrVcVd2XL3d8dS8lxSCB944AE1b948vb6nZeuZZ55R3bp1s3P16tWL2B42Ft27d48oisEvfoQkMgDgi3JRth555BG7bgqNlIkTTzzRRgwYMEA99thj6tNPP1XnnXeePWbGjBl2fNRRRxU57osvvtBXiqpUqWL3K03ZkqtnYuTIkeqQQw5RH374oRoyZIiaPn16xL4llS1zhc2QK1tGtMcOomyRsg4A+KJcl60gd869GuSWLfPWXJhZs2apnj176vFf/vKXwNbdohUeU7YqV65s5+Ttz70tW9nZ2RGFi7JFUjEA4ItyW7aEFAoTkZubq8e1atVSN9xwg93PLVtyX5c5pm7dunpOrmrJutyDZZi5MNEKjylbZl4yevTovS5bhilclC2SigEAX6R82ULpULZIWQcAfEHZ2kfmCpVJ9erVg7uUWvDc48aNC+5SapQtUtYBAF9QtjxF2SJlHQDwBWXL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WAQAUDLKlqfKy71nAAAkO8qWpyhbAAAkBmXLU5QtAAASg7LlKcoWAACJQdnyFDfIk+ICAIgdypanKFukuAAAYoey5SnKFikuAIDYoWx5irJFigsAIHYoW56ibJHiAgCInYSXrVh8F1xeXp5q166dHpvzzZs3T911113ubjFx9NFHB6fKxPjx4/UyFh8/QdkixQUAEDtlWrYWL16sunbtqvLz8509lLrmmmvUVVddpQoLC/X6o48+qo455hi73S1bxr6ULSlU2dnZEXP9+/cvcr5du3apLl266OdmvPzyy6pjx45FXkOQHGt+0fXYsWNVt27dIrY/8cQTqnPnzhFzo0ePVmeccYYem7LlGjZsWJFjxHvvvaduv/324HQEyhYpLgCA2CmzstW2bVs1Z84cPZbCNXPmTD1u0KCB3TcoIyNDL0u6siUlLTMz8/eDnH2CZP7DDz/U41NOOUUtWLDAzhvula2pU6fasejUqZMdm/OEcc8XNm7atKmdGzlypF5OmzbNls2333672CtbYecU7nnFb7/9plauXKlD2SLFBQAQO2VWtoKloWHDhqpnz55q9uzZEfNCSpbsb44pqWyJihUr6uVJJ52kryqFCT6Hgw46KGIpTNlyr2YZcny1atVUlSpVVOXKldXy5cuDu2jRypD7sahatao+T6VKlVRBQUGRK3dhZSs9PT3i4yKuu+46O3YLp1i0aJH69ttvdShbpLgAAGInKcrWzp07dcG5++671b///W87L9z9pIiIPSlbS5Ysidgext2Wm5urjjjiCD0+9NBD7bwpWyNGjLBzRnHndoUVLHccdp5TTz01Yj1YtuStVsM9vm/fvnYcLFsuyhYpLgCA2CmzsrVx40Y9/vTTTyPKglytGTJkiBo+fLhatWqVLkBXXHGFmjx5crFlS1SoUEGtXr3arsu2kt7ek0Iyffr0iPOElS0h+8hVIVO8pKDVrl1bLVu2rNh7pMIKljuWK28tWrTQV8Z69+5tt8tzk6tRcmUvWLY++eQTXU7l/i953QZli8QiAIDYSXjZSiS32IQpaXt5RtkixQUAEDvltmxJmXDv//rqq6+KJB5lK/gYyYqyRYoLACB2ym3ZQvEoW6S4AABih7LlKcoWKS4AgNihbHmKskWKCwAgdihbnorH/WoAAKAoypanKFsAACQGZctTlC0AABKDsuUpyhYAAIlB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>