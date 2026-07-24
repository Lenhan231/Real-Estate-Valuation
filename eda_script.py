import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client, Client
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "Raw_Features")

print(f"Connecting to Supabase at {SUPABASE_URL}")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_supabase_data(table_name):
    print(f"Fetching full data from table: {table_name}")
    all_data = []
    page_size = 1000
    start = 0
    while True:
        end = start + page_size - 1
        response = supabase.table(table_name).select("*").range(start, end).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        print(f"Fetched rows {start} to {start + len(data) - 1} (Total: {len(all_data)})")
        if len(data) < page_size:
            break
        start += page_size
    return pd.DataFrame(all_data)

df_raw = fetch_supabase_data(SUPABASE_TABLE)
print(f"Supabase data loaded. Shape: {df_raw.shape}")

print("Loading processed data...")
df_processed = pd.read_csv("data/processed/model_training_data.csv")
print(f"Processed data loaded. Shape: {df_processed.shape}")

# Create output directories
os.makedirs("Reports/eda_figures", exist_ok=True)

# Generate Markdown
md_content = f"# Exploratory Data Analysis Report\n\n"
md_content += "This report presents an Exploratory Data Analysis (EDA) on the raw dataset retrieved from Supabase and the processed model training data from the local codebase.\n\n"

def generate_eda(df, title, prefix):
    md = f"## {title}\n\n"
    md += f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns\n\n"
    
    md += "### Missing Values\n\n"
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        md += "| Feature | Missing Count | Missing Percentage |\n"
        md += "|---|---|---|\n"
        for idx, val in missing.items():
            pct = (val / len(df)) * 100
            md += f"| {idx} | {val} | {pct:.2f}% |\n"
    else:
        md += "No missing values found.\n"
    md += "\n"
        
    md += "### Data Types\n\n"
    dtypes_df = df.dtypes.astype(str).value_counts().reset_index()
    dtypes_df.columns = ['Data Type', 'Count']
    md += dtypes_df.to_markdown(index=False) + "\n\n"
    
    # Save descriptive statistics
    md += "### Descriptive Statistics (Numeric Features)\n\n"
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        desc = df[numeric_cols].describe().round(2).T
        md += desc.to_markdown() + "\n\n"
    
    # Visualizations
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        # Compute correlation on numeric columns only, drop nas just for correlation
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
        plt.title(f"Correlation Matrix - {title}")
        plt.tight_layout()
        corr_path = f"Reports/eda_figures/{prefix}_correlation.png"
        plt.savefig(corr_path)
        plt.close()
        md += f"### Correlation Matrix\n\n"
        # Relative path for the markdown file in the same or parent folder
        md += f"![Correlation Matrix](eda_figures/{prefix}_correlation.png)\n\n"

    # Histograms for top 6 numeric features
    if len(numeric_cols) > 0:
        top_cols = numeric_cols[:6]
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(top_cols):
            plt.subplot(2, 3, i+1)
            # handle NaNs for plotting
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
        plt.tight_layout()
        hist_path = f"Reports/eda_figures/{prefix}_distributions.png"
        plt.savefig(hist_path)
        plt.close()
        md += f"### Feature Distributions (Sample)\n\n"
        md += f"![Distributions](eda_figures/{prefix}_distributions.png)\n\n"
        
    return md

md_content += generate_eda(df_raw, "Raw Data (Supabase)", "raw")
md_content += generate_eda(df_processed, "Processed Data (Model Training)", "processed")

with open("Reports/EDA_Report.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print("EDA completed and Reports/EDA_Report.md generated.")
