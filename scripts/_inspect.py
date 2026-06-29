import pandas as pd
df = pd.read_csv('data/processed/alonhadat_features_cleaned_optB.csv')
print('Categorical value counts:')
for col in ['direction','listing_type','property_type','legal_status']:
    print(f'\n{col}:')
    print(df[col].value_counts())
print('\nlocality_square sample:', df['locality_square'].head(10).tolist())
