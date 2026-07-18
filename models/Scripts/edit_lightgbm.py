import re

with open('train_xgboost.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace header
content = content.replace('XGBoost Training Pipeline', 'LightGBM Training Pipeline')
content = content.replace('Trains an XGBoost regressor', 'Trains a LightGBM regressor')

# Replace import
content = content.replace('import xgboost as xgb', 'from lightgbm import LGBMRegressor')

# Replace the training block
content = content.replace('Training XGBoost Model...', 'Training LightGBM Model...')
content = content.replace('Training single XGBoost model', 'Training single LightGBM model')

content = content.replace('xgb_params = {', 'lgb_params = {')
content = content.replace('**xgb_params,', '**lgb_params,')

content = content.replace('xgb.XGBRegressor(**xgb_params)', 'LGBMRegressor(**lgb_params)')
content = content.replace('model.fit(X_train, y_log_train, eval_set=eval_set, verbose=False)', 'model.fit(X_train, y_log_train, eval_set=eval_set, callbacks=[])')

content = content.replace('xgboost_', 'lgbm_')
content = content.replace('xgboost-', 'lgbm-')

with open('train_lightgbm.py', 'w', encoding='utf-8') as f:
    f.write(content)
