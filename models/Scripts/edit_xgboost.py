import re

with open('train_xgboost.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace header
content = content.replace('XGBoost & CatBoost Training Pipeline', 'XGBoost Training Pipeline')
content = content.replace('Trains an ensemble regressor (LGBM + CatBoost)', 'Trains an XGBoost regressor')

# Replace the training block
start_marker = 'print("[4/5] Training Ensembles (LGBM + CatBoost)...")'
end_marker = 'print(f"  All buckets trained in {time.time() - t0:.2f}s")'

training_logic = '''print("[4/5] Training XGBoost Model...")
    import time
    import joblib

    xgb_params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    print(f"  Training single XGBoost model (Train: {len(X_train)}, Test: {len(X_test)})...")
    
    model = xgb.XGBRegressor(**xgb_params)
    eval_set = [(X_test, y_log_test)] if len(X_test) > 0 else None
    
    model.fit(X_train, y_log_train, eval_set=eval_set, verbose=False)
    
    if len(X_test) > 0:
        y_log_pred_b = model.predict(X_test)
        
        global_y_log_pred = y_log_pred_b
        global_y_log_test = y_log_test.values
        
        y_pred_b = np.expm1(y_log_pred_b)
        y_pred_b = np.clip(y_pred_b, 0, None)
        
        global_y_pred = y_pred_b
        global_y_test = np.expm1(y_log_test.values)
        
    model_path = MODEL_DIR / f"xgboost_{dataset_label}_{data_source}.pkl"
    joblib.dump(model, model_path)
            
    print(f"  Model trained in {time.time() - t0:.2f}s")'''

content = content.split(start_marker)[0] + training_logic + content.split(end_marker)[1]

# Remove per-segment MAPE breakdown since seg_test uses global_y_test but it's simpler to just replace that part
# Wait, seg_test uses BINS_VND which might not be defined if I removed it.
# Let's define BINS_VND
bins_def = '''
    BINS_VND = [0, 5e9, 20e9, float('inf')]
    print("[5/5] Evaluating Globally...")
'''
content = content.replace('print("[5/5] Evaluating Globally...")', bins_def)

# Fix plot feature importance
content = content.replace('if "mid_property_type_nha_trong_hem" in models:', 'if True:')
content = content.replace('models["mid_property_type_nha_trong_hem"]', 'model')
content = content.replace('feature_importance_lgbm', 'feature_importance_xgboost')
content = content.replace('_mid_alley', '')

# Fix W&B logging
content = content.replace('ensemble-6bucket-', 'xgboost-')
content = content.replace('**lgb_params,', '**xgb_params,')

with open('train_xgboost.py', 'w', encoding='utf-8') as f:
    f.write(content)
