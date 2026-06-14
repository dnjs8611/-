import os
import pickle
import pandas as pd
import numpy as np
import config
from indicators import calculate_indicators

print("============================================================")
print("🔍 종목별 예측 확률 분포 분석 (테스트셋 기준)")
print("============================================================")

for symbol in config.SYMBOLS:
    sym_clean = symbol.replace('/', '_')
    csv_path = os.path.join(config.DATA_DIR, f"{sym_clean}.csv")
    xgb_l_path = os.path.join(config.MODELS_DIR, f"xgb_{sym_clean}_long.pkl")
    xgb_s_path = os.path.join(config.MODELS_DIR, f"xgb_{sym_clean}_short.pkl")
    rf_long_path = os.path.join(config.MODELS_DIR, f"rf_{sym_clean}_long.pkl")
    rf_short_path = os.path.join(config.MODELS_DIR, f"rf_{sym_clean}_short.pkl")
    
    if not os.path.exists(xgb_l_path):
        continue
        
    with open(xgb_l_path, 'rb') as f: xgb_l = pickle.load(f)
    with open(xgb_s_path, 'rb') as f: xgb_s = pickle.load(f)
    
    df_raw = pd.read_csv(csv_path)
    df = calculate_indicators(df_raw)
    
    total_len = len(df)
    val_end = int(total_len * 0.85)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    X_test = test_df[config.FEATURES]
    
    xgb_lp = xgb_l.predict_proba(X_test)[:, 1]
    xgb_sp = xgb_s.predict_proba(X_test)[:, 1]
    
    # Random Forest 예측
    with open(rf_long_path, 'rb') as f: rf_l = pickle.load(f)
    with open(rf_short_path, 'rb') as f: rf_s = pickle.load(f)
    rf_lp = rf_l.predict_proba(X_test)[:, 1]
    rf_sp = rf_s.predict_proba(X_test)[:, 1]
    
    print(f"[{symbol}]")
    print(f"  LONG 예측 확률  - Max: {xgb_lp.max()*100:.2f}%, 99%: {np.percentile(xgb_lp, 99)*100:.2f}%, Mean: {xgb_lp.mean()*100:.2f}%")
    print(f"  SHORT 예측 확률 - Max: {xgb_sp.max()*100:.2f}%, 99%: {np.percentile(xgb_sp, 99)*100:.2f}%, Mean: {xgb_sp.mean()*100:.2f}%")
    
    # 트리거 횟수 검사
    xgb_l_triggers = (xgb_lp > config.LONG_THRESHOLD).sum()
    rf_l_triggers = (rf_lp > config.LONG_THRESHOLD).sum()
    ens_l_triggers = ((xgb_lp > config.LONG_THRESHOLD) & (rf_lp > config.LONG_THRESHOLD)).sum()
    
    xgb_s_triggers = (xgb_sp > config.LONG_THRESHOLD).sum()
    rf_s_triggers = (rf_sp > config.LONG_THRESHOLD).sum()
    ens_s_triggers = ((xgb_sp > config.LONG_THRESHOLD) & (rf_sp > config.LONG_THRESHOLD)).sum()
    
    print(f"  LONG 트리거 횟수  - XGB: {xgb_l_triggers}회 | RF: {rf_l_triggers}회 | 앙상블: {ens_l_triggers}회")
    print(f"  SHORT 트리거 횟수 - XGB: {xgb_s_triggers}회 | RF: {rf_s_triggers}회 | 앙상블: {ens_s_triggers}회")
    print("------------------------------------------------------------")
