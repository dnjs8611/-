import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import config
from indicators import calculate_indicators

def build_unified_target(df):
    """
    3-클래스 통합 타겟 생성
    라벨: 2 = UP(LONG 신호), 1 = SIDEWAYS, 0 = DOWN(SHORT 신호)
    기준: 15분봉 1봉 후 종가 기준 0.5% 이상 상승/하락
    => predict_proba 결과: [P(DOWN), P(SIDE), P(UP)] 합산 = 100%
    """
    tf_minutes = int(config.TIMEFRAME.replace('m', ''))
    shift_bars = max(1, 15 // tf_minutes)  # 15분에 해당하는 봉 수

    future_close = df['close'].shift(-shift_bars)
    up   = future_close > df['close'] * 1.005   # 0.5% 이상 상승
    down = future_close < df['close'] * 0.995   # 0.5% 이상 하락

    target = np.ones(len(df), dtype=int)  # 기본: SIDEWAYS(1)
    target[up.values]   = 2               # UP(LONG)
    target[down.values] = 0               # DOWN(SHORT)
    return target

def train_and_evaluate_gb_unified(df, symbol, features):
    df = df.copy()
    df['target_unified'] = build_unified_target(df)
    df.dropna(subset=features + ['target_unified'], inplace=True)
    X = df[features]
    y = df['target_unified'].values
    total_len = len(df)
    val_end   = int(total_len * 0.85)
    X_train_val = X.iloc[:val_end]
    y_train_val = y[:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    
    params = config.GB_PARAMS.copy()
    model = GradientBoostingClassifier(**params)
    model.fit(X_train_val, y_train_val)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc


def train_and_evaluate_regime(df, symbol, features):
    """
    지정된 데이터프레임에 대해 시장 장세(상승/하락/횡보) 모델 학습 및 평가 진행
    """
    is_bull = (df['ema20'] > df['ema60']) & (df['ema60'] > df['ema200']) & (df['ema20'] > df['ema20'].shift(1))
    is_bear = (df['ema20'] < df['ema60']) & (df['ema60'] < df['ema200']) & (df['ema20'] < df['ema20'].shift(1))
    
    target = np.ones(len(df), dtype=int)
    target[is_bull.values] = 2
    target[is_bear.values] = 0
    
    df['target_regime'] = target
    df.dropna(subset=features + ['target_regime'], inplace=True)
    
    X = df[features]
    y = df['target_regime'].values
    
    total_len = len(df)
    train_end = int(total_len * 0.70)
    val_end = int(total_len * 0.85)
    
    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return model, acc

def train_symbol_model(symbol, is_retrain=False):
    """
    특정 종목의 기본 전략(180일, 21피처)과 현재 전략(365일, 23피처) GradBoost 모델 세트를 모두 학습
    """
    sym_clean = symbol.replace('/', '_')
    csv_path = os.path.join(config.DATA_DIR, f"{sym_clean}.csv")

    if not os.path.exists(csv_path):
        print(f"[ERROR] [{symbol}] 필요한 데이터 파일이 누락되었습니다.")
        return None

    print("============================================================")
    print(f"[{symbol}] {config.TIMEFRAME} GradBoost 기본(Basic) 및 현재(Current) 모델 학습 시작")

    df_raw = pd.read_csv(csv_path)
    df = calculate_indicators(df_raw)

    # 1. 기본 전략 모델 학습 (최근 180일, FEATURES_BASIC)
    # 15분봉 기준 180일 = 180 * 24 * 4 = 17280개 봉
    df_basic = df.tail(180 * 24 * 4).copy()
    gb_basic_u, acc_gb_basic_u = train_and_evaluate_gb_unified(df_basic, symbol, config.FEATURES_BASIC)
    gb_basic_r, acc_gb_basic_r = train_and_evaluate_regime(df_basic, symbol, config.FEATURES_BASIC)

    # 2. 현재 전략 모델 학습 (최근 365일, FEATURES_CURRENT)
    # 15분봉 기준 365일 = 365 * 24 * 4 = 35040개 봉
    df_current = df.tail(365 * 24 * 4).copy()
    gb_current_u, acc_gb_current_u = train_and_evaluate_gb_unified(df_current, symbol, config.FEATURES_CURRENT)
    gb_current_r, acc_gb_current_r = train_and_evaluate_regime(df_current, symbol, config.FEATURES_CURRENT)

    print(f"  [결과 요약]")
    print(f"    - 기본 전략 통합 정확도: {acc_gb_basic_u*100:.1f}% | 장세 정확도: {acc_gb_basic_r*100:.1f}%")
    print(f"    - 현재 전략 통합 정확도: {acc_gb_current_u*100:.1f}% | 장세 정확도: {acc_gb_current_r*100:.1f}%")
 
    # 모델 파일 경로 정의
    gb_basic_unified_path  = os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_basic_unified.pkl")
    gb_basic_regime_path   = os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_basic_regime.pkl")
    gb_current_unified_path = os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_current_unified.pkl")
    gb_current_regime_path  = os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_current_regime.pkl")

    # 백업 처리 및 모델 파일 저장
    for src, model in [
        (gb_basic_unified_path, gb_basic_u),
        (gb_basic_regime_path, gb_basic_r),
        (gb_current_unified_path, gb_current_u),
        (gb_current_regime_path, gb_current_r)
    ]:
        if os.path.exists(src):
            bak = src.replace(".pkl", "_backup.pkl")
            if os.path.exists(bak):
                os.remove(bak)
            os.rename(src, bak)
        with open(src, 'wb') as f:
            pickle.dump(model, f)

    # 메타데이터 로드 및 업데이트
    metadata_file = os.path.join(config.MODELS_DIR, 'model_metadata.json')
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}

    old_avg = metadata.get(symbol, {}).get('gb_current_unified_accuracy', 0.5)

    metadata[symbol] = {
        "last_trained"                : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "gb_basic_unified_accuracy"   : float(acc_gb_basic_u),
        "gb_basic_regime_accuracy"    : float(acc_gb_basic_r),
        "gb_current_unified_accuracy" : float(acc_gb_current_u),
        "gb_current_regime_accuracy"  : float(acc_gb_current_r),
        # 하위 호환성 필드
        "gb_unified_accuracy"         : float(acc_gb_current_u),
        "ensemble_accuracy"           : float(acc_gb_current_u),
        "train_data_days"             : config.TRAIN_DAYS,
        "train_data_count"            : len(df),
        "status"                      : "healthy"
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f"[{symbol}] 기본/현재 GradBoost 모델 저장 및 업데이트 완료 [OK]")
    print("============================================================\n")

    return {
        'symbol'  : symbol,
        'old_xgb' : old_avg,
        'new_xgb' : acc_gb_current_u,
        'action'  : "SUCCESS 기본/현재 GradBoost 통합 모델 저장 및 교체"
    }

def train_all_models(is_retrain=False):
    results = []
    for symbol in config.SYMBOLS:
        res = train_symbol_model(symbol, is_retrain=is_retrain)
        if res:
            results.append(res)
    return results

if __name__ == '__main__':
    train_all_models(is_retrain=False)
