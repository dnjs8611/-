import os
import sys
import time
import json
import pickle
import warnings
import requests
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

warnings.filterwarnings('ignore')

# Import project configurations and modules
import config
import notifier
from indicators import calculate_indicators

# File and Directory Settings
MODELS_15M_DIR = os.path.join(config.MODELS_DIR, '15m')
os.makedirs(MODELS_15M_DIR, exist_ok=True)
PREDICTIONS_FILE = os.path.join(config.DATA_DIR, 'predictions_15m.json')

# Lock for thread-safe file operations on predictions
predictions_lock = threading.Lock()
last_loss_streak_trigger_time = 0

def load_predictions() -> list:
    import data_manager
    with predictions_lock:
        return data_manager.db_load_predictions()

def save_predictions(preds: list):
    import data_manager
    with predictions_lock:
        try:
            for p in preds:
                data_manager.db_save_prediction(p)
        except Exception as e:
            print(f"[15mBot] Error saving predictions to SQLite: {e}")

# CCXT Binance Setup
def get_ccxt_exchange():
    import ccxt
    return ccxt.binanceusdm({
        'apiKey': config.API_KEY,
        'secret': config.API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }
    })

# Fetch historical public futures data from Binance (similar to 1_collect_data.py)
def fetch_futures_data_hist(endpoint, symbol, period='5m', days=90):
    url = f"https://fapi.binance.com{endpoint}"
    now_ms = int(time.time() * 1000)
    # Cap lookback to 28 days to avoid Binance API 400 error (parameter 'startTime' is invalid)
    fetch_days = min(days, 28)
    start_ms = now_ms - (fetch_days * 24 * 60 * 60 * 1000)
    
    all_data = []
    until = now_ms
    limit = 500
    
    while until > start_ms:
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit,
            'endTime': until
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code != 200:
                break
            data = res.json()
            if not data:
                break
            
            # Sort ascending by timestamp to ensure oldest is first
            data.sort(key=lambda x: x['timestamp'])
            all_data.extend(data)
            
            oldest_timestamp = data[0]['timestamp']
            if oldest_timestamp >= until:
                break
            until = oldest_timestamp - 1
            time.sleep(0.1)
        except Exception as e:
            print(f"  [API Exception] {endpoint}: {e}. Retrying...")
            time.sleep(2)
            
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    return df

def fetch_funding_rate_hist(symbol, days=90):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    params = {
        'symbol': symbol,
        'startTime': start_ms,
        'limit': 1000
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data:
                df = pd.DataFrame(data)
                df.drop_duplicates(subset=['fundingTime'], inplace=True)
                df.sort_values(by='fundingTime', inplace=True)
                return df
    except Exception as e:
        print(f"  [API Exception] fundingRate: {e}")
    return pd.DataFrame()

def collect_symbol_data_15m(exchange, symbol, days=90):
    """
    Collects 15-minute candles and merges them with 5-minute sentiment indicators.
    """
    symbol_clean = symbol.replace('/', '_')
    print(f"[{symbol}] 15분봉 데이터 수집 시작 (기간: 최근 {days}일)")
    
    limit = 1000
    interval_ms = 15 * 60 * 1000
    
    now_ms = exchange.milliseconds()
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    since = start_ms
    
    while since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '15m', since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + interval_ms
            time.sleep(exchange.rateLimit / 1000)
            if len(ohlcv) < limit:
                break
        except Exception as e:
            print(f"  [Warning] Candle fetch error: {e}. Waiting 5s...")
            time.sleep(5)
            
    if not all_ohlcv:
        print(f"[Error] [{symbol}] 15분봉 수집 실패")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    binance_symbol = symbol.replace('/', '')
    
    # Fetch 5m interval sentiment features
    df_oi = fetch_futures_data_hist('/futures/data/openInterestHist', binance_symbol, period='5m', days=days)
    if not df_oi.empty:
        df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
        df_oi = df_oi[['timestamp', 'sumOpenInterest']].rename(columns={'sumOpenInterest': 'open_interest'})
        df = pd.merge_asof(df, df_oi, on='timestamp', direction='backward')
    else:
        df['open_interest'] = np.nan

    df_ls = fetch_futures_data_hist('/futures/data/topLongShortPositionRatio', binance_symbol, period='5m', days=days)
    if not df_ls.empty:
        df_ls['longShortRatio'] = df_ls['longShortRatio'].astype(float)
        df_ls = df_ls[['timestamp', 'longShortRatio']].rename(columns={'longShortRatio': 'long_short_ratio'})
        df = pd.merge_asof(df, df_ls, on='timestamp', direction='backward')
    else:
        df['long_short_ratio'] = np.nan

    df_fr = fetch_funding_rate_hist(binance_symbol, days=days)
    if not df_fr.empty:
        df_fr['fundingRate'] = df_fr['fundingRate'].astype(float)
        df_fr['timestamp'] = df_fr['fundingTime'].astype(int)
        df_fr = df_fr[['timestamp', 'fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
        df.sort_values(by='timestamp', inplace=True)
        df_fr.sort_values(by='timestamp', inplace=True)
        df = pd.merge_asof(df, df_fr, on='timestamp', direction='backward')
    else:
        df['funding_rate'] = np.nan

    df['open_interest'] = df['open_interest'].ffill().bfill().fillna(0.0)
    df['long_short_ratio'] = df['long_short_ratio'].ffill().bfill().fillna(1.0)
    df['funding_rate'] = df['funding_rate'].ffill().bfill().fillna(0.0)
    
    csv_path = os.path.join(config.DATA_DIR, f"{symbol_clean}_15m.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] [{symbol}] 15분봉 {len(df)}개 저장 완료: {csv_path}")
    return df

def update_model_metadata(symbol, accuracies: dict):
    metadata_file = os.path.join(config.MODELS_DIR, 'model_metadata.json')
    try:
        meta = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        
        if symbol not in meta:
            meta[symbol] = {}
            
        update_data = {
            "last_trained": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "xgb_long_accuracy": accuracies.get('xgb', 0.50),
            "xgb_short_accuracy": accuracies.get('rf', 0.50),
            "xgb_accuracy": accuracies.get('ens', 0.50),
            "status": "healthy"
        }
        for m, val in accuracies.items():
            update_data[f"{m}_accuracy_val"] = val
            
        meta[symbol].update(update_data)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4)
        print(f"[{symbol}] Updated model_metadata.json with validation accuracies.")
    except Exception as e:
        print(f"[15mBot] Error updating model_metadata.json: {e}")

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
    acc = np.mean(preds == y_test)
    
    return model, acc

def train_15m_model(symbol):
    """
    Trains GradBoost model and MLP model separately (both 180d, FEATURES_BASIC).
    Each model predicts independently; results shown as separate cards on dashboard.
    """
    symbol_clean = symbol.replace('/', '_')
    csv_path = os.path.join(config.DATA_DIR, f"{symbol_clean}_15m.csv")
    
    if not os.path.exists(csv_path):
        print(f"[Train] [{symbol}] Data missing.")
        return False
        
    df_raw = pd.read_csv(csv_path)
    if len(df_raw) < 300:
        print(f"[Train] [{symbol}] Insufficient data ({len(df_raw)} rows).")
        return False
        
    df = calculate_indicators(df_raw)
    df['target_15m'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 공통 데이터 준비 (최근 180일, FEATURES_BASIC 21개)
    df_train = df.tail(180 * 24 * 4).copy()
    df_train.dropna(subset=config.FEATURES_BASIC + ['target_15m'], inplace=True)
    X_all = df_train[config.FEATURES_BASIC]
    y_all = df_train['target_15m'].values
    train_end = int(len(df_train) * 0.80)
    X_train_raw = X_all.iloc[:train_end]
    X_val_raw   = X_all.iloc[train_end:]
    y_train     = y_all[:train_end]
    y_val       = y_all[train_end:]
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_val_sc   = scaler.transform(X_val_raw)
    
    # 2일 전과 동일하게 GradBoost는 clipping이 없는 오리지널 스케일링 데이터를 사용
    X_train_gb = pd.DataFrame(X_train_sc, columns=config.FEATURES_BASIC, index=X_train_raw.index)
    X_val_gb   = pd.DataFrame(X_val_sc,   columns=config.FEATURES_BASIC, index=X_val_raw.index)
    
    # MLP는 확률 포화(0%, 100% 쏠림) 방지를 위해 clipping이 적용된 데이터를 사용
    X_train_sc_mlp = np.clip(X_train_sc, -5.0, 5.0)
    X_val_sc_mlp   = np.clip(X_val_sc, -5.0, 5.0)
    X_train_mlp = pd.DataFrame(X_train_sc_mlp, columns=config.FEATURES_BASIC, index=X_train_raw.index)
    X_val_mlp   = pd.DataFrame(X_val_sc_mlp,   columns=config.FEATURES_BASIC, index=X_val_raw.index)
    
    scaler_path = os.path.join(MODELS_15M_DIR, f"scaler_{symbol_clean}_gb.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # 1. GradBoost 모델 학습
    gb_model = GradientBoostingClassifier(**config.GB_PARAMS)
    gb_model.fit(X_train_gb, y_train)
    acc_gb = np.mean(gb_model.predict(X_val_gb) == y_val)
    gb_regime, acc_regime = train_and_evaluate_regime(df_train.copy(), symbol, config.FEATURES_BASIC)
    
    with open(os.path.join(MODELS_15M_DIR, f"gb_{symbol_clean}_gb.pkl"), 'wb') as f: pickle.dump(gb_model, f)
    with open(os.path.join(MODELS_15M_DIR, f"gb_{symbol_clean}_regime.pkl"), 'wb') as f: pickle.dump(gb_regime, f)
    
    # 2. MLP 모델 학습 (같은 데이터, 다른 모델)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=150,
        alpha=2.0,  # L2 규제 강화를 통해 확률 포화(0%, 100% 쏠림) 방지
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_model.fit(X_train_mlp, y_train)
    acc_mlp = np.mean(mlp_model.predict(X_val_mlp) == y_val)
    
    with open(os.path.join(MODELS_15M_DIR, f"gb_{symbol_clean}_mlp.pkl"), 'wb') as f: pickle.dump(mlp_model, f)
    
    print(f"[{symbol}] 15m Models Trained. GB: {acc_gb*100:.1f}% | MLP: {acc_mlp*100:.1f}% | Regime: {acc_regime*100:.1f}%")
    
    update_model_metadata(symbol, {
        'gb': acc_gb,
        'mlp': acc_mlp,
        'regime': acc_regime,
        'xgb': 0.5, 'rf': 0.5, 'lgb': 0.5, 'cat': 0.5, 'et': 0.5, 'svm': 0.5, 'ens': (acc_gb + acc_mlp) / 2
    })
    
    return True

def load_15m_models(symbol):
    symbol_clean = symbol.replace('/', '_')
    gb_path     = os.path.join(MODELS_15M_DIR, f"gb_{symbol_clean}_gb.pkl")
    regime_path = os.path.join(MODELS_15M_DIR, f"gb_{symbol_clean}_regime.pkl")
    mlp_path    = os.path.join(MODELS_15M_DIR, f"gb_{symbol_clean}_mlp.pkl")
    
    if all(os.path.exists(p) for p in [gb_path, regime_path, mlp_path]):
        try:
            with open(gb_path, 'rb') as f:     gb_model  = pickle.load(f)
            with open(regime_path, 'rb') as f: gb_regime = pickle.load(f)
            with open(mlp_path, 'rb') as f:    mlp_model = pickle.load(f)
            return gb_model, gb_regime, mlp_model
        except Exception as e:
            print(f"[15mBot] Error loading models for {symbol}: {e}")
    return None

def fetch_realtime_features_15m(exchange, symbol):
    """
    Fetches real-time OI, Long/Short ratio, and Funding rate from API.
    """
    binance_symbol = symbol.replace('/', '')
    funding_rate = 0.0
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        res = requests.get(url, params={'symbol': binance_symbol}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict):
                funding_rate = float(data.get('lastFundingRate', 0.0))
            elif isinstance(data, list) and len(data) > 0:
                funding_rate = float(data[0].get('lastFundingRate', 0.0))
    except Exception:
        pass
        
    open_interest = 0.0
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        res = requests.get(url, params={'symbol': binance_symbol}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            open_interest = float(data.get('openInterest', 0.0))
    except Exception:
        pass
        
    long_short_ratio = 1.0
    try:
        url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
        res = requests.get(url, params={'symbol': binance_symbol, 'period': '5m', 'limit': 1}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0:
                long_short_ratio = float(data[0].get('longShortRatio', 1.0))
    except Exception:
        pass
        
    return {
        'funding_rate': funding_rate,
        'open_interest': open_interest,
        'long_short_ratio': long_short_ratio
    }

def retry_api(func, *args, retries=3, delay=2, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(delay)
def get_current_capital(preds, start_capital=100000.0):
    total_pnl = sum(float(p.get('pnl_krw') or 0.0) for p in preds if p.get('status') == 'COMPLETED')
    return start_capital + total_pnl

def update_model_regime_metadata(symbol, regime):
    metadata_file = os.path.join(config.MODELS_DIR, 'model_metadata.json')
    try:
        meta = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        if symbol not in meta:
            meta[symbol] = {}
        meta[symbol]['regime'] = regime
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4)
    except Exception as e:
        print(f"[15mBot] Error updating metadata regime for {symbol}: {e}")

def execute_predictions_for_cycle(exchange, current_cycle_time):
    """
    Runs prediction for all symbols for the recently closed candle.
    """
    print(f"\n[15mBot] === 예측 사이클 시작: {current_cycle_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    preds = load_predictions()
    
    for symbol in config.SYMBOLS:
        symbol_clean = symbol.replace('/', '_')
        models_tuple = load_15m_models(symbol)
        if models_tuple is None:
            print(f"[15mBot] [{symbol}] 학습된 모델이 없어 예측을 스킵합니다.")
            continue
        gb_model, gb_regime, mlp_model = models_tuple
            
        try:
            # Fetch latest 300 candles (15m)
            ohlcv = retry_api(exchange.fetch_ohlcv, symbol, '15m', limit=300)
            df_raw = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = calculate_indicators(df_raw)
            
            # Fetch live sentiment and merge into the dataframe
            rt = fetch_realtime_features_15m(exchange, symbol)
            df['funding_rate'] = rt['funding_rate']
            df['open_interest'] = rt['open_interest']
            df['long_short_ratio'] = rt['long_short_ratio']
            
            # Ensure the row we predict on is the recently completed 15m candle.
            candle_time = int(df['timestamp'].iloc[-2])
            candle_dt = datetime.fromtimestamp(candle_time / 1000)
            entry_price = float(df['close'].iloc[-2])

            # Load scaler and transform features
            X_raw = df[config.FEATURES_BASIC].iloc[-2:-1]
            scaler_path = os.path.join(MODELS_15M_DIR, f"scaler_{symbol_clean}_gb.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_scaled = scaler.transform(X_raw)
                
                # GradBoost와 regime 모델은 2일 전과 똑같이 clip이 없는 원본 스케일링 데이터 사용
                X_gb = pd.DataFrame(X_scaled, columns=config.FEATURES_BASIC, index=X_raw.index)
                
                # MLP 모델은 확률 포화 방지를 위해 clip이 적용된 데이터 사용
                X_scaled_mlp = np.clip(X_scaled, -5.0, 5.0)
                X_mlp = pd.DataFrame(X_scaled_mlp, columns=config.FEATURES_BASIC, index=X_raw.index)
            else:
                X_gb = X_raw
                X_mlp = X_raw

            # Predict market regime (GradBoost 기반이므로 X_gb 사용)
            regime_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
            regime_probs = gb_regime.predict_proba(X_gb)[0]
            current_regime = regime_map[int(np.argmax(regime_probs))]
            update_model_regime_metadata(symbol, current_regime)

            # --- GradBoost 예측 ---
            gb_prob = float(gb_model.predict_proba(X_gb)[0][1])
            gb_side = 'PASS'
            if current_regime == 'BULL':
                if gb_prob >= 0.75:
                    gb_side = 'LONG'
            elif current_regime == 'BEAR':
                if gb_prob <= 0.25:
                    gb_side = 'SHORT'
            else:  # SIDEWAYS
                if gb_prob >= 0.75:
                    gb_side = 'LONG'
                elif gb_prob <= 0.25:
                    gb_side = 'SHORT'

            # --- MLP 예측 ---
            mlp_prob = float(mlp_model.predict_proba(X_mlp)[0][1])
            mlp_side = 'PASS'
            if current_regime == 'BULL':
                if mlp_prob >= 0.75:
                    mlp_side = 'LONG'
            elif current_regime == 'BEAR':
                if mlp_prob <= 0.25:
                    mlp_side = 'SHORT'
            else:  # SIDEWAYS
                if mlp_prob >= 0.75:
                    mlp_side = 'LONG'
                elif mlp_prob <= 0.25:
                    mlp_side = 'SHORT'

            # Check duplicate entry
            duplicate = any(p['symbol'] == symbol and p['timestamp'] == candle_time for p in preds)
            if duplicate:
                continue

            print(f"  [{symbol}] GB: {gb_side} ({gb_prob*100:.1f}%) | MLP: {mlp_side} ({mlp_prob*100:.1f}%) | 장세: {current_regime}")

            # If both PASS, skip saving
            if gb_side == 'PASS' and mlp_side == 'PASS':
                continue

            # Ensemble side: majority or GB fallback
            if gb_side != 'PASS' and mlp_side != 'PASS':
                ensemble_side = gb_side if gb_side == mlp_side else gb_side
            elif gb_side != 'PASS':
                ensemble_side = gb_side
            else:
                ensemble_side = mlp_side

            # Max positions check
            active_preds = [p for p in preds if p['status'] == 'PENDING']
            active_count = len(active_preds)
            has_active = any(p['symbol'] == symbol for p in active_preds)
            if not has_active and active_count >= config.MAX_POSITIONS:
                print(f"  [{symbol}] 진입 패스 - 포지션 한도 초과 ({active_count}/{config.MAX_POSITIONS})")
                continue

            # Capital allocation
            current_capital = get_current_capital(preds, start_capital=100000.0)
            allocation_ratio = 0.90 / config.MAX_POSITIONS
            margin_krw = current_capital * allocation_ratio
            exchange_rate = 1350.0
            margin_usdt = margin_krw / exchange_rate

            target_time = candle_time + 15 * 60 * 1000
            target_time_str = datetime.fromtimestamp(target_time / 1000).strftime('%Y-%m-%d %H:%M:%S')

            pred_item = {
                'symbol': symbol,
                'predict_time': candle_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': candle_time,
                'entry_price': entry_price,

                # Ensemble/main side
                'predicted_side': ensemble_side,
                'predicted_regime': current_regime,

                # GradBoost fields
                'gb_predicted_side': gb_side,
                'gb_prob': gb_prob,
                'gb_basic_predicted_side': gb_side,
                'gb_basic_prob': gb_prob,
                'gb_basic_result': 'PENDING' if gb_side != 'PASS' else 'PASS',
                'gb_basic_pnl_usdt': 0.0,
                'gb_basic_pnl_krw': 0.0,

                # MLP fields
                'mlp_predicted_side': mlp_side,
                'mlp_prob': mlp_prob,
                'gb_current_predicted_side': mlp_side,
                'gb_current_prob': mlp_prob,
                'gb_current_result': 'PENDING' if mlp_side != 'PASS' else 'PASS',
                'gb_current_pnl_usdt': 0.0,
                'gb_current_pnl_krw': 0.0,

                # Unused model placeholders
                'xgb_predicted_side': 'PASS', 'xgb_prob': 0.5,
                'rf_predicted_side':  'PASS', 'rf_prob':  0.5,
                'lgb_predicted_side': 'PASS', 'lgb_prob': 0.5,
                'cat_predicted_side': 'PASS', 'cat_prob': 0.5,
                'et_predicted_side':  'PASS', 'et_prob':  0.5,
                'svm_predicted_side': 'PASS', 'svm_prob': 0.5,
                'ensemble_prob': (gb_prob + mlp_prob) / 2,

                'entry_margin_krw': margin_krw,
                'entry_margin_usdt': margin_usdt,
                'target_time': target_time,
                'target_time_str': target_time_str,
                'status': 'PENDING',
                'actual_price': None,
                'result': 'PENDING'
            }

            preds.append(pred_item)
            print(f"  [{symbol}] 진입 | GB: {gb_side} ({gb_prob*100:.1f}%) | MLP: {mlp_side} ({mlp_prob*100:.1f}%) | 기준가: ${entry_price:,.4f}")
            
        except Exception as e:
            print(f"  [Error] [{symbol}] 예측 중 에러 발생: {e}")
            
    save_predictions(preds)



def calculate_profit_lock_return(pred_side, entry_price, high_price, low_price, close_price, sl_pct):
    # sl_pct = 0.010 (1.0% Stop Loss)
    # Define 7 stages of Profit Lock
    # (Trigger price change, Lock price change)
    stages = [
        (0.0020, 0.0010), # Stage 1
        (0.0030, 0.0015), # Stage 2
        (0.0040, 0.0020), # Stage 3
        (0.0050, 0.0025), # Stage 4
        (0.0060, 0.0030), # Stage 5
        (0.0080, 0.0040), # Stage 6
        (0.0100, 0.0050)  # Stage 7
    ]
    
    if pred_side == 'LONG':
        max_move = (high_price - entry_price) / entry_price
        max_loss = (entry_price - low_price) / entry_price
        
        locked_ret = None
        stage_num = 0
        for idx, (trigger, lock) in enumerate(stages, 1):
            if max_move >= trigger:
                locked_ret = lock
                stage_num = idx
                
        if max_loss >= sl_pct:
            if locked_ret is not None:
                return True, locked_ret, f"PROFIT_LOCK_STAGE_{stage_num}"
            else:
                return False, -sl_pct, "STOP_LOSS"
        else:
            if locked_ret is not None:
                close_ret = (close_price - entry_price) / entry_price
                final_ret = max(locked_ret, close_ret)
                exit_type = f"PROFIT_LOCK_STAGE_{stage_num}" if final_ret == locked_ret else "CANDLE_CLOSE"
                return True, final_ret, exit_type
            else:
                close_ret = (close_price - entry_price) / entry_price
                return close_ret > 0, close_ret, "CANDLE_CLOSE"
                
    elif pred_side == 'SHORT':
        max_move = (entry_price - low_price) / entry_price
        max_loss = (high_price - entry_price) / entry_price
        
        locked_ret = None
        stage_num = 0
        for idx, (trigger, lock) in enumerate(stages, 1):
            if max_move >= trigger:
                locked_ret = lock
                stage_num = idx
                
        if max_loss >= sl_pct:
            if locked_ret is not None:
                return True, locked_ret, f"PROFIT_LOCK_STAGE_{stage_num}"
            else:
                return False, -sl_pct, "STOP_LOSS"
        else:
            if locked_ret is not None:
                close_ret = (entry_price - close_price) / entry_price
                final_ret = max(locked_ret, close_ret)
                exit_type = f"PROFIT_LOCK_STAGE_{stage_num}" if final_ret == locked_ret else "CANDLE_CLOSE"
                return True, final_ret, exit_type
            else:
                close_ret = (entry_price - close_price) / entry_price
                return close_ret > 0, close_ret, "CANDLE_CLOSE"
    return False, 0.0, "PASS"
def evaluate_pending_predictions(exchange):
    """
    Checks completed 15m candles to resolve pending predictions and calculate simulated profits.
    """
    preds = load_predictions()
    pending = [p for p in preds if p['status'] == 'PENDING']
    
    if not pending:
        return
        
    updated = False
    
    # Group pending predictions by symbol to minimize CCXT calls
    from collections import defaultdict
    pending_by_symbol = defaultdict(list)
    for p in pending:
        pending_by_symbol[p['symbol']].append(p)
        
    for symbol, sym_preds in pending_by_symbol.items():
        try:
            # Fetch latest 15m candles
            ohlcv = retry_api(exchange.fetch_ohlcv, symbol, '15m', limit=50)
            # Exclude the last (incomplete/active) candle to avoid premature evaluation
            # candles_map format: timestamp -> {'high': high, 'low': low, 'close': close}
            candles_map = {
                int(c[0]): {
                    'high': float(c[2]),
                    'low': float(c[3]),
                    'close': float(c[4])
                } for c in ohlcv[:-1]
            }
            
            for p in sym_preds:
                target_ts = p['target_time']
                
                # Check if the target candle exists
                if target_ts in candles_map:
                    candle_data = candles_map[target_ts]
                    target_price = candle_data['close']
                    high_price = candle_data['high']
                    low_price = candle_data['low']
                    entry_price = p['entry_price']
                    
                    # Fetch dynamic margin saved in the prediction item
                    # default fallback: margin 30,000 KRW (30% of 100k capital), 5x leverage
                    margin_krw = p.get('entry_margin_krw', 30000.0)
                    exchange_rate = 1350.0
                    margin_usdt = p.get('entry_margin_usdt', margin_krw / exchange_rate)
                    
                    # Use individual stop loss from config
                    sl_pct = config.STOP_LOSS.get(symbol, 0.008)
                    
                    main_side = p['predicted_side']
                    
                    # Evaluate SL/TP for the main trade
                    win, ret, exit_type = calculate_profit_lock_return(
                        main_side, entry_price, high_price, low_price, target_price, sl_pct
                    )
                    
                    is_closed = exit_type in ["STOP_LOSS"] or exit_type.startswith("PROFIT_LOCK_STAGE")
                    
                    # Check if there is a new prediction for the same symbol to decide rollover/reversal
                    p2 = None
                    if not is_closed:
                        # Find the new prediction in the current cycle
                        for other in preds:
                            if other != p and other['symbol'] == symbol and other['timestamp'] == target_ts and other['status'] == 'PENDING':
                                p2 = other
                                break
                                
                    if not is_closed and p2 is not None and p2['predicted_side'] == main_side:
                        # Rollover! Extend the target time and keep PENDING
                        p['target_time'] = p2['target_time']
                        p['target_time_str'] = p2['target_time_str']
                        # Remove the duplicate new prediction so it's not run as a separate trade
                        if p2 in preds:
                            preds.remove(p2)
                        import data_manager
                        data_manager.db_delete_prediction(symbol, p2['timestamp'])
                        print(f"  [포지션연장] [{symbol}] 신호 유지 ➔ {main_side} 포지션 연장 (기준가: ${entry_price:,.4f})")
                        updated = True
                        continue
                        
                    # Otherwise, the position is closed (either by SL/TP, reversal, or PASS)
                    p['actual_price'] = target_price
                    p['status'] = 'COMPLETED'
                    
                    # Compute PnL for all models
                    for m in ['ensemble', 'xgb', 'rf', 'lgb', 'cat', 'et', 'gb', 'mlp', 'svm']:
                        side_k = 'predicted_side' if m == 'ensemble' else f'{m}_predicted_side'
                        
                        # Set up list of result and PnL keys to update for both standard and dashboard-compatible fields
                        res_keys = []
                        pnl_usdt_keys = []
                        pnl_krw_keys = []
                        
                        if m == 'ensemble':
                            res_keys = ['result']
                            pnl_usdt_keys = ['pnl_usdt']
                            pnl_krw_keys = ['pnl_krw']
                        elif m == 'gb':
                            res_keys = ['gb_result', 'gb_basic_result']
                            pnl_usdt_keys = ['gb_pnl_usdt', 'gb_basic_pnl_usdt']
                            pnl_krw_keys = ['gb_pnl_krw', 'gb_basic_pnl_krw']
                        elif m == 'mlp':
                            res_keys = ['mlp_result', 'gb_current_result']
                            pnl_usdt_keys = ['mlp_pnl_usdt', 'gb_current_pnl_usdt']
                            pnl_krw_keys = ['mlp_pnl_krw', 'gb_current_pnl_krw']
                        else:
                            res_keys = [f'{m}_result']
                            pnl_usdt_keys = [f'{m}_pnl_usdt']
                            pnl_krw_keys = [f'{m}_pnl_krw']
                            
                        pred_side = p.get(side_k, 'LONG')
                        
                        if pred_side == 'PASS':
                            for k in res_keys: p[k] = 'PASS'
                            for k in pnl_usdt_keys: p[k] = 0.0
                            for k in pnl_krw_keys: p[k] = 0.0
                            net_ret = 0.0
                        else:
                            # Evaluate SL and Profit Lock
                            m_win, m_ret, m_exit_type = calculate_profit_lock_return(
                                pred_side, entry_price, high_price, low_price, target_price, sl_pct
                            )
                            
                            net_ret = m_ret - 0.0008  # 0.08% fee
                            pnl_usdt = margin_usdt * net_ret * 5  # 5x leverage
                            pnl_krw = margin_krw * net_ret * 5
                            
                            for k in res_keys: p[k] = 'WIN' if m_win else 'LOSS'
                            for k in pnl_usdt_keys: p[k] = pnl_usdt
                            for k in pnl_krw_keys: p[k] = pnl_krw
                            
                        if m == 'ensemble':
                            p['net_pnl_pct'] = net_ret
                            
                    # Print log
                    print(f"  [결과확정] [{symbol}] | Ensemble: {p['predicted_side']} ({p['result']}) | 기준가: ${entry_price:,.4f} -> 결과가: ${target_price:,.4f}")
                    updated = True
                else:
                    p = None # placeholder
        except Exception as e:
            print(f"  [Error] [{symbol}] 결과 채점 중 에러 발생: {e}")
            
    if updated:
        save_predictions(preds)
        
        global last_loss_streak_trigger_time
        completed_all = [p for p in preds if p.get('status') == 'COMPLETED']
        completed_all.sort(key=lambda x: x.get('target_time', 0))
        if len(completed_all) >= 3:
            recent_3 = completed_all[-3:]
            latest_trade_time = recent_3[-1].get('target_time', 0)
            if all(x.get('result') == 'LOSS' for x in recent_3) and latest_trade_time > last_loss_streak_trigger_time:
                last_loss_streak_trigger_time = latest_trade_time
                print("[15mBot] 3 consecutive losses detected! Triggering immediate emergency retraining...")
                notifier.send_telegram_message("⚠️ <b>[ALERT] 3회 연속 손절 발생!</b>\n\n최근 3개 거래가 연속으로 손절되었습니다. 시장 장세 급변에 대응하기 위해 전체 모델 즉시 재학습을 백그라운드에서 개시합니다.")
                
                def run_emergency_retrain():
                    print("[15mBot] Emergency retraining started...")
                    for symbol in config.SYMBOLS:
                        try:
                            collect_symbol_data_15m(exchange, symbol, days=180)
                            train_15m_model(symbol)
                        except Exception as e:
                            print(f"  [Error] Emergency retraining failed for {symbol}: {e}")
                    print("[15mBot] Emergency retraining completed successfully.")
                    notifier.send_telegram_message("✅ <b>[ALERT] 3연패 대응 긴급 재학습 완료!</b>\n\n전체 모델의 갱신이 완료되었습니다.")
                
                threading.Thread(target=run_emergency_retrain, daemon=True).start()

def generate_report_and_retrain(exchange, start_time, end_time):
    """
    Generates a 1-hour summary report with financial stats, sends it to Telegram, and retrains the models.
    """
    print(f"\n[15mBot] === 정각 통계 정산 및 재학습 시작 ===")
    
    preds = load_predictions()
    
    # Filter predictions that matured during this period
    completed_in_period = []
    for p in preds:
        if p['status'] == 'COMPLETED':
            try:
                target_dt = datetime.strptime(p['target_time_str'], '%Y-%m-%d %H:%M:%S')
                if start_time <= target_dt < end_time:
                    completed_in_period.append(p)
            except Exception:
                pass
                
    # Period stats
    period_count = len(completed_in_period)
    
    # Calculate stats for all models (including Ensemble)
    completed_all = [p for p in preds if p['status'] == 'COMPLETED']
    total_count = len(completed_all)
    
    model_names = {
        'gb_basic': 'GradBoost 기본',
        'gb_current': 'GradBoost 현재'
    }
    
    stats = {}
    for m in ['gb_basic', 'gb_current']:
        res_k = f'{m}_result'
        pnl_k = f'{m}_pnl_krw'
        pnl_usdt_k = f'{m}_pnl_usdt'
        
        p_trades = sum(1 for p in completed_in_period if p.get(res_k) in ['WIN', 'LOSS'])
        p_wins = sum(1 for p in completed_in_period if p.get(res_k) == 'WIN')
        p_wr = (p_wins / p_trades) if p_trades > 0 else 0.0
        p_pnl = sum(float(p.get(pnl_k) or 0.0) for p in completed_in_period)
        
        c_trades = sum(1 for p in completed_all if p.get(res_k) in ['WIN', 'LOSS'])
        c_wins = sum(1 for p in completed_all if p.get(res_k) == 'WIN')
        c_wr = (c_wins / c_trades) if c_trades > 0 else 0.0
        c_pnl = sum(float(p.get(pnl_k) or 0.0) for p in completed_all)
        
        stats[m] = {
            'period_trades': p_trades,
            'period_wins': p_wins,
            'period_wr': p_wr,
            'period_pnl': p_pnl,
            'cum_trades': c_trades,
            'cum_wins': c_wins,
            'cum_wr': c_wr,
            'cum_pnl': c_pnl
        }
        
    pnl_sign = lambda x: "+" if x >= 0 else ""
    
    msg_lines = [
        f"<b>📊 [15분봉 예측 1시간 정각 리포트]</b>",
        f"  기간: {start_time.strftime('%H:%M')} ~ {end_time.strftime('%H:%M')}\n",
        f"<b>⏱ 최근 1시간 성과 (총 {period_count}회 예측):</b>"
    ]
    
    for m in ['gb_basic', 'gb_current']:
        st = stats[m]
        msg_lines.append(
            f"  - <b>{model_names[m]}</b>: {st['period_wr']*100:.1f}% ({st['period_wins']}승/{st['period_trades']}회) | {pnl_sign(st['period_pnl'])}{st['period_pnl']:,.0f}원"
        )
        
    msg_lines.append(f"\n<b>📈 누적 시뮬레이션 성과 (총 {total_count}회 예측):</b>")
    for m in ['gb_basic', 'gb_current']:
        st = stats[m]
        msg_lines.append(
            f"  - <b>{model_names[m]}</b>: {st['cum_wr']*100:.1f}% ({st['cum_wins']}승/{st['cum_trades']}회) | {pnl_sign(st['cum_pnl'])}{st['cum_pnl']:,.0f}원"
        )
        
    msg_lines.append(f"\n<i>※ 1시간 주기 자동 재학습을 시작합니다...</i>")
    msg = "\n".join(msg_lines)
    
    print("[15mBot] Sending report to Telegram...")
    notifier.send_telegram_message(msg)
    
    # Save text report to file
    report_file = os.path.join(config.LOGS_DIR, 'predict_15m_reports.log')
    try:
        with open(report_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== REPORT {start_time} to {end_time} ===\n")
            f.write(f"Period: {period_count} trades.\n")
            for m in ['gb_basic', 'gb_current']:
                st = stats[m]
                f.write(f"  {m:10}: Period WR: {st['period_wr']*100:.1f}% ({st['period_wins']}/{st['period_trades']} trades), PnL: {st['period_pnl']:,.0f} KRW | Cum WR: {st['cum_wr']*100:.1f}% ({st['cum_wins']}/{st['cum_trades']} trades), PnL: {st['cum_pnl']:,.0f} KRW\n")
    except Exception as e:
        print(f"[15mBot] Error logging text report: {e}")
        
    print("[15mBot] Starting periodic retraining for all symbols...")
    for symbol in config.SYMBOLS:
        try:
            collect_symbol_data_15m(exchange, symbol, days=180)
            train_15m_model(symbol)
        except Exception as e:
            print(f"  [Error] Retraining failed for {symbol}: {e}")
            
    print("[15mBot] Retraining completed successfully.")

def main():
    print("[15mBot] Starting 15-minute Candle Prediction Bot...")
    
    exchange = get_ccxt_exchange()
    
    def start_proactive_time_sync(exchange):
        def run_sync():
            while True:
                try:
                    exchange.load_time_difference()
                except Exception:
                    pass
                time.sleep(15 * 60)
        threading.Thread(target=run_sync, daemon=True).start()

    start_proactive_time_sync(exchange)
    
    # 1. Initial Check / Train (Forced initial training with 180 days)
    print("[15mBot] Performing initial data collection and training (180 days)...")
    for symbol in config.SYMBOLS:
        try:
            collect_symbol_data_15m(exchange, symbol, days=180)
            train_15m_model(symbol)
        except Exception as e:
            print(f"[15mBot] Error in initial training for {symbol}: {e}")
                
    print("[15mBot] Initial training check complete. Starting prediction loop.")
    notifier.send_telegram_message("🤖 <b>15분봉 자동 예측 봇 가동</b>\n15분마다 예측, 정각마다 재학습 및 성과 보고를 수행합니다.")
    
    last_checked_tf = None
    last_retrain_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    while True:
        try:
            now = datetime.now()
            
            # Align to 15-minute boundaries (0, 15, 30, 45)
            current_tf_timestamp = now.replace(minute=now.minute - (now.minute % 15), second=0, microsecond=0)
            
            if current_tf_timestamp != last_checked_tf:
                time.sleep(8)
                execute_predictions_for_cycle(exchange, current_tf_timestamp)
                last_checked_tf = current_tf_timestamp
                
            evaluate_pending_predictions(exchange)
            
            # 정각 5분 전에 다음 시간대 모델을 위한 재학습 및 성과 리포트 전송 수행 (예: 15:55에 16:00 모델 대상)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            retrain_trigger_time = next_hour - timedelta(minutes=5)
            
            if now >= retrain_trigger_time and next_hour != last_retrain_hour:
                start_period = last_retrain_hour
                end_period = next_hour
                
                t = threading.Thread(target=generate_report_and_retrain, args=(exchange, start_period, end_period), daemon=True)
                t.start()
                
                last_retrain_hour = next_hour
                
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("[15mBot] Exiting program...")
            break
        except Exception as e:
            print(f"[15mBot] Main loop error: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()
