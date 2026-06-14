import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

import os
import time
import json
import requests
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import config
import notifier
from indicators import calculate_indicators
from risk_manager import RiskManager
from executor import Executor
from trade_logger import TradeLogger
from auto_retrain import AutoRetrainer
import dashboard
from regime_evaluator import RegimeEvaluator
from loss_streak_handler import check_and_handle_loss_streak

# 액티브 포지션 저장 파일 경로
ACTIVE_POSITIONS_FILE = os.path.join(config.LOGS_DIR, 'active_positions.json')

def load_active_positions() -> dict:
    if os.path.exists(ACTIVE_POSITIONS_FILE):
        try:
            with open(ACTIVE_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_active_positions(active_pos: dict):
    try:
        with open(ACTIVE_POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(active_pos, f, indent=4)
    except Exception as e:
        print(f"[LiveBot] Error saving active positions: {e}")

def fetch_realtime_futures_features(symbol, timeframe='15m'):
    """
    바이낸스 REST API를 통해 실시간 수급 지표 조회 (OI, Long/Short Ratio, Funding Rate)
    """
    binance_symbol = symbol.replace('/', '')
    
    # 1. 펀딩 피 (Default: 0.0)
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
    except Exception as e:
        print(f"[LiveBot] Error fetching real-time funding rate for {symbol}: {e}")
        
    # 2. 미체결 약정 (Default: 0.0)
    open_interest = 0.0
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        res = requests.get(url, params={'symbol': binance_symbol}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            open_interest = float(data.get('openInterest', 0.0))
    except Exception as e:
        print(f"[LiveBot] Error fetching real-time open interest for {symbol}: {e}")
        
    # 3. 롱숏 비율 (Default: 1.0)
    long_short_ratio = 1.0
    try:
        url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
        res = requests.get(url, params={'symbol': binance_symbol, 'period': timeframe, 'limit': 1}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0:
                long_short_ratio = float(data[0].get('longShortRatio', 1.0))
    except Exception as e:
        print(f"[LiveBot] Error fetching real-time {timeframe} long/short ratio for {symbol}: {e}")
        
    return {
        'funding_rate': funding_rate,
        'open_interest': open_interest,
        'long_short_ratio': long_short_ratio
    }

# 웹소켓 실시간 가격 수집기 구현
import websocket
realtime_prices = {}
realtime_prices_lock = threading.Lock()

def start_binance_websocket():
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
                symbol_raw = stream_data.get('s')
                price = float(stream_data.get('c', 0.0))
                if symbol_raw:
                    for standard_symbol in config.SYMBOLS:
                        if standard_symbol.replace('/', '') == symbol_raw:
                            with realtime_prices_lock:
                                realtime_prices[standard_symbol] = price
                            break
        except Exception as e:
            pass

    def on_error(ws, error):
        pass

    def on_close(ws, close_status_code, close_msg):
        time.sleep(5)
        connect_ws()

    def connect_ws():
        streams = "/".join([f"{s.replace('/', '').lower()}@ticker" for s in config.SYMBOLS])
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()

    threading.Thread(target=connect_ws, daemon=True).start()

def start_proactive_time_sync(executor):
    def run_sync():
        while True:
            try:
                executor.exchange.load_time_difference()
            except Exception as e:
                pass
            time.sleep(15 * 60) # 15분마다 동기화
            
    threading.Thread(target=run_sync, daemon=True).start()

def check_entry_signal(df, symbol, models_dict):
    """
    단일 3-클래스 통합 모델 기반 진입 판단 로직 (GradBoost 단독 사용)
    predict_proba 출력: [P(DOWN), P(SIDEWAYS), P(UP)] → 합산 = 100%
    라벨: 0=DOWN(SHORT), 1=SIDEWAYS, 2=UP(LONG)
    """
    X = df[config.FEATURES].iloc[-2:-1]

    # 1. 장세 예측 (0: BEAR, 1: SIDEWAYS, 2: BULL)
    regime_probs = models_dict['xgb_regime'].predict_proba(X)[0]
    regime_idx = int(np.argmax(regime_probs))
    regime_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
    regime = regime_map[regime_idx]

    # 2. 통합 모델 확률 추출
    gb_probs = models_dict['gb_unified'].predict_proba(X)[0]

    gb_long_p  = float(gb_probs[2])  # P(UP)
    gb_short_p = float(gb_probs[0])  # P(DOWN)

    # GradBoost 단독 사용
    ens_long_p  = gb_long_p
    ens_short_p = gb_short_p

    signal = None
    ensemble_prob = 0.0

    # 3. 신호 감지: GradBoost 임계값 충족 및 우위 조건
    raw_signal = None
    if (gb_long_p > config.GB_THRESHOLD and 
        ens_long_p > ens_short_p):
        raw_signal = 'LONG'
        ensemble_prob = ens_long_p
    elif (gb_short_p > config.GB_THRESHOLD and 
          ens_short_p > ens_long_p):
        raw_signal = 'SHORT'
        ensemble_prob = ens_short_p

    # 4. 장세 필터 적용
    if raw_signal == 'LONG':
        if regime in ['BULL', 'SIDEWAYS']:
            signal = 'LONG'
        else:
            print(f"[LiveBot] Filtered LONG signal for {symbol} due to {regime} regime prediction.")
    elif raw_signal == 'SHORT':
        if regime in ['BEAR', 'SIDEWAYS']:
            signal = 'SHORT'
        else:
            print(f"[LiveBot] Filtered SHORT signal for {symbol} due to {regime} regime prediction.")

    return {
        'signal'       : signal,
        'xgb_prob'     : gb_long_p if signal == 'LONG' else gb_short_p,
        'rf_prob'      : 0.0,
        'ensemble_prob': ensemble_prob,
        'xgb_long_prob': gb_long_p,
        'rf_long_prob' : 0.0,
        'xgb_short_prob': gb_short_p,
        'rf_short_prob' : 0.0,
        'regime'       : regime
    }


def parse_ccxt_position(pos) -> dict:
    """
    CCXT 포지션 데이터 파싱 및 표준화
    """
    if not pos:
        return {
            'size': 0.0,
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'side': None,
            'liquidation_price': 0.0
        }
    size = float(pos.get('contracts', 0.0) or 0.0)
    side_str = pos.get('side', None) # 'long' or 'short' or None
    entry_price = float(pos.get('entryPrice', 0.0) or 0.0)
    unrealized_pnl = float(pos.get('unrealizedPnl', 0.0) or 0.0)
    liquidation_price = float(pos.get('liquidationPrice', 0.0) or 0.0)
    
    side = None
    if size > 0:
        if side_str == 'long':
            side = 'LONG'
        elif side_str == 'short':
            side = 'SHORT'
            size = -size
            
    return {
        'size': size,
        'entry_price': entry_price,
        'unrealized_pnl': unrealized_pnl,
        'side': side,
        'liquidation_price': liquidation_price
    }

def retry_api(func, *args, retries=3, delay=5, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            # Timestamp 동기화 오류 (-1021) 감지 시 자동으로 시간 차이 재측정 및 동기화 수행
            if "Timestamp" in err_msg or "-1021" in err_msg:
                print("[LiveBot] Binance Timestamp mismatch detected. Synchronizing clock difference...")
                try:
                    if hasattr(func, '__self__'):
                        obj = func.__self__
                        if hasattr(obj, 'load_time_difference'):
                            obj.load_time_difference()
                        elif hasattr(obj, 'exchange') and hasattr(obj.exchange, 'load_time_difference'):
                            obj.exchange.load_time_difference()
                except Exception as ex:
                    print(f"[LiveBot] Failed to synchronize clock difference: {ex}")

            if i == retries - 1:
                raise e
            print(f"[LiveBot] CCXT API Error: {e}. Retrying {i+2}/{retries} in {delay}s...")
            time.sleep(delay)

def main_loop():
    print("[LiveBot] Initializing trading bot components...")
    
    tf_minutes = int(config.TIMEFRAME.replace('m', ''))
    risk_manager = RiskManager()
    trade_logger = TradeLogger()
    auto_retrainer = AutoRetrainer()
    executor = Executor()
    regime_evaluator = RegimeEvaluator()
    
    models = {}
    last_model_trained_times = {}
    cached_dfs = {}

    start_binance_websocket()
    start_proactive_time_sync(executor)

    def load_all_models():
        for symbol in config.SYMBOLS:
            xgb_u, rf_u, lgb_u, cat_u, et_u, gb_u, mlp_u, svm_u, xgb_r = auto_retrainer.load_models_with_fallback(symbol)
            models[symbol] = {
                'gb_unified': gb_u,
                'mlp_unified': mlp_u,
                'xgb_regime' : xgb_r
            }

            meta = auto_retrainer.load_metadata()
            last_model_trained_times[symbol] = meta.get(symbol, {}).get('last_trained', '')
        print("[LiveBot] GB/MLP and Regime models loaded successfully.")

    load_all_models()

    print(f"[LiveBot] Starting Flask dashboard on http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}...")
    dashboard_thread = threading.Thread(target=dashboard.start_dashboard_server, daemon=True)
    dashboard_thread.start()

    notifier.send_alert("🤖 라이브 봇 시작 | GradBoost 진입(15m) + 1s 실시간 트레일링 스탑 & 수수료 보호 청산 탑재")

    active_positions = load_active_positions()
    loss_streak_wait_until = None
    
    is_first_run = True
    last_checked_tf = None
    wallet_balance = 0.0
    last_balance_check = None
    
    # 오픈 알고리즘 주문 캐싱 변수 선언
    last_algo_check_time = None
    active_algo_orders = []

    while True:
        if not dashboard.state.is_running:
            print("[LiveBot] Bot is currently in STOPPED state via dashboard. Waiting...")
            time.sleep(5)
            continue

        now = datetime.now()
        
        # 1분 단위로 open algo orders 캐싱 갱신 (Rate limit 보호용)
        if last_algo_check_time is None or (now - last_algo_check_time).total_seconds() >= 60:
            try:
                active_algo_orders = retry_api(executor.exchange.fapiPrivateGetOpenAlgoOrders)
                last_algo_check_time = now
            except Exception as e:
                print(f"[LiveBot] Error caching open algo orders: {e}")
        
        # 설정된 봉 주기에 따라 정밀하게 완성봉이 끝났는지 체크 (Timing Skip 버그 원천 차단)
        current_tf_timestamp = now.replace(minute=now.minute - (now.minute % tf_minutes), second=0, microsecond=0)
        is_tf_cycle = (current_tf_timestamp != last_checked_tf) or is_first_run

        # 매 완성봉 사이클이거나 1분 단위 디버그용으로 주기적 로깅 (1초 주기 출력 폭탄 방지)
        if is_tf_cycle or (now.second == 0):
            print(f"\n[LiveBot] --- Trading Cycle Check: {now.strftime('%Y-%m-%d %H:%M:%S')} ({config.TIMEFRAME} Cycle: {is_tf_cycle}) ---")
        
        try:
            # 10초에 한 번만 잔고를 체크하여 API 소모량 감소
            if last_balance_check is None or (now - last_balance_check).total_seconds() >= 10:
                wallet_balance = retry_api(executor.get_balance)
                dashboard.state.current_capital = wallet_balance
                last_balance_check = now
            
            if risk_manager.check_daily_limit():
                msg = f"[ALERT] 당일 손실 한도 초과 (-{config.DAILY_LOSS_LIMIT*100}%)로 인해 봇을 강제 정지합니다."
                print(f"[LiveBot] {msg}")
                notifier.send_alert(msg)
                dashboard.state.is_running = False
                time.sleep(10)
                continue

            # 3회 연속 손절 시 1시간 매매 정지 로직 비활성화

            if auto_retrainer.check_retrain_needed():
                threading.Thread(target=auto_retrainer.run_retrain, args=(config.SYMBOLS,), daemon=True).start()

            meta = auto_retrainer.load_metadata()
            need_reload = False
            for symbol in config.SYMBOLS:
                current_time = meta.get(symbol, {}).get('last_trained', '')
                if current_time != last_model_trained_times.get(symbol, ''):
                    need_reload = True
                    break
            
            if need_reload:
                print("[LiveBot] New models detected. Reloading all models...")
                load_all_models()
                dashboard.state.load_initial_metadata()

            # 웹소켓 실시간 가격 캐시 조회
            tickers = {}
            missing_symbols = []
            with realtime_prices_lock:
                for symbol in config.SYMBOLS:
                    price = realtime_prices.get(symbol)
                    if price is not None:
                        tickers[symbol] = {'last': price}
                        tickers[f"{symbol}:USDT"] = {'last': price}
                    else:
                        missing_symbols.append(symbol)
            
            # 캐시가 빈 경우 거래소 API 호출하여 채움 (대안책)
            if missing_symbols:
                try:
                    if len(missing_symbols) > len(config.SYMBOLS) / 2:
                        rest_tickers = retry_api(executor.exchange.fetch_tickers)
                        for sym in config.SYMBOLS:
                            ticker_key = f"{sym}:USDT"
                            t_val = rest_tickers.get(ticker_key) or rest_tickers.get(sym)
                            if t_val and t_val.get('last') is not None:
                                with realtime_prices_lock:
                                    realtime_prices[sym] = float(t_val['last'])
                                tickers[sym] = {'last': float(t_val['last'])}
                                tickers[f"{sym}:USDT"] = {'last': float(t_val['last'])}
                    else:
                        for sym in missing_symbols:
                            t_val = retry_api(executor.exchange.fetch_ticker, sym)
                            if t_val and t_val.get('last') is not None:
                                with realtime_prices_lock:
                                    realtime_prices[sym] = float(t_val['last'])
                                tickers[sym] = {'last': float(t_val['last'])}
                                tickers[f"{sym}:USDT"] = {'last': float(t_val['last'])}
                except Exception as e:
                    print(f"[LiveBot] Error fetching fallback tickers: {e}")

            # 1초 주기 감시를 위해 실시간 전체 포지션을 fetch_positions 1회 호출로 캐싱
            pos_map = {}
            try:
                all_positions = retry_api(executor.exchange.fetch_positions)
                for p in all_positions:
                    sym = p.get('symbol')
                    if sym:
                        pos_map[sym] = p
            except Exception as e:
                if "banned until" in str(e) or "418" in str(e):
                    raise e
            for symbol in config.SYMBOLS:
                if any(models[symbol].get(k) is None for k in ['gb_unified', 'mlp_unified', 'xgb_regime']):
                    continue
                try:

                    # 포지션 확인 (캐싱된 pos_map 데이터 활용하여 REST 호출 최소화)
                    ticker_key = f"{symbol}:USDT"
                    raw_pos = pos_map.get(ticker_key) or pos_map.get(symbol)
                    ex_pos = parse_ccxt_position(raw_pos)
                
                    size = ex_pos['size']
                    entry_price = ex_pos['entry_price']
                    unpnl = ex_pos['unrealized_pnl']
                
                    dashboard.state.positions[symbol] = {
                        'size': size,
                        'entry_price': entry_price,
                        'unrealized_pnl': unpnl,
                        'side': ex_pos['side'],
                        'liquidation_price': ex_pos['liquidation_price'],
                        'leverage': config.SYMBOL_LEVERAGE.get(symbol, config.LEVERAGE)
                    }

                    # 완성봉 단위 K라인 캐싱 및 1초 주기 실시간 장세 예측
                    current_regime = 'SIDEWAYS'
                    df = None
                    
                    if is_tf_cycle:
                        try:
                            # 5m K라인 데이터 수집 및 지표 계산
                            ohlcv = retry_api(executor.exchange.fetch_ohlcv, symbol, config.TIMEFRAME, limit=config.LIVE_LIMIT)
                            df_raw = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df = calculate_indicators(df_raw)
                        
                            # 실시간 수급 데이터 결합
                            rt_feats = fetch_realtime_futures_features(symbol, config.TIMEFRAME)
                            df['funding_rate'] = rt_feats['funding_rate']
                            df['open_interest'] = rt_feats['open_interest']
                            df['long_short_ratio'] = rt_feats['long_short_ratio']

                            # 캐시에 저장
                            cached_dfs[symbol] = df.copy()

                            # 장세 모델 예측 (iloc[-2] 완성봉 기준으로 5분마다 기본 예측)
                            X = df[config.FEATURES].iloc[-2:-1]
                            xgb_r = models[symbol]['xgb_regime']
                            regime_probs = xgb_r.predict_proba(X)[0]
                            regime_idx = int(np.argmax(regime_probs))
                            regime_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
                            current_regime = regime_map[regime_idx]
                            
                            if symbol in dashboard.state.models_status:
                                dashboard.state.models_status[symbol]['regime'] = current_regime

                            # 5분 완성봉 예측 내역 로깅 및 실시간 평가 검증 진행
                            pred_time = df.index[-2].strftime('%Y-%m-%d %H:%M:%S') if isinstance(df.index, pd.DatetimeIndex) else str(df.index[-2])
                            pred_price = float(df['close'].iloc[-2])
                            regime_evaluator.log_prediction(symbol, current_regime, pred_time, pred_price)
                            regime_evaluator.evaluate_predictions(symbol, df)
                        except Exception as e:
                            print(f"[LiveBot] Error predicting regime/indicators on 5m cycle for {symbol}: {e}")
                    else:
                        # 15분 주기가 아닐 때, 캐시 데이터를 바탕으로 1초 실시간 가격 결합하여 실시간 장세 예측
                        if symbol in cached_dfs:
                            try:
                                df_temp = cached_dfs[symbol].copy()
                                
                                # 실시간 시세 조회
                                ticker_key = f"{symbol}:USDT"
                                ticker = tickers.get(ticker_key) or tickers.get(symbol)
                                if ticker and ticker.get('last') is not None:
                                    current_close = float(ticker['last'])
                                    
                                    # K라인의 마지막 행(실시간 미완성봉) 업데이트
                                    df_temp.loc[df_temp.index[-1], 'close'] = current_close
                                    if current_close > df_temp.loc[df_temp.index[-1], 'high']:
                                        df_temp.loc[df_temp.index[-1], 'high'] = current_close
                                    if current_close < df_temp.loc[df_temp.index[-1], 'low']:
                                        df_temp.loc[df_temp.index[-1], 'low'] = current_close
                                        
                                    # 지표 재계산
                                    df_temp = calculate_indicators(df_temp)
                                    
                                    # 실시간 수급 지표도 채워줌 (ffill)
                                    df_temp['funding_rate'] = df_temp['funding_rate'].ffill()
                                    df_temp['open_interest'] = df_temp['open_interest'].ffill()
                                    df_temp['long_short_ratio'] = df_temp['long_short_ratio'].ffill()
                                    
                                    # 실시간 미완성봉(iloc[-1]) 기준으로 장세 예측
                                    X = df_temp[config.FEATURES].iloc[-1:]
                                    xgb_r = models[symbol]['xgb_regime']
                                    regime_probs = xgb_r.predict_proba(X)[0]
                                    regime_idx = int(np.argmax(regime_probs))
                                    regime_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
                                    current_regime = regime_map[regime_idx]
                                    
                                    if symbol in dashboard.state.models_status:
                                        dashboard.state.models_status[symbol]['regime'] = current_regime
                            except Exception as e:
                                pass
                                
                        if symbol in dashboard.state.models_status and 'regime' not in dashboard.state.models_status[symbol]:
                            dashboard.state.models_status[symbol]['regime'] = 'SIDEWAYS'
                            
                        current_regime = dashboard.state.models_status.get(symbol, {}).get('regime', 'SIDEWAYS')
                        
                        # CASE 1에서 사용할 df 복원 (완성봉 기준)
                        df = cached_dfs.get(symbol)

                    # ------------------ [CASE 1] 포지션 미보유시 (15m 진입 탐색) ------------------
                    if size == 0:
                        if symbol in active_positions:
                            pos_info = active_positions.pop(symbol)
                            save_active_positions(active_positions)
                            retry_api(executor.cancel_all_orders, symbol)
                            
                            # 거래소 체결 내역 조회하여 실제 청산 정보 복구
                            exit_price = pos_info['entry_price']
                            exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            exit_reason = "거래소청산"
                            
                            try:
                                # 최근 15분 내의 체결 중 포지션 종료 반대 방향 체결 조회
                                since_ms = int((datetime.now() - timedelta(minutes=15)).timestamp() * 1000)
                                trades = retry_api(executor.exchange.fetch_my_trades, symbol=symbol, since=since_ms, limit=5)
                                if trades:
                                    opp_side = 'sell' if pos_info['side'] == 'LONG' else 'buy'
                                    match_trades = [t for t in trades if t['side'] == opp_side]
                                    if match_trades:
                                        total_qty = sum(t['amount'] for t in match_trades)
                                        avg_exit_price = sum(t['price'] * t['amount'] for t in match_trades) / total_qty
                                        exit_price = avg_exit_price
                                        
                                        last_t = match_trades[-1]
                                        exit_time = datetime.fromtimestamp(last_t['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                                        
                                        is_long = pos_info['side'] == 'LONG'
                                        is_profit = (avg_exit_price > pos_info['entry_price']) if is_long else (avg_exit_price < pos_info['entry_price'])
                                        exit_reason = "거래소익절" if is_profit else "거래소손절"
                            except Exception as ex_err:
                                print(f"[LiveBot] Error fetching exchange exit details for {symbol}: {ex_err}")
                                
                            qty = pos_info['quantity']
                            pnl_usdt = (exit_price - pos_info['entry_price']) * qty if pos_info['side'] == 'LONG' else (pos_info['entry_price'] - exit_price) * qty
                            fee_usdt = (pos_info['entry_price'] + exit_price) * qty * config.FEE_RATE
                            
                            final_pnl_usdt = pnl_usdt - fee_usdt
                            final_pnl_pct = final_pnl_usdt / (pos_info['entry_price'] * qty + 1e-9)
                            
                            risk_manager.update_trade_result(final_pnl_pct)
                            
                            pos_leverage = pos_info.get('leverage', config.SYMBOL_LEVERAGE.get(symbol, config.LEVERAGE))
                            trade_record = {
                                'id': str(int(time.time())),
                                'symbol': symbol,
                                'side': pos_info['side'],
                                'entry_time': pos_info['entry_time'],
                                'exit_time': exit_time,
                                'entry_price': pos_info['entry_price'],
                                'exit_price': exit_price,
                                'quantity': qty,
                                'pnl_usdt': final_pnl_usdt,
                                'pnl_pct': final_pnl_pct,
                                'leverage': pos_leverage,
                                'exit_reason': exit_reason,
                                'xgb_prob': pos_info.get('xgb_prob', 0.5),
                                'rf_prob': 0.0,
                                'ensemble_prob': pos_info.get('ensemble_prob', 0.5),
                                'hold_minutes': (datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(pos_info['entry_time'], '%Y-%m-%d %H:%M:%S')).total_seconds() / 60.0,
                                'fee_usdt': fee_usdt
                            }
                            trade_logger.log_trade(trade_record)
                            check_and_handle_loss_streak(risk_manager, trade_logger, auto_retrainer)
                            
                            notifier.send_exit(symbol, pos_info['side'], pos_info['entry_price'], exit_price, final_pnl_pct, exit_reason, final_pnl_usdt, leverage=pos_leverage)
                            print(f"[LiveBot] Closed {pos_info['side']} position on exchange for {symbol}. Reason: {exit_reason}, PnL: {final_pnl_pct*100:.2f}%")

                        if not is_tf_cycle or df is None:
                            continue  # 완성봉 주기 시점이 아니거나 데이터 없으면 진입 계산 패스

                        # 해당 종목이 학습 중이면 진입하지 않음
                        if auto_retrainer.is_symbol_training(symbol):
                            print(f"[LiveBot] Skipping entry evaluation for {symbol} because it is currently undergoing retraining.")
                            continue

                        # 예측 모델 결과 계산 (완성봉 기준)
                        sig_res = check_entry_signal(df, symbol, models[symbol])
                        signal = sig_res['signal']
                        xgb_prob = sig_res['xgb_prob']
                    
                        if signal is not None:
                            # 신규 진입 시 포지션 개수 한도 체크
                            if len(active_positions) >= config.MAX_POSITIONS:
                                print(f"[LiveBot] 최대 포지션 수({config.MAX_POSITIONS}) 초과로 {symbol} {signal} 진입 스킵.")
                                continue
                            
                            # 필터 지표 추출 (완성봉 iloc[-2] 기준)
                            vol_ratio = df['vol_ratio'].iloc[-2]
                            rsi = df['rsi'].iloc[-2]
                        
                            is_filtered = True
                            if signal == 'LONG':
                                is_filtered = not (
                                    vol_ratio >= 1.0 and
                                    rsi > 40 and
                                    not risk_manager.check_daily_limit() and
                                    not risk_manager.check_loss_streak()
                                )
                            else:  # SHORT
                                is_filtered = not (
                                    vol_ratio >= 1.0 and
                                    rsi < 60 and
                                    not risk_manager.check_daily_limit() and
                                    not risk_manager.check_loss_streak()
                                )
                            
                            if not is_filtered:
                                current_close = df['close'].iloc[-2]
                                is_high_conf = (xgb_prob > 0.65)
                                qty = risk_manager.calc_position_size(wallet_balance, current_close, symbol, is_high_conf)
                            
                                # 15분봉 최근 5개 완성봉의 저가/고가 기준 구조적 손절가 계산
                                # df.iloc[-6:-1]이 최근 5개 완성봉을 의미 (iloc[-1]은 현재 실시간 미완성봉)
                                if signal == 'LONG':
                                    support_price = df['low'].iloc[-6:-1].min()
                                    sl_price = support_price * 0.9995  # 0.05% 버퍼
                                    # 손절폭 제한 (최소 0.3%, 최대 2.0%)
                                    min_sl = current_close * 0.997
                                    max_sl = current_close * 0.980
                                    sl_price = max(max_sl, min(min_sl, sl_price))
                                else:
                                    resistance_price = df['high'].iloc[-6:-1].max()
                                    sl_price = resistance_price * 1.0005  # 0.05% 버퍼
                                    # 손절폭 제한 (최소 0.3%, 최대 2.0%)
                                    min_sl = current_close * 1.003
                                    max_sl = current_close * 1.020
                                    sl_price = min(max_sl, max(min_sl, sl_price))
                                
                                # 레버리지 설정
                                symbol_leverage = config.SYMBOL_LEVERAGE.get(symbol, config.LEVERAGE)
                                retry_api(executor.set_leverage, symbol, symbol_leverage)
                            
                                if signal == 'LONG':
                                    order = retry_api(executor.open_long, symbol, qty)
                                else:
                                    order = retry_api(executor.open_short, symbol, qty)
                                
                                retry_api(executor.set_stop_loss, symbol, signal, sl_price, qty)
                            
                                active_positions[symbol] = {
                                    'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'entry_price': current_close,
                                    'quantity': qty,
                                    'side': signal,
                                    'tp1_hit': False,
                                    'xgb_prob': xgb_prob,
                                    'rf_prob': sig_res['rf_prob'],
                                    'ensemble_prob': sig_res['ensemble_prob'],
                                    'max_price': current_close,  # 최고점 기록 초기화
                                    'min_price': current_close,  # 최저점 기록 초기화
                                    'sl_price': sl_price,        # 구조적 손절가 저장
                                    'leverage': symbol_leverage, # 레버리지 저장
                                    'checked_15m_exit': False,    # 15분 조기 청산 여부 초기화
                                    'profit_lock_stage': 0,      # 3단계 익절보존 단계 초기화
                                    'price_history': [current_close] # 실시간 가격 히스토리 초기화
                                }
                                save_active_positions(active_positions)
                            
                                notifier.send_entry(symbol, signal, xgb_prob, sig_res['rf_prob'], sig_res['ensemble_prob'], current_close, qty, leverage=symbol_leverage)
                                print(f"[LiveBot] Entered {signal} position for {symbol}. Price: {current_close}, Qty: {qty}")
                            
                    # ------------------ [CASE 2] 포지션 보유시 (1초 실시간 감시 및 트레일링 스탑) ------------------
                    else:
                        pos_info = active_positions.get(symbol)
                        if not pos_info:
                            pos_side = 'LONG' if size > 0 else 'SHORT'
                            symbol_leverage = config.SYMBOL_LEVERAGE.get(symbol, config.LEVERAGE)
                            pos_info = {
                                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'entry_price': entry_price,
                                'quantity': abs(size),
                                'side': pos_side,
                                'tp1_hit': False,
                                'xgb_prob': 0.5,
                                'rf_prob': 0.0,
                                'ensemble_prob': 0.5,
                                'max_price': entry_price,
                                'min_price': entry_price,
                                'leverage': symbol_leverage,
                                'checked_15m_exit': False,
                                'profit_lock_stage': 0,
                                'price_history': [entry_price]
                            }
                            active_positions[symbol] = pos_info
                            save_active_positions(active_positions)
                        
                        pos_side = pos_info['side']
                        qty = pos_info['quantity']

                        # 거래소 스탑로스 주문 유실 감지 및 자동 복원 로직
                        binance_symbol = symbol.replace('/', '')
                        has_sl_on_exchange = False
                        for a in active_algo_orders:
                            if a.get('symbol') == binance_symbol and a.get('orderType') == 'STOP_MARKET' and a.get('algoStatus') == 'NEW':
                                has_sl_on_exchange = True
                                break
                        
                        if not has_sl_on_exchange:
                            sl_price = pos_info.get('sl_price')
                            if sl_price:
                                print(f"[LiveBot] ⚠️ Missing Stop Loss order detected for {symbol} on exchange. Restoring stop loss at {sl_price}...")
                                restore_order = retry_api(executor.set_stop_loss, symbol, pos_side, sl_price, qty)
                                if restore_order:
                                    # 중복 방지를 위해 active_algo_orders 캐시에 수동 추가
                                    new_algo_item = {
                                        'symbol': binance_symbol,
                                        'orderType': 'STOP_MARKET',
                                        'algoStatus': 'NEW',
                                        'algoId': restore_order.get('id') or 'restored'
                                    }
                                    active_algo_orders.append(new_algo_item)
                        
                        entry_dt = datetime.strptime(pos_info['entry_time'], '%Y-%m-%d %H:%M:%S')
                        hold_minutes = (now - entry_dt).total_seconds() / 60.0
                    
                        # 캐시된 시세에서 현재가 파싱
                        current_close = entry_price
                        ticker_key = f"{symbol}:USDT"
                        ticker = tickers.get(ticker_key) or tickers.get(symbol)
                        if ticker and ticker.get('last') is not None:
                            current_close = float(ticker['last'])
                        else:
                            try:
                                current_close = float(retry_api(executor.exchange.fetch_ticker, symbol)['last'])
                            except Exception:
                                pass

                        # 실시간 가격 히스토리 저장 (최대 30초)
                        if 'price_history' not in pos_info:
                            pos_info['price_history'] = []
                        pos_info['price_history'].append(current_close)
                        if len(pos_info['price_history']) > 30:
                            pos_info['price_history'].pop(0)

                        # 최고점 / 최저점 실시간 갱신 및 파일 저장 (1초 단위)
                        updated_peak = False
                        if pos_side == 'LONG':
                            if current_close > pos_info.get('max_price', entry_price):
                                pos_info['max_price'] = current_close
                                updated_peak = True
                            current_roi = ((current_close - entry_price) / entry_price) * pos_info.get('leverage', 5)
                        else:
                            if current_close < pos_info.get('min_price', entry_price):
                                pos_info['min_price'] = current_close
                                updated_peak = True
                            current_roi = ((entry_price - current_close) / entry_price) * pos_info.get('leverage', 5)

                        # Profit Lock 단계 상향 조정 (대안 B 적용)
                        old_stage = pos_info.get('profit_lock_stage', 0)
                        if current_roi >= 0.04 and old_stage < 5.0:
                            pos_info['profit_lock_stage'] = 5.0
                            updated_peak = True
                            print(f"[LiveBot] {symbol} Profit Lock STAGE 5 activated (+4.0% ROI reached, locking +3.0% ROI).")
                        elif current_roi >= 0.03 and old_stage < 4.0:
                            pos_info['profit_lock_stage'] = 4.0
                            updated_peak = True
                            print(f"[LiveBot] {symbol} Profit Lock STAGE 4 activated (+3.0% ROI reached, locking +2.2% ROI).")
                        elif current_roi >= 0.02 and old_stage < 3.0:
                            pos_info['profit_lock_stage'] = 3.0
                            updated_peak = True
                            print(f"[LiveBot] {symbol} Profit Lock STAGE 3 activated (+2.0% ROI reached, locking +1.4% ROI).")
                        elif current_roi >= 0.016 and old_stage < 2.0:
                            pos_info['profit_lock_stage'] = 2.0
                            updated_peak = True
                            print(f"[LiveBot] {symbol} Profit Lock STAGE 2 activated (+1.6% ROI reached, locking +1.1% ROI).")
                        elif current_roi >= 0.012 and old_stage < 1.0:
                            pos_info['profit_lock_stage'] = 1.0
                            updated_peak = True
                            print(f"[LiveBot] {symbol} Profit Lock STAGE 1 activated (+1.2% ROI reached, locking +0.8% ROI).")
                        elif current_roi >= 0.008 and old_stage < 0.5:
                            pos_info['profit_lock_stage'] = 0.5
                            updated_peak = True
                            print(f"[LiveBot] {symbol} Profit Lock STAGE 0.5 activated (+0.8% ROI reached, locking +0.5% ROI).")
                            
                        # 실시간 수학적 추세 붕괴 여부 판정 (기울기 계산)
                        is_unprofitable = (pos_side == 'LONG' and current_close < entry_price) or (pos_side == 'SHORT' and current_close > entry_price)
                        math_exit_triggered = False
                        exit_reason_str = ""
                        
                        if is_unprofitable and len(pos_info.get('price_history', [])) >= 15:
                            ph = pos_info['price_history']
                            n = len(ph)
                            x = np.arange(n)
                            y = np.array(ph)
                            
                            x_mean = (n - 1) / 2.0
                            y_mean = np.mean(y)
                            num = np.sum((x - x_mean) * (y - y_mean))
                            den = np.sum((x - x_mean) ** 2)
                            slope = num / den if den > 0 else 0.0
                            norm_slope = slope / entry_price
                            
                            # 초당 0.01% 이상 가격이 불리한 방향으로 이동할 때 (기울기 감지)
                            if pos_side == 'LONG' and norm_slope <= -0.0001:
                                math_exit_triggered = True
                                exit_reason_str = f"실시간 추세 붕괴 청산 (LONG) | 기울기: {norm_slope*100:.4f}%/s"
                            elif pos_side == 'SHORT' and norm_slope >= 0.0001:
                                math_exit_triggered = True
                                exit_reason_str = f"실시간 추세 붕괴 청산 (SHORT) | 기울기: {norm_slope*100:.4f}%/s"

                        if updated_peak:
                            save_active_positions(active_positions)
                    
                        exit_reason = None
                        exit_price = current_close
                    
                        # 완성봉 단위 역방향 신호 예측 확률 조회 (스마트 청산용)
                        gb_lp = 0.0
                        mlp_lp = 0.0
                        gb_sp = 0.0
                        mlp_sp = 0.0
                        if is_tf_cycle and df is not None:
                            try:
                                X_completed = df[config.FEATURES].iloc[-2:-1]
                                gb_probs = models[symbol]['gb_unified'].predict_proba(X_completed)[0]
                                mlp_probs = models[symbol]['mlp_unified'].predict_proba(X_completed)[0]
                                
                                gb_lp = float(gb_probs[2])
                                mlp_lp = float(mlp_probs[2])
                                
                                gb_sp = float(gb_probs[0])
                                mlp_sp = float(mlp_probs[0])
                            except Exception as pe:
                                print(f"[LiveBot] Error predicting counter-signal probabilities for {symbol}: {pe}")

                        # 수수료 고려 손실 방지 기준선 계산 (왕복 수수료 0.08% + 최소 여유 버퍼 0.02% = 0.10%)
                        fee_protect_ratio = 2 * config.FEE_RATE + config.FEE_BUFFER  # 0.0010 (0.10%)
                    
                        # A. 120분 시간 초과 강제 청산
                        if hold_minutes >= config.MAX_HOLD_MINUTES:
                            exit_reason = "시간초과"

                        # H. 실시간 수학적 추세 붕괴 조기 청산
                        elif math_exit_triggered:
                            exit_reason = exit_reason_str
                        
                        # G. 레버리지 수익률 기준 단계별 익절보존 (Profit Lock) 감시 (대안 B)
                        elif (
                            (pos_info.get('profit_lock_stage', 0) == 5.0 and current_roi <= 0.030) or
                            (pos_info.get('profit_lock_stage', 0) == 4.0 and current_roi <= 0.022) or
                            (pos_info.get('profit_lock_stage', 0) == 3.0 and current_roi <= 0.014) or
                            (pos_info.get('profit_lock_stage', 0) == 2.0 and current_roi <= 0.011) or
                            (pos_info.get('profit_lock_stage', 0) == 1.0 and current_roi <= 0.008) or
                            (pos_info.get('profit_lock_stage', 0) == 0.5 and current_roi <= 0.005)
                        ) and (
                            (pos_side == 'LONG' and current_close >= entry_price * (1.0 + fee_protect_ratio)) or
                            (pos_side == 'SHORT' and current_close <= entry_price * (1.0 - fee_protect_ratio))
                        ):
                            stage = pos_info.get('profit_lock_stage', 0)
                            exit_reason = f"익절보존 ({stage}단계)"
                            exit_price = current_close
                        
                        # F. 15분 경과 시점에 본전 혹은 손실 상태일 때 조기 청산
                        elif hold_minutes >= 15.0 and not pos_info.get('checked_15m_exit', False):
                            is_loss = False
                            if pos_side == 'LONG' and current_close <= entry_price:
                                is_loss = True
                            elif pos_side == 'SHORT' and current_close >= entry_price:
                                is_loss = True

                            if is_loss:
                                exit_reason = "15분 손실 청산"
                                exit_price = current_close
                            else:
                                # 수익 상태이므로 청산하지 않고, 체크 완료 상태만 기록
                                pos_info['checked_15m_exit'] = True
                                save_active_positions(active_positions)
                                print(f"[LiveBot] {symbol} {pos_side} position is profitable at 15m. Holding for trend.")
                        
                        # B. 하드 손절 체크 (구조적 손절라인 sl_price 보호)
                        elif pos_side == 'LONG' and current_close <= pos_info.get('sl_price', entry_price * (1.0 - sl_pct)):
                            exit_reason = "손절"
                            exit_price = pos_info.get('sl_price', entry_price * (1.0 - sl_pct))
                        elif pos_side == 'SHORT' and current_close >= pos_info.get('sl_price', entry_price * (1.0 + sl_pct)):
                            exit_reason = "손절"
                            exit_price = pos_info.get('sl_price', entry_price * (1.0 + sl_pct))
                        
                        # C. AI 역방향 신호 감지 즉시 탈출 (스마트 청산)
                        elif pos_side == 'LONG' and is_tf_cycle and gb_sp > config.GB_THRESHOLD and mlp_sp > config.MLP_THRESHOLD:
                            exit_reason = "역방향 청산 (AI)"
                            exit_price = current_close
                            print(f"[LiveBot] AI Counter-Signal detected for {symbol} LONG position: gb_short={gb_sp:.2f}, mlp_short={mlp_sp:.2f}")
                        elif pos_side == 'SHORT' and is_tf_cycle and gb_lp > config.GB_THRESHOLD and mlp_lp > config.MLP_THRESHOLD:
                            exit_reason = "역방향 청산 (AI)"
                            exit_price = current_close
                            print(f"[LiveBot] AI Counter-Signal detected for {symbol} SHORT position: gb_long={gb_lp:.2f}, mlp_long={mlp_lp:.2f}")
                        
                        # D. 트레일링 스탑 및 수수료 보호 검사 (설정된 콜백 비율 기준)
                        elif pos_side == 'LONG':
                            max_p = pos_info.get('max_price', entry_price)
                            retracement = (max_p - current_close) / max_p
                            # 최고점 대비 설정된 콜백 비율만큼 하락했고, 현재가가 수수료 보장 가격 이상일 때 청산
                            if retracement >= config.TRAILING_CALLBACK_RATE:
                                min_exit_price = entry_price * (1.0 + fee_protect_ratio)
                                if current_close >= min_exit_price:
                                    exit_reason = "트레일링스탑"
                                    exit_price = current_close
                        elif pos_side == 'SHORT':
                            min_p = pos_info.get('min_price', entry_price)
                            retracement = (current_close - min_p) / min_p
                            # 최저점 대비 설정된 콜백 비율만큼 반등했고, 현재가가 수수료 보장 가격 이하일 때 청산
                            if retracement >= config.TRAILING_CALLBACK_RATE:
                                max_exit_price = entry_price * (1.0 - fee_protect_ratio)
                                if current_close <= max_exit_price:
                                    exit_reason = "트레일링스탑"
                                    exit_price = current_close
                                
                        # E. 하드 익절 백업선 보호 (+3.0%)
                        elif pos_side == 'LONG' and current_close >= entry_price * 1.03:
                            exit_reason = "하드익절"
                            exit_price = entry_price * 1.03
                        elif pos_side == 'SHORT' and current_close <= entry_price * 0.97:
                            exit_reason = "하드익절"
                            exit_price = entry_price * 0.97

                        # 청산 실행
                        if exit_reason is not None:
                            current_qty = abs(size)
                        
                            retry_api(executor.close_position, symbol, current_qty, pos_side)
                            retry_api(executor.cancel_all_orders, symbol)
                        
                            pnl_usdt = (exit_price - entry_price) * current_qty if pos_side == 'LONG' else (entry_price - exit_price) * current_qty
                            fee_usdt = (entry_price + exit_price) * current_qty * config.FEE_RATE
                            
                            final_pnl_usdt = pnl_usdt - fee_usdt
                            final_pnl_pct = final_pnl_usdt / (entry_price * qty + 1e-9)
                        
                            risk_manager.update_trade_result(final_pnl_pct)
                        
                            pos_leverage = pos_info.get('leverage', config.SYMBOL_LEVERAGE.get(symbol, config.LEVERAGE))
                            trade_record = {
                                'id': str(int(time.time())),
                                'symbol': symbol,
                                'side': pos_side,
                                'entry_time': pos_info['entry_time'],
                                'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'quantity': qty,
                                'pnl_usdt': final_pnl_usdt,
                                'pnl_pct': final_pnl_pct,
                                'leverage': pos_leverage,
                                'exit_reason': exit_reason,
                                'xgb_prob': pos_info['xgb_prob'],
                                'rf_prob': pos_info.get('rf_prob', 0.0),
                                'ensemble_prob': pos_info['ensemble_prob'],
                                'hold_minutes': hold_minutes,
                                'fee_usdt': fee_usdt
                            }
                            trade_logger.log_trade(trade_record)
                            auto_retrainer.monitor_performance(symbol)
                            check_and_handle_loss_streak(risk_manager, trade_logger, auto_retrainer)
                        
                            notifier.send_exit(symbol, pos_side, entry_price, exit_price, final_pnl_pct, exit_reason, final_pnl_usdt, leverage=pos_leverage)
                        
                            active_positions.pop(symbol)
                            save_active_positions(active_positions)
                            print(f"[LiveBot] Closed {pos_side} position for {symbol}. Reason: {exit_reason}, PnL: {final_pnl_pct*100:.2f}%")

                except Exception as e:
                    if "banned until" in str(e) or "418" in str(e):
                        raise e
                    print(f"[LiveBot] Error processing symbol {symbol}: {e}")
            if is_tf_cycle:
                last_checked_tf = current_tf_timestamp
            is_first_run = False
            
            # 주기적인 장세 예측 정확도 검증 보고서 발송 체크
            regime_evaluator.check_and_send_report()

        except Exception as e:
            error_msg = f"[ERROR] 메인 루프 실행 중 에러 발생: {e}"
            print(f"[LiveBot] {error_msg}")
            
            err_msg = str(e)
            if "banned until" in err_msg or "418" in err_msg:
                import re
                match = re.search(r"banned until (\d+)", err_msg)
                if match:
                    ban_ts = int(match.group(1)) / 1000.0
                    sleep_secs = max(30, int(ban_ts - time.time()) + 10)
                    ban_time_str = datetime.fromtimestamp(ban_ts).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[LiveBot] [WARNING] IP Banned until {ban_time_str}. Sleeping for {sleep_secs}s to avoid ban extension...")
                    time.sleep(sleep_secs)
                else:
                    print("[LiveBot] [WARNING] IP Banned detected. Sleeping for 60s...")
                    time.sleep(60)
            else:
                try:
                    error_log_path = os.path.join(config.LOGS_DIR, 'error.log')
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"[{datetime.now()}] {error_msg}\n")
                except Exception:
                    pass
                    
                notifier.send_alert(error_msg)
                time.sleep(5)
 
        time.sleep(config.MONITOR_INTERVAL)  # 1초 단위 감시

if __name__ == '__main__':
    main_loop()
