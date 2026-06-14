import os
import pickle
import pandas as pd
import numpy as np
import config
from indicators import calculate_indicators

# 10가지 전략 매개변수 정의
STRATEGIES = {
    1: {
        "name": "기본 앙상블 (XGB 10% / RF 65%)",
        "xgb_th": 0.10, "rf_th": 0.65, "long_only": False,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 4, "use_filters": True
    },
    2: {
        "name": "완전 앙상블 (XGB 65% / RF 65%)",
        "xgb_th": 0.65, "rf_th": 0.65, "long_only": False,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 4, "use_filters": True
    },
    3: {
        "name": "롱 단방향 전략 (LONG Only)",
        "xgb_th": 0.10, "rf_th": 0.65, "long_only": True,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 4, "use_filters": True
    },
    4: {
        "name": "고신뢰 롱 단방향 (XGB 15% / RF 75%)",
        "xgb_th": 0.15, "rf_th": 0.75, "long_only": True,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 4, "use_filters": True
    },
    5: {
        "name": "초보수 양방향 (XGB 15% / RF 75%)",
        "xgb_th": 0.15, "rf_th": 0.75, "long_only": False,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 4, "use_filters": True
    },
    6: {
        "name": "느슨한 양방향 (XGB 8% / RF 55%)",
        "xgb_th": 0.08, "rf_th": 0.55, "long_only": False,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 4, "use_filters": True
    },
    7: {
        "name": "타이트 손절형 (SL 50% 축소)",
        "xgb_th": 0.10, "rf_th": 0.65, "long_only": False,
        "sl_mult": 0.5, "tp_mult": 1.0, "hold_bars": 4, "use_filters": True
    },
    8: {
        "name": "익절 폭 확장형 (TP 1.5배)",
        "xgb_th": 0.10, "rf_th": 0.65, "long_only": False,
        "sl_mult": 1.0, "tp_mult": 1.5, "hold_bars": 4, "use_filters": True
    },
    9: {
        "name": "시간 청산 단축형 (10분 청산)",
        "xgb_th": 0.10, "rf_th": 0.65, "long_only": False,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 2, "use_filters": True
    },
    10: {
        "name": "노 필터형 (ML 신호 단독 진입)",
        "xgb_th": 0.10, "rf_th": 0.65, "long_only": False,
        "sl_mult": 1.0, "tp_mult": 1.0, "hold_bars": 4, "use_filters": False
    }
}

def run_simulation(df, xgb_long_probs, rf_long_probs, xgb_short_probs, rf_short_probs, symbol, params):
    """
    주어진 전략 매개변수 집합(params)에 따른 개별 백테스트 시뮬레이션
    """
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # 보조지표 필터 데이터 추출
    vol_ratios = df['vol_ratio'].values
    ema_bullish = df['ema_bullish'].values
    vwaps = df['vwap'].values
    rsis = df['rsi'].values
    
    sl_pct = config.STOP_LOSS[symbol] * params['sl_mult']
    tp1_pct = config.TAKE_PROFIT_1[symbol] * params['tp_mult']
    tp2_pct = config.TAKE_PROFIT_2[symbol] * params['tp_mult']
    fee_rate = config.FEE_RATE
    
    capital = 1000000.0  # 시작 자본
    position = 0  # 0: 무, 1: 롱, -1: 숏
    entry_price = 0.0
    entry_idx = 0
    tp1_hit = False
    qty = 0.0
    
    trades = []
    
    for i in range(len(df)):
        if position == 0:
            xgb_lp = xgb_long_probs[i]
            rf_lp = rf_long_probs[i]
            xgb_sp = xgb_short_probs[i]
            rf_sp = rf_short_probs[i]
            
            is_long = False
            is_short = False
            
            # ML 신호 체크
            if xgb_lp > params['xgb_th'] and rf_lp > params['rf_th']:
                is_long = True
            elif not params['long_only'] and xgb_sp > params['xgb_th'] and rf_sp > params['rf_th']:
                is_short = True
                
            # 보조지표 필터 체크
            if params['use_filters']:
                if is_long:
                    # 롱 보조지표 조건: 거래량 >= 1.5, 정배열, close > vwap, rsi > 45
                    if not (vol_ratios[i] >= config.VOLUME_MULTIPLIER and 
                            ema_bullish[i] == 1 and 
                            close_prices[i] > vwaps[i] and 
                            rsis[i] > 45):
                        is_long = False
                elif is_short:
                    # 숏 보조지표 조건: 거래량 >= 1.5, 역배열, close < vwap, rsi < 55
                    if not (vol_ratios[i] >= config.VOLUME_MULTIPLIER and 
                            ema_bullish[i] == 0 and 
                            close_prices[i] < vwaps[i] and 
                            rsis[i] < 55):
                        is_short = False
            
            if is_long:
                position = 1
                entry_price = close_prices[i]
                entry_idx = i
                tp1_hit = False
                risk_amount = capital * config.RISK_PER_TRADE
                pos_value = risk_amount / sl_pct
                qty = pos_value / entry_price
            elif is_short:
                position = -1
                entry_price = close_prices[i]
                entry_idx = i
                tp1_hit = False
                risk_amount = capital * config.RISK_PER_TRADE
                pos_value = risk_amount / sl_pct
                qty = pos_value / entry_price
                
        else:
            current_close = close_prices[i]
            current_high = high_prices[i]
            current_low = low_prices[i]
            bars_held = i - entry_idx
            
            sl_hit = False
            tp1_triggered = False
            tp2_triggered = False
            time_up = False
            
            if position == 1:
                sl_price = entry_price * (1.0 - sl_pct)
                tp1_price = entry_price * (1.0 + tp1_pct)
                tp2_price = entry_price * (1.0 + tp2_pct)
                
                if current_low <= sl_price:
                    sl_hit = True
                elif not tp1_hit and current_high >= tp1_price:
                    tp1_triggered = True
                elif tp1_hit and current_high >= tp2_price:
                    tp2_triggered = True
                elif bars_held >= params['hold_bars']:
                    time_up = True
            elif position == -1:
                sl_price = entry_price * (1.0 + sl_pct)
                tp1_price = entry_price * (1.0 - tp1_pct)
                tp2_price = entry_price * (1.0 - tp2_pct)
                
                if current_high >= sl_price:
                    sl_hit = True
                elif not tp1_hit and current_low <= tp1_price:
                    tp1_triggered = True
                elif tp1_hit and current_low <= tp2_price:
                    tp2_triggered = True
                elif bars_held >= params['hold_bars']:
                    time_up = True
            
            if sl_hit:
                pnl = (sl_price - entry_price) * qty if position == 1 else (entry_price - sl_price) * qty
                fee = (entry_price + sl_price) * qty * fee_rate
                trade_pnl = pnl - fee
                capital += trade_pnl
                trades.append({'pnl': trade_pnl, 'pnl_pct': trade_pnl / (entry_price * qty), 'win': 0})
                position = 0
            elif tp2_triggered:
                current_qty = qty * 0.5 if tp1_hit else qty
                pnl = (tp2_price - entry_price) * current_qty if position == 1 else (entry_price - tp2_price) * current_qty
                fee = (entry_price + tp2_price) * current_qty * fee_rate
                trade_pnl = pnl - fee
                capital += trade_pnl
                trades.append({'pnl': trade_pnl, 'pnl_pct': trade_pnl / (entry_price * qty), 'win': 1})
                position = 0
            elif tp1_triggered:
                pnl1 = (tp1_price - entry_price) * (qty * 0.5) if position == 1 else (entry_price - tp1_price) * (qty * 0.5)
                fee1 = (entry_price + tp1_price) * (qty * 0.5) * fee_rate
                capital += (pnl1 - fee1)
                tp1_hit = True
            elif time_up:
                current_qty = qty * 0.5 if tp1_hit else qty
                pnl = (current_close - entry_price) * current_qty if position == 1 else (entry_price - current_close) * current_qty
                fee = (entry_price + current_close) * current_qty * fee_rate
                trade_pnl = pnl - fee
                capital += trade_pnl
                win_status = 1 if (trade_pnl > 0 or tp1_hit) else 0
                trades.append({'pnl': trade_pnl, 'pnl_pct': trade_pnl / (entry_price * qty), 'win': win_status})
                position = 0
                
    if not trades:
        return {'trades': 0, 'win_rate': 0.0, 'total_return': 0.0, 'mdd': 0.0}
        
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = wins / total_trades
    total_return = (capital - 1000000.0) / 1000000.0
    
    # MDD
    capital_history = [1000000.0]
    current = 1000000.0
    for t in trades:
        current += t['pnl']
        capital_history.append(current)
    peaks = np.maximum.accumulate(capital_history)
    drawdowns = (capital_history - peaks) / peaks
    mdd = np.min(drawdowns)
    
    return {
        'trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'mdd': mdd
    }

def main():
    print("============================================================")
    print("🚀 10가지 전략 백테스트 벤치마크 시작 (5대 종목 통합 포트폴리오)")
    print("============================================================\n")
    
    # 데이터 로드
    datasets = {}
    models = {}
    
    for symbol in config.SYMBOLS:
        sym_clean = symbol.replace('/', '_')
        csv_path = os.path.join(config.DATA_DIR, f"{sym_clean}.csv")
        xgb_l_path = os.path.join(config.MODELS_DIR, f"xgb_{sym_clean}_long.pkl")
        rf_l_path = os.path.join(config.MODELS_DIR, f"rf_{sym_clean}_long.pkl")
        xgb_s_path = os.path.join(config.MODELS_DIR, f"xgb_{sym_clean}_short.pkl")
        rf_s_path = os.path.join(config.MODELS_DIR, f"rf_{sym_clean}_short.pkl")
        
        if not all([os.path.exists(p) for p in [csv_path, xgb_l_path, rf_l_path, xgb_s_path, rf_s_path]]):
            print(f"⚠️ [{symbol}] 모델 혹은 데이터가 누락되어 제외합니다.")
            continue
            
        with open(xgb_l_path, 'rb') as f: xgb_l = pickle.load(f)
        with open(rf_l_path, 'rb') as f: rf_l = pickle.load(f)
        with open(xgb_s_path, 'rb') as f: xgb_s = pickle.load(f)
        with open(rf_s_path, 'rb') as f: rf_s = pickle.load(f)
            
        df_raw = pd.read_csv(csv_path)
        df = calculate_indicators(df_raw)
        
        # 마지막 15% Test셋 분리
        total_len = len(df)
        val_end = int(total_len * 0.85)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)
        X_test = test_df[config.FEATURES]
        
        xgb_lp = xgb_l.predict_proba(X_test)[:, 1]
        rf_lp = rf_l.predict_proba(X_test)[:, 1]
        xgb_sp = xgb_s.predict_proba(X_test)[:, 1]
        rf_sp = rf_s.predict_proba(X_test)[:, 1]
        
        datasets[symbol] = test_df
        models[symbol] = {
            'xgb_lp': xgb_lp, 'rf_lp': rf_lp,
            'xgb_sp': xgb_sp, 'rf_sp': rf_sp
        }

    # 전략별 백테스트 실행
    results = []
    
    for sid, params in STRATEGIES.items():
        print(f"⌛ 전략 {sid} 테스트 중: {params['name']}...")
        symbol_returns = []
        symbol_win_rates = []
        symbol_trades = []
        symbol_mdds = []
        
        for symbol in datasets.keys():
            res = run_simulation(
                datasets[symbol],
                models[symbol]['xgb_lp'], models[symbol]['rf_lp'],
                models[symbol]['xgb_sp'], models[symbol]['rf_sp'],
                symbol, params
            )
            symbol_returns.append(res['total_return'])
            symbol_win_rates.append(res['win_rate'])
            symbol_trades.append(res['trades'])
            symbol_mdds.append(res['mdd'])
            
        # 포트폴리오(5개 종목 평균) 기준 성능 계산
        portfolio_return = np.mean(symbol_returns)
        portfolio_win_rate = np.mean(symbol_win_rates)
        portfolio_trades = np.sum(symbol_trades)
        portfolio_mdd = np.mean(symbol_mdds)
        
        results.append({
            'id': sid,
            'name': params['name'],
            'trades': int(portfolio_trades),
            'win_rate': portfolio_win_rate,
            'return': portfolio_return,
            'mdd': portfolio_mdd
        })

    # 최종 결과 출력
    print("\n" + "="*80)
    print(f"{'ID':<3} | {'전략명':<30} | {'총거래수':<8} | {'평균승률':<8} | {'평균수익률':<10} | {'평균MDD':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['id']:<3} | {r['name']:<30} | {r['trades']:^8} | {r['win_rate']*100:^7.1f}% | {r['return']*100:^+9.2f}% | {r['mdd']*100:^7.1f}%")
    print("="*80)
    print("\n💡 [분석 코멘트]")
    # 수익률 기준 정렬 후 추천 전략 출력
    best_strat = max(results, key=lambda x: x['return'])
    print(f"가장 성과가 뛰어난 전략은 전략 {best_strat['id']} [ {best_strat['name']} ] 입니다.")
    print(f"  - 총 거래수: {best_strat['trades']}회")
    print(f"  - 평균 승률: {best_strat['win_rate']*100:.1f}%")
    print(f"  - 평균 수익률: {best_strat['return']*100:+.2f}%")
    print(f"  - 평균 MDD: {best_strat['mdd']*100:.1f}%")

if __name__ == '__main__':
    main()
