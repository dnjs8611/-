import os
import pickle
import pandas as pd
import numpy as np
import config
from indicators import calculate_indicators

def run_simulation(df, gb_model, gb_regime_model, symbol):
    """
    GradBoost 단독 모델 시뮬레이터 (시장 장세 필터 및 트레일링 스탑/익절 보존 로직 반영)
    """
    sl_pct = config.STOP_LOSS.get(symbol, 0.008)
    tp1_pct = config.TAKE_PROFIT_1.get(symbol, 0.016)
    tp2_pct = config.TAKE_PROFIT_2.get(symbol, 0.016)
    fee_rate = config.FEE_RATE
    leverage = config.SYMBOL_LEVERAGE.get(symbol, config.LEVERAGE)
    
    X = df[config.FEATURES]
    
    # 1. 장세 및 방향 예측 확률 계산
    regime_probs = gb_regime_model.predict_proba(X)
    regimes_idx = np.argmax(regime_probs, axis=1) # 0: BEAR, 1: SIDEWAYS, 2: BULL
    regime_map = {0: 'BEAR', 1: 'SIDEWAYS', 2: 'BULL'}
    regimes = [regime_map[idx] for idx in regimes_idx]
    
    gb_probs = gb_model.predict_proba(X) # [P(DOWN), P(SIDE), P(UP)]
    
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    capital = 1000000.0  # 시작 자본: 1,000,000원
    position = 0         # 0: 없음, 1: 롱, -1: 숏
    entry_price = 0.0
    entry_idx = 0
    tp1_hit = False
    qty = 0.0
    profit_lock_stage = 0.0
    
    hold_bars = max(2, config.MAX_HOLD_MINUTES // 15) # 15분봉 개수 기준
    
    trades = []
    
    for i in range(len(df) - 1):
        if position == 0:
            regime = regimes[i]
            p_down, p_side, p_up = gb_probs[i]
            
            is_long = (p_up > config.GB_THRESHOLD and p_up > p_down)
            is_short = (p_down > config.GB_THRESHOLD and p_down > p_up)
            
            signal = None
            if is_long and regime in ['BULL', 'SIDEWAYS']:
                signal = 'LONG'
            elif is_short and regime in ['BEAR', 'SIDEWAYS']:
                signal = 'SHORT'
                
            if signal == 'LONG':
                position = 1
                entry_price = close_prices[i]
                entry_idx = i
                tp1_hit = False
                qty = (capital * config.PORTFOLIO_ALLOCATION) / entry_price
                profit_lock_stage = 0.0
            elif signal == 'SHORT':
                position = -1
                entry_price = close_prices[i]
                entry_idx = i
                tp1_hit = False
                qty = (capital * config.PORTFOLIO_ALLOCATION) / entry_price
                profit_lock_stage = 0.0
        else:
            current_close = close_prices[i]
            current_high = high_prices[i]
            current_low = low_prices[i]
            bars_held = i - entry_idx
            
            sl_hit = False
            tp1_triggered = False
            tp2_triggered = False
            time_up = False
            profit_lock_hit = False
            exit_price = current_close
            
            # ROI 계산
            if position == 1:
                max_roi = ((current_high - entry_price) / entry_price) * leverage
                current_roi = ((current_close - entry_price) / entry_price) * leverage
            else:
                max_roi = ((entry_price - current_low) / entry_price) * leverage
                current_roi = ((entry_price - current_close) / entry_price) * leverage
                
            # Profit Lock 단계 갱신
            if max_roi >= 0.04:
                profit_lock_stage = max(profit_lock_stage, 5.0)
            elif max_roi >= 0.03:
                profit_lock_stage = max(profit_lock_stage, 4.0)
            elif max_roi >= 0.02:
                profit_lock_stage = max(profit_lock_stage, 3.0)
            elif max_roi >= 0.016:
                profit_lock_stage = max(profit_lock_stage, 2.0)
            elif max_roi >= 0.012:
                profit_lock_stage = max(profit_lock_stage, 1.0)
            elif max_roi >= 0.008:
                profit_lock_stage = max(profit_lock_stage, 0.5)
                
            # Profit Lock 청산선 감시
            if profit_lock_stage == 5.0 and current_roi <= 0.030:
                profit_lock_hit = True
                exit_price = entry_price * (1.0 + 0.030/leverage) if position == 1 else entry_price * (1.0 - 0.030/leverage)
            elif profit_lock_stage == 4.0 and current_roi <= 0.022:
                profit_lock_hit = True
                exit_price = entry_price * (1.0 + 0.022/leverage) if position == 1 else entry_price * (1.0 - 0.022/leverage)
            elif profit_lock_stage == 3.0 and current_roi <= 0.014:
                profit_lock_hit = True
                exit_price = entry_price * (1.0 + 0.014/leverage) if position == 1 else entry_price * (1.0 - 0.014/leverage)
            elif profit_lock_stage == 2.0 and current_roi <= 0.011:
                profit_lock_hit = True
                exit_price = entry_price * (1.0 + 0.011/leverage) if position == 1 else entry_price * (1.0 - 0.011/leverage)
            elif profit_lock_stage == 1.0 and current_roi <= 0.008:
                profit_lock_hit = True
                exit_price = entry_price * (1.0 + 0.008/leverage) if position == 1 else entry_price * (1.0 - 0.008/leverage)
            elif profit_lock_stage == 0.5 and current_roi <= 0.005:
                profit_lock_hit = True
                exit_price = entry_price * (1.0 + 0.005/leverage) if position == 1 else entry_price * (1.0 - 0.005/leverage)
                
            if position == 1:
                sl_price = entry_price * (1.0 - sl_pct)
                tp1_price = entry_price * (1.0 + tp1_pct)
                tp2_price = entry_price * (1.0 + tp2_pct)
                
                if current_low <= sl_price:
                    sl_hit = True
                    exit_price = sl_price
                elif not tp1_hit and current_high >= tp1_price:
                    tp1_triggered = True
                elif tp1_hit and current_high >= tp2_price:
                    tp2_triggered = True
                    exit_price = tp2_price
                elif bars_held >= hold_bars:
                    time_up = True
                elif bars_held == 1 and current_close <= entry_price:
                    time_up = True
            else: # position == -1
                sl_price = entry_price * (1.0 + sl_pct)
                tp1_price = entry_price * (1.0 - tp1_pct)
                tp2_price = entry_price * (1.0 - tp2_pct)
                
                if current_high >= sl_price:
                    sl_hit = True
                    exit_price = sl_price
                elif not tp1_hit and current_low <= tp1_price:
                    tp1_triggered = True
                elif tp1_hit and current_low <= tp2_price:
                    tp2_triggered = True
                    exit_price = tp2_price
                elif bars_held >= hold_bars:
                    time_up = True
                elif bars_held == 1 and current_close >= entry_price:
                    time_up = True
                    
            # 청산 집계
            if profit_lock_hit:
                curr_qty = qty * 0.5 if tp1_hit else qty
                pnl = (exit_price - entry_price) * curr_qty if position == 1 else (entry_price - exit_price) * curr_qty
                fee = (entry_price + exit_price) * curr_qty * fee_rate
                trade_pnl = pnl - fee
                capital += trade_pnl
                trades.append({'side': 'LONG' if position == 1 else 'SHORT', 'pnl': trade_pnl, 'pnl_pct': trade_pnl / (entry_price * qty + 1e-9), 'win': 1})
                position = 0
            elif sl_hit:
                curr_qty = qty * 0.5 if tp1_hit else qty
                pnl = (exit_price - entry_price) * curr_qty if position == 1 else (entry_price - exit_price) * curr_qty
                fee = (entry_price + exit_price) * curr_qty * fee_rate
                trade_pnl = pnl - fee
                capital += trade_pnl
                trades.append({'side': 'LONG' if position == 1 else 'SHORT', 'pnl': trade_pnl, 'pnl_pct': trade_pnl / (entry_price * qty + 1e-9), 'win': 0})
                position = 0
            elif tp2_triggered:
                curr_qty = qty * 0.5 if tp1_hit else qty
                pnl = (exit_price - entry_price) * curr_qty if position == 1 else (entry_price - exit_price) * curr_qty
                fee = (entry_price + exit_price) * curr_qty * fee_rate
                trade_pnl = pnl - fee
                capital += trade_pnl
                trades.append({'side': 'LONG' if position == 1 else 'SHORT', 'pnl': trade_pnl, 'pnl_pct': trade_pnl / (entry_price * qty + 1e-9), 'win': 1})
                position = 0
            elif tp1_triggered:
                pnl1 = (tp1_price - entry_price) * (qty * 0.5) if position == 1 else (entry_price - tp1_price) * (qty * 0.5)
                fee1 = (entry_price + tp1_price) * (qty * 0.5) * fee_rate
                capital += (pnl1 - fee1)
                tp1_hit = True
            elif time_up:
                curr_qty = qty * 0.5 if tp1_hit else qty
                pnl = (exit_price - entry_price) * curr_qty if position == 1 else (entry_price - exit_price) * curr_qty
                fee = (entry_price + exit_price) * curr_qty * fee_rate
                trade_pnl = pnl - fee
                capital += trade_pnl
                win_status = 1 if (trade_pnl > 0 or tp1_hit) else 0
                trades.append({'side': 'LONG' if position == 1 else 'SHORT', 'pnl': trade_pnl, 'pnl_pct': trade_pnl / (entry_price * qty + 1e-9), 'win': win_status})
                position = 0

    if not trades:
        return {
            'long_trades': 0, 'long_win_rate': 0.0,
            'short_trades': 0, 'short_win_rate': 0.0,
            'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
            'mdd': 0.0, 'final_capital': capital
        }
        
    longs = [t for t in trades if t['side'] == 'LONG']
    shorts = [t for t in trades if t['side'] == 'SHORT']
    
    long_trades = len(longs)
    long_win_rate = sum(1 for t in longs if t['pnl'] > 0) / long_trades if long_trades > 0 else 0.0
    
    short_trades = len(shorts)
    short_win_rate = sum(1 for t in shorts if t['pnl'] > 0) / short_trades if short_trades > 0 else 0.0
    
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['win'] == 1)
    win_rate = wins / total_trades
    total_return = (capital - 1000000.0) / 1000000.0
    
    capital_history = [1000000.0]
    current = 1000000.0
    for t in trades:
        current += t['pnl']
        capital_history.append(current)
    peaks = np.maximum.accumulate(capital_history)
    drawdowns = (capital_history - peaks) / peaks
    mdd = np.min(drawdowns)
    
    return {
        'long_trades': long_trades, 'long_win_rate': long_win_rate,
        'short_trades': short_trades, 'short_win_rate': short_win_rate,
        'total_trades': total_trades, 'win_rate': win_rate, 'total_return': total_return,
        'mdd': mdd, 'final_capital': capital
    }

def run_backtest_for_symbol(symbol):
    sym_clean = symbol.replace('/', '_')
    csv_path = os.path.join(config.DATA_DIR, f"{sym_clean}.csv")
    
    gb_path = os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_unified.pkl")
    gb_regime_path = os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_regime.pkl")
    
    if not all([os.path.exists(p) for p in [csv_path, gb_path, gb_regime_path]]):
        print(f"⚠️ [{symbol}] 모델 또는 데이터가 누락되어 백테스트를 건너뜁니다.")
        return
        
    with open(gb_path, 'rb') as f: gb_model = pickle.load(f)
    with open(gb_regime_path, 'rb') as f: gb_regime_model = pickle.load(f)
        
    df_raw = pd.read_csv(csv_path)
    df = calculate_indicators(df_raw)
    
    total_len = len(df)
    val_end = int(total_len * 0.85)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    res = run_simulation(test_df, gb_model, gb_regime_model, symbol)
    
    print(f"========== [{symbol}] GradBoost 백테스트 결과 (최근 15% 검증구간) ==========")
    print(f"   - 총 거래 횟수:  {res['total_trades']:^5}회 (롱: {res['long_trades']}회 / 숏: {res['short_trades']}회)")
    print(f"   - 롱/숏 승률:    롱: {res['long_win_rate']*100:^5.1f}% | 숏: {res['short_win_rate']*100:^5.1f}%")
    print(f"   - 통합 승률:      {res['win_rate']*100:^5.1f}%")
    print(f"   - 통합 총 수익률: {res['total_return']*100:^+5.2f}%")
    print(f"   - 최대 낙폭(MDD): {res['mdd']*100:^5.1f}%")
    print(f"   - 최종 자본:      {res['final_capital']:,.0f}원")
    print(f"========================================================================\n")

def run_all_backtests():
    for symbol in config.SYMBOLS:
        run_backtest_for_symbol(symbol)

if __name__ == '__main__':
    run_all_backtests()
