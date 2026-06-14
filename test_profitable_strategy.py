import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import config
from indicators import calculate_indicators

# 리샘플링 규칙 정의
RESAMPLE_RULES = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}

def build_and_backtest_profitable_strategy():
    print("============================================================")
    print("💎 수수료 극복 추세 추종 전략 (15분봉 리샘플링 + 1% 타겟) 검증 시작")
    print("============================================================\n")
    
    portfolio_results = []
    
    # 5개 종목 순회
    for symbol in config.SYMBOLS:
        sym_clean = symbol.replace('/', '_')
        csv_path = os.path.join(config.DATA_DIR, f"{sym_clean}.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        # 1. 5분봉 데이터 로드 및 15분봉 리샘플링
        df_5m = pd.read_csv(csv_path)
        df_5m['datetime'] = pd.to_datetime(df_5m['timestamp'], unit='ms')
        df_5m.set_index('datetime', inplace=True)
        
        # 15분봉으로 집계
        df_15m = df_5m.resample('15min').agg(RESAMPLE_RULES).dropna().reset_index()
        df_15m['timestamp'] = df_15m['datetime'].astype(np.int64) // 10**6
        
        # 기술 지표 계산
        df = calculate_indicators(df_15m)
        
        # 2. 1.0% 상승/하락 타겟 레이블링 (4봉 후 = 1시간 후)
        df['target_long'] = (df['close'].shift(-4) > df['close'] * 1.010).astype(int)
        df['target_short'] = (df['close'].shift(-4) < df['close'] * 0.990).astype(int)
        
        df.dropna(subset=config.FEATURES + ['target_long', 'target_short'], inplace=True)
        
        X = df[config.FEATURES]
        y_long = df['target_long'].values
        y_short = df['target_short'].values
        
        # 데이터 분할
        total_len = len(df)
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.85)
        
        X_train, y_long_train, y_short_train = X.iloc[:train_end], y_long[:train_end], y_short[:train_end]
        X_val, y_long_val, y_short_val = X.iloc[train_end:val_end], y_long[train_end:val_end], y_short[train_end:val_end]
        X_test, y_long_test, y_short_test = X.iloc[val_end:], y_long[val_end:], y_short[val_end:]
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)
        
        # XGBoost Dynamic scale_pos_weight 계산
        pos_w_l = (y_long_train == 0).sum() / (y_long_train == 1).sum() if (y_long_train == 1).sum() > 0 else 1.0
        pos_w_s = (y_short_train == 0).sum() / (y_short_train == 1).sum() if (y_short_train == 1).sum() > 0 else 1.0
        
        # 모델 학습
        xgb_l = XGBClassifier(scale_pos_weight=pos_w_l, **config.XGB_PARAMS)
        xgb_l.fit(X_train, y_long_train, eval_set=[(X_val, y_long_val)], verbose=False)
        
        rf_l = RandomForestClassifier(**config.RF_PARAMS)
        rf_l.fit(pd.concat([X_train, X_val]), np.concatenate([y_long_train, y_long_val]))
        
        xgb_s = XGBClassifier(scale_pos_weight=pos_w_s, **config.XGB_PARAMS)
        xgb_s.fit(X_train, y_short_train, eval_set=[(X_val, y_short_val)], verbose=False)
        
        rf_s = RandomForestClassifier(**config.RF_PARAMS)
        rf_s.fit(pd.concat([X_train, X_val]), np.concatenate([y_short_train, y_short_val]))
        
        # 테스트셋 예측
        xgb_lp = xgb_l.predict_proba(X_test)[:, 1]
        rf_lp = rf_l.predict_proba(X_test)[:, 1]
        xgb_sp = xgb_s.predict_proba(X_test)[:, 1]
        rf_sp = rf_s.predict_proba(X_test)[:, 1]
        
        # 백테스트 파라미터 적용 (익절가 1.0%/2.0%, 손절가 1.0%, 최대 4봉=1시간 홀딩)
        sl_pct = 0.010
        tp1_pct = 0.010
        tp2_pct = 0.020
        fee_rate = config.FEE_RATE
        
        # 앙상블 롱 단방향 시뮬레이션 (수익 극대화 모델)
        capital = 1000000.0
        position = 0
        entry_price = 0.0
        entry_idx = 0
        tp1_hit = False
        qty = 0.0
        trades = []
        
        test_close = test_df['close'].values
        test_high = test_df['high'].values
        test_low = test_df['low'].values
        
        for i in range(len(test_df)):
            if position == 0:
                xgb_val = xgb_lp[i]
                rf_val = rf_lp[i]
                
                # 롱 진입 장벽 65% 설정
                if xgb_val > 0.65 and rf_val > 0.65:
                    position = 1
                    entry_price = test_close[i]
                    entry_idx = i
                    tp1_hit = False
                    
                    risk_amount = capital * config.RISK_PER_TRADE
                    pos_value = risk_amount / sl_pct
                    qty = pos_value / entry_price
            else:
                current_close = test_close[i]
                current_high = test_high[i]
                current_low = test_low[i]
                bars_held = i - entry_idx
                
                sl_hit = False
                tp1_triggered = False
                tp2_triggered = False
                time_up = False
                
                # 롱 청산 체크
                sl_price = entry_price * (1.0 - sl_pct)
                tp1_price = entry_price * (1.0 + tp1_pct)
                tp2_price = entry_price * (1.0 + tp2_pct)
                
                if current_low <= sl_price:
                    sl_hit = True
                elif not tp1_hit and current_high >= tp1_price:
                    tp1_triggered = True
                elif tp1_hit and current_high >= tp2_price:
                    tp2_triggered = True
                elif bars_held >= 4:  # 15분봉 4개 = 1시간
                    time_up = True
                    
                if sl_hit:
                    pnl = (sl_price - entry_price) * qty
                    fee = (entry_price + sl_price) * qty * fee_rate
                    trade_pnl = pnl - fee
                    capital += trade_pnl
                    trades.append({'pnl': trade_pnl, 'win': 0})
                    position = 0
                elif tp2_triggered:
                    current_qty = qty * 0.5 if tp1_hit else qty
                    pnl = (tp2_price - entry_price) * current_qty
                    fee = (entry_price + tp2_price) * current_qty * fee_rate
                    trade_pnl = pnl - fee
                    capital += trade_pnl
                    trades.append({'pnl': trade_pnl, 'win': 1})
                    position = 0
                elif tp1_triggered:
                    pnl1 = (tp1_price - entry_price) * (qty * 0.5)
                    fee1 = (entry_price + tp1_price) * (qty * 0.5) * fee_rate
                    capital += (pnl1 - fee1)
                    tp1_hit = True
                elif time_up:
                    current_qty = qty * 0.5 if tp1_hit else qty
                    pnl = (current_close - entry_price) * current_qty
                    fee = (entry_price + current_close) * current_qty * fee_rate
                    trade_pnl = pnl - fee
                    capital += trade_pnl
                    trades.append({'pnl': trade_pnl, 'win': 1 if (trade_pnl > 0 or tp1_hit) else 0})
                    position = 0
                    
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        total_return = (capital - 1000000.0) / 1000000.0
        
        # MDD 계산
        capital_history = [1000000.0]
        curr = 1000000.0
        for t in trades:
            curr += t['pnl']
            capital_history.append(curr)
        peaks = np.maximum.accumulate(capital_history)
        drawdowns = (capital_history - peaks) / peaks
        mdd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        portfolio_results.append({
            'symbol': symbol,
            'trades': total_trades,
            'win_rate': win_rate,
            'return': total_return,
            'mdd': mdd
        })
        
        print(f"[{symbol}] 백테스트 완료")
        print(f"  - 거래수: {total_trades}회 | 승률: {win_rate*100:.1f}% | 수익률: {total_return*100:+.2f}% | MDD: {mdd*100:.1f}%\n")
        
    print("============================================================")
    print("📊 최종 포트폴리오 결과 요약")
    print("============================================================")
    avg_trades = np.mean([r['trades'] for r in portfolio_results])
    avg_win = np.mean([r['win_rate'] for r in portfolio_results])
    avg_ret = np.mean([r['return'] for r in portfolio_results])
    avg_mdd = np.mean([r['mdd'] for r in portfolio_results])
    
    print(f"평균 거래 횟수: {avg_trades:.1f}회")
    print(f"평균 승률     : {avg_win*100:.1f}%")
    print(f"평균 포트폴리오 수익률: {avg_ret*100:+.2f}%")
    print(f"평균 MDD      : {avg_mdd*100:.1f}%")
    print("============================================================\n")

if __name__ == '__main__':
    build_and_backtest_profitable_strategy()
