"""
fix_pending.py - PENDING 상태로 멈춰있는 예측들을 강제 확정시킵니다.
"""
import sys, os, time, json
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except:
        pass

sys.path.insert(0, os.path.dirname(__file__))
import config
import data_manager
import ccxt

def get_exchange():
    return ccxt.binanceusdm({
        'apiKey': config.API_KEY,
        'secret': config.API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
    })

def calculate_pnl(pred_side, entry_price, high_price, low_price, close_price, sl_pct=0.008):
    stages = [
        (0.0020, 0.0010), (0.0030, 0.0015), (0.0040, 0.0020),
        (0.0050, 0.0025), (0.0060, 0.0030), (0.0080, 0.0040), (0.0100, 0.0050)
    ]
    if pred_side == 'LONG':
        max_move = (high_price - entry_price) / entry_price
        max_loss = (entry_price - low_price) / entry_price
    elif pred_side == 'SHORT':
        max_move = (entry_price - low_price) / entry_price
        max_loss = (high_price - entry_price) / entry_price
    else:
        return False, 0.0, "PASS"

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
            if pred_side == 'LONG':
                close_ret = (close_price - entry_price) / entry_price
            else:
                close_ret = (entry_price - close_price) / entry_price
            final_ret = max(locked_ret, close_ret)
            exit_type = f"PROFIT_LOCK_STAGE_{stage_num}" if final_ret == locked_ret else "CANDLE_CLOSE"
            return True, final_ret, exit_type
        else:
            if pred_side == 'LONG':
                close_ret = (close_price - entry_price) / entry_price
            else:
                close_ret = (entry_price - close_price) / entry_price
            return close_ret > 0, close_ret, "CANDLE_CLOSE"

def main():
    exchange = get_exchange()
    preds = data_manager.db_load_predictions()
    pending = [p for p in preds if p.get('status') == 'PENDING']
    
    print(f"[fix_pending] PENDING 예측 {len(pending)}개 발견")
    if not pending:
        print("[fix_pending] 처리할 항목 없음.")
        return

    from collections import defaultdict
    by_symbol = defaultdict(list)
    for p in pending:
        by_symbol[p['symbol']].append(p)

    for symbol, sym_preds in by_symbol.items():
        print(f"\n[{symbol}] {len(sym_preds)}개 처리 중...")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=100)
            # Build map: all candles except last (incomplete)
            candles_map = {
                int(c[0]): {'high': float(c[2]), 'low': float(c[3]), 'close': float(c[4])}
                for c in ohlcv[:-1]
            }
            now_ms = int(time.time() * 1000)

            for p in sym_preds:
                target_ts = int(p['target_time'])
                entry_price = float(p['entry_price'])
                margin_krw = float(p.get('entry_margin_krw', 30000.0))
                exchange_rate = 1350.0
                margin_usdt = float(p.get('entry_margin_usdt', margin_krw / exchange_rate))
                sl_pct = config.STOP_LOSS.get(symbol, 0.008)

                # Find matching candle - exact or closest within 30 min
                if target_ts not in candles_map:
                    closest_ts = min(candles_map.keys(), key=lambda k: abs(k - target_ts), default=None)
                    if closest_ts and abs(closest_ts - target_ts) < 30 * 60 * 1000:
                        print(f"  → target_ts 보정: {target_ts} → {closest_ts}")
                        target_ts = closest_ts
                    else:
                        # Use current price as fallback
                        ticker = exchange.fetch_ticker(symbol)
                        cur_price = float(ticker['last'])
                        candles_map[target_ts] = {'high': cur_price, 'low': cur_price, 'close': cur_price}
                        print(f"  → 캔들 없음, 현재가 {cur_price:.4f} 사용")

                if target_ts not in candles_map:
                    print(f"  → [{symbol}] 캔들 찾기 실패, 스킵")
                    continue

                candle = candles_map[target_ts]
                high_price = candle['high']
                low_price = candle['low']
                close_price = candle['close']

                p['actual_price'] = close_price
                p['status'] = 'COMPLETED'

                for m in ['ensemble', 'xgb', 'rf', 'lgb', 'cat', 'et', 'gb', 'mlp', 'svm', 'gb_basic', 'gb_current']:
                    side_k = 'predicted_side' if m == 'ensemble' else f'{m}_predicted_side'
                    res_k = 'result' if m == 'ensemble' else f'{m}_result'
                    pnl_usdt_k = 'pnl_usdt' if m == 'ensemble' else f'{m}_pnl_usdt'
                    pnl_krw_k = 'pnl_krw' if m == 'ensemble' else f'{m}_pnl_krw'
                    pred_side = p.get(side_k, 'PASS')

                    if pred_side == 'PASS':
                        p[res_k] = 'PASS'
                        p[pnl_usdt_k] = 0.0
                        p[pnl_krw_k] = 0.0
                    else:
                        m_win, m_ret, m_exit = calculate_pnl(pred_side, entry_price, high_price, low_price, close_price, sl_pct)
                        net_ret = m_ret - 0.0008
                        pnl_usdt = margin_usdt * net_ret * 5
                        pnl_krw = margin_krw * net_ret * 5
                        p[res_k] = 'WIN' if m_win else 'LOSS'
                        p[pnl_usdt_k] = round(pnl_usdt, 4)
                        p[pnl_krw_k] = round(pnl_krw, 2)

                basic_side = p.get('gb_basic_predicted_side', 'PASS')
                current_side = p.get('gb_current_predicted_side', 'PASS')
                basic_res = p.get('gb_basic_result', '-')
                current_res = p.get('gb_current_result', '-')
                basic_pnl = p.get('gb_basic_pnl_krw', 0.0)
                current_pnl = p.get('gb_current_pnl_krw', 0.0)
                print(f"  ✅ [{symbol}] {p['predict_time']} | 기본: {basic_side}→{basic_res} ({basic_pnl:+,.0f}원) | 현재: {current_side}→{current_res} ({current_pnl:+,.0f}원) | 진입: {entry_price:.2f} → 결과: {close_price:.2f}")

                data_manager.db_save_prediction(p)

        except Exception as e:
            print(f"  [Error] [{symbol}] 처리 중 에러: {e}")

    print("\n[fix_pending] 완료! 대시보드를 새로고침하세요.")

if __name__ == '__main__':
    main()
