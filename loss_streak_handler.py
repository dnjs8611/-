import os
import json
import threading
from datetime import datetime
import config
import notifier

def check_and_handle_loss_streak(risk_manager, trade_logger, auto_retrainer):
    consecutive_losses = risk_manager.state.get('consecutive_losses', 0)
    print(f"[LossStreakHandler] Current consecutive losses: {consecutive_losses}")
    
    # 최근 1개 거래 확인하여 패배(손실) 여부 판정
    recent_trades_1 = trade_logger.get_recent_trades(1)
    if not recent_trades_1:
        return
        
    last_trade = recent_trades_1[0]
    last_symbol = last_trade.get('symbol')
    last_pnl_pct = last_trade.get('pnl_pct', 0.0)
    is_last_trade_loss = last_pnl_pct < 0.0
    
    # 1. 3회 연속 손절 시 전체 종목 모델 백그라운드 재학습
    if consecutive_losses >= 3:
        print("[LossStreakHandler] 3 consecutive losses detected! Creating incorrect answers note and retraining all symbols...")
        recent_trades_3 = trade_logger.get_recent_trades(3)
        
        try:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file_now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            note_filename = f"오답노트_3연속손절_{file_now_str}.md"
            note_path = os.path.join(config.LOGS_DIR, note_filename)
            
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(f"# 📝 3회 연속 손절 오답노트 ({now_str})\n\n")
                f.write("> [!WARNING]\n")
                f.write("> 최근 3회 연속 손실이 발생하여 자동으로 작성된 분석 오답노트입니다.\n")
                f.write("> 이 알림과 동시에 전체 모델의 자동 재학습이 시작되었습니다.\n\n")
                
                f.write("## 📊 최근 손절 거래 내역\n\n")
                f.write("| ID | 종목 | 구분 (Side) | 진입시간 | 청산시간 | 진입가 | 청산가 | 손익 (USDT) | 손익률 | 청산사유 |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                
                for t in recent_trades_3:
                    pnl_usdt = t.get('pnl_usdt', 0.0)
                    pnl_pct = t.get('pnl_pct', 0.0)
                    f.write(f"| {t.get('id')} | {t.get('symbol')} | {t.get('side')} | {t.get('entry_time')} | {t.get('exit_time')} | {t.get('entry_price')} | {t.get('exit_price')} | {pnl_usdt:.4f} | {pnl_pct*100:.2f}% | {t.get('exit_reason')} |\n")
                
                f.write("\n## 🔍 오답 요약 및 진단\n\n")
                f.write("1. **손실 요인**: 가격이 진입 방향과 다르게 급격히 움직였거나, 손실 청산 선에 걸려 청산되었습니다.\n")
                f.write("2. **수수료 비중**: 15분봉 단기 매매 특성상 진입 횟수가 빈번하며, 슬리피지 및 거래소 수수료가 손익에 직접적인 영향을 줄 수 있습니다.\n")
                f.write("3. **장세 평가**: 앙상블 모델의 신호가 급변하는 시장 장세(예: 횡보장 횡보 구간에서 급등락)에 과적합되어 예측 오차가 발생했을 가능성이 높습니다.\n\n")
                f.write("## 🔄 대응 조치 (전체 자동 재학습)\n\n")
                f.write("* 3회 연속 손절 트리거가 작동하여 **전체 종목에 대한 모델 백그라운드 재학습**을 수행합니다.\n")
                f.write("* 재학습 완료 시 성능이 우수한 모델로 자동 갱신됩니다.\n")
                
            print(f"[LossStreakHandler] Incorrect answers note written to {note_path}")
            
            symbols_str = ", ".join([t.get('symbol', '') for t in recent_trades_3])
            alert_msg = f"⚠️ [ALERT] 3회 연속 손절 발생 ({symbols_str})!\n\n오답노트 작성 완료: {note_filename}\n백그라운드 전체 모델 자동 재학습을 즉시 개시합니다."
            notifier.send_alert(alert_msg)
            
        except Exception as e:
            print(f"[LossStreakHandler] Error writing incorrect answers note: {e}")
            
        try:
            threading.Thread(target=auto_retrainer.run_retrain, args=(config.SYMBOLS,), daemon=True).start()
        except Exception as e:
            print(f"[LossStreakHandler] Error starting retrain thread: {e}")

    # 2. 1회 손절 시 해당 종목만 백그라운드 재학습 (3연패인 경우 전체 재학습이 돌므로 제외)
    elif is_last_trade_loss:
        print(f"[LossStreakHandler] Trade loss detected for {last_symbol}! Triggering single-symbol retraining...")
        try:
            alert_msg = f"⚠️ [ALERT] {last_symbol} 거래 손실 발생!\n\n해당 종목만 백그라운드 모델 재학습을 개시합니다. 학습 완료 전까지 해당 종목의 진입이 제한됩니다."
            notifier.send_alert(alert_msg)
            
            # 해당 종목만 재학습 시작
            threading.Thread(target=auto_retrainer.run_retrain, args=([last_symbol],), daemon=True).start()
        except Exception as e:
            print(f"[LossStreakHandler] Error starting single-symbol retrain thread for {last_symbol}: {e}")
