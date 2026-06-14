import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import requests
import config

def send_telegram_message(text):
    """
    공통 텔레그램 메시지 발송 함수 (requests 동기 방식)
    """
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        print(f"[Notifier] Telegram config missing or disabled. Msg: {text}")
        return False
        
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': config.TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': 'HTML'  # 마크다운보다 HTML이 기호 오류(특히 / 나 _)가 발생하지 않아 안전함
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"[Notifier] Failed to send Telegram: {response.text}")
            return False
    except Exception as e:
        print(f"[Notifier] Telegram network exception: {e}")
        return False

def send_entry(symbol, side, xgb_prob, rf_prob, ensemble_prob, price, quantity, leverage=1):
    """
    진입 알림
    """
    emoji = "🟢" if side == "LONG" else "🔴"
    position_value = price * quantity
    margin_usdt = position_value / leverage
    text = (
        f"<b>{emoji} [{symbol}] {side} 진입 ({leverage}x)</b>\n"
        f"   신뢰도(XGB): {xgb_prob*100:.1f}%\n"
        f"   진입가: ${price:,.4f} | 수량: {quantity:.4f}\n"
        f"   포지션 규모: ${position_value:,.2f} | 진입 보증금: ${margin_usdt:,.2f}"
    )
    return send_telegram_message(text)

def send_exit(symbol, side, entry_price, exit_price, pnl_pct, reason, pnl_usdt=0.0, leverage=1):
    """
    청산 알림
    """
    # 이모지 설정
    if "익절" in reason or pnl_pct > 0:
        emoji = "✅"
    elif "손절" in reason or pnl_pct < 0:
        emoji = "🔴"
    else:
        emoji = "⏱" if "시간" in reason else "ℹ️"
        
    pnl_sign = "+" if pnl_pct >= 0 else ""
    pnl_usdt_sign = "+" if pnl_usdt >= 0 else ""
    
    leveraged_pnl_pct = pnl_pct * leverage
    
    text = (
        f"<b>{emoji} [{symbol}] {reason} ({leverage}x)</b>\n"
        f"   레버리지 수익률: <b>{pnl_sign}{leveraged_pnl_pct*100:.2f}%</b> (기본: {pnl_sign}{pnl_pct*100:.2f}%)\n"
        f"   수익금: <b>{pnl_usdt_sign}${pnl_usdt:,.2f}</b>\n"
        f"   진입가: ${entry_price:,.4f} → 청산가: ${exit_price:,.4f}"
    )
    return send_telegram_message(text)

def send_daily_summary(trades_count, win_rate, total_pnl_usdt):
    """
    일일 요약 알림
    """
    pnl_sign = "+" if total_pnl_usdt >= 0 else ""
    text = (
        f"<b>📊 일일 요약</b>\n"
        f"   거래 횟수: {trades_count}회\n"
        f"   승률: {win_rate*100:.1f}%\n"
        f"   당일 손익: {pnl_sign}${total_pnl_usdt:,.2f}"
    )
    return send_telegram_message(text)

def send_retrain_result(symbol, old_xgb, new_xgb, action, old_rf=None, new_rf=None):
    """
    재학습 결과 알림
    """
    text = (
        f"<b>🔄 [{symbol}] 재학습 완료</b>\n"
        f"   XGB: {old_xgb*100:.1f}% → {new_xgb*100:.1f}%\n"
        f"   결과: <b>{action}</b>"
    )
    return send_telegram_message(text)

def send_alert(message):
    """
    긴급 시스템 경고
    """
    text = f"<b>⚠️ [긴급 시스템 알림]</b>\n{message}"
    return send_telegram_message(text)
