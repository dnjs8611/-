import os
import ccxt
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

api_key = os.getenv('BINANCE_API_KEY') or os.getenv('BINANCE_ACCESS_KEY')
api_secret = os.getenv('BINANCE_API_SECRET') or os.getenv('BINANCE_SECRET_KEY')
telegram_token = os.getenv('TELEGRAM_TOKEN')
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

print("============================================================")
print("🔑 설정된 환경 변수 정보 검사")
print("============================================================")
print(f"Binance API Key   : {api_key[:6] if api_key else 'None'}... (길이: {len(api_key) if api_key else 0})")
print(f"Binance API Secret: {api_secret[:6] if api_secret else 'None'}... (길이: {len(api_secret) if api_secret else 0})")
print(f"Telegram Token    : {telegram_token[:6] if telegram_token else 'None'}... (길이: {len(telegram_token) if telegram_token else 0})")
print(f"Telegram Chat ID  : {telegram_chat_id if telegram_chat_id else 'None'} (길이: {len(telegram_chat_id) if telegram_chat_id else 0})")
print("============================================================\n")

print("⚡ 1. Binance Futures API 연결 테스트 중...")
try:
    exchange = ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }
    })
    
    # 선물 계좌 잔고 가져오기
    balance = exchange.fetch_balance()
    usdt_balance = balance['total'].get('USDT', 0.0)
    print(f"✅ Binance Futures API 연결 성공!")
    print(f"   현재 USDT 선물 지갑 총 잔고: ${usdt_balance:,.2f} USDT")
    
except Exception as e:
    print(f"❌ Binance Futures API 연결 실패: {e}")
    print("   (팁: API 키 권한에 'Enable Futures'가 활성화되어 있는지 확인해 주세요.)")

print("\n🔔 2. Telegram 알림 연동 테스트 중...")
if telegram_token and telegram_chat_id:
    import requests
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {
        'chat_id': telegram_chat_id,
        'text': "🔔 <b>[XGB+RF 앙상블 봇]</b> API 및 텔레그램 연동 검증 완료! 봇이 실전 투입 가능 상태입니다.",
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram 테스트 메시지 전송 성공!")
        else:
            print(f"❌ Telegram 전송 실패 (HTTP {response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Telegram 전송 중 네트워크 오류: {e}")
else:
    print("⚠️ Telegram 환경 변수가 비어 있어 테스트를 건너뜁니다.")
print("============================================================")
