import os
import ccxt
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

api_key = os.getenv('BINANCE_API_KEY') or os.getenv('BINANCE_ACCESS_KEY')
api_secret = os.getenv('BINANCE_API_SECRET') or os.getenv('BINANCE_SECRET_KEY')

if not api_key or not api_secret:
    print("❌ .env 파일에서 Binance API 키를 찾을 수 없습니다.")
    exit(1)

print("============================================================")
print("🚨 선물 포지션 전량 긴급 청산 프로그램 작동 시작")
print("============================================================")

try:
    # 바이낸스 USD-M 선물 클라이언트 초기화
    exchange = ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }
    })
    
    print("⌛ 마켓 정보를 로드하고 있습니다...")
    exchange.load_markets()
    
    print("⌛ 현재 활성화된 포지션을 조회하고 있습니다...")
    positions = exchange.fetch_positions()
    
    active_positions = []
    for pos in positions:
        # contracts(계약 수량)가 0보다 큰 경우에 활성화된 포지션으로 간주
        size = float(pos.get('contracts', 0.0) or 0.0)
        if size > 0:
            active_positions.append(pos)
            
    if not active_positions:
        print("✅ 현재 보유 중인 선물 포지션이 없습니다.")
    else:
        print(f"🔍 보유 중인 포지션 {len(active_positions)}개를 발견했습니다. 청산을 집행합니다.")
        
        for pos in active_positions:
            symbol = pos['symbol']
            side = pos['side']  # 'long' 또는 'short'
            size = float(pos['contracts'])
            
            print(f"\n⚡ [{symbol}] {side.upper()} 포지션 (수량: {size}) 청산 중...")
            
            # 롱 포지션은 매도(sell)로, 숏 포지션은 매수(buy)로 청산 주문 실행
            order_side = 'sell' if (side == 'long' or side == 'buy') else 'buy'
            qty_str = exchange.amount_to_precision(symbol, size)
            
            # 시장가 청산 주문 접수
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=order_side,
                amount=float(qty_str),
                params={'reduceOnly': True}
            )
            print(f"   ➡️ 시장가 청산 완료! 주문 ID: {order['id']}")
            
            # 해당 종목의 미체결 예약 주문(손절/익절 등) 전체 취소
            try:
                exchange.cancel_all_orders(symbol)
                print(f"   ➡️ [{symbol}]의 모든 미체결 예약 주문을 취소했습니다.")
            except Exception as e:
                print(f"   ⚠️ 예약 주문 취소 중 오류 발생: {e}")
                
except Exception as e:
    print(f"❌ 포지션 긴급 청산 중 에러 발생: {e}")

print("============================================================")
print("✅ 긴급 청산 절차가 완료되었습니다.")
print("============================================================")
