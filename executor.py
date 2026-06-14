import ccxt
import config

class Executor:
    def __init__(self):
        # ccxt binanceusdm 객체 초기화
        self.exchange = ccxt.binanceusdm({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        self.markets = None
        self.init_markets()

    def init_markets(self):
        try:
            self.markets = self.exchange.load_markets()
            print("[Executor] Binance Futures markets loaded successfully.")
        except Exception as e:
            print(f"[Executor] Error loading markets: {e}")

    def set_leverage(self, symbol, leverage):
        """
        레버리지 설정 API 호출
        """
        try:
            res = self.exchange.set_leverage(leverage, symbol)
            print(f"[Executor] Set leverage to {leverage}x for {symbol} successfully.")
            return res
        except Exception as e:
            print(f"[Executor] Error setting leverage to {leverage} for {symbol}: {e}")
            raise e

    def open_long(self, symbol, quantity):
        """
        롱 포지션 진입 (시장가 매수)
        """
        try:
            if not self.markets:
                self.init_markets()
            
            qty_str = self.exchange.amount_to_precision(symbol, quantity)
            order = self.exchange.create_market_buy_order(
                symbol=symbol,
                amount=float(qty_str)
            )
            print(f"[Executor] Open LONG order executed for {symbol}. Qty: {qty_str}")
            return order
        except Exception as e:
            print(f"[Executor] Error opening LONG for {symbol}: {e}")
            raise e

    def open_short(self, symbol, quantity):
        """
        숏 포지션 진입 (시장가 매도)
        """
        try:
            if not self.markets:
                self.init_markets()
                
            qty_str = self.exchange.amount_to_precision(symbol, quantity)
            order = self.exchange.create_market_sell_order(
                symbol=symbol,
                amount=float(qty_str)
            )
            print(f"[Executor] Open SHORT order executed for {symbol}. Qty: {qty_str}")
            return order
        except Exception as e:
            print(f"[Executor] Error opening SHORT for {symbol}: {e}")
            raise e

    def close_position(self, symbol, quantity, side):
        """
        포지션 청산 (시장가)
        side: 'LONG' (롱포지션 청산 -> 매도) 또는 'SHORT' (숏포지션 청산 -> 매수)
        """
        try:
            if not self.markets:
                self.init_markets()
                
            qty_str = self.exchange.amount_to_precision(symbol, quantity)
            order_side = 'sell' if side == 'LONG' else 'buy'
            
            params = {'reduceOnly': True}
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=order_side,
                amount=float(qty_str),
                price=None,
                params=params
            )
            print(f"[Executor] Close {side} position order executed for {symbol}. Qty: {qty_str}")
            return order
        except Exception as e:
            print(f"[Executor] Error closing position for {symbol} ({side}): {e}")
            raise e

    def get_position(self, symbol) -> dict:
        """
        현재 포지션 정보 조회
        반환 형식:
        {
            'size': float (양수: 롱, 음수: 숏, 0: 없음),
            'entry_price': float,
            'unrealized_pnl': float,
            'side': 'LONG' / 'SHORT' / None,
            'liquidation_price': float
        }
        """
        try:
            positions = self.exchange.fetch_positions(symbols=[symbol])
            if not positions:
                return {
                    'size': 0.0,
                    'entry_price': 0.0,
                    'unrealized_pnl': 0.0,
                    'side': None,
                    'liquidation_price': 0.0
                }
            pos = positions[0]
            
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
                    size = -size # 숏은 내부적으로 음수로 표시
            
            return {
                'size': size,
                'entry_price': entry_price,
                'unrealized_pnl': unrealized_pnl,
                'side': side,
                'liquidation_price': liquidation_price
            }
        except Exception as e:
            print(f"[Executor] Error fetching position for {symbol}: {e}")
            raise e

    def set_stop_loss(self, symbol, side, stop_price, quantity=None):
        """
        손절가 설정 (STOP_MARKET 주문)
        side: 'LONG' (롱포지션 손절 -> 매도) 또는 'SHORT' (숏포지션 손절 -> 매수)
        """
        try:
            if not self.markets:
                self.init_markets()
                
            # 수량이 명시되지 않으면 현재 포지션 크기만큼 설정
            if quantity is None:
                pos = self.get_position(symbol)
                qty = abs(pos['size'])
            else:
                qty = quantity
                
            if qty == 0:
                print(f"[Executor] Zero quantity for Stop Loss on {symbol}, order skipped.")
                return None
                
            order_side = 'sell' if side == 'LONG' else 'buy'
            qty_str = self.exchange.amount_to_precision(symbol, qty)
            price_str = self.exchange.price_to_precision(symbol, stop_price)
            
            params = {
                'stopPrice': price_str,
                'reduceOnly': True
            }
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='stop_market',
                side=order_side,
                amount=float(qty_str),
                price=None,
                params=params
            )
            print(f"[Executor] Set Stop Loss for {symbol} ({side}) at {price_str}. Qty: {qty_str}")
            return order
        except Exception as e:
            print(f"[Executor] Error setting Stop Loss for {symbol}: {e}")
            raise e

    def set_take_profit(self, symbol, side, tp_price, quantity=None):
        """
        익절가 설정 (TAKE_PROFIT_MARKET 주문)
        side: 'LONG' (롱포지션 익절 -> 매도) 또는 'SHORT' (숏포지션 익절 -> 매수)
        """
        try:
            if not self.markets:
                self.init_markets()
                
            if quantity is None:
                pos = self.get_position(symbol)
                qty = abs(pos['size'])
            else:
                qty = quantity
                
            if qty == 0:
                print(f"[Executor] Zero quantity for Take Profit on {symbol}, order skipped.")
                return None
                
            order_side = 'sell' if side == 'LONG' else 'buy'
            qty_str = self.exchange.amount_to_precision(symbol, qty)
            price_str = self.exchange.price_to_precision(symbol, tp_price)
            
            params = {
                'stopPrice': price_str,
                'reduceOnly': True
            }
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='take_profit_market',
                side=order_side,
                amount=float(qty_str),
                price=None,
                params=params
            )
            print(f"[Executor] Set Take Profit for {symbol} ({side}) at {price_str}. Qty: {qty_str}")
            return order
        except Exception as e:
            print(f"[Executor] Error setting Take Profit for {symbol}: {e}")
            raise e

    def cancel_all_orders(self, symbol):
        """
        미체결 주문(일반 대기 주문 및 스탑로스/익절 알고리즘 주문) 전체 취소
        """
        try:
            orders = self.exchange.cancel_all_orders(symbol)
            print(f"[Executor] Canceled all open regular orders for {symbol}.")
        except Exception as e:
            print(f"[Executor] Error canceling regular orders for {symbol}: {e}")
            raise e

        try:
            # 바이낸스 선물 알고리즘 오픈 주문(Stop Market, Take Profit 등) 조회 후 해당 심볼의 주문 삭제
            algos = self.exchange.fapiPrivateGetOpenAlgoOrders()
            binance_symbol = symbol.replace('/', '')
            for a in algos:
                if a.get('symbol') == binance_symbol:
                    algo_id = a.get('algoId')
                    if algo_id:
                        self.exchange.fapiPrivateDeleteAlgoOrder({'algoId': algo_id})
                        print(f"[Executor] Canceled algo order {algo_id} for {symbol}.")
        except Exception as e:
            print(f"[Executor] Error canceling algo orders for {symbol}: {e}")
            raise e

        return orders

    def get_balance(self) -> float:
        """
        선물 지갑의 가용 USDT 잔고 조회
        """
        try:
            bal = self.exchange.fetch_balance()
            # binance usd-m futures balance has USDT
            usdt_bal = float(bal['total'].get('USDT', 0.0))
            return usdt_bal
        except Exception as e:
            print(f"[Executor] Error fetching USDT balance: {e}")
            raise e
