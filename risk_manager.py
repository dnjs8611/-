import os
import json
from datetime import datetime
import config

class RiskManager:
    def __init__(self):
        self.risk_per_trade = config.RISK_PER_TRADE
        self.daily_loss_limit = config.DAILY_LOSS_LIMIT
        self.max_loss_streak = config.MAX_LOSS_STREAK
        self.state_file = os.path.join(config.LOGS_DIR, 'risk_state.json')
        self.load_state()

    def load_state(self):
        self.state = {
            'today': datetime.now().strftime('%Y-%m-%d'),
            'daily_pnl_pct': 0.0,
            'consecutive_losses': 0
        }
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    if saved.get('today') == self.state['today']:
                        self.state = saved
                    else:
                        # 날짜가 바뀌었으면 일일 PnL만 리셋하고 연속 손실 횟수는 유지 (승리할 때까지 유지)
                        self.state['today'] = datetime.now().strftime('%Y-%m-%d')
                        self.state['daily_pnl_pct'] = 0.0
                        self.state['consecutive_losses'] = saved.get('consecutive_losses', 0)
            except Exception as e:
                print(f"[RiskManager] Error loading state: {e}")

    def save_state(self):
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"[RiskManager] Error saving state: {e}")

    def calc_position_size(self, capital, entry_price, symbol, is_high_confidence=False) -> float:
        """
        고정 보증금 또는 자산 대비 비율을 사용하여 포지션 수량 계산 (늘리지 않고 고정)
        """
        if getattr(config, 'USE_FIXED_MARGIN', False):
            margin_amount = config.FIXED_MARGIN_USDT
        else:
            margin_pct = config.PORTFOLIO_ALLOCATION
            margin_amount = capital * margin_pct
            
        # 보증금이 전체 자본을 초과하지 않도록 제한
        margin_amount = min(capital * 0.95, margin_amount)
        
        leverage = config.SYMBOL_LEVERAGE.get(symbol, config.LEVERAGE)
        position_value = margin_amount * leverage
        quantity = position_value / entry_price
        return quantity

    def check_daily_limit(self) -> bool:
        """
        당일 손실 한도 초과 여부 체크
        """
        # daily_pnl_pct가 -3% 이하이면 True 반환
        return self.state['daily_pnl_pct'] <= -self.daily_loss_limit

    def check_loss_streak(self) -> bool:
        """
        연속 손절 초과 여부 체크 (비활성화)
        """
        return False

    def update_trade_result(self, pnl_pct: float):
        """
        거래 결과 반영 (일일 누적 손익률 및 연속 손절 횟수 갱신)
        """
        self.state['daily_pnl_pct'] += pnl_pct
        
        if pnl_pct < 0:
            self.state['consecutive_losses'] += 1
        else:
            self.state['consecutive_losses'] = 0
            
        self.save_state()

    def reset_daily(self):
        """
        일일 제한 초기화
        """
        self.state['today'] = datetime.now().strftime('%Y-%m-%d')
        self.state['daily_pnl_pct'] = 0.0
        self.save_state()
