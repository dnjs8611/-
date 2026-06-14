import os
import json
import csv
from datetime import datetime, timedelta
import pandas as pd
import config

class TradeLogger:
    def __init__(self):
        self.csv_file = os.path.join(config.LOGS_DIR, 'trades.csv')
        self.json_file = os.path.join(config.LOGS_DIR, 'trades.json')
        self.summary_file = os.path.join(config.LOGS_DIR, 'daily_summary.json')
        self.equity_file = os.path.join(config.LOGS_DIR, 'equity_curve.json')
        self._init_files()

    def _init_files(self):
        # CSV 파일 헤더 생성
        if not os.path.exists(self.csv_file):
            headers = [
                'id', 'symbol', 'side', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'quantity', 'pnl_usdt', 'pnl_pct',
                'exit_reason', 'xgb_prob', 'rf_prob', 'ensemble_prob',
                'hold_minutes', 'fee_usdt', 'leverage'
            ]
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        # JSON 파일 생성
        if not os.path.exists(self.json_file):
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def log_trade(self, trade: dict):
        """
        거래 기록 로깅 (CSV 및 JSON 저장 + 데일리 서머리 & 에퀴티 갱신)
        """
        # 1. CSV 저장
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.get('id', ''),
                    trade.get('symbol', ''),
                    trade.get('side', ''),
                    trade.get('entry_time', ''),
                    trade.get('exit_time', ''),
                    trade.get('entry_price', 0.0),
                    trade.get('exit_price', 0.0),
                    trade.get('quantity', 0.0),
                    trade.get('pnl_usdt', 0.0),
                    trade.get('pnl_pct', 0.0),
                    trade.get('exit_reason', ''),
                    trade.get('xgb_prob', 0.0),
                    trade.get('rf_prob', 0.0),
                    trade.get('ensemble_prob', 0.0),
                    trade.get('hold_minutes', 0.0),
                    trade.get('fee_usdt', 0.0),
                    trade.get('leverage', 1)
                ])
        except Exception as e:
            print(f"[TradeLogger] CSV log write error: {e}")

        # 2. JSON 저장
        trades = []
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                trades = json.load(f)
        except Exception:
            trades = []

        trades.append(trade)
        
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(trades, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[TradeLogger] JSON log write error: {e}")

        # 3. 요약 및 자산 곡선 업데이트
        self._update_daily_summary()
        self._update_equity_curve()

    def _update_daily_summary(self):
        """
        오늘 하루의 거래 요약 파일 생성/업데이트
        """
        trades = self.get_recent_trades(1000) # 최근 거래를 가져와서 오늘 거래 필터링
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        today_trades = []
        for t in trades:
            exit_time_str = t.get('exit_time', '')
            if exit_time_str.startswith(today_str):
                today_trades.append(t)
                
        trades_count = len(today_trades)
        wins = sum(1 for t in today_trades if t.get('pnl_usdt', 0.0) > 0)
        win_rate = wins / trades_count if trades_count > 0 else 0.0
        total_pnl = sum(t.get('pnl_usdt', 0.0) for t in today_trades)
        
        summary = {
            'date': today_str,
            'trades_count': trades_count,
            'win_rate': win_rate,
            'total_pnl_usdt': total_pnl
        }
        
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4)
        except Exception as e:
            print(f"[TradeLogger] Summary write error: {e}")

    def _update_equity_curve(self):
        """
        누적 손익 차트용 데이터 갱신
        """
        try:
            if not os.path.exists(self.json_file):
                return
            
            with open(self.json_file, 'r', encoding='utf-8') as f:
                trades = json.load(f)
                
            if not trades:
                with open(self.equity_file, 'w', encoding='utf-8') as ef:
                    json.dump([], ef)
                return

            # 날짜별 PnL 누적 계산
            df = pd.DataFrame(trades)
            df['exit_date'] = pd.to_datetime(df['exit_time']).dt.date
            daily_pnl = df.groupby('exit_date')['pnl_usdt'].sum().reset_index()
            daily_pnl['cumulative_pnl'] = daily_pnl['pnl_usdt'].cumsum()
            
            # JSON 포맷으로 저장
            equity_curve = []
            for _, row in daily_pnl.iterrows():
                equity_curve.append({
                    'date': str(row['exit_date']),
                    'pnl': float(row['pnl_usdt']),
                    'cumulative_pnl': float(row['cumulative_pnl'])
                })
                
            with open(self.equity_file, 'w', encoding='utf-8') as ef:
                json.dump(equity_curve, ef, indent=4)
        except Exception as e:
            print(f"[TradeLogger] Equity curve update error: {e}")

    def get_daily_summary(self) -> dict:
        """
        현재 누적 일일 요약본 반환
        """
        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trades_count': 0,
            'win_rate': 0.0,
            'total_pnl_usdt': 0.0
        }

    def get_equity_curve(self, days=30) -> list:
        """
        누적 자산곡선 데이터 반환
        """
        if os.path.exists(self.equity_file):
            try:
                with open(self.equity_file, 'r', encoding='utf-8') as f:
                    curve = json.load(f)
                    # 최근 days 일수만큼 잘라서 리턴
                    cutoff_date = datetime.now() - timedelta(days=days)
                    filtered_curve = [c for c in curve if datetime.strptime(c['date'], '%Y-%m-%d') >= cutoff_date]
                    return filtered_curve
            except Exception:
                pass
        return []

    def get_recent_trades(self, n=20) -> list:
        """
        최근 n건 거래 내역 반환
        """
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
                    return trades[-n:]
            except Exception:
                pass
        return []

    def get_model_performance(self) -> dict:
        """
        전체 통계 분석 및 모델별 승률 기여도 계산
        """
        try:
            if not os.path.exists(self.json_file):
                return {}
            with open(self.json_file, 'r', encoding='utf-8') as f:
                trades = json.load(f)
            if not trades:
                return {}
                
            df = pd.DataFrame(trades)
            total_trades = len(df)
            wins = len(df[df['pnl_usdt'] > 0])
            win_rate = wins / total_trades if total_trades > 0 else 0.0
            
            # 앙상블 평균 예측 확률 통계
            avg_xgb = float(df['xgb_prob'].mean()) if 'xgb_prob' in df.columns else 0.0
            avg_rf = float(df['rf_prob'].mean()) if 'rf_prob' in df.columns else 0.0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_xgb_prob': avg_xgb,
                'avg_rf_prob': avg_rf
            }
        except Exception as e:
            print(f"[TradeLogger] Error reading model performance: {e}")
            return {}
