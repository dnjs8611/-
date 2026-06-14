import os
import json
import pickle
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import config
import notifier
from trade_logger import TradeLogger

class AutoRetrainer:
    def __init__(self):
        self.logger = TradeLogger()
        self.metadata_file = os.path.join(config.MODELS_DIR, 'model_metadata.json')
        self._init_metadata()
        self.is_training = False
        self.training_symbols = set()
        self.queue = []
        self.lock = threading.Lock()
        threading.Thread(target=self._process_queue, daemon=True).start()

    def _init_metadata(self):
        if not os.path.exists(self.metadata_file):
            initial_metadata = {}
            for s in config.SYMBOLS:
                initial_metadata[s] = {
                    "last_trained": "2026-05-01 00:00:00",
                    "xgb_long_accuracy": 0.500,
                    "xgb_short_accuracy": 0.500,
                    
                    # 하위 호환성용 평균값
                    "xgb_accuracy": 0.500,
                    "ensemble_accuracy": 0.500,
                    
                    "train_data_days": 180,
                    "train_data_count": 0,
                    "recent_win_rate": 0.500,
                    "trades_since_train": 0,
                    "retrain_count": 0,
                    "xgb_top_features": [],
                    "status": "initial"
                }
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(initial_metadata, f, indent=4)

    def load_metadata(self) -> dict:
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            self._init_metadata()
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    def save_metadata(self, metadata: dict):
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"[AutoRetrainer] Error saving metadata: {e}")

    def check_retrain_needed(self) -> bool:
        """
        재학습 필요 여부 검사
        1. 날짜 기반: 매월 1일 자정 (이번 달 재학습이 안 되어있을 때)
        2. 성능 저하: 최근 50거래 실제 승률이 기존 학습 앙상블 정확도보다 5%p 이상 낮을 때
        3. 시장 급변: BTC 7일 변동성이 30일 평균의 2배 초과 시
        4. 3시간 주기 자동 재학습: 마지막 학습 시점으로부터 3시간 경과 시
        """
        if self.is_training:
            return False

        now = datetime.now()
        metadata = self.load_metadata()

        # 조건 4 — 3시간 주기 자동 재학습
        btc_meta = metadata.get('BTC/USDT', {})
        last_trained_str = btc_meta.get('last_trained', '')
        if last_trained_str:
            try:
                last_trained = datetime.strptime(last_trained_str, '%Y-%m-%d %H:%M:%S')
                if (now - last_trained).total_seconds() >= 3 * 3600:
                    print(f"[AutoRetrainer] Trigger: 3-Hour Periodic Retraining. Last trained: {last_trained_str}")
                    return True
            except Exception as e:
                print(f"[AutoRetrainer] Time parsing error in 3-hour check: {e}")
                return True
        else:
            return True

        # 조건 1 — 날짜 기반
        if now.day == config.RETRAIN_DAY:
            btc_meta = metadata.get('BTC/USDT', {})
            last_trained_str = btc_meta.get('last_trained', '')
            try:
                last_trained = datetime.strptime(last_trained_str, '%Y-%m-%d %H:%M:%S')
                if last_trained.month != now.month or last_trained.year != now.year:
                    print("[AutoRetrainer] Trigger: Monthly Retrain Day.")
                    return True
            except Exception:
                return True

        # 조건 2 — 성능 저하 감지
        trades = self.logger.get_recent_trades(1000)
        if len(trades) >= 50:
            for symbol in config.SYMBOLS:
                symbol_trades = [t for t in trades if t.get('symbol') == symbol][-50:]
                if len(symbol_trades) >= 20:
                    wins = sum(1 for t in symbol_trades if t.get('pnl_usdt', 0.0) > 0)
                    recent_win_rate = wins / len(symbol_trades)
                    
                    symbol_meta = metadata.get(symbol, {})
                    ensemble_acc = symbol_meta.get('ensemble_accuracy', 0.55)
                    
                    if (ensemble_acc - recent_win_rate) >= config.RETRAIN_TRIGGER_DROP:
                        print(f"[AutoRetrainer] Trigger: Performance Drop for {symbol} ({ensemble_acc:.3f} -> {recent_win_rate:.3f})")
                        return True

        # 조건 3 — 시장 급변 감지 (BTC 7일 변동성 급증)
        btc_csv = os.path.join(config.DATA_DIR, 'BTC_USDT.csv')
        if os.path.exists(btc_csv):
            try:
                df = pd.read_csv(btc_csv)
                if len(df) >= 8640:
                    close_pct = df['close'].pct_change().dropna()
                    vol_7d = close_pct.tail(2016).std()
                    vol_30d = close_pct.tail(8640).std()
                    
                    if vol_7d > 2.0 * vol_30d:
                        print(f"[AutoRetrainer] Trigger: Market Volatility Spike. 7D Std: {vol_7d:.6f} | 30D Std: {vol_30d:.6f}")
                        return True
            except Exception as e:
                print(f"[AutoRetrainer] Volatility check error: {e}")

        return False

    def is_symbol_training(self, symbol: str) -> bool:
        with self.lock:
            return symbol in self.training_symbols

    def run_retrain(self, symbols: list):
        self.trigger_retrain(symbols)

    def trigger_retrain(self, symbols: list):
        with self.lock:
            self.is_training = True
            # 5연패 또는 3시간 주기인 경우 전체 symbols를 큐에 넣거나 기존 큐를 대체
            if set(symbols) == set(config.SYMBOLS):
                self.queue = [config.SYMBOLS]
                for sym in config.SYMBOLS:
                    self.training_symbols.add(sym)
            else:
                # 개별 재학습인 경우 큐에 추가
                for sym in symbols:
                    if sym not in self.training_symbols and not any(sym in item for item in self.queue if isinstance(item, list)):
                        self.queue.append([sym])
                        self.training_symbols.add(sym)
        print(f"[AutoRetrainer] Added retrain request to queue for: {symbols}. Current queue size: {len(self.queue)}")

    def _process_queue(self):
        import time
        while True:
            symbols_to_train = None
            with self.lock:
                if self.queue:
                    symbols_to_train = self.queue.pop(0)
            
            if symbols_to_train:
                print(f"[AutoRetrainer] Queue Processor: Starting retrain for {symbols_to_train}")
                try:
                    from train_pipeline import collect_and_train_all
                    collect_and_train_all(is_retrain=True, symbols=symbols_to_train)
                    print(f"[AutoRetrainer] Queue Processor: Completed retrain for {symbols_to_train}")
                except Exception as e:
                    error_msg = f"❌ 재학습 중 오류 발생 ({symbols_to_train}): {e}"
                    print(f"[AutoRetrainer] {error_msg}")
                    notifier.send_alert(error_msg)
                finally:
                    with self.lock:
                        # 학습 완료된 심볼 제거
                        for sym in symbols_to_train:
                            self.training_symbols.discard(sym)
                        self.is_training = len(self.training_symbols) > 0 or len(self.queue) > 0
            
            time.sleep(1)

    def load_models_with_fallback(self, symbol: str) -> tuple:
        """
        주어진 종목에 대해 (xgb_unified, rf_unified, lgb_unified, cat_unified, et_unified, gb_unified, mlp_unified, svm_unified, xgb_regime) 통합 모델 로드.
        로드 실패 시 백업 본을 가져오는 Fallback 메커니즘 제공.
        반환: (xgb_unified, rf_unified, lgb_unified, cat_unified, et_unified, gb_unified, mlp_unified, svm_unified, xgb_regime)
        """
        sym_clean = symbol.replace('/', '_')

        targets = [
            ("xgb", None, None),
            ("rf", None, None),
            ("lgb", None, None),
            ("cat", None, None),
            ("et", None, None),
            ("gb", os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_unified.pkl"), os.path.join(config.MODELS_DIR, f"gb_{sym_clean}_unified_backup.pkl")),
            ("mlp", os.path.join(config.MODELS_DIR, f"mlp_{sym_clean}_unified.pkl"), os.path.join(config.MODELS_DIR, f"mlp_{sym_clean}_unified_backup.pkl")),
            ("svm", None, None),
            ("regime", os.path.join(config.MODELS_DIR, f"xgb_{sym_clean}_regime.pkl"), os.path.join(config.MODELS_DIR, f"xgb_{sym_clean}_regime_backup.pkl"))
        ]

        models = []

        for name, path, bak in targets:
            if path is None:
                models.append(None)
                continue
            model = None
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                print(f"[AutoRetrainer] Failed to load {path}, trying backup. Error: {e}")
                try:
                    with open(bak, 'rb') as f:
                        model = pickle.load(f)
                except Exception as ex:
                    print(f"[AutoRetrainer] CRITICAL: Backup model also failed for {path}: {ex}")
            models.append(model)

        return tuple(models)

    def monitor_performance(self, symbol: str):
        metadata = self.load_metadata()
        trades = self.logger.get_recent_trades(1000)
        
        symbol_trades = [t for t in trades if t.get('symbol') == symbol]
        trades_since_train = len(symbol_trades)
        
        recent_trades_50 = symbol_trades[-50:]
        recent_win_rate = 0.50
        if recent_trades_50:
            wins = sum(1 for t in recent_trades_50 if t.get('pnl_usdt', 0.0) > 0)
            recent_win_rate = wins / len(recent_trades_50)
            
        if symbol in metadata:
            metadata[symbol]['trades_since_train'] = trades_since_train
            metadata[symbol]['recent_win_rate'] = recent_win_rate
            
            ensemble_acc = metadata[symbol].get('ensemble_accuracy', 0.55)
            if (ensemble_acc - recent_win_rate) >= config.RETRAIN_TRIGGER_DROP:
                metadata[symbol]['status'] = 'degraded'
            else:
                metadata[symbol]['status'] = 'healthy'
                
            self.save_metadata(metadata)
