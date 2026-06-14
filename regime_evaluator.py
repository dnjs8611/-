import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import config
import notifier

REGIME_HISTORY_FILE = os.path.join(config.LOGS_DIR, 'regime_history.json')
REPORT_STATE_FILE = os.path.join(config.LOGS_DIR, 'regime_report_state.json')

class RegimeEvaluator:
    def __init__(self):
        self._init_files()

    def _init_files(self):
        if not os.path.exists(REGIME_HISTORY_FILE):
            try:
                with open(REGIME_HISTORY_FILE, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except Exception as e:
                print(f"[RegimeEvaluator] Init history file error: {e}")
                
        if not os.path.exists(REPORT_STATE_FILE):
            try:
                with open(REPORT_STATE_FILE, 'w', encoding='utf-8') as f:
                    json.dump({'last_report_time': ''}, f)
            except Exception as e:
                print(f"[RegimeEvaluator] Init report state file error: {e}")

    def load_history(self) -> list:
        try:
            with open(REGIME_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save_history(self, history: list):
        try:
            with open(REGIME_HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[RegimeEvaluator] Error saving history: {e}")

    def log_prediction(self, symbol: str, predicted_regime: str, timestamp: str, price: float):
        """
        새로운 장세 예측 기록을 저장
        """
        history = self.load_history()
        
        # 중복 기록 방지
        for h in history:
            if h.get('symbol') == symbol and h.get('timestamp') == timestamp:
                return
                
        history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'predicted': predicted_regime,
            'entry_price': float(price),
            'actual': None,
            'evaluated': False,
            'success': None,
            'eval_time': None
        })
        
        # 파일이 너무 커지지 않도록 최근 3000개만 유지
        if len(history) > 3000:
            history = history[-3000:]
            
        self.save_history(history)

    def evaluate_predictions(self, symbol: str, df: pd.DataFrame):
        """
        특정 코인에 대해 캐싱된 DataFrame의 데이터를 활용하여, 
        과거 30분 전에 이루어진 예측들 중 미평가된 건들을 실시간으로 매칭 평가 진행
        """
        if df is None or len(df) < 50:
            return
            
        tf_minutes = int(config.TIMEFRAME.replace('m', ''))
        regime_bars = 30 // tf_minutes
            
        history = self.load_history()
        updated = False
        
        # 미평가 예측 필터링
        unevaluated = [h for h in history if h.get('symbol') == symbol and not h.get('evaluated')]
        if not unevaluated:
            return
            
        # DataFrame index를 문자열 리스트로 변환 (datetime 인덱스인 경우 처리)
        if isinstance(df.index, pd.DatetimeIndex):
            df_index_times = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        else:
            df_index_times = [str(x) for x in df.index.tolist()]
            
        for h in unevaluated:
            pred_time_str = h.get('timestamp')
            
            # 예측 시점이 DataFrame 내에 있는지 확인
            if pred_time_str not in df_index_times:
                continue
                
            pred_idx = df_index_times.index(pred_time_str)
            
            # 30분(regime_bars개 완성봉)이 흘렀는지 확인 (즉, 예측 인덱스 뒤로 최소 regime_bars개의 완성봉이 더 있어야 함)
            # df.iloc[-1]은 실시간 미완성봉이므로, 완성봉만 고려하기 위해 index <= len(df) - 2 - regime_bars
            if pred_idx + regime_bars > len(df) - 2:
                continue
                
            # 예측 시점의 가격
            entry_p = h.get('entry_price')
            
            # 예측 시점 이후의 regime_bars개 완성봉 범위 슬라이싱 (pred_idx + 1 ~ pred_idx + regime_bars)
            df_after = df.iloc[pred_idx + 1 : pred_idx + 1 + regime_bars]
            
            max_p = df_after['high'].max()
            min_p = df_after['low'].min()
            
            max_return = (max_p - entry_p) / entry_p
            min_return = (min_p - entry_p) / entry_p
            
            # 실제 장세 판정 규칙
            is_bull = (max_return >= 0.003) & (min_return > -0.003)
            is_bear = (min_return <= -0.003) & (max_return < 0.003)
            
            if is_bull:
                actual_regime = 'BULL'
            elif is_bear:
                actual_regime = 'BEAR'
            else:
                actual_regime = 'SIDEWAYS'
                
            # 평가 기록 갱신
            h['actual'] = actual_regime
            h['evaluated'] = True
            h['success'] = (h.get('predicted') == actual_regime)
            h['eval_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            updated = True
            print(f"[RegimeEvaluator] Evaluated {symbol} predicted at {pred_time_str}: Pred={h.get('predicted')} | Actual={actual_regime} | Result={h.get('success')}")
            
        if updated:
            self.save_history(history)

    def check_and_send_report(self, force=False):
        """
        주기적으로 (예: 6시간 마다) 최근 평가된 예측들의 정확도를 Telegram으로 발송
        """
        try:
            with open(REPORT_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except Exception:
            state = {'last_report_time': ''}
            
        now = datetime.now()
        last_report_str = state.get('last_report_time', '')
        
        should_report = False
        if force or not last_report_str:
            should_report = True
        else:
            try:
                last_report = datetime.strptime(last_report_str, '%Y-%m-%d %H:%M:%S')
                # 6시간 경과 여부 체크
                if (now - last_report).total_seconds() >= 6 * 3600:
                    should_report = True
            except Exception:
                should_report = True
                
        if not should_report:
            return
            
        # 최근 6시간 이내에 평가 완료된 기록들 수집
        history = self.load_history()
        eval_list = [h for h in history if h.get('evaluated') and h.get('eval_time')]
        
        # 최근 6시간 이내 평가된 건수 필터링
        recent_evals = []
        six_hours_ago = now - timedelta(hours=6)
        for h in eval_list:
            try:
                eval_t = datetime.strptime(h.get('eval_time'), '%Y-%m-%d %H:%M:%S')
                if eval_t >= six_hours_ago:
                    recent_evals.append(h)
            except Exception:
                pass
                
        if not recent_evals:
            print("[RegimeEvaluator] No predictions evaluated in the last 6 hours to report.")
            # 리포트 타임스탬프는 갱신하여 빈 메시지가 계속 울리는 것을 방지
            state['last_report_time'] = now.strftime('%Y-%m-%d %H:%M:%S')
            try:
                with open(REPORT_STATE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=4)
            except Exception:
                pass
            return
            
        total = len(recent_evals)
        successes = sum(1 for h in recent_evals if h.get('success'))
        accuracy = (successes / total) * 100 if total > 0 else 0.0
        
        # 종목별 통계 계산
        symbol_stats = {}
        for h in recent_evals:
            sym = h.get('symbol')
            if sym not in symbol_stats:
                symbol_stats[sym] = {'total': 0, 'success': 0}
            symbol_stats[sym]['total'] += 1
            if h.get('success'):
                symbol_stats[sym]['success'] += 1
                
        # 텔레그램 메시지 빌드
        msg = (
            f"<b>📊 [AI 장세 예측 실시간 검증 리포트]</b>\n"
            f"최근 6시간 동안 만기 평가된 예측 결과입니다.\n\n"
            f"● <b>전체 요약</b>\n"
            f"   - 검증 건수: {total}건\n"
            f"   - 예측 적중: {successes}건\n"
            f"   - <b>최종 예측 정확도: {accuracy:.1f}%</b>\n\n"
            f"● <b>코인별 상세 정확도</b>\n"
        )
        
        for sym, stat in sorted(symbol_stats.items()):
            sym_acc = (stat['success'] / stat['total']) * 100
            msg += f"   - {sym}: {sym_acc:.1f}% ({stat['success']}/{stat['total']}건)\n"
            
        # 텔레그램 알림 발송
        notifier.send_telegram_message(msg)
        print(f"[RegimeEvaluator] Sent periodic trend accuracy report. Accuracy: {accuracy:.1f}%")
        
        # 상태 저장
        state['last_report_time'] = now.strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(REPORT_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
        except Exception:
            pass
