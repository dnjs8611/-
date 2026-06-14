import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 디렉토리 생성
for path in [MODELS_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(path, exist_ok=True)

# 대상 종목 및 봉 설정
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SUI/USDT',
    'LINK/USDT', 'APT/USDT'
]
TIMEFRAME = '15m'

# 데이터 수집량
LIVE_LIMIT = 200
TRAIN_LIMIT = 52000
TRAIN_DAYS = 180

# XGBoost 파라미터
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42,
    'early_stopping_rounds': 30,
}

# 랜덤포레스트 파라미터 추가
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,           # 멀티코어 활용
    'class_weight': 'balanced',
}

# 기존 앙상블 설정 제거 (아래에 8개 모델 기준으로 재정의됨)
# 자금 분할 및 레버리지 설정
USE_FIXED_MARGIN = False         # True: 고정 USDT 금액 사용, False: 자산 대비 비율 사용
FIXED_MARGIN_USDT = 10.0        # 고정 진입 보증금 (USDT)
PORTFOLIO_ALLOCATION = 0.20  # 한 코인당 내 자산의 진입 비중 (보증금 기준 20%)
MAX_POSITIONS = 4            # 동시에 가질 수 있는 최대 포지션 수 (증거금 버퍼 고려 4개로 조정)
LEVERAGE = 5                 # 기본 레버리지
SYMBOL_LEVERAGE = {
    'BTC/USDT':  8,
    'ETH/USDT':  5,
    'SOL/USDT':  5,
    'BNB/USDT':  5,
    'XRP/USDT':  5,
    'DOGE/USDT': 5,
    'ADA/USDT':  5,
    'AVAX/USDT': 5,
    'NEAR/USDT': 5,
    'SUI/USDT':  5,
    'LINK/USDT': 5,
    'APT/USDT':  5,
    'STX/USDT':  5,
}

# 앙상블 진입 임계값 (3-클래스 단일 통합 모델 기준)
# 3클래스 랜덤 확률 = 33%, 0.45면 랜덤 대비 +12%p 우위
XGB_THRESHOLD = 0.45    # XGBoost 임계값 (3클래스 기준)
RF_THRESHOLD = 0.45     # Random Forest 임계값 (3클래스 기준)
LGB_THRESHOLD = 0.45    # LightGBM 임계값
CAT_THRESHOLD = 0.45    # CatBoost 임계값
ET_THRESHOLD = 0.45     # Extra Trees 임계값
GB_THRESHOLD = 0.45     # Gradient Boosting 임계값
MLP_THRESHOLD = 0.45    # MLP 임계값
SVM_THRESHOLD = 0.45    # SVM 임계값

LONG_THRESHOLD = 0.45   # 하위 호환성 유지용 임계값
SHORT_THRESHOLD = 0.45
COUNTER_SIGNAL_THRESHOLD = 0.45  # 역방향 신호 감지 탈출 임계값 (스마트 청산)

# 앙상블 가중치 (GradBoost, MLP 두 개만 남기고 합산 1.0)
XGB_WEIGHT = 0.0
RF_WEIGHT = 0.0
LGB_WEIGHT = 0.0
CAT_WEIGHT = 0.0
ET_WEIGHT = 0.0
GB_WEIGHT = 0.50
MLP_WEIGHT = 0.50
SVM_WEIGHT = 0.0

# 신규 모델 파라미터 추가
LGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

CAT_PARAMS = {
    'iterations': 300,
    'depth': 4,
    'learning_rate': 0.05,
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1
}

ET_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'random_state': 42,
    'n_jobs': -1
}

GB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.05,
    'random_state': 42
}

MLP_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'max_iter': 100,
    'random_state': 42,
    'early_stopping': True
}

SVM_PARAMS = {
    'probability': True,
    'kernel': 'linear',
    'max_iter': 1000,
    'random_state': 42
}

# 리스크 관리
RISK_PER_TRADE = 0.02
DAILY_LOSS_LIMIT = 0.03
MAX_HOLD_MINUTES = 120
MAX_LOSS_STREAK = 3

# 거래량 필터
VOLUME_MULTIPLIER = 1.5

# 종목별 손절 기준 (Stop Loss) - 5m 봉 변동성 반영
STOP_LOSS = {
    'BTC/USDT':  0.005,
    'ETH/USDT':  0.006,
    'SOL/USDT':  0.008,
    'BNB/USDT':  0.006,
    'XRP/USDT':  0.007,
    'DOGE/USDT': 0.008,
    'ADA/USDT':  0.007,
    'AVAX/USDT': 0.008,
    'NEAR/USDT': 0.008,
    'SUI/USDT':  0.008,
    'LINK/USDT': 0.007,
    'APT/USDT':  0.008,
    'STX/USDT':  0.009,
}

# 1차 익절 (50% 청산) - 5m 봉 변동성 반영
TAKE_PROFIT_1 = {
    'BTC/USDT':  0.005,
    'ETH/USDT':  0.006,
    'SOL/USDT':  0.008,
    'BNB/USDT':  0.006,
    'XRP/USDT':  0.007,
    'DOGE/USDT': 0.008,
    'ADA/USDT':  0.007,
    'AVAX/USDT': 0.008,
    'NEAR/USDT': 0.008,
    'SUI/USDT':  0.008,
    'LINK/USDT': 0.007,
    'APT/USDT':  0.008,
    'STX/USDT':  0.009,
}

# 2차 익절 (나머지 전량 청산) - 5m 봉 변동성 반영
TAKE_PROFIT_2 = {
    'BTC/USDT':  0.010,
    'ETH/USDT':  0.012,
    'SOL/USDT':  0.015,
    'BNB/USDT':  0.012,
    'XRP/USDT':  0.014,
    'DOGE/USDT': 0.015,
    'ADA/USDT':  0.014,
    'AVAX/USDT': 0.015,
    'NEAR/USDT': 0.015,
    'SUI/USDT':  0.015,
    'LINK/USDT': 0.014,
    'APT/USDT':  0.015,
    'STX/USDT':  0.018,
}

FEE_RATE = 0.0004
RETRAIN_DAY = 1
MIN_ACCURACY = 0.54
MAX_ACCURACY = 0.68
RETRAIN_TRIGGER_DROP = 0.05

DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 8080

# 1초 감시 및 트레일링 스탑 설정
MONITOR_INTERVAL = 1             # 감시 주기 (1초)
TRAILING_CALLBACK_RATE = 0.002   # 고점 대비 역방향 하락 비율 (0.2%)
FEE_BUFFER = 0.0002              # 수수료 차감 후 최소 마진 버퍼 (0.02%)

# API 키 및 알림 키 로드 (실전 바로 적용)
API_KEY          = (os.getenv('BINANCE_API_KEY') or os.getenv('BINANCE_ACCESS_KEY') or '').strip()
API_SECRET       = (os.getenv('BINANCE_API_SECRET') or os.getenv('BINANCE_SECRET_KEY') or '').strip()
TELEGRAM_TOKEN   = (os.getenv('TELEGRAM_TOKEN') or '').strip()
TELEGRAM_CHAT_ID = (os.getenv('TELEGRAM_CHAT_ID') or '').strip()

# 학습에 사용할 기술 지표 목록
FEATURES = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'ema20', 'ema60', 'ema200', 'ema_bullish',
    'bb_upper', 'bb_mid', 'bb_lower', 'bb_width',
    'vwap', 'above_vwap',
    'vol_ratio', 'price_change', 'hl_range', 'volume',
    'funding_rate', 'open_interest', 'long_short_ratio'
]

# 기본 및 현재 피처 세트 설정 (하위 호환성 유지)
FEATURES_BASIC = FEATURES
FEATURES_CURRENT = FEATURES
