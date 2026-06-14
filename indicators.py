import pandas as pd
import numpy as np

def calculate_indicators(df_input):
    """
    지표 계산 함수.
    입력: 'timestamp', 'open', 'high', 'low', 'close', 'volume' 컬럼이 포함된 DataFrame
    반환: 보조지표 컬럼들이 추가된 복사본 DataFrame
    """
    df = df_input.copy()
    
    # timestamp 컬럼이 있으면 인덱스를 datetime으로 변경
    if 'timestamp' in df.columns:
        if not isinstance(df.index, pd.DatetimeIndex):
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
    # 1. EMA 단기 & 중기 & 장기 & 배열
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema60'] = df['close'].ewm(span=60, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_bullish'] = (df['ema20'] > df['ema60']).astype(int)
    
    # 2. RSI (14) - Wilder's Smoothing 적용
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 4. 볼린저밴드 (20, 2.0)
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_mid'] = rolling_mean
    df['bb_upper'] = rolling_mean + (rolling_std * 2.0)
    df['bb_lower'] = rolling_mean - (rolling_std * 2.0)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-9)
    
    # 5. VWAP (Daily Reset)
    dates = df.index.date
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = tp * df['volume']
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = df['volume'].groupby(dates).cumsum()
    df['vwap'] = cum_pv / (cum_vol + 1e-9)
    df['above_vwap'] = (df['close'] > df['vwap']).astype(int)
    
    # 6. 거래량 MA & 비율
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-9)
    
    # 7. 가격 변화율
    df['price_change'] = df['close'].pct_change(1)
    
    # 8. 고저 범위
    df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
    
    # 결측치 제거
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    return df
