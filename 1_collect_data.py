import ccxt
import pandas as pd
import time
import os
import requests
from datetime import datetime
import numpy as np
import config

# CCXT 심볼을 바이낸스 선물 API용으로 변환 (e.g., BTC/USDT -> BTCUSDT)
def symbol_ccxt_to_binance(symbol):
    return symbol.replace('/', '')

def fetch_futures_data_hist(endpoint, symbol, period='15m', days=30):
    """
    바이낸스 선물 정보 퍼블릭 데이터 (Open Interest Hist, Long/Short Ratio) 페이지네이션 수집
    """
    url = f"https://fapi.binance.com{endpoint}"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    all_data = []
    since = start_ms
    limit = 500
    
    while since < now_ms:
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit,
            'startTime': since
        }
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code != 200:
                print(f"  [API Error] {endpoint} status: {res.status_code} | Body: {res.text}. Breaking...")
                break
            data = res.json()
            if not data:
                break
            
            all_data.extend(data)
            
            # 다음 루프를 위해 마지막 타임스탬프 업데이트
            last_timestamp = data[-1]['timestamp']
            if last_timestamp <= since:
                break
            since = last_timestamp + 1
            
            time.sleep(0.1)  # Rate Limit 준수
        except Exception as e:
            print(f"  [API Exception] {endpoint}: {e}. Retrying...")
            time.sleep(2)
            
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    return df

def fetch_funding_rate_hist(symbol, days=30):
    """
    바이낸스 선물 과거 펀딩 피 히스토리 수집 (최근 1000개 수집)
    """
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    params = {
        'symbol': symbol,
        'startTime': start_ms,
        'limit': 1000
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data:
                df = pd.DataFrame(data)
                df.drop_duplicates(subset=['fundingTime'], inplace=True)
                df.sort_values(by='fundingTime', inplace=True)
                return df
    except Exception as e:
        print(f"  [API Exception] fundingRate: {e}")
    return pd.DataFrame()

def collect_symbol_data(exchange, symbol, timeframe, train_limit, days=30):
    """
    특정 종목의 최근 days 일치 데이터를 바이낸스 선물 API로부터 수집하고,
    미체결 약정(OI), 롱숏비율, 펀딩 피 데이터를 결합하여 CSV에 저장
    """
    symbol_clean = symbol.replace('/', '_')
    suffix = "_1m" if timeframe == '1m' else ""
    print(f"============================================================")
    print(f"[{symbol}] {timeframe} 데이터 수집 시작 | 목표 기간: 최근 {days}일")
    
    limit = 1000
    if timeframe == '1m':
        # 1분봉의 경우 tf_minutes = 1
        interval_ms = 1 * 60 * 1000
    else:
        tf_minutes = int(timeframe.replace('m', ''))
        interval_ms = tf_minutes * 60 * 1000
    
    # 시작 시간 계산
    now_ms = exchange.milliseconds()
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    since = start_ms
    
    while since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + interval_ms
            
            # 레이트 리밋 준수
            time.sleep(exchange.rateLimit / 1000)
            
            if len(ohlcv) < limit:
                break
                
        except Exception as e:
            print(f"  [Warning] 에러 발생: {e}. 5초 대기 후 재시도...")
            time.sleep(5)
            
    if not all_ohlcv:
        print(f"[Error] [{symbol}] {timeframe} 수집된 캔들 데이터가 없습니다.")
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 최근 train_limit개만 슬라이싱
    if len(df) > train_limit:
        df = df.tail(train_limit).reset_index(drop=True)
        
    # 바이낸스 선물 시장정보 수집 및 병합
    binance_symbol = symbol_ccxt_to_binance(symbol)
    print(f"  [{symbol}] {timeframe} 선물 시장 수급 데이터 수집 중 (OI, Long/Short Ratio, Funding Rate)...")
    
    # 1. 미체결 약정(OI)
    df_oi = fetch_futures_data_hist('/futures/data/openInterestHist', binance_symbol, period=timeframe, days=days)
    if not df_oi.empty:
        df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
        df_oi = df_oi[['timestamp', 'sumOpenInterest']].rename(columns={'sumOpenInterest': 'open_interest'})
        df = pd.merge(df, df_oi, on='timestamp', how='left')
    else:
        df['open_interest'] = np.nan
        print("  [Warning] 미체결 약정 수집 실패 (NaN 처리)")

    # 2. 상위 트레이더 롱/숏 비율
    df_ls = fetch_futures_data_hist('/futures/data/topLongShortPositionRatio', binance_symbol, period=timeframe, days=days)
    if not df_ls.empty:
        df_ls['longShortRatio'] = df_ls['longShortRatio'].astype(float)
        df_ls = df_ls[['timestamp', 'longShortRatio']].rename(columns={'longShortRatio': 'long_short_ratio'})
        df = pd.merge(df, df_ls, on='timestamp', how='left')
    else:
        df['long_short_ratio'] = np.nan
        print("  [Warning] 롱숏 비율 수집 실패 (NaN 처리)")

    # 3. 펀딩 피 (Funding Rate)
    df_fr = fetch_funding_rate_hist(binance_symbol, days=days)
    if not df_fr.empty:
        df_fr['fundingRate'] = df_fr['fundingRate'].astype(float)
        df_fr['timestamp'] = df_fr['fundingTime'].astype(int)
        df_fr = df_fr[['timestamp', 'fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
        
        # merge_asof 연산을 위해 정렬 상태 확인
        df.sort_values(by='timestamp', inplace=True)
        df_fr.sort_values(by='timestamp', inplace=True)
        
        # 캔들 시간에 맞춰 가장 가까운 직전 펀딩 피 결합
        df = pd.merge_asof(df, df_fr, on='timestamp', direction='backward')
    else:
        df['funding_rate'] = np.nan
        print("  [Warning] 펀딩 피 수집 실패 (NaN 처리)")

    # 결측치 보간 (Forward fill 후 Backward fill)
    df['open_interest'] = df['open_interest'].ffill().bfill()
    df['long_short_ratio'] = df['long_short_ratio'].ffill().bfill()
    df['funding_rate'] = df['funding_rate'].ffill().bfill()
    
    # 극단적인 결측치 대비 기본값 채우기
    df['open_interest'] = df['open_interest'].fillna(0.0)
    df['long_short_ratio'] = df['long_short_ratio'].fillna(1.0)
    df['funding_rate'] = df['funding_rate'].fillna(0.0)
    
    # CSV 저장
    csv_path = os.path.join(config.DATA_DIR, f"{symbol_clean}{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] [{symbol}] {timeframe} 수집 및 결합 완료 | 총 {len(df)}개 봉 | 저장: {csv_path}")
    print(f"============================================================")
    return df

def collect_all_data(days=30, symbols=None):
    exchange = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    if symbols is None:
        symbols = config.SYMBOLS
        
    # 데이터 수집 (config.TIMEFRAME)
    print(f"\n--- {config.TIMEFRAME}봉 데이터 수집 시작 ---")
    for symbol in symbols:
        collect_symbol_data(exchange, symbol, config.TIMEFRAME, config.TRAIN_LIMIT, days=days)

if __name__ == '__main__':
    collect_all_data(config.TRAIN_DAYS)
