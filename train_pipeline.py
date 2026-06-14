import importlib
import config
import notifier

def collect_and_train_all(is_retrain=True, symbols=None):
    """
    1_collect_data와 2_train_model을 순차적으로 호출하여 전체 혹은 특정 종목의 학습 파이프라인 진행.
    파일명이 숫자로 시작하므로 importlib을 사용하여 동적으로 임포트함.
    """
    try:
        collect_data = importlib.import_module("1_collect_data")
        train_model = importlib.import_module("2_train_model")
    except ModuleNotFoundError as e:
        print(f"[Pipeline] Import error: {e}")
        return []

    print("[Pipeline] ==================== 파이프라인 시작 ====================")
    
    if symbols is None:
        symbols = config.SYMBOLS
    
    # 1. 데이터 수집 실행
    print(f"[Pipeline] STEP 1/2: 종목 데이터 수집 중... ({symbols})")
    collect_data.collect_all_data(config.TRAIN_DAYS, symbols=symbols)
    
    # 2. 모델 재학습 실행 및 정확도 판정
    print("[Pipeline] STEP 2/2: 모델 학습 및 교체/롤백 판정 중...")
    results = []
    for symbol in symbols:
        res = train_model.train_symbol_model(symbol, is_retrain=is_retrain)
        if res:
            results.append(res)
            
            # 각 종목 결과별로 텔레그램 알림 발송
            notifier.send_retrain_result(
                symbol=res['symbol'],
                old_xgb=res['old_xgb'],
                new_xgb=res['new_xgb'],
                action=res['action']
            )
            
    print("[Pipeline] ==================== 파이프라인 종료 ====================")
    return results

if __name__ == '__main__':
    # 독립 실행 시 초기 학습으로 간주하여 교체 판정
    collect_and_train_all(is_retrain=False)
