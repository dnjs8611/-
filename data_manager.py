import os
import json
import sqlite3
import config

DB_PATH = os.path.join(config.DATA_DIR, 'predictions_15m.db')
JSON_PATH = os.path.join(config.DATA_DIR, 'predictions_15m.json')

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = dict_factory
    return conn

def init_db():
    """
    Initializes the SQLite database and migrates existing JSON predictions if needed.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            predict_time TEXT,
            timestamp INTEGER,
            entry_price REAL,
            predicted_side TEXT,
            predicted_regime TEXT,
            xgb_predicted_side TEXT,
            rf_predicted_side TEXT,
            lgb_predicted_side TEXT,
            cat_predicted_side TEXT,
            et_predicted_side TEXT,
            gb_predicted_side TEXT,
            mlp_predicted_side TEXT,
            svm_predicted_side TEXT,
            xgb_prob REAL,
            rf_prob REAL,
            lgb_prob REAL,
            cat_prob REAL,
            et_prob REAL,
            gb_prob REAL,
            mlp_prob REAL,
            svm_prob REAL,
            ensemble_prob REAL,
            entry_margin_krw REAL,
            entry_margin_usdt REAL,
            target_time INTEGER,
            target_time_str TEXT,
            status TEXT,
            actual_price REAL,
            result TEXT,
            xgb_result TEXT,
            rf_result TEXT,
            lgb_result TEXT,
            cat_result TEXT,
            et_result TEXT,
            gb_result TEXT,
            mlp_result TEXT,
            svm_result TEXT,
            pnl_usdt REAL,
            pnl_krw REAL,
            net_pnl_pct REAL,
            xgb_pnl_usdt REAL,
            xgb_pnl_krw REAL,
            rf_pnl_usdt REAL,
            rf_pnl_krw REAL,
            lgb_pnl_usdt REAL,
            lgb_pnl_krw REAL,
            cat_pnl_usdt REAL,
            cat_pnl_krw REAL,
            et_pnl_usdt REAL,
            et_pnl_krw REAL,
            gb_pnl_usdt REAL,
            gb_pnl_krw REAL,
            mlp_pnl_usdt REAL,
            mlp_pnl_krw REAL,
            svm_pnl_usdt REAL,
            svm_pnl_krw REAL,
            gb_basic_predicted_side TEXT,
            gb_basic_prob REAL,
            gb_basic_result TEXT,
            gb_basic_pnl_usdt REAL,
            gb_basic_pnl_krw REAL,
            gb_current_predicted_side TEXT,
            gb_current_prob REAL,
            gb_current_result TEXT,
            gb_current_pnl_usdt REAL,
            gb_current_pnl_krw REAL,
            UNIQUE(symbol, timestamp)
        )
    """)
    conn.commit()
    
    # Check if we should migrate data from JSON
    cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
    cnt = cursor.fetchone()['cnt']
    
    if cnt == 0 and os.path.exists(JSON_PATH):
        print(f"[DataManager] Database empty. Starting migration from {JSON_PATH}...")
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                preds = json.load(f)
            
            keys = [
                'symbol', 'predict_time', 'timestamp', 'entry_price', 'predicted_side', 'predicted_regime',
                'xgb_predicted_side', 'rf_predicted_side', 'lgb_predicted_side', 'cat_predicted_side', 'et_predicted_side', 'gb_predicted_side', 'mlp_predicted_side', 'svm_predicted_side',
                'xgb_prob', 'rf_prob', 'lgb_prob', 'cat_prob', 'et_prob', 'gb_prob', 'mlp_prob', 'svm_prob', 'ensemble_prob',
                'entry_margin_krw', 'entry_margin_usdt', 'target_time', 'target_time_str', 'status', 'actual_price',
                'result', 'xgb_result', 'rf_result', 'lgb_result', 'cat_result', 'et_result', 'gb_result', 'mlp_result', 'svm_result',
                'pnl_usdt', 'pnl_krw', 'net_pnl_pct', 'xgb_pnl_usdt', 'xgb_pnl_krw', 'rf_pnl_usdt', 'rf_pnl_krw', 'lgb_pnl_usdt', 'lgb_pnl_krw', 'cat_pnl_usdt', 'cat_pnl_krw', 'et_pnl_usdt', 'et_pnl_krw', 'gb_pnl_usdt', 'gb_pnl_krw', 'mlp_pnl_usdt', 'mlp_pnl_krw', 'svm_pnl_usdt', 'svm_pnl_krw'
            ]
            
            inserted = 0
            for p in preds:
                try:
                    full_item = {k: p.get(k, None) for k in keys}
                    for k in ['xgb_predicted_side', 'rf_predicted_side', 'lgb_predicted_side', 'cat_predicted_side', 'et_predicted_side', 'gb_predicted_side', 'mlp_predicted_side', 'svm_predicted_side']:
                        if full_item[k] is None: full_item[k] = p.get('predicted_side', 'PASS')
                    for k in ['xgb_prob', 'rf_prob', 'lgb_prob', 'cat_prob', 'et_prob', 'gb_prob', 'mlp_prob', 'svm_prob', 'ensemble_prob']:
                        if full_item[k] is None: full_item[k] = p.get('ensemble_prob', 0.5)
                    for k in ['result', 'xgb_result', 'rf_result', 'lgb_result', 'cat_result', 'et_result', 'gb_result', 'mlp_result', 'svm_result']:
                        if full_item[k] is None: full_item[k] = p.get('result', 'PASS' if p.get('predicted_side') == 'PASS' else 'PENDING')
                    for k in ['pnl_usdt', 'xgb_pnl_usdt', 'rf_pnl_usdt', 'lgb_pnl_usdt', 'cat_pnl_usdt', 'et_pnl_usdt', 'gb_pnl_usdt', 'mlp_pnl_usdt', 'svm_pnl_usdt']:
                        if full_item[k] is None: full_item[k] = p.get('pnl_usdt', 0.0)
                    for k in ['pnl_krw', 'xgb_pnl_krw', 'rf_pnl_krw', 'lgb_pnl_krw', 'cat_pnl_krw', 'et_pnl_krw', 'gb_pnl_krw', 'mlp_pnl_krw', 'svm_pnl_krw']:
                        if full_item[k] is None: full_item[k] = p.get('pnl_krw', 0.0)

                    cursor.execute("""
                        INSERT OR IGNORE INTO predictions (
                            symbol, predict_time, timestamp, entry_price, predicted_side, predicted_regime,
                            xgb_predicted_side, rf_predicted_side, lgb_predicted_side, cat_predicted_side, et_predicted_side, gb_predicted_side, mlp_predicted_side, svm_predicted_side,
                            xgb_prob, rf_prob, lgb_prob, cat_prob, et_prob, gb_prob, mlp_prob, svm_prob, ensemble_prob,
                            entry_margin_krw, entry_margin_usdt, target_time, target_time_str, status, actual_price,
                            result, xgb_result, rf_result, lgb_result, cat_result, et_result, gb_result, mlp_result, svm_result,
                            pnl_usdt, pnl_krw, net_pnl_pct, xgb_pnl_usdt, xgb_pnl_krw, rf_pnl_usdt, rf_pnl_krw, lgb_pnl_usdt, lgb_pnl_krw, cat_pnl_usdt, cat_pnl_krw, et_pnl_usdt, et_pnl_krw, gb_pnl_usdt, gb_pnl_krw, mlp_pnl_usdt, mlp_pnl_krw, svm_pnl_usdt, svm_pnl_krw
                        ) VALUES (
                            :symbol, :predict_time, :timestamp, :entry_price, :predicted_side, :predicted_regime,
                            :xgb_predicted_side, :rf_predicted_side, :lgb_predicted_side, :cat_predicted_side, :et_predicted_side, :gb_predicted_side, :mlp_predicted_side, :svm_predicted_side,
                            :xgb_prob, :rf_prob, :lgb_prob, :cat_prob, :et_prob, :gb_prob, :mlp_prob, :svm_prob, :ensemble_prob,
                            :entry_margin_krw, :entry_margin_usdt, :target_time, :target_time_str, :status, :actual_price,
                            :result, :xgb_result, :rf_result, :lgb_result, :cat_result, :et_result, :gb_result, :mlp_result, :svm_result,
                            :pnl_usdt, :pnl_krw, :net_pnl_pct, :xgb_pnl_usdt, :xgb_pnl_krw, :rf_pnl_usdt, :rf_pnl_krw, :lgb_pnl_usdt, :lgb_pnl_krw, :cat_pnl_usdt, :cat_pnl_krw, :et_pnl_usdt, :et_pnl_krw, :gb_pnl_usdt, :gb_pnl_krw, :mlp_pnl_usdt, :mlp_pnl_krw, :svm_pnl_usdt, :svm_pnl_krw
                        )
                    """, full_item)
                    inserted += 1
                except Exception as ie:
                    pass
            
            conn.commit()
            print(f"[DataManager] Successfully migrated {inserted} records from JSON to SQLite database.")
            
            # Backup the JSON file
            bak_path = JSON_PATH + ".bak"
            if os.path.exists(bak_path):
                os.remove(bak_path)
            os.rename(JSON_PATH, bak_path)
            print(f"[DataManager] Backed up original JSON file to {bak_path}")
            
        except Exception as e:
            print(f"[DataManager] Migration failed: {e}")
            
    conn.close()

def db_load_predictions() -> list:
    """
    Loads all predictions from the database sorted by timestamp.
    """
    init_db()  # Ensure DB and tables are initialized
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def db_save_prediction(item: dict) -> bool:
    """
    Saves/inserts a new prediction item.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Generate default values for missing keys to avoid sqlite errors
        keys = [
            'symbol', 'predict_time', 'timestamp', 'entry_price', 'predicted_side', 'predicted_regime',
            'xgb_predicted_side', 'rf_predicted_side', 'lgb_predicted_side', 'cat_predicted_side', 'et_predicted_side', 'gb_predicted_side', 'mlp_predicted_side', 'svm_predicted_side',
            'xgb_prob', 'rf_prob', 'lgb_prob', 'cat_prob', 'et_prob', 'gb_prob', 'mlp_prob', 'svm_prob', 'ensemble_prob',
            'entry_margin_krw', 'entry_margin_usdt', 'target_time', 'target_time_str', 'status', 'actual_price',
            'result', 'xgb_result', 'rf_result', 'lgb_result', 'cat_result', 'et_result', 'gb_result', 'mlp_result', 'svm_result',
            'pnl_usdt', 'pnl_krw', 'net_pnl_pct', 'xgb_pnl_usdt', 'xgb_pnl_krw', 'rf_pnl_usdt', 'rf_pnl_krw', 'lgb_pnl_usdt', 'lgb_pnl_krw', 'cat_pnl_usdt', 'cat_pnl_krw', 'et_pnl_usdt', 'et_pnl_krw', 'gb_pnl_usdt', 'gb_pnl_krw', 'mlp_pnl_usdt', 'mlp_pnl_krw', 'svm_pnl_usdt', 'svm_pnl_krw',
            'gb_basic_predicted_side', 'gb_basic_prob', 'gb_basic_result', 'gb_basic_pnl_usdt', 'gb_basic_pnl_krw',
            'gb_current_predicted_side', 'gb_current_prob', 'gb_current_result', 'gb_current_pnl_usdt', 'gb_current_pnl_krw'
        ]
        full_item = {k: item.get(k, None) for k in keys}
        
        # Put default PASS/0.5/0.0 values if not provided
        for k in ['xgb_predicted_side', 'rf_predicted_side', 'lgb_predicted_side', 'cat_predicted_side', 'et_predicted_side', 'gb_predicted_side', 'mlp_predicted_side', 'svm_predicted_side', 'gb_basic_predicted_side', 'gb_current_predicted_side']:
            if full_item[k] is None: full_item[k] = 'PASS'
        for k in ['xgb_prob', 'rf_prob', 'lgb_prob', 'cat_prob', 'et_prob', 'gb_prob', 'mlp_prob', 'svm_prob', 'ensemble_prob', 'gb_basic_prob', 'gb_current_prob']:
            if full_item[k] is None: full_item[k] = 0.5
        for k in ['result', 'xgb_result', 'rf_result', 'lgb_result', 'cat_result', 'et_result', 'gb_result', 'mlp_result', 'svm_result', 'gb_basic_result', 'gb_current_result']:
            if full_item[k] is None: full_item[k] = 'PASS' if (p_side := full_item.get(k.replace('_result', '_predicted_side')) or 'PASS') == 'PASS' else 'PENDING'
            
        cursor.execute("""
            INSERT OR REPLACE INTO predictions (
                symbol, predict_time, timestamp, entry_price, predicted_side, predicted_regime,
                xgb_predicted_side, rf_predicted_side, lgb_predicted_side, cat_predicted_side, et_predicted_side, gb_predicted_side, mlp_predicted_side, svm_predicted_side,
                xgb_prob, rf_prob, lgb_prob, cat_prob, et_prob, gb_prob, mlp_prob, svm_prob, ensemble_prob,
                entry_margin_krw, entry_margin_usdt, target_time, target_time_str, status, actual_price,
                result, xgb_result, rf_result, lgb_result, cat_result, et_result, gb_result, mlp_result, svm_result,
                pnl_usdt, pnl_krw, net_pnl_pct, xgb_pnl_usdt, xgb_pnl_krw, rf_pnl_usdt, rf_pnl_krw, lgb_pnl_usdt, lgb_pnl_krw, cat_pnl_usdt, cat_pnl_krw, et_pnl_usdt, et_pnl_krw, gb_pnl_usdt, gb_pnl_krw, mlp_pnl_usdt, mlp_pnl_krw, svm_pnl_usdt, svm_pnl_krw,
                gb_basic_predicted_side, gb_basic_prob, gb_basic_result, gb_basic_pnl_usdt, gb_basic_pnl_krw,
                gb_current_predicted_side, gb_current_prob, gb_current_result, gb_current_pnl_usdt, gb_current_pnl_krw
            ) VALUES (
                :symbol, :predict_time, :timestamp, :entry_price, :predicted_side, :predicted_regime,
                :xgb_predicted_side, :rf_predicted_side, :lgb_predicted_side, :cat_predicted_side, :et_predicted_side, :gb_predicted_side, :mlp_predicted_side, :svm_predicted_side,
                :xgb_prob, :rf_prob, :lgb_prob, :cat_prob, :et_prob, :gb_prob, :mlp_prob, :svm_prob, :ensemble_prob,
                :entry_margin_krw, :entry_margin_usdt, :target_time, :target_time_str, :status, :actual_price,
                :result, :xgb_result, :rf_result, :lgb_result, :cat_result, :et_result, :gb_result, :mlp_result, :svm_result,
                :pnl_usdt, :pnl_krw, :net_pnl_pct, :xgb_pnl_usdt, :xgb_pnl_krw, :rf_pnl_usdt, :rf_pnl_krw, :lgb_pnl_usdt, :lgb_pnl_krw, :cat_pnl_usdt, :cat_pnl_krw, :et_pnl_usdt, :et_pnl_krw, :gb_pnl_usdt, :gb_pnl_krw, :mlp_pnl_usdt, :mlp_pnl_krw, :svm_pnl_usdt, :svm_pnl_krw,
                :gb_basic_predicted_side, :gb_basic_prob, :gb_basic_result, :gb_basic_pnl_usdt, :gb_basic_pnl_krw,
                :gb_current_predicted_side, :gb_current_prob, :gb_current_result, :gb_current_pnl_usdt, :gb_current_pnl_krw
            )
        """, full_item)
        conn.commit()
        success = True
    except Exception as e:
        print(f"[DataManager] Error saving prediction: {e}")
        success = False
    finally:
        conn.close()
    return success

def db_update_prediction(item: dict) -> bool:
    """
    Updates an existing prediction item matching symbol and timestamp.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Build dynamic query update string based on keys in item
        update_keys = [k for k in item.keys() if k not in ['id', 'symbol', 'timestamp']]
        set_clause = ", ".join([f"{k} = :{k}" for k in update_keys])
        
        cursor.execute(f"""
            UPDATE predictions 
            SET {set_clause}
            WHERE symbol = :symbol AND timestamp = :timestamp
        """, item)
        conn.commit()
        success = True
    except Exception as e:
        print(f"[DataManager] Error updating prediction: {e}")
        success = False
    finally:
        conn.close()
    return success

def db_delete_prediction(symbol: str, timestamp: int) -> bool:
    """
    Deletes a prediction item matching symbol and timestamp.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM predictions WHERE symbol = ? AND timestamp = ?", (symbol, timestamp))
        conn.commit()
        success = True
    except Exception as e:
        print(f"[DataManager] Error deleting prediction: {e}")
        success = False
    finally:
        conn.close()
    return success

# Initialize DB when first imported to guarantee setup
init_db()
