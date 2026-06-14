import os
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template_string
import config
from trade_logger import TradeLogger
import data_manager

app = Flask(__name__)

# 플래스크 로깅 레벨 조정 (출력창 간소화)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class BotState:
    def __init__(self):
        self.is_running = True
        self.positions = {symbol: {
            'size': 0.0,
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'side': None,
            'liquidation_price': 0.0,
            'leverage': 5
        } for symbol in config.SYMBOLS}
        
        self.models_status = {symbol: {
            'xgb_long_accuracy': 0.50,
            'xgb_short_accuracy': 0.50,
            'xgb_accuracy': 0.50,
            'xgb_accuracy_val': 0.50,
            'rf_accuracy_val': 0.50,
            'lgb_accuracy_val': 0.50,
            'cat_accuracy_val': 0.50,
            'et_accuracy_val': 0.50,
            'gb_accuracy_val': 0.50,
            'mlp_accuracy_val': 0.50,
            'svm_accuracy_val': 0.50,
            'ens_accuracy_val': 0.50,
            'status': 'healthy',
            'regime': 'SIDEWAYS'
        } for symbol in config.SYMBOLS}
        
        self.current_capital = 0.0
        self.load_initial_metadata()

    def load_initial_metadata(self):
        metadata_file = os.path.join(config.MODELS_DIR, 'model_metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    for symbol, data in meta.items():
                        if symbol in self.models_status:
                            # Load values with fallback logic
                            xgb_val = data.get('xgb_accuracy_val') or data.get('xgb_long_accuracy') or data.get('xgb_accuracy') or 0.50
                            rf_val = data.get('rf_accuracy_val') or data.get('xgb_short_accuracy') or data.get('rf_accuracy') or 0.50
                            lgb_val = data.get('lgb_accuracy_val') or 0.50
                            cat_val = data.get('cat_accuracy_val') or 0.50
                            et_val = data.get('et_accuracy_val') or 0.50
                            gb_val = data.get('gb_accuracy_val') or 0.50
                            mlp_val = data.get('mlp_accuracy_val') or 0.50
                            svm_val = data.get('svm_accuracy_val') or 0.50
                            ens_val = data.get('ens_accuracy_val') or data.get('ensemble_accuracy') or data.get('xgb_accuracy') or 0.50
                            
                            self.models_status[symbol] = {
                                'xgb_long_accuracy': xgb_val,
                                'xgb_short_accuracy': rf_val,
                                'xgb_accuracy': ens_val,
                                
                                'xgb_accuracy_val': xgb_val,
                                'rf_accuracy_val': rf_val,
                                'lgb_accuracy_val': lgb_val,
                                'cat_accuracy_val': cat_val,
                                'et_accuracy_val': et_val,
                                'gb_accuracy_val': gb_val,
                                'mlp_accuracy_val': mlp_val,
                                'svm_accuracy_val': svm_val,
                                'ens_accuracy_val': ens_val,
                                
                                'last_trained': data.get('last_trained', '-'),
                                'status': data.get('status', 'healthy'),
                                'regime': 'SIDEWAYS'
                            }
            except Exception as e:
                print(f"[Dashboard State] Error loading metadata: {e}")

# 싱글톤 상태 객체
state = BotState()
logger = TradeLogger()

HTML_TEMPLATE_ENS = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Ensemble 모의투자 및 자동매매 대시보드</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #060814;
            --card-bg: #0e132b;
            --border-color: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #00d4ff;
            --accent-purple: #a855f7;
            --color-green: #10b981;
            --color-red: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body {
            background-color: var(--bg-color); color: var(--text-main); padding: 20px; padding-bottom: 120px; min-height: 100vh;
            background-image: radial-gradient(circle at 10% 20%, rgba(12, 18, 45, 0.5) 0%, rgba(6, 8, 20, 0.5) 90%);
        }
        .container { max-width: 1250px; margin: 0 auto; }
        header {
            display: flex; justify-content: space-between; align-items: center; padding: 16px 24px;
            background: linear-gradient(135deg, var(--card-bg), #131b3e); border: 1px solid var(--border-color);
            border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); backdrop-filter: blur(10px);
        }
        header h1 { font-size: 1.5rem; font-weight: 700; background: linear-gradient(to right, var(--accent-cyan), var(--accent-purple), #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .nav-links { display: flex; gap: 16px; }
        .nav-links a { color: var(--text-muted); text-decoration: none; font-size: 0.9rem; font-weight: 600; padding: 8px 16px; border-radius: 8px; border: 1px solid transparent; transition: all 0.2s; background: rgba(255, 255, 255, 0.02); }
        .nav-links a:hover { color: var(--accent-cyan); border-color: rgba(0, 212, 255, 0.2); background: rgba(0, 212, 255, 0.05); }
        .nav-links a.active { color: var(--accent-cyan); border-color: rgba(0, 212, 255, 0.4); background: rgba(0, 212, 255, 0.1); box-shadow: 0 0 15px rgba(0, 212, 255, 0.1); }
        .status-badge { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 0.95rem; padding: 6px 14px; border-radius: 9999px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); color: var(--color-green); }
        .status-badge.stopped { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); color: var(--color-red); }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background-color: var(--color-green); box-shadow: 0 0 10px var(--color-green); }
        .status-badge.stopped .status-dot { background-color: var(--color-red); box-shadow: 0 0 10px var(--color-red); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); transition: transform 0.2s, border-color 0.2s; }
        .card:hover { transform: translateY(-2px); border-color: rgba(0, 212, 255, 0.3); }
        .card-title { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; margin-bottom: 8px; }
        .card-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
        .main-layout { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; margin-bottom: 24px; }
        .chart-container { min-height: 300px; }
        .position-list { display: flex; flex-direction: column; gap: 12px; }
        .pos-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: rgba(30, 41, 59, 0.3); border: 1px solid var(--border-color); border-radius: 12px; }
        .pos-symbol { font-weight: 600; }
        .pos-badge { font-size: 0.8rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }
        .pos-badge.long { background-color: rgba(16, 185, 129, 0.2); color: var(--color-green); }
        .pos-badge.short { background-color: rgba(239, 68, 68, 0.2); color: var(--color-red); }
        .pos-badge.flat { background-color: rgba(148, 163, 184, 0.1); color: var(--text-muted); }
        .section-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
        th { color: var(--text-muted); font-weight: 500; padding: 12px 16px; border-bottom: 1px solid var(--border-color); }
        td { padding: 14px 16px; border-bottom: 1px solid var(--border-color); color: #e2e8f0; }
        .table-wrapper { overflow-x: auto; border-radius: 12px; border: 1px solid var(--border-color); background: var(--card-bg); }
        .emergency-bar { position: fixed; bottom: 0; left: 0; width: 100%; background: rgba(10, 14, 35, 0.95); border-top: 1px solid var(--border-color); padding: 16px 24px; display: flex; justify-content: center; align-items: center; backdrop-filter: blur(8px); z-index: 1000; }
        .btn { font-size: 1.1rem; font-weight: 700; padding: 14px 48px; border-radius: 12px; cursor: pointer; border: none; transition: all 0.3s ease; display: flex; align-items: center; gap: 10px; }
        .btn-stop { background: linear-gradient(135deg, var(--color-red), #b91c1c); color: white; }
        .btn-stop:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        .btn-start { background: linear-gradient(135deg, var(--color-green), #047857); color: white; }
        .btn-start:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
    </style>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <header>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <h1>🤖 AI Ensemble 자동매매 & 모의투자</h1>
                <div class="nav-links">
                    <a href="/" class="active">🔗 Ensemble 홈</a>
                    <a href="/gb">🔗 GradBoost 전용</a>
                    <a href="/mlp">🔗 MLP 전용</a>
                </div>
            </div>
            <div id="bot-status-badge" class="status-badge">
                <div class="status-dot"></div>
                <span id="bot-status-text">실행중</span>
            </div>
        </header>

        <!-- KPI 서머리 -->
        <div class="summary-grid">
            <div class="card">
                <div class="card-title">오늘 손익</div>
                <div id="kpi-pnl" class="card-value">+$0.00</div>
            </div>
            <div class="card">
                <div class="card-title">오늘 승률</div>
                <div id="kpi-winrate" class="card-value">0.0%</div>
            </div>
            <div class="card">
                <div class="card-title">오늘 거래 횟수</div>
                <div id="kpi-trades" class="card-value">0회</div>
            </div>
            <div class="card">
                <div class="card-title">현재 자본</div>
                <div id="kpi-capital" class="card-value">$0.00</div>
            </div>
        </div>

        <!-- 2열 레이아웃 -->
        <div class="main-layout">
            <div class="card">
                <div class="section-title">📉 누적 손익 차트 (최근 30일)</div>
                <div class="chart-container"><canvas id="equityChart"></canvas></div>
            </div>
            <div class="card">
                <div class="section-title">⚡ 종목별 포지션 현황</div>
                <div id="position-list" class="position-list"></div>
            </div>
        </div>

        <!-- 모델 상태 -->
        <div class="card" style="margin-bottom: 24px;">
            <div class="section-title">📊 15분봉 AI 모델별 검증 정확도 및 상태 (9-Model Ensemble)</div>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>종목</th>
                            <th style="color: var(--accent-cyan); font-weight: 700;">Ensemble 검증 정확도</th>
                            <th>GradBoost 정확도</th>
                            <th>MLP 정확도</th>
                            <th>XGBoost 정확도</th>
                            <th>마지막 학습 시간</th>
                            <th>시장 장세</th>
                            <th>모델 성능 상태</th>
                        </tr>
                    </thead>
                    <tbody id="model-table-body"></tbody>
                </table>
            </div>
        </div>

        <!-- 최근 거래 내역 -->
        <div class="card" style="margin-bottom: 24px;">
            <div class="section-title">📜 최근 매매 내역 (20건)</div>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>청산 시간</th><th>종목</th><th>구분</th><th>진입 확률</th><th>수익률 (레버리지)</th><th>수익금 (USDT)</th><th>청산 이유</th>
                        </tr>
                    </thead>
                    <tbody id="trades-table-body"></tbody>
                </table>
            </div>
        </div>

        <!-- Ensemble 모의 투자 현황 -->
        <div class="card" style="margin-bottom: 24px; border: 1px solid rgba(0, 212, 255, 0.3); background: linear-gradient(135deg, var(--card-bg), #0f153a);">
            <div class="section-title">🤖 15분봉 Ensemble 모의 투자 현황 (3시간 주기 재학습)</div>
            <div id="gb-cards-container" class="summary-grid" style="grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px;"></div>
            
            <div style="margin-bottom: 24px; padding: 0 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px;">
                    <span>목표 승률 70% 달성도 (Ensemble 앙상블 전략 기준)</span>
                    <span id="ens-winrate-progress-text" style="font-weight: 600;">0% (0.0% / 70.0%)</span>
                </div>
                <div style="width: 100%; height: 12px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden; border: 1px solid var(--border-color);">
                    <div id="ens-winrate-progress-bar" style="width: 0%; height: 100%; background: linear-gradient(to right, var(--accent-purple), var(--accent-cyan)); box-shadow: 0 0 10px var(--accent-cyan); transition: width 0.5s ease;"></div>
                </div>
            </div>

            <div class="section-title" style="font-size: 0.95rem; margin-top: 20px;">📜 최근 15분봉 예측 내역 (Ensemble vs 단독 전략 대조)</div>
            <div class="table-wrapper" style="background: rgba(10, 14, 26, 0.4);">
                <table>
                    <thead>
                        <tr>
                            <th rowspan="2" style="vertical-align: middle;">예측 시간</th>
                            <th rowspan="2" style="vertical-align: middle;">종목</th>
                            <th rowspan="2" style="vertical-align: middle;">진입 / 결과가</th>
                            <th colspan="3" style="text-align: center; border-bottom: 1px solid rgba(0,212,255,0.15); background: rgba(0,212,255,0.15); color: var(--accent-cyan);">🤖 Ensemble (타이트)</th>
                            <th colspan="3" style="text-align: center; border-bottom: 1px solid rgba(255,160,0,0.15); background: rgba(255,160,0,0.1); color: #ffa000;">📈 GradBoost (완화)</th>
                            <th colspan="3" style="text-align: center; border-bottom: 1px solid rgba(168,85,247,0.15); background: rgba(168,85,247,0.1); color: #c084fc;">🧠 MLP 단독 (타이트)</th>
                        </tr>
                        <tr>
                            <th style="background: rgba(0,212,255,0.05);">진입</th><th style="background: rgba(0,212,255,0.05);">확률</th><th style="background: rgba(0,212,255,0.05);">결과/손익</th>
                            <th style="background: rgba(255,160,0,0.05);">진입</th><th style="background: rgba(255,160,0,0.05);">확률</th><th style="background: rgba(255,160,0,0.05);">결과/손익</th>
                            <th style="background: rgba(168,85,247,0.05);">진입</th><th style="background: rgba(168,85,247,0.05);">확률</th><th style="background: rgba(168,85,247,0.05);">결과/손익</th>
                        </tr>
                    </thead>
                    <tbody id="gb-table-body">
                        <tr><td colspan="12" style="text-align:center; color:var(--text-muted);">데이터 로딩 중...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="emergency-bar">
        <button id="ctrl-btn" class="btn btn-stop" onclick="toggleBot()">
            <span id="ctrl-btn-text">🛑 봇 긴급 정지</span>
        </button>
    </div>

    <script>
        let equityChart = null;
        async function updateDashboard() {
            try {
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                const isRunning = statusData.status === 'running';
                
                const badge = document.getElementById('bot-status-badge');
                const badgeText = document.getElementById('bot-status-text');
                const ctrlBtn = document.getElementById('ctrl-btn');
                const ctrlBtnText = document.getElementById('ctrl-btn-text');

                if (isRunning) {
                    badge.className = 'status-badge'; badgeText.innerText = '실행중';
                    ctrlBtn.className = 'btn btn-stop'; ctrlBtnText.innerText = '🛑 봇 긴급 정지';
                } else {
                    badge.className = 'status-badge stopped'; badgeText.innerText = '정지됨';
                    ctrlBtn.className = 'btn btn-start'; ctrlBtnText.innerText = '🟢 봇 매매 재개';
                }

                const summaryRes = await fetch('/api/summary');
                const summary = await summaryRes.json();
                const pnlEl = document.getElementById('kpi-pnl');
                const pnlVal = parseFloat(summary.total_pnl_usdt);
                pnlEl.innerText = (pnlVal >= 0 ? '+' : '') + '$' + pnlVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                pnlEl.style.color = pnlVal >= 0 ? 'var(--color-green)' : 'var(--color-red)';
                
                document.getElementById('kpi-winrate').innerText = (parseFloat(summary.win_rate) * 100).toFixed(1) + '%';
                document.getElementById('kpi-trades').innerText = summary.trades_count + '회';

                const positionsRes = await fetch('/api/positions');
                const positions = await positionsRes.json();
                const capRes = await fetch('/api/status');
                const capData = await capRes.json();
                document.getElementById('kpi-capital').innerText = '$' + capData.capital.toLocaleString(undefined, {minimumFractionDigits: 2});

                const posContainer = document.getElementById('position-list');
                posContainer.innerHTML = '';
                for (const [symbol, pos] of Object.entries(positions)) {
                    const size = parseFloat(pos.size);
                    let sideText = '대기'; let badgeClass = 'pos-badge flat';
                    if (size > 0) { sideText = '🟢 LONG'; badgeClass = 'pos-badge long'; }
                    else if (size < 0) { sideText = '🔴 SHORT'; badgeClass = 'pos-badge short'; }
                    
                    let pnlText = '';
                    if (size !== 0) {
                        const leverage = pos.leverage ? parseInt(pos.leverage) : 5;
                        const positionValue = parseFloat(pos.entry_price) * Math.abs(size);
                        const margin = positionValue / leverage;
                        const pnlPctLeveraged = (parseFloat(pos.unrealized_pnl) / (margin + 1e-9)) * 100;
                        pnlText = ` <span style="color:${pos.unrealized_pnl >= 0 ? 'var(--color-green)' : 'var(--color-red)'}; font-size:0.85rem;">(${pos.unrealized_pnl >= 0 ? '+' : ''}$${parseFloat(pos.unrealized_pnl).toFixed(2)} / ${pos.unrealized_pnl >= 0 ? '+' : ''}${pnlPctLeveraged.toFixed(2)}% [${leverage}x])</span>`;
                    }
                    posContainer.innerHTML += `<div class="pos-item"><span class="pos-symbol">${symbol}</span><div><span class="${badgeClass}">${sideText}</span>${pnlText}</div></div>`;
                }

                const modelsRes = await fetch('/api/models');
                const models = await modelsRes.json();
                const modelTable = document.getElementById('model-table-body');
                modelTable.innerHTML = '';
                for (const [symbol, m] of Object.entries(models)) {
                    const ensAcc = parseFloat(m.ens_accuracy_val) || parseFloat(m.ensemble_accuracy) || 0.50;
                    const gbAcc = parseFloat(m.gb_accuracy_val) || 0.50;
                    const mlpAcc = parseFloat(m.mlp_accuracy_val) || 0.50;
                    const xgbAcc = parseFloat(m.xgb_accuracy_val) || 0.50;
                    
                    const regimeVal = m.regime || 'SIDEWAYS';
                    let regimeText = regimeVal === 'BULL' ? '🟢 상승장' : (regimeVal === 'BEAR' ? '🔴 하락장' : '🟡 횡보장');
                    let regimeColor = regimeVal === 'BULL' ? 'var(--color-green)' : (regimeVal === 'BEAR' ? 'var(--color-red)' : '#eab308');
                    let regimeBg = regimeVal === 'BULL' ? 'rgba(16, 185, 129, 0.1)' : (regimeVal === 'BEAR' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(234, 179, 8, 0.1)');

                    modelTable.innerHTML += `
                        <tr>
                            <td style="font-weight:600;">${symbol}</td>
                            <td style="font-weight:700; color:var(--accent-cyan);">${(ensAcc * 100).toFixed(1)}%</td>
                            <td>${(gbAcc * 100).toFixed(1)}%</td>
                            <td>${(mlpAcc * 100).toFixed(1)}%</td>
                            <td style="color:var(--text-muted);">${(xgbAcc * 100).toFixed(1)}%</td>
                            <td style="font-size:0.85rem; color:var(--text-muted);">${m.last_trained || '-'}</td>
                            <td><span style="color:${regimeColor}; font-weight:600; background:${regimeBg}; border:1px solid rgba(255,255,255,0.05); padding:3px 8px; border-radius:4px; font-size:0.8rem;">${regimeText}</span></td>
                            <td><span style="color:${m.status === 'healthy' ? 'var(--color-green)' : 'var(--color-red)'}; font-weight:600;">${m.status === 'healthy' ? '✅ 정상' : '⚠️ 성능 저하'}</span></td>
                        </tr>
                    `;
                }

                const tradesRes = await fetch('/api/trades');
                const trades = await tradesRes.json();
                const tradesTable = document.getElementById('trades-table-body');
                tradesTable.innerHTML = '';
                if (trades.length === 0) {
                    tradesTable.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">최근 거래 내역이 없습니다.</td></tr>`;
                } else {
                    trades.forEach(t => {
                        const pnl = parseFloat(t.pnl_pct);
                        const leverage = t.leverage ? parseInt(t.leverage) : 5;
                        const pnlUsdt = parseFloat(t.pnl_usdt);
                        tradesTable.innerHTML += `
                            <tr>
                                <td style="font-size:0.8rem; color:var(--text-muted);">${t.exit_time.substring(5, 16)}</td>
                                <td style="font-weight:600;">${t.symbol}</td>
                                <td><span class="pos-badge ${t.side === 'LONG' ? 'long' : 'short'}">${t.side}</span></td>
                                <td>${(parseFloat(t.ensemble_prob || t.xgb_prob || 0.5)*100).toFixed(0)}%</td>
                                <td style="color:${pnl >= 0 ? 'var(--color-green)' : 'var(--color-red)'}; font-weight:600;">${pnl >= 0 ? '+' : ''}${(pnl * leverage * 100).toFixed(2)}% <span style="font-size:0.75rem; color:var(--text-muted);">(${leverage}x)</span></td>
                                <td style="color:${pnlUsdt >= 0 ? 'var(--color-green)' : 'var(--color-red)'}; font-weight:600;">${pnlUsdt >= 0 ? '+' : ''}$${pnlUsdt.toFixed(2)}</td>
                                <td style="font-size:0.85rem;">${t.exit_reason}</td>
                            </tr>
                        `;
                    });
                }

                const equityRes = await fetch('/api/equity');
                const equityData = await equityRes.json();
                updateChart(equityData);

                try {
                    const res15m = await fetch('/api/predict15m');
                    const data15m = await res15m.json();
                    const stats = data15m.stats;
                    const history = data15m.history;
                    
                    const empty_model_stats = { total: 0, wins: 0, losses: 0, win_rate: 0.0, total_pnl_krw: 0.0, total_pnl_usdt: 0.0, capital_krw: 100000.0, capital_usdt: 100000.0/1350.0, recent_wins:0, recent_losses:0, recent_win_rate:0.0, recent_pnl_krw:0.0 };
                    const ensStats = stats['ens'] || empty_model_stats;
                    const gbStats = stats['gb'] || empty_model_stats;
                    const mlpStats = stats['mlp'] || empty_model_stats;
                    
                    document.getElementById('gb-cards-container').innerHTML = `
                        <!-- Ensemble Card -->
                        <div class="card" style="background: rgba(14, 20, 47, 0.8); border: 1px solid rgba(0, 212, 255, 0.4); padding: 20px; border-radius: 12px; width: 100%; box-shadow: 0 0 15px rgba(0, 212, 255, 0.15);">
                            <div style="font-weight: 700; font-size: 1.1rem; color: var(--accent-cyan); margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; display:flex; justify-content:space-between; align-items:center;">
                                <span>🤖 Ensemble (현재 타이트) (Main)</span>
                                <span style="font-size:0.75rem; background:rgba(0,212,255,0.15); padding:2px 8px; border-radius:4px; color:var(--accent-cyan); font-weight:bold;">Main</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 승률</div><div style="font-size:1.4rem; font-weight:700;">${(ensStats.win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${ensStats.total}회 (${ensStats.wins}승/${ensStats.losses}패)</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 손익</div><div style="font-size:1.4rem; font-weight:700; color:${ensStats.total_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${ensStats.total_pnl_krw>=0?'+':''}${ensStats.total_pnl_krw.toLocaleString(undefined, {maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">${ensStats.total_pnl_usdt>=0?'+':''}$${ensStats.total_pnl_usdt.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">평가 자산</div><div style="font-size:1.1rem; font-weight:600;">${ensStats.capital_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">$${ensStats.capital_usdt.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">최근 1시간 현황</div><div style="font-size:1.1rem; font-weight:600; color:${ensStats.recent_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${(ensStats.recent_win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${ensStats.recent_wins}승/${ensStats.recent_losses}패 | <span style="font-weight:600;">${ensStats.recent_pnl_krw>=0?'+':''}${ensStats.recent_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span></div></div>
                            </div>
                        </div>
                        <!-- GradBoost Card -->
                        <div class="card" style="background: rgba(14, 20, 47, 0.8); border: 1px solid rgba(255, 160, 0, 0.3); padding: 20px; border-radius: 12px; width: 100%;">
                            <div style="font-weight: 700; font-size: 1.1rem; color: #ffa000; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                                <span>📈 GradBoost (기본 완화)</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 승률</div><div style="font-size:1.4rem; font-weight:700;">${(gbStats.win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${gbStats.total}회 (${gbStats.wins}승/${gbStats.losses}패)</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 손익</div><div style="font-size:1.4rem; font-weight:700; color:${gbStats.total_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${gbStats.total_pnl_krw>=0?'+':''}${gbStats.total_pnl_krw.toLocaleString(undefined, {maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">${gbStats.total_pnl_usdt>=0?'+':''}$${gbStats.total_pnl_usdt.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">평가 자산</div><div style="font-size:1.1rem; font-weight:600;">${gbStats.capital_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">$${gbStats.capital_usdt.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">최근 1시간 현황</div><div style="font-size:1.1rem; font-weight:600; color:${gbStats.recent_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${(gbStats.recent_win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${gbStats.recent_wins}승/${gbStats.recent_losses}패 | <span style="font-weight:600;">${gbStats.recent_pnl_krw>=0?'+':''}${gbStats.recent_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span></div></div>
                            </div>
                        </div>
                        <!-- MLP Card -->
                        <div class="card" style="background: rgba(14, 20, 47, 0.8); border: 1px solid rgba(168, 85, 247, 0.3); padding: 20px; border-radius: 12px; width: 100%;">
                            <div style="font-weight: 700; font-size: 1.1rem; color: #c084fc; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                                <span>🧠 MLP 단독 (타이트)</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 승률</div><div style="font-size:1.4rem; font-weight:700;">${(mlpStats.win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${mlpStats.total}회 (${mlpStats.wins}승/${mlpStats.losses}패)</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 손익</div><div style="font-size:1.4rem; font-weight:700; color:${mlpStats.total_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${mlpStats.total_pnl_krw>=0?'+':''}${mlpStats.total_pnl_krw.toLocaleString(undefined, {maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">${mlpStats.total_pnl_usdt>=0?'+':''}$${mlpStats.total_pnl_usdt.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">평가 자산</div><div style="font-size:1.1rem; font-weight:600;">${mlpStats.capital_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">$${mlpStats.capital_usdt.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</div></div>
                                <div><div style="font-size:0.8rem; color:var(--text-muted);">최근 1시간 현황</div><div style="font-size:1.1rem; font-weight:600; color:${mlpStats.recent_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${(mlpStats.recent_win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${mlpStats.recent_wins}승/${mlpStats.recent_losses}패 | <span style="font-weight:600;">${mlpStats.recent_pnl_krw>=0?'+':''}${mlpStats.recent_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span></div></div>
                            </div>
                        </div>
                    `;
                    
                    const ensWinRatePct = (ensStats.win_rate || 0.0) * 100;
                    const ensProgressPercent = Math.min(100, (ensWinRatePct / 70.0) * 100);
                    const ensProgressTextEl = document.getElementById('ens-winrate-progress-text');
                    const ensProgressBarEl  = document.getElementById('ens-winrate-progress-bar');
                    ensProgressBarEl.style.width = ensProgressPercent.toFixed(1) + '%';
                    if (ensWinRatePct >= 70.0 && ensStats.total >= 30) {
                        ensProgressTextEl.innerHTML = `🔥 <span style="color:var(--color-green); font-weight:700;">목표 달성! (${ensWinRatePct.toFixed(1)}% / 70.0%)</span>`;
                    } else {
                        ensProgressTextEl.innerText = `${ensProgressPercent.toFixed(1)}% (${ensWinRatePct.toFixed(1)}% / 70.0%)`;
                    }
                    
                    const gbTbody = document.getElementById('gb-table-body');
                    gbTbody.innerHTML = '';
                    if (history.length === 0) {
                        gbTbody.innerHTML = `<tr><td colspan="12" style="text-align:center; color:var(--text-muted);">예측 데이터가 없습니다.</td></tr>`;
                    } else {
                        history.slice(0, 30).forEach(p => {
                            const entryPrice = p.entry_price ? p.entry_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4}) : '-';
                            const actualPrice = p.actual_price ? p.actual_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4}) : '-';

                            // 1. Ensemble (ens)
                            let ensSide = p.predicted_side || 'PASS';
                            let ensResultBadge = p.result === 'WIN' ? '<span style="color:var(--color-green); font-weight:600;">승</span>' : (p.result === 'LOSS' ? '<span style="color:var(--color-red); font-weight:600;">패</span>' : '<span style="color:var(--text-muted);">대기</span>');
                            let ensPnlText = (ensSide !== 'PASS' && p.pnl_krw !== null && p.pnl_krw !== undefined) ? `<span style="color:${p.pnl_krw>=0?'var(--color-green)':'var(--color-red)'}; font-weight:600;">${p.pnl_krw>=0?'+':''}${p.pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span>` : '<span style="color:var(--text-muted);">-</span>';

                            // 2. GradBoost (gb)
                            let gbSide = p.gb_basic_predicted_side || 'PASS';
                            let gbResultBadge = p.gb_basic_result === 'WIN' ? '<span style="color:var(--color-green); font-weight:600;">승</span>' : (p.gb_basic_result === 'LOSS' ? '<span style="color:var(--color-red); font-weight:600;">패</span>' : '<span style="color:var(--text-muted);">대기</span>');
                            let gbPnlText = (gbSide !== 'PASS' && p.gb_basic_pnl_krw !== null && p.gb_basic_pnl_krw !== undefined) ? `<span style="color:${p.gb_basic_pnl_krw>=0?'var(--color-green)':'var(--color-red)'}; font-weight:600;">${p.gb_basic_pnl_krw>=0?'+':''}${p.gb_basic_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span>` : '<span style="color:var(--text-muted);">-</span>';

                            // 3. MLP (mlp)
                            let mlpSide = p.gb_current_predicted_side || 'PASS';
                            let mlpResultBadge = p.gb_current_result === 'WIN' ? '<span style="color:var(--color-green); font-weight:600;">승</span>' : (p.gb_current_result === 'LOSS' ? '<span style="color:var(--color-red); font-weight:600;">패</span>' : '<span style="color:var(--text-muted);">대기</span>');
                            let mlpPnlText = (mlpSide !== 'PASS' && p.gb_current_pnl_krw !== null && p.gb_current_pnl_krw !== undefined) ? `<span style="color:${p.gb_current_pnl_krw>=0?'var(--color-green)':'var(--color-red)'}; font-weight:600;">${p.gb_current_pnl_krw>=0?'+':''}${p.gb_current_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span>` : '<span style="color:var(--text-muted);">-</span>';

                            gbTbody.innerHTML += `
                                <tr>
                                    <td style="font-size:0.8rem; color:var(--text-muted);">${p.predict_time.substring(5, 16)}</td>
                                    <td style="font-weight:600;">${p.symbol}</td>
                                    <td style="font-size:0.85rem;"><span style="color:var(--text-muted);">진입:</span> $${entryPrice}<br><span style="color:var(--text-muted);">결과:</span> $${actualPrice}</td>
                                    
                                    <td style="background:rgba(0,212,255,0.02);"><span class="pos-badge ${ensSide==='LONG'?'long':(ensSide==='SHORT'?'short':'flat')}">${ensSide}</span></td>
                                    <td style="background:rgba(0,212,255,0.02); font-weight:bold;">${ensSide!=='PASS'? ((p.ensemble_prob >= 0.5 ? p.ensemble_prob : 1.0 - p.ensemble_prob)*100).toFixed(0)+'%' : '-'}</td>
                                    <td style="background:rgba(0,212,255,0.02);">${ensResultBadge} ${ensPnlText}</td>
                                    
                                    <td><span class="pos-badge ${gbSide==='LONG'?'long':(gbSide==='SHORT'?'short':'flat')}">${gbSide}</span></td>
                                    <td>${gbSide!=='PASS'? ((p.gb_basic_prob >= 0.5 ? p.gb_basic_prob : 1.0 - p.gb_basic_prob)*100).toFixed(0)+'%' : '-'}</td>
                                    <td>${gbResultBadge} ${gbPnlText}</td>
                                    
                                    <td><span class="pos-badge ${mlpSide==='LONG'?'long':(mlpSide==='SHORT'?'short':'flat')}">${mlpSide}</span></td>
                                    <td>${mlpSide!=='PASS'? ((p.gb_current_prob >= 0.5 ? p.gb_current_prob : 1.0 - p.gb_current_prob)*100).toFixed(0)+'%' : '-'}</td>
                                    <td>${mlpResultBadge} ${mlpPnlText}</td>
                                </tr>
                            `;
                        });
                    }
                } catch (err15m) { console.error(err15m); }
            } catch (err) { console.error(err); }
        }

        function updateChart(equityData) {
            const ctx = document.getElementById('equityChart').getContext('2d');
            const labels = equityData.map(d => d.date.substring(5));
            const points = equityData.map(d => d.cumulative_pnl);
            if (equityChart) {
                equityChart.data.labels = labels; equityChart.data.datasets[0].data = points; equityChart.update();
            } else {
                equityChart = new Chart(ctx, {
                    type: 'line', data: { labels: labels, datasets: [{ label: '누적 수익금 (USDT)', data: points, borderColor: '#00d4ff', backgroundColor: 'rgba(0, 212, 255, 0.1)', borderWidth: 2, fill: true, tension: 0.3 }] },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } } } }
                });
            }
        }

        async function toggleBot() {
            const statusRes = await fetch('/api/status'); const statusData = await statusRes.json();
            let url = statusData.status === 'stopped' ? '/api/bot/start' : '/api/bot/stop';
            try { await fetch(url, { method: 'POST' }); updateDashboard(); } catch (err) { alert("봇 제어 실패"); }
        }
        setInterval(updateDashboard, 10000); updateDashboard();
    </script>
</body>
</html>"""

HTML_TEMPLATE_GB = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GradBoost 단독 전략 대시보드</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #060814;
            --card-bg: #0e132b;
            --border-color: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #00d4ff;
            --accent-orange: #ffa000;
            --color-green: #10b981;
            --color-red: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body {
            background-color: var(--bg-color); color: var(--text-main); padding: 20px; padding-bottom: 120px; min-height: 100vh;
            background-image: radial-gradient(circle at 10% 20%, rgba(12, 18, 45, 0.5) 0%, rgba(6, 8, 20, 0.5) 90%);
        }
        .container { max-width: 1250px; margin: 0 auto; }
        header {
            display: flex; justify-content: space-between; align-items: center; padding: 16px 24px;
            background: linear-gradient(135deg, var(--card-bg), #131b3e); border: 1px solid var(--border-color);
            border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); backdrop-filter: blur(10px);
        }
        header h1 { font-size: 1.5rem; font-weight: 700; background: linear-gradient(to right, var(--accent-orange), #ffb300, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .nav-links { display: flex; gap: 16px; }
        .nav-links a { color: var(--text-muted); text-decoration: none; font-size: 0.9rem; font-weight: 600; padding: 8px 16px; border-radius: 8px; border: 1px solid transparent; transition: all 0.2s; background: rgba(255, 255, 255, 0.02); }
        .nav-links a:hover { color: var(--accent-orange); border-color: rgba(255, 160, 0, 0.2); background: rgba(255, 160, 0, 0.05); }
        .nav-links a.active { color: var(--accent-orange); border-color: rgba(255, 160, 0, 0.4); background: rgba(255, 160, 0, 0.1); box-shadow: 0 0 15px rgba(255, 160, 0, 0.1); }
        .status-badge { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 0.95rem; padding: 6px 14px; border-radius: 9999px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); color: var(--color-green); }
        .status-badge.stopped { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); color: var(--color-red); }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background-color: var(--color-green); box-shadow: 0 0 10px var(--color-green); }
        .status-badge.stopped .status-dot { background-color: var(--color-red); box-shadow: 0 0 10px var(--color-red); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); transition: transform 0.2s, border-color 0.2s; }
        .card:hover { transform: translateY(-2px); border-color: rgba(255, 160, 0, 0.3); }
        .card-title { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; margin-bottom: 8px; }
        .card-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
        .main-layout { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; margin-bottom: 24px; }
        .chart-container { min-height: 300px; }
        .position-list { display: flex; flex-direction: column; gap: 12px; }
        .pos-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: rgba(30, 41, 59, 0.3); border: 1px solid var(--border-color); border-radius: 12px; }
        .pos-symbol { font-weight: 600; }
        .pos-badge { font-size: 0.8rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }
        .pos-badge.long { background-color: rgba(16, 185, 129, 0.2); color: var(--color-green); }
        .pos-badge.short { background-color: rgba(239, 68, 68, 0.2); color: var(--color-red); }
        .pos-badge.flat { background-color: rgba(148, 163, 184, 0.1); color: var(--text-muted); }
        .section-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
        th { color: var(--text-muted); font-weight: 500; padding: 12px 16px; border-bottom: 1px solid var(--border-color); }
        td { padding: 14px 16px; border-bottom: 1px solid var(--border-color); color: #e2e8f0; }
        .table-wrapper { overflow-x: auto; border-radius: 12px; border: 1px solid var(--border-color); background: var(--card-bg); }
        .emergency-bar { position: fixed; bottom: 0; left: 0; width: 100%; background: rgba(10, 14, 35, 0.95); border-top: 1px solid var(--border-color); padding: 16px 24px; display: flex; justify-content: center; align-items: center; backdrop-filter: blur(8px); z-index: 1000; }
        .btn { font-size: 1.1rem; font-weight: 700; padding: 14px 48px; border-radius: 12px; cursor: pointer; border: none; transition: all 0.3s ease; display: flex; align-items: center; gap: 10px; }
        .btn-stop { background: linear-gradient(135deg, var(--color-red), #b91c1c); color: white; }
        .btn-stop:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        .btn-start { background: linear-gradient(135deg, var(--color-green), #047857); color: white; }
        .btn-start:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
    </style>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <header>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <h1>📈 GradBoost 단독 전략 대시보드</h1>
                <div class="nav-links">
                    <a href="/">🔗 Ensemble 홈</a>
                    <a href="/gb" class="active">🔗 GradBoost 전용</a>
                    <a href="/mlp">🔗 MLP 전용</a>
                </div>
            </div>
            <div id="bot-status-badge" class="status-badge">
                <div class="status-dot"></div>
                <span id="bot-status-text">실행중</span>
            </div>
        </header>

        <!-- KPI 서머리 (GradBoost 단독 스태츠 투입) -->
        <div class="summary-grid">
            <div class="card">
                <div class="card-title">GradBoost 누적 손익</div>
                <div id="kpi-pnl" class="card-value">+$0.00</div>
            </div>
            <div class="card">
                <div class="card-title">GradBoost 승률</div>
                <div id="kpi-winrate" class="card-value">0.0%</div>
            </div>
            <div class="card">
                <div class="card-title">GradBoost 거래 횟수</div>
                <div id="kpi-trades" class="card-value">0회</div>
            </div>
            <div class="card">
                <div class="card-title">GradBoost 평가 자산</div>
                <div id="kpi-capital" class="card-value">$0.00</div>
            </div>
        </div>

        <!-- 2열 레이아웃 -->
        <div class="main-layout">
            <div class="card">
                <div class="section-title">📉 GradBoost 독자 누적 손익 차트 (15m 예측 기준)</div>
                <div class="chart-container"><canvas id="equityChart"></canvas></div>
            </div>
            <div class="card">
                <div class="section-title">⚡ 실시간 포지션 현황 (참고용)</div>
                <div id="position-list" class="position-list"></div>
            </div>
        </div>

        <!-- GradBoost 모의 투자 상세 현황 카드 -->
        <div class="card" style="margin-bottom: 24px; border: 1px solid rgba(255, 160, 0, 0.3); background: linear-gradient(135deg, var(--card-bg), #1a150c);">
            <div class="section-title" style="color: #ffa000;">📈 15분봉 GradBoost 단독 전략 상세 현황</div>
            <div id="gb-cards-container" class="summary-grid" style="grid-template-columns: 1fr; margin-bottom: 20px;"></div>
            
            <div style="margin-bottom: 24px; padding: 0 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px;">
                    <span>목표 승률 70% 달성도 (GradBoost 전략 기준)</span>
                    <span id="gb-winrate-progress-text" style="font-weight: 600;">0% (0.0% / 70.0%)</span>
                </div>
                <div style="width: 100%; height: 12px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden; border: 1px solid var(--border-color);">
                    <div id="gb-winrate-progress-bar" style="width: 0%; height: 100%; background: linear-gradient(to right, #ffa000, #ffc107); box-shadow: 0 0 10px #ffa000; transition: width 0.5s ease;"></div>
                </div>
            </div>

            <div class="section-title" style="font-size: 0.95rem; margin-top: 20px;">📜 최근 15분봉 GradBoost 예측 기록 (최근 30건)</div>
            <div class="table-wrapper" style="background: rgba(10, 14, 26, 0.4);">
                <table>
                    <thead>
                        <tr>
                            <th>예측 시간</th><th>종목</th><th>진입 / 결과가</th><th style="color: #ffa000;">GradBoost 진입 구분</th><th style="color: #ffa000;">진입 확률</th><th style="color: #ffa000;">결과</th><th style="color: #ffa000;">수익금 (KRW)</th>
                        </tr>
                    </thead>
                    <tbody id="gb-table-body">
                        <tr><td colspan="7" style="text-align:center; color:var(--text-muted);">데이터 로딩 중...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="emergency-bar">
        <button id="ctrl-btn" class="btn btn-stop" onclick="toggleBot()">
            <span id="ctrl-btn-text">🛑 봇 긴급 정지</span>
        </button>
    </div>

    <script>
        let equityChart = null;
        async function updateDashboard() {
            try {
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                const isRunning = statusData.status === 'running';
                
                const badge = document.getElementById('bot-status-badge');
                const badgeText = document.getElementById('bot-status-text');
                const ctrlBtn = document.getElementById('ctrl-btn');
                const ctrlBtnText = document.getElementById('ctrl-btn-text');

                if (isRunning) {
                    badge.className = 'status-badge'; badgeText.innerText = '실행중';
                    ctrlBtn.className = 'btn btn-stop'; ctrlBtnText.innerText = '🛑 봇 긴급 정지';
                } else {
                    badge.className = 'status-badge stopped'; badgeText.innerText = '정지됨';
                    ctrlBtn.className = 'btn btn-start'; ctrlBtnText.innerText = '🟢 봇 매매 재개';
                }

                const positionsRes = await fetch('/api/positions');
                const positions = await positionsRes.json();
                const posContainer = document.getElementById('position-list');
                posContainer.innerHTML = '';
                for (const [symbol, pos] of Object.entries(positions)) {
                    const size = parseFloat(pos.size);
                    let sideText = '대기'; let badgeClass = 'pos-badge flat';
                    if (size > 0) { sideText = '🟢 LONG'; badgeClass = 'pos-badge long'; }
                    else if (size < 0) { sideText = '🔴 SHORT'; badgeClass = 'pos-badge short'; }
                    posContainer.innerHTML += `<div class="pos-item"><span class="pos-symbol">${symbol}</span><div><span class="${badgeClass}">${sideText}</span></div></div>`;
                }

                const res15m = await fetch('/api/predict15m');
                const data15m = await res15m.json();
                const stats = data15m.stats;
                const history = data15m.history;

                const empty_model_stats = { total: 0, wins: 0, losses: 0, win_rate: 0.0, total_pnl_krw: 0.0, total_pnl_usdt: 0.0, capital_krw: 100000.0, capital_usdt: 100000.0/1350.0, recent_wins:0, recent_losses:0, recent_win_rate:0.0, recent_pnl_krw:0.0 };
                const gbStats = stats['gb'] || empty_model_stats;

                const pnlEl = document.getElementById('kpi-pnl');
                const pnlVal = parseFloat(gbStats.total_pnl_usdt);
                pnlEl.innerText = (pnlVal >= 0 ? '+' : '') + '$' + pnlVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                pnlEl.style.color = pnlVal >= 0 ? 'var(--color-green)' : 'var(--color-red)';
                
                document.getElementById('kpi-winrate').innerText = (parseFloat(gbStats.win_rate) * 100).toFixed(1) + '%';
                document.getElementById('kpi-trades').innerText = gbStats.total + '회';
                document.getElementById('kpi-capital').innerText = '$' + gbStats.capital_usdt.toLocaleString(undefined, {minimumFractionDigits: 2});

                // GradBoost Card 상세
                document.getElementById('gb-cards-container').innerHTML = `
                    <div class="card" style="background: rgba(14, 20, 47, 0.8); border: 1px solid rgba(255, 160, 0, 0.5); padding: 20px; border-radius: 12px; width: 100%;">
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 전적</div><div style="font-size:1.4rem; font-weight:700;">${gbStats.wins}승 ${gbStats.losses}패</div><div style="font-size:0.75rem; color:var(--text-muted);">총 거래 ${gbStats.total}회</div></div>
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 손익 (원)</div><div style="font-size:1.4rem; font-weight:700; color:${gbStats.total_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${gbStats.total_pnl_krw>=0?'+':''}${gbStats.total_pnl_krw.toLocaleString(undefined, {maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">USDT: $${gbStats.total_pnl_usdt.toFixed(2)}</div></div>
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">평가 자산 (원)</div><div style="font-size:1.4rem; font-weight:700;">${gbStats.capital_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">USDT: $${gbStats.capital_usdt.toFixed(2)}</div></div>
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">최근 1시간 현황</div><div style="font-size:1.4rem; font-weight:700; color:${gbStats.recent_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${(gbStats.recent_win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${gbStats.recent_wins}승/${gbStats.recent_losses}패 | <span style="font-weight:600;">${gbStats.recent_pnl_krw>=0?'+':''}${gbStats.recent_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span></div></div>
                        </div>
                    </div>
                `;

                const gbWinRatePct = (gbStats.win_rate || 0.0) * 100;
                const gbProgressPercent = Math.min(100, (gbWinRatePct / 70.0) * 100);
                const gbProgressTextEl = document.getElementById('gb-winrate-progress-text');
                const gbProgressBarEl  = document.getElementById('gb-winrate-progress-bar');
                gbProgressBarEl.style.width = gbProgressPercent.toFixed(1) + '%';
                if (gbWinRatePct >= 70.0 && gbStats.total >= 30) {
                    gbProgressTextEl.innerHTML = `🔥 <span style="color:var(--color-green); font-weight:700;">목표 달성! (${gbWinRatePct.toFixed(1)}% / 70.0%)</span>`;
                } else {
                    gbProgressTextEl.innerText = `${gbProgressPercent.toFixed(1)}% (${gbWinRatePct.toFixed(1)}% / 70.0%)`;
                }

                // GradBoost 예측만 추출해서 테이블에 그리기
                const gbTbody = document.getElementById('gb-table-body');
                gbTbody.innerHTML = '';
                if (history.length === 0) {
                    gbTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">예측 데이터가 없습니다.</td></tr>`;
                } else {
                    history.slice(0, 30).forEach(p => {
                        const entryPrice = p.entry_price ? p.entry_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4}) : '-';
                        const actualPrice = p.actual_price ? p.actual_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4}) : '-';

                        let basicSide = p.gb_basic_predicted_side || 'PASS';
                        let basicResultBadge = p.gb_basic_result === 'WIN' ? '<span style="color:var(--color-green); font-weight:600;">승</span>' : (p.gb_basic_result === 'LOSS' ? '<span style="color:var(--color-red); font-weight:600;">패</span>' : '<span style="color:var(--text-muted);">대기</span>');
                        let basicPnlText = (basicSide !== 'PASS' && p.gb_basic_pnl_krw !== null && p.gb_basic_pnl_krw !== undefined) ? `<span style="color:${p.gb_basic_pnl_krw>=0?'var(--color-green)':'var(--color-red)'}; font-weight:600;">${p.gb_basic_pnl_krw>=0?'+':''}${p.gb_basic_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span>` : '<span style="color:var(--text-muted);">-</span>';

                        gbTbody.innerHTML += `
                            <tr>
                                <td style="font-size:0.8rem; color:var(--text-muted);">${p.predict_time.substring(5, 16)}</td>
                                <td style="font-weight:600;">${p.symbol}</td>
                                <td style="font-size:0.85rem;"><span style="color:var(--text-muted);">진입:</span> $${entryPrice}<br><span style="color:var(--text-muted);">결과:</span> $${actualPrice}</td>
                                <td><span class="pos-badge ${basicSide==='LONG'?'long':(basicSide==='SHORT'?'short':'flat')}">${basicSide}</span></td>
                                <td>${basicSide!=='PASS'? ((p.gb_basic_prob >= 0.5 ? p.gb_basic_prob : 1.0 - p.gb_basic_prob)*100).toFixed(0)+'%' : '-'}</td>
                                <td>${basicResultBadge}</td>
                                <td>${basicPnlText}</td>
                            </tr>
                        `;
                    });
                }

                // GradBoost Dynamic Equity curve
                const revHistory = [...history].reverse();
                let currentPnl = 0.0;
                const chartLabels = ["Start"];
                const chartPoints = [0];
                revHistory.forEach(p => {
                    if (p.gb_basic_predicted_side !== 'PASS' && p.gb_basic_pnl_krw !== null && p.gb_basic_pnl_krw !== undefined) {
                        currentPnl += p.gb_basic_pnl_krw / 1350.0; // USDT
                        chartLabels.push(p.predict_time.substring(5, 16));
                        chartPoints.push(parseFloat(currentPnl.toFixed(2)));
                    }
                });
                updateChart(chartLabels, chartPoints);

            } catch (err) { console.error(err); }
        }

        function updateChart(labels, points) {
            const ctx = document.getElementById('equityChart').getContext('2d');
            if (equityChart) {
                equityChart.data.labels = labels; equityChart.data.datasets[0].data = points; equityChart.update();
            } else {
                equityChart = new Chart(ctx, {
                    type: 'line', data: { labels: labels, datasets: [{ label: 'GradBoost 누적 손익 (USDT)', data: points, borderColor: '#ffa000', backgroundColor: 'rgba(255, 160, 0, 0.1)', borderWidth: 2, fill: true, tension: 0.3 }] },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } } } }
                });
            }
        }

        async function toggleBot() {
            const statusRes = await fetch('/api/status'); const statusData = await statusRes.json();
            let url = statusData.status === 'stopped' ? '/api/bot/start' : '/api/bot/stop';
            try { await fetch(url, { method: 'POST' }); updateDashboard(); } catch (err) { alert("봇 제어 실패"); }
        }
        setInterval(updateDashboard, 10000); updateDashboard();
    </script>
</body>
</html>"""

HTML_TEMPLATE_MLP = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLP 단독 전략 대시보드</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #060814;
            --card-bg: #0e132b;
            --border-color: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #00d4ff;
            --accent-purple: #c084fc;
            --color-green: #10b981;
            --color-red: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body {
            background-color: var(--bg-color); color: var(--text-main); padding: 20px; padding-bottom: 120px; min-height: 100vh;
            background-image: radial-gradient(circle at 10% 20%, rgba(12, 18, 45, 0.5) 0%, rgba(6, 8, 20, 0.5) 90%);
        }
        .container { max-width: 1250px; margin: 0 auto; }
        header {
            display: flex; justify-content: space-between; align-items: center; padding: 16px 24px;
            background: linear-gradient(135deg, var(--card-bg), #131b3e); border: 1px solid var(--border-color);
            border-radius: 16px; margin-bottom: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); backdrop-filter: blur(10px);
        }
        header h1 { font-size: 1.5rem; font-weight: 700; background: linear-gradient(to right, var(--accent-purple), #a855f7, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .nav-links { display: flex; gap: 16px; }
        .nav-links a { color: var(--text-muted); text-decoration: none; font-size: 0.9rem; font-weight: 600; padding: 8px 16px; border-radius: 8px; border: 1px solid transparent; transition: all 0.2s; background: rgba(255, 255, 255, 0.02); }
        .nav-links a:hover { color: var(--accent-purple); border-color: rgba(168, 85, 247, 0.2); background: rgba(168, 85, 247, 0.05); }
        .nav-links a.active { color: var(--accent-purple); border-color: rgba(168, 85, 247, 0.4); background: rgba(168, 85, 247, 0.1); box-shadow: 0 0 15px rgba(168, 85, 247, 0.1); }
        .status-badge { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 0.95rem; padding: 6px 14px; border-radius: 9999px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); color: var(--color-green); }
        .status-badge.stopped { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); color: var(--color-red); }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background-color: var(--color-green); box-shadow: 0 0 10px var(--color-green); }
        .status-badge.stopped .status-dot { background-color: var(--color-red); box-shadow: 0 0 10px var(--color-red); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); transition: transform 0.2s, border-color 0.2s; }
        .card:hover { transform: translateY(-2px); border-color: rgba(168, 85, 247, 0.3); }
        .card-title { font-size: 0.85rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; margin-bottom: 8px; }
        .card-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
        .main-layout { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; margin-bottom: 24px; }
        .chart-container { min-height: 300px; }
        .position-list { display: flex; flex-direction: column; gap: 12px; }
        .pos-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: rgba(30, 41, 59, 0.3); border: 1px solid var(--border-color); border-radius: 12px; }
        .pos-symbol { font-weight: 600; }
        .pos-badge { font-size: 0.8rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }
        .pos-badge.long { background-color: rgba(16, 185, 129, 0.2); color: var(--color-green); }
        .pos-badge.short { background-color: rgba(239, 68, 68, 0.2); color: var(--color-red); }
        .pos-badge.flat { background-color: rgba(148, 163, 184, 0.1); color: var(--text-muted); }
        .section-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
        th { color: var(--text-muted); font-weight: 500; padding: 12px 16px; border-bottom: 1px solid var(--border-color); }
        td { padding: 14px 16px; border-bottom: 1px solid var(--border-color); color: #e2e8f0; }
        .table-wrapper { overflow-x: auto; border-radius: 12px; border: 1px solid var(--border-color); background: var(--card-bg); }
        .emergency-bar { position: fixed; bottom: 0; left: 0; width: 100%; background: rgba(10, 14, 35, 0.95); border-top: 1px solid var(--border-color); padding: 16px 24px; display: flex; justify-content: center; align-items: center; backdrop-filter: blur(8px); z-index: 1000; }
        .btn { font-size: 1.1rem; font-weight: 700; padding: 14px 48px; border-radius: 12px; cursor: pointer; border: none; transition: all 0.3s ease; display: flex; align-items: center; gap: 10px; }
        .btn-stop { background: linear-gradient(135deg, var(--color-red), #b91c1c); color: white; }
        .btn-stop:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
        .btn-start { background: linear-gradient(135deg, var(--color-green), #047857); color: white; }
        .btn-start:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
    </style>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <header>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <h1>🧠 MLP 단독 전략 대시보드</h1>
                <div class="nav-links">
                    <a href="/">🔗 Ensemble 홈</a>
                    <a href="/gb">🔗 GradBoost 전용</a>
                    <a href="/mlp" class="active">🔗 MLP 전용</a>
                </div>
            </div>
            <div id="bot-status-badge" class="status-badge">
                <div class="status-dot"></div>
                <span id="bot-status-text">실행중</span>
            </div>
        </header>

        <!-- KPI 서머리 (MLP 단독 스태츠 투입) -->
        <div class="summary-grid">
            <div class="card">
                <div class="card-title">MLP 누적 손익</div>
                <div id="kpi-pnl" class="card-value">+$0.00</div>
            </div>
            <div class="card">
                <div class="card-title">MLP 승률</div>
                <div id="kpi-winrate" class="card-value">0.0%</div>
            </div>
            <div class="card">
                <div class="card-title">MLP 거래 횟수</div>
                <div id="kpi-trades" class="card-value">0회</div>
            </div>
            <div class="card">
                <div class="card-title">MLP 평가 자산</div>
                <div id="kpi-capital" class="card-value">$0.00</div>
            </div>
        </div>

        <!-- 2열 레이아웃 -->
        <div class="main-layout">
            <div class="card">
                <div class="section-title">📉 MLP 독자 누적 손익 차트 (15m 예측 기준)</div>
                <div class="chart-container"><canvas id="equityChart"></canvas></div>
            </div>
            <div class="card">
                <div class="section-title">⚡ 실시간 포지션 현황 (참고용)</div>
                <div id="position-list" class="position-list"></div>
            </div>
        </div>

        <!-- MLP 모의 투자 상세 현황 카드 -->
        <div class="card" style="margin-bottom: 24px; border: 1px solid rgba(168, 85, 247, 0.3); background: linear-gradient(135deg, var(--card-bg), #170c1f);">
            <div class="section-title" style="color: #c084fc;">🧠 15분봉 MLP 단독 전략 상세 현황</div>
            <div id="mlp-cards-container" class="summary-grid" style="grid-template-columns: 1fr; margin-bottom: 20px;"></div>
            
            <div style="margin-bottom: 24px; padding: 0 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px;">
                    <span>목표 승률 70% 달성도 (MLP 전략 기준)</span>
                    <span id="mlp-winrate-progress-text" style="font-weight: 600;">0% (0.0% / 70.0%)</span>
                </div>
                <div style="width: 100%; height: 12px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden; border: 1px solid var(--border-color);">
                    <div id="mlp-winrate-progress-bar" style="width: 0%; height: 100%; background: linear-gradient(to right, #a855f7, #c084fc); box-shadow: 0 0 10px #a855f7; transition: width 0.5s ease;"></div>
                </div>
            </div>

            <div class="section-title" style="font-size: 0.95rem; margin-top: 20px;">📜 최근 15분봉 MLP 예측 기록 (최근 30건)</div>
            <div class="table-wrapper" style="background: rgba(10, 14, 26, 0.4);">
                <table>
                    <thead>
                        <tr>
                            <th>예측 시간</th><th>종목</th><th>진입 / 결과가</th><th style="color: #c084fc;">MLP 진입 구분</th><th style="color: #c084fc;">진입 확률</th><th style="color: #c084fc;">결과</th><th style="color: #c084fc;">수익금 (KRW)</th>
                        </tr>
                    </thead>
                    <tbody id="mlp-table-body">
                        <tr><td colspan="7" style="text-align:center; color:var(--text-muted);">데이터 로딩 중...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="emergency-bar">
        <button id="ctrl-btn" class="btn btn-stop" onclick="toggleBot()">
            <span id="ctrl-btn-text">🛑 봇 긴급 정지</span>
        </button>
    </div>

    <script>
        let equityChart = null;
        async function updateDashboard() {
            try {
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                const isRunning = statusData.status === 'running';
                
                const badge = document.getElementById('bot-status-badge');
                const badgeText = document.getElementById('bot-status-text');
                const ctrlBtn = document.getElementById('ctrl-btn');
                const ctrlBtnText = document.getElementById('ctrl-btn-text');

                if (isRunning) {
                    badge.className = 'status-badge'; badgeText.innerText = '실행중';
                    ctrlBtn.className = 'btn btn-stop'; ctrlBtnText.innerText = '🛑 봇 긴급 정지';
                } else {
                    badge.className = 'status-badge stopped'; badgeText.innerText = '정지됨';
                    ctrlBtn.className = 'btn btn-start'; ctrlBtnText.innerText = '🟢 봇 매매 재개';
                }

                const positionsRes = await fetch('/api/positions');
                const positions = await positionsRes.json();
                const posContainer = document.getElementById('position-list');
                posContainer.innerHTML = '';
                for (const [symbol, pos] of Object.entries(positions)) {
                    const size = parseFloat(pos.size);
                    let sideText = '대기'; let badgeClass = 'pos-badge flat';
                    if (size > 0) { sideText = '🟢 LONG'; badgeClass = 'pos-badge long'; }
                    else if (size < 0) { sideText = '🔴 SHORT'; badgeClass = 'pos-badge short'; }
                    posContainer.innerHTML += `<div class="pos-item"><span class="pos-symbol">${symbol}</span><div><span class="${badgeClass}">${sideText}</span></div></div>`;
                }

                const res15m = await fetch('/api/predict15m');
                const data15m = await res15m.json();
                const stats = data15m.stats;
                const history = data15m.history;

                const empty_model_stats = { total: 0, wins: 0, losses: 0, win_rate: 0.0, total_pnl_krw: 0.0, total_pnl_usdt: 0.0, capital_krw: 100000.0, capital_usdt: 100000.0/1350.0, recent_wins:0, recent_losses:0, recent_win_rate:0.0, recent_pnl_krw:0.0 };
                const mlpStats = stats['mlp'] || empty_model_stats;

                const pnlEl = document.getElementById('kpi-pnl');
                const pnlVal = parseFloat(mlpStats.total_pnl_usdt);
                pnlEl.innerText = (pnlVal >= 0 ? '+' : '') + '$' + pnlVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                pnlEl.style.color = pnlVal >= 0 ? 'var(--color-green)' : 'var(--color-red)';
                
                document.getElementById('kpi-winrate').innerText = (parseFloat(mlpStats.win_rate) * 100).toFixed(1) + '%';
                document.getElementById('kpi-trades').innerText = mlpStats.total + '회';
                document.getElementById('kpi-capital').innerText = '$' + mlpStats.capital_usdt.toLocaleString(undefined, {minimumFractionDigits: 2});

                // MLP Card 상세
                document.getElementById('mlp-cards-container').innerHTML = `
                    <div class="card" style="background: rgba(14, 20, 47, 0.8); border: 1px solid rgba(168, 85, 247, 0.5); padding: 20px; border-radius: 12px; width: 100%;">
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 전적</div><div style="font-size:1.4rem; font-weight:700;">${mlpStats.wins}승 ${mlpStats.losses}패</div><div style="font-size:0.75rem; color:var(--text-muted);">총 거래 ${mlpStats.total}회</div></div>
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">누적 손익 (원)</div><div style="font-size:1.4rem; font-weight:700; color:${mlpStats.total_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${mlpStats.total_pnl_krw>=0?'+':''}${mlpStats.total_pnl_krw.toLocaleString(undefined, {maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">USDT: $${mlpStats.total_pnl_usdt.toFixed(2)}</div></div>
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">평가 자산 (원)</div><div style="font-size:1.4rem; font-weight:700;">${mlpStats.capital_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</div><div style="font-size:0.75rem; color:var(--text-muted);">USDT: $${mlpStats.capital_usdt.toFixed(2)}</div></div>
                            <div><div style="font-size:0.8rem; color:var(--text-muted);">최근 1시간 현황</div><div style="font-size:1.4rem; font-weight:700; color:${mlpStats.recent_pnl_krw>=0?'var(--color-green)':'var(--color-red)'};">${(mlpStats.recent_win_rate*100).toFixed(1)}%</div><div style="font-size:0.75rem; color:var(--text-muted);">${mlpStats.recent_wins}승/${mlpStats.recent_losses}패 | <span style="font-weight:600;">${mlpStats.recent_pnl_krw>=0?'+':''}${mlpStats.recent_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span></div></div>
                        </div>
                    </div>
                `;

                const mlpWinRatePct = (mlpStats.win_rate || 0.0) * 100;
                const mlpProgressPercent = Math.min(100, (mlpWinRatePct / 70.0) * 100);
                const mlpProgressTextEl = document.getElementById('mlp-winrate-progress-text');
                const mlpProgressBarEl  = document.getElementById('mlp-winrate-progress-bar');
                mlpProgressBarEl.style.width = mlpProgressPercent.toFixed(1) + '%';
                if (mlpWinRatePct >= 70.0 && mlpStats.total >= 30) {
                    mlpProgressTextEl.innerHTML = `🔥 <span style="color:var(--color-green); font-weight:700;">목표 달성! (${mlpWinRatePct.toFixed(1)}% / 70.0%)</span>`;
                } else {
                    mlpProgressTextEl.innerText = `${mlpProgressPercent.toFixed(1)}% (${mlpWinRatePct.toFixed(1)}% / 70.0%)`;
                }

                // MLP 예측만 추출해서 테이블에 그리기
                const mlpTbody = document.getElementById('mlp-table-body');
                mlpTbody.innerHTML = '';
                if (history.length === 0) {
                    mlpTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted);">예측 데이터가 없습니다.</td></tr>`;
                } else {
                    history.slice(0, 30).forEach(p => {
                        const entryPrice = p.entry_price ? p.entry_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4}) : '-';
                        const actualPrice = p.actual_price ? p.actual_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4}) : '-';

                        let currentSide = p.gb_current_predicted_side || p.mlp_predicted_side || 'PASS';
                        let currentResultBadge = p.gb_current_result === 'WIN' ? '<span style="color:var(--color-green); font-weight:600;">승</span>' : (p.gb_current_result === 'LOSS' ? '<span style="color:var(--color-red); font-weight:600;">패</span>' : '<span style="color:var(--text-muted);">대기</span>');
                        let currentPnlText = (currentSide !== 'PASS' && p.gb_current_pnl_krw !== null && p.gb_current_pnl_krw !== undefined) ? `<span style="color:${p.gb_current_pnl_krw>=0?'var(--color-green)':'var(--color-red)'}; font-weight:600;">${p.gb_current_pnl_krw>=0?'+':''}${p.gb_current_pnl_krw.toLocaleString(undefined,{maximumFractionDigits:0})}원</span>` : '<span style="color:var(--text-muted);">-</span>';

                        mlpTbody.innerHTML += `
                            <tr>
                                <td style="font-size:0.8rem; color:var(--text-muted);">${p.predict_time.substring(5, 16)}</td>
                                <td style="font-weight:600;">${p.symbol}</td>
                                <td style="font-size:0.85rem;"><span style="color:var(--text-muted);">진입:</span> $${entryPrice}<br><span style="color:var(--text-muted);">결과:</span> $${actualPrice}</td>
                                <td><span class="pos-badge ${currentSide==='LONG'?'long':(currentSide==='SHORT'?'short':'flat')}">${currentSide}</span></td>
                                <td>${currentSide!=='PASS'? ((p.gb_current_prob >= 0.5 ? p.gb_current_prob : 1.0 - p.gb_current_prob)*100).toFixed(0)+'%' : '-'}</td>
                                <td>${currentResultBadge}</td>
                                <td>${currentPnlText}</td>
                            </tr>
                        `;
                    });
                }

                // MLP Dynamic Equity curve
                const revHistory = [...history].reverse();
                let currentPnl = 0.0;
                const chartLabels = ["Start"];
                const chartPoints = [0];
                revHistory.forEach(p => {
                    if (p.gb_current_predicted_side !== 'PASS' && p.gb_current_pnl_krw !== null && p.gb_current_pnl_krw !== undefined) {
                        currentPnl += p.gb_current_pnl_krw / 1350.0; // USDT
                        chartLabels.push(p.predict_time.substring(5, 16));
                        chartPoints.push(parseFloat(currentPnl.toFixed(2)));
                    }
                });
                updateChart(chartLabels, chartPoints);
            } catch (err) { console.error(err); }
        }

        function updateChart(labels, points) {
            const ctx = document.getElementById('equityChart').getContext('2d');
            if (equityChart) {
                equityChart.data.labels = labels; equityChart.data.datasets[0].data = points; equityChart.update();
            } else {
                equityChart = new Chart(ctx, {
                    type: 'line', data: { labels: labels, datasets: [{ label: 'MLP 누적 손익 (USDT)', data: points, borderColor: '#a855f7', backgroundColor: 'rgba(168, 85, 247, 0.1)', borderWidth: 2, fill: true, tension: 0.3 }] },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } } } }
                });
            }
        }

        async function toggleBot() {
            const statusRes = await fetch('/api/status'); const statusData = await statusRes.json();
            let url = statusData.status === 'stopped' ? '/api/bot/start' : '/api/bot/stop';
            try { await fetch(url, { method: 'POST' }); updateDashboard(); } catch (err) { alert("봇 제어 실패"); }
        }
        setInterval(updateDashboard, 10000); updateDashboard();
    </script>
</body>
</html>"""
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE_ENS)

@app.route('/gb')
def gb_dashboard():
    return render_template_string(HTML_TEMPLATE_GB)

@app.route('/mlp')
def mlp_dashboard():
    return render_template_string(HTML_TEMPLATE_MLP)

@app.route('/api/status', methods=['GET'])
def get_status():
    capital = state.current_capital
    if capital <= 0.0:
        try:
            preds = data_manager.db_load_predictions()
            completed = [p for p in preds if p.get('status') == 'COMPLETED']
            total_pnl_usdt = sum(float(p.get('pnl_usdt') or 0.0) for p in completed)
            capital = (100000.0 / 1350.0) + total_pnl_usdt
        except Exception as e:
            print(f"[Dashboard API] Error calculating capital fallback: {e}")
            capital = 100000.0 / 1350.0
            
    return jsonify({
        'status': 'running' if state.is_running else 'stopped',
        'capital': capital
    })

@app.route('/api/summary', methods=['GET'])
def get_summary():
    summary = logger.get_daily_summary()
    return jsonify(summary)

@app.route('/api/equity', methods=['GET'])
def get_equity():
    equity = logger.get_equity_curve(30)
    return jsonify(equity)

@app.route('/api/trades', methods=['GET'])
def get_trades():
    trades = logger.get_recent_trades(20)
    return jsonify(trades)

@app.route('/api/positions', methods=['GET'])
def get_positions():
    is_all_flat = all(pos['size'] == 0.0 for pos in state.positions.values())
    if is_all_flat:
        try:
            preds = data_manager.db_load_predictions()
            pending = [p for p in preds if p.get('status') == 'PENDING']
            mock_positions = {symbol: {
                'size': 0.0,
                'entry_price': 0.0,
                'unrealized_pnl': 0.0,
                'side': None,
                'liquidation_price': 0.0,
                'leverage': 5
            } for symbol in config.SYMBOLS}
            
            for p in pending:
                symbol = p.get('symbol')
                if symbol in mock_positions:
                    side = p.get('predicted_side')
                    if side in ['LONG', 'SHORT']:
                        mock_positions[symbol]['side'] = side
                        mock_positions[symbol]['size'] = 1.0 if side == 'LONG' else -1.0
                        mock_positions[symbol]['entry_price'] = float(p.get('entry_price') or 0.0)
                        
            return jsonify(mock_positions)
        except Exception as e:
            print(f"[Dashboard API] Error loading positions fallback: {e}")
            
    return jsonify(state.positions)

@app.route('/api/models', methods=['GET'])
def get_models():
    state.load_initial_metadata()
    return jsonify(state.models_status)

def get_15m_stats_and_history():
    empty_model_stats = {
        'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
        'recent_wins': 0, 'recent_losses': 0, 'recent_win_rate': 0.0, 'recent_total': 0,
        'total_pnl_krw': 0.0, 'total_pnl_usdt': 0.0,
        'recent_pnl_krw': 0.0, 'recent_pnl_usdt': 0.0,
        'capital_krw': 100000.0, 'capital_usdt': 100000.0 / 1350.0
    }
    
    models = ['ens', 'gb', 'mlp']
    default_res = {
        'stats': {m: empty_model_stats.copy() for m in models},
        'history': []
    }
    
    try:
        preds = data_manager.db_load_predictions()
    except Exception as e:
        print(f"[Dashboard] Error loading predictions from DB: {e}")
        return default_res
        
    completed = [p for p in preds if p.get('status') == 'COMPLETED']
    margin_krw = 1000000.0
    exchange_rate = 1350.0
    margin_usdt = margin_krw / exchange_rate
    
    for p in completed:
        entry_price = float(p.get('entry_price', 0.0))
        actual_price = float(p.get('actual_price', 0.0))
        if entry_price <= 0.0 or actual_price <= 0.0: continue
            
        for m in models:
            if m == 'ens':
                side_k, res_k, pnl_krw_k, pnl_usdt_k, prob_k = 'predicted_side', 'result', 'pnl_krw', 'pnl_usdt', 'ensemble_prob'
            elif m == 'gb':
                side_k, res_k, pnl_krw_k, pnl_usdt_k, prob_k = 'gb_basic_predicted_side', 'gb_basic_result', 'gb_basic_pnl_krw', 'gb_basic_pnl_usdt', 'gb_basic_prob'
            else:  # mlp -> maps to gb_current in db
                side_k, res_k, pnl_krw_k, pnl_usdt_k, prob_k = 'gb_current_predicted_side', 'gb_current_result', 'gb_current_pnl_krw', 'gb_current_pnl_usdt', 'gb_current_prob'
            
            m_side = p.get(side_k)
            if not m_side:
                m_prob = p.get(prob_k, 0.5)
                m_side = 'LONG' if m_prob >= 0.55 else ('SHORT' if m_prob <= 0.45 else 'PASS')
                p[side_k] = m_side
                
            if p.get(res_k) is None:
                if m_side == 'PASS': p[res_k] = 'PASS'
                else: p[res_k] = 'WIN' if (m_side == 'LONG' and actual_price > entry_price) or (m_side == 'SHORT' and actual_price < entry_price) else 'LOSS'
                
            if p.get(pnl_krw_k) is None:
                if m_side == 'PASS': p[pnl_usdt_k], p[pnl_krw_k] = 0.0, 0.0
                else:
                    ret = ((actual_price - entry_price) / entry_price) if m_side == 'LONG' else ((entry_price - actual_price) / entry_price)
                    net = ret - 0.0008
                    p[pnl_usdt_k], p[pnl_krw_k] = margin_usdt * net * 5, margin_krw * net * 5
 
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_completed = [p for p in completed if datetime.strptime(p.get('target_time_str', '1970-01-01 00:00:00'), '%Y-%m-%d %H:%M:%S') >= one_hour_ago]
    
    stats_out = {}
    for m in models:
        if m == 'ens':
            res_k, pnl_krw_k, pnl_usdt_k = 'result', 'pnl_krw', 'pnl_usdt'
        elif m == 'gb':
            res_k, pnl_krw_k, pnl_usdt_k = 'gb_basic_result', 'gb_basic_pnl_krw', 'gb_basic_pnl_usdt'
        else:  # mlp -> maps to gb_current in db
            res_k, pnl_krw_k, pnl_usdt_k = 'gb_current_result', 'gb_current_pnl_krw', 'gb_current_pnl_usdt'
            
        # Cumulative
        wins = sum(1 for p in completed if p.get(res_k) == 'WIN')
        losses = sum(1 for p in completed if p.get(res_k) == 'LOSS')
        active_trades = wins + losses
        win_rate = wins / active_trades if active_trades > 0 else 0.0
        total_pnl_krw = sum(float(p.get(pnl_krw_k) or 0.0) for p in completed)
        total_pnl_usdt = sum(float(p.get(pnl_usdt_k) or 0.0) for p in completed)
        capital_krw = 100000.0 + total_pnl_krw
        capital_usdt = capital_krw / exchange_rate
        
        # Recent
        recent_wins = sum(1 for p in recent_completed if p.get(res_k) == 'WIN')
        recent_losses = sum(1 for p in recent_completed if p.get(res_k) == 'LOSS')
        recent_active_trades = recent_wins + recent_losses
        recent_win_rate = recent_wins / recent_active_trades if recent_active_trades > 0 else 0.0
        recent_pnl_krw = sum(float(p.get(pnl_krw_k) or 0.0) for p in recent_completed)
        recent_pnl_usdt = sum(float(p.get(pnl_usdt_k) or 0.0) for p in recent_completed)
        
        stats_out[m] = {
            'total': active_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'recent_total': recent_active_trades,
            'recent_wins': recent_wins,
            'recent_losses': recent_losses,
            'recent_win_rate': recent_win_rate,
            'total_pnl_krw': total_pnl_krw,
            'total_pnl_usdt': total_pnl_usdt,
            'recent_pnl_krw': recent_pnl_krw,
            'recent_pnl_usdt': recent_pnl_usdt,
            'capital_krw': capital_krw,
            'capital_usdt': capital_usdt
        }
        
    sorted_preds = sorted(preds, key=lambda x: x.get('target_time', 0), reverse=True)
    recent_history = sorted_preds[:50]
    
    return {
        'stats': stats_out,
        'history': recent_history
    }

@app.route('/api/predict15m', methods=['GET'])
def get_predict15m():
    state.load_initial_metadata()
    data = get_15m_stats_and_history()
    return jsonify(data)

@app.route('/api/bot/stop', methods=['POST'])
def bot_stop():
    state.is_running = False
    print("[Dashboard] Emergency STOP triggered by User via Dashboard.")
    return jsonify({'status': 'stopped'})

@app.route('/api/bot/start', methods=['POST'])
def bot_start():
    # 모의투자 거래내역 및 관련 로그 파일 초기화
    data_manager.clear_db_predictions()
    logger.clear_logs()
    
    # 가용 포지션 초기화
    state.positions = {symbol: {
        'size': 0.0,
        'entry_price': 0.0,
        'unrealized_pnl': 0.0,
        'side': None,
        'liquidation_price': 0.0,
        'leverage': 5
    } for symbol in config.SYMBOLS}
    
    state.is_running = True
    print("[Dashboard] Bot resume triggered by User via Dashboard. Reset mock investment trade history & logs.")
    return jsonify({'status': 'running'})

def start_dashboard_server():
    """
    백그라운드에서 구동하기 위한 서버 시작 함수
    """
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=False, use_reloader=False)

if __name__ == '__main__':
    start_dashboard_server()
