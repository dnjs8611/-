with open('dashboard.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'def get_15m_stats_and_history' in line:
        print(f"Line {i+1}: {line.strip()}")
