with open('dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
for i, line in enumerate(lines):
    if 'gb_basic' in line or 'gb_current' in line:
        print(f"Line {i+1}: {line.strip()}")
