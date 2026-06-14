with open('dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()
import re
matches = re.findall(r'models = .*', content)
print(matches)
