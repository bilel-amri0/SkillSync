#!/usr/bin/env python3
"""Show lines around get_template_by_id"""

with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== Line 2286 area ===")
for i in range(2282, 2300):
    if i < len(lines):
        print(f'{i+1}: {lines[i]}', end='')
