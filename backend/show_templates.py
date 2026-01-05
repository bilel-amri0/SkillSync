#!/usr/bin/env python3
"""Show portfolio templates definition"""

with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== Portfolio templates definition ===")
for i in range(2078, 2110):
    if i < len(lines):
        print(f'{i+1}: {lines[i]}', end='')
