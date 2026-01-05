#!/usr/bin/env python3
"""Show exact line content"""

with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print('Line 1213 (index 1212):')
print(repr(lines[1212]))
