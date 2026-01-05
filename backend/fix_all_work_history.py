#!/usr/bin/env python
"""Fix all work_history access issues"""

with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_count = 0
for i, line in enumerate(lines):
    # Fix any line that accesses work_history without the 'or []' guard
    if 'cv_result.work_history)' in line and 'or []' not in line:
        lines[i] = line.replace('cv_result.work_history)', 'cv_result.work_history or [])')
        print(f"Fixed line {i+1}: {lines[i].strip()}")
        fixed_count += 1
    elif '.work_history)' in line and 'or []' not in line:
        lines[i] = line.replace('.work_history)', '.work_history or [])')
        print(f"Fixed line {i+1}: {lines[i].strip()}")
        fixed_count += 1

with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n✅ Fixed {fixed_count} occurrences")
