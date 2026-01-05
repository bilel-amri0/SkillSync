#!/usr/bin/env python
"""Fix work_history access issue"""

with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic line
old_line = 'logger.info(f"   • Work experiences: {len(cv_result.work_history)}")'
new_line = 'logger.info(f"   • Work experiences: {len(cv_result.work_history or [])}")'

if old_line in content:
    content = content.replace(old_line, new_line)
    with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('✅ Fixed work_history access')
else:
    print('Pattern not found - checking alternative...')
    # Try another pattern
    old_line2 = 'len(cv_result.work_history)'
    if old_line2 in content and 'or []' not in content[content.find(old_line2)-10:content.find(old_line2)+50]:
        content = content.replace(old_line2, 'len(cv_result.work_history or [])')
        with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print('✅ Fixed work_history access (alternative pattern)')
    else:
        print('❌ Could not find pattern to fix')
