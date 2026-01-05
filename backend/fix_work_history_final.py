#!/usr/bin/env python
"""Fix work_history issue properly"""

with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the problematic line at the end of analyze_cv_text
old_pattern = 'logger.info(f"   • Work experiences: {len(cv_result.work_history or [])}")'
new_pattern = 'logger.info(f"   • Work experiences: {len(getattr(cv_result, \'work_history\', None) or [])}")'

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print(f"Fixed pattern 1")

# Also add work_history to create_cv_analysis response
# Find where CVAnalysisResponse is created
old_constructor = '''        certifications=certifications,
        recommendations=recommendations
    )'''
new_constructor = '''        certifications=certifications,
        recommendations=recommendations,
        work_history=[]  # Initialize to empty list
    )'''

if old_constructor in content:
    content = content.replace(old_constructor, new_constructor)
    print(f"Fixed pattern 2 - added work_history to constructor")

with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Done")
