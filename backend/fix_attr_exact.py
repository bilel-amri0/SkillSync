#!/usr/bin/env python3
"""Fix the work_history attribute access issue - exact string matching"""

file_path = 'main_simple_for_frontend.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: work_history access (note: 4 spaces before "Work")
old1 = 'logger.info(f"    Work experiences: {len(cv_result.work_history or [])}")'
new1 = 'logger.info(f"    Work experiences: {len(getattr(cv_result, \'work_history\', None) or [])}")'

# Fix 2: education access (note: 4 spaces before "Education")  
old2 = 'logger.info(f"    Education: {len(cv_result.education or [])}")'
new2 = 'logger.info(f"    Education: {len(getattr(cv_result, \'education\', None) or [])}")'

# Fix 3: projects access (note: 4 spaces before "Projects")
old3 = 'logger.info(f"    Projects: {len(cv_result.projects or [])}")'
new3 = 'logger.info(f"    Projects: {len(getattr(cv_result, \'projects\', None) or [])}")'

print(f"Looking for: {repr(old1)}")
print(f"Found: {old1 in content}")

if old1 in content:
    content = content.replace(old1, new1)
    print("✅ Fixed work_history access")
else:
    print("⚠️ work_history pattern not found or already fixed")

if old2 in content:
    content = content.replace(old2, new2)
    print("✅ Fixed education access")
else:
    print("⚠️ education pattern not found or already fixed")

if old3 in content:
    content = content.replace(old3, new3)
    print("✅ Fixed projects access")
else:
    print("⚠️ projects pattern not found or already fixed")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Verify
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

if "getattr(cv_result, 'work_history'" in content:
    print("✅ Verification passed - work_history fixed!")
else:
    print("❌ work_history fix not applied correctly")
