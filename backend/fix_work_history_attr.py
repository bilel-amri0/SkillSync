#!/usr/bin/env python3
"""Fix the work_history attribute access issue"""

file_path = 'main_simple_for_frontend.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already fixed
if "getattr(cv_result, 'work_history'" in content:
    print("Already fixed!")
else:
    # Fix work_history access
    old1 = "logger.info(f\"   • Work experiences: {len(cv_result.work_history or [])}\")"
    new1 = "logger.info(f\"   • Work experiences: {len(getattr(cv_result, 'work_history', None) or [])}\")"
    
    # Fix education access
    old2 = "logger.info(f\"   • Education: {len(cv_result.education or [])}\")"
    new2 = "logger.info(f\"   • Education: {len(getattr(cv_result, 'education', None) or [])}\")"
    
    # Fix projects access
    old3 = "logger.info(f\"   • Projects: {len(cv_result.projects or [])}\")"
    new3 = "logger.info(f\"   • Projects: {len(getattr(cv_result, 'projects', None) or [])}\")"
    
    content = content.replace(old1, new1)
    content = content.replace(old2, new2)
    content = content.replace(old3, new3)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed!")
    
# Verify
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

if "getattr(cv_result, 'work_history'" in content:
    print("✅ Verification passed!")
else:
    print("❌ Fix not applied correctly")
