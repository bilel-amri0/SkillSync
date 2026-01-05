#!/usr/bin/env python3
"""Fix get_template_by_id to use dynamic templates"""

file_path = 'main_simple_for_frontend.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the correct templates list to use in get_template_by_id
old_func = '''def get_template_by_id(template_id):
    """Return matching template or default to the first entry."""
    for template in portfolio_templates:
        if template.id == template_id:
            return template
    return portfolio_templates[0] if portfolio_templates else None'''

new_func = '''def get_template_by_id(template_id):
    """Return matching template or default to the first entry."""
    # Use dynamic templates (same as API returns)
    templates = [
        PortfolioTemplate(
            id="professional",
            name="Professional",
            description="Clean and professional layout",
            category="modern",
            features=["Responsive", "ATS-friendly", "Customizable"],
            preview_url=None
        ),
        PortfolioTemplate(
            id="creative",
            name="Creative",
            description="Bold and creative design",
            category="creative",
            features=["Visual impact", "Portfolio showcase", "Interactive"],
            preview_url=None
        ),
        PortfolioTemplate(
            id="minimal",
            name="Minimal",
            description="Simple and elegant",
            category="minimal",
            features=["Typography-focused", "Clean layout", "Fast loading"],
            preview_url=None
        ),
        # Also support "modern" as alias for professional
        PortfolioTemplate(
            id="modern",
            name="Modern Spotlight",
            description="Clean layout with hero header",
            category="modern",
            features=["Hero banner", "Skill radar", "Case study cards"],
            preview_url=None
        )
    ]
    for template in templates:
        if template.id == template_id:
            return template
    # Return first template as default
    return templates[0]'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed get_template_by_id function")
else:
    print("❌ Could not find the exact function to replace")
    # Try to show what's there
    import re
    match = re.search(r'def get_template_by_id.*?(?=\ndef |\Z)', content, re.DOTALL)
    if match:
        print("Current function:")
        print(match.group()[:500])
