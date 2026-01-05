#!/usr/bin/env python
"""Script to add the missing portfolio generate endpoints to main_simple_for_frontend.py"""

import re

# Read the file
with open('main_simple_for_frontend.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The code to add after the list_portfolios function
portfolio_generate_code = '''

# Portfolio Generation Implementation
async def _generate_portfolio_impl(request):
    """Internal implementation for portfolio generation."""
    from datetime import datetime
    import uuid
    
    logger.info("🎨 Portfolio generation requested")
    cv_data = get_cv_data_for_portfolio(request.cv_id)
    template = get_template_by_id(request.template_id)
    html_content = generate_portfolio_html(cv_data, template.id)
    customization = request.customization or {
        "color_scheme": "blue",
        "font_family": "Inter",
        "layout_style": "clean",
        "sections_visible": ["about", "experience", "skills", "projects", "education"]
    }
    generated_id = str(uuid.uuid4())
    now_iso = datetime.utcnow().isoformat()
    portfolio_item = PortfolioItem(
        id=generated_id,
        name=f"{cv_data.get('personal_info', {}).get('name', 'Professional')} - {template.name}",
        cv_id=request.cv_id,
        template_id=template.id,
        customization=customization,
        generated_date=now_iso,
        last_modified=now_iso,
        status="published",
        metrics=PortfolioMetrics(
            views=0,
            downloads=0,
            likes=0
        )
    )
    portfolio_store[generated_id] = portfolio_item
    portfolio_export_store[generated_id] = html_content
    logger.info(f"✅ Portfolio {generated_id} generated with template {template.id}")
    return PortfolioGenerateResponse(portfolio=portfolio_item, html_content=html_content)

@app.post("/api/v1/portfolio/generate", response_model=PortfolioGenerateResponse)
async def generate_portfolio(request: PortfolioGenerateRequest):
    """Generate a portfolio entry from existing CV analysis data."""
    try:
        return await _generate_portfolio_impl(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Portfolio generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio generation failed: {str(e)}")

@app.post("/api/v1/generate-portfolio", response_model=PortfolioGenerateResponse)
async def generate_portfolio_legacy(request: PortfolioGenerateRequest):
    """Generate a portfolio entry from existing CV analysis data (legacy endpoint)."""
    try:
        return await _generate_portfolio_impl(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Portfolio generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio generation failed: {str(e)}")

@app.get("/api/v1/portfolio/export/{portfolio_id}", response_model=PortfolioExportResponse)
async def export_portfolio(portfolio_id: str, format: str = Query("html", pattern="^(html|pdf)$")):
    """Return stored HTML (or placeholder PDF) content for a portfolio."""
    portfolio = portfolio_store.get(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    html_content = portfolio_export_store.get(portfolio_id)
    if not html_content:
        cv_data = get_cv_data_for_portfolio(portfolio.cv_id)
        html_content = generate_portfolio_html(cv_data, portfolio.template_id)
        portfolio_export_store[portfolio_id] = html_content
    if format == "pdf":
        exported_content = (
            f"PDF export placeholder for {portfolio.name}\\n"
            f"Generated: {datetime.utcnow().isoformat()}"
        )
    else:
        exported_content = html_content
    return PortfolioExportResponse(
        portfolio_id=portfolio_id,
        format=format,
        content=exported_content,
        exported_at=datetime.utcnow().isoformat()
    )

def generate_portfolio_html(cv_data, template):
    """Generate professional portfolio HTML with modern design"""
    
    # Extract CV data
    personal_info = cv_data.get('personal_info', {})
    name = personal_info.get('name', 'Professional')
    email = cv_data.get('contact_info', {}).get('email', personal_info.get('email', ''))
    phone = cv_data.get('contact_info', {}).get('phone', personal_info.get('phone', ''))
    location = personal_info.get('location', '')
    
    job_titles = cv_data.get('job_titles', [])
    title = job_titles[0] if job_titles else 'Software Professional'
    
    skills = cv_data.get('skills', [])
    experience_years = cv_data.get('experience_years', 0)
    education = cv_data.get('education', [])
    summary = cv_data.get('summary', f'Experienced {title} with {experience_years}+ years of expertise.')
    
    # Build skills HTML
    skills_html = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in skills[:15]])
    
    # Build education HTML
    education_html = ''.join([f'<li>🎓 {edu}</li>' for edu in education]) if education else '<li>🎓 Advanced studies</li>'
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} | Portfolio</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 40px 20px; }}
        .header {{ background: linear-gradient(135deg, #2563eb, #8b5cf6); color: white; padding: 60px 40px; border-radius: 16px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.2em; opacity: 0.9; }}
        .section {{ background: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        .section h2 {{ color: #2563eb; margin-bottom: 20px; font-size: 1.5em; }}
        .skill-tag {{ display: inline-block; background: #e0e7ff; color: #3730a3; padding: 8px 16px; border-radius: 20px; margin: 5px; font-size: 0.9em; }}
        .contact-item {{ margin: 10px 0; color: #64748b; }}
        ul {{ list-style: none; }}
        li {{ margin: 10px 0; padding-left: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{name}</h1>
            <p>{title}</p>
        </div>
        <div class="section">
            <h2>About</h2>
            <p>{summary}</p>
            <p class="contact-item">📧 {email}</p>
            <p class="contact-item">📱 {phone}</p>
            <p class="contact-item">📍 {location}</p>
        </div>
        <div class="section">
            <h2>Skills</h2>
            <div>{skills_html}</div>
        </div>
        <div class="section">
            <h2>Education</h2>
            <ul>{education_html}</ul>
        </div>
    </div>
</body>
</html>"""
    
    return html_content

def get_cv_data_for_portfolio(cv_id):
    """Fetch CV analysis data for portfolio generation."""
    stored = cv_analysis_storage.get(cv_id)
    if not stored:
        raise HTTPException(
            status_code=404,
            detail=f"CV analysis {cv_id} not found. Run /api/v1/analyze-cv first."
        )
    return stored

def get_template_by_id(template_id):
    """Return matching template or default to the first entry."""
    for template in portfolio_templates:
        if template.id == template_id:
            return template
    return portfolio_templates[0] if portfolio_templates else None

'''

# Find the position after "return items" in list_portfolios function
# The pattern is to find "@app.get("/api/v1/portfolio/list"..." followed by the function body
# and insert after it but before the next function definition

# Find the line "    return items" after portfolio/list endpoint
pattern = r'(@app\.get\("/api/v1/portfolio/list".*?\n(?:.*?\n)*?    return items\n)'
match = re.search(pattern, content)

if match:
    insert_pos = match.end()
    # Insert the code
    new_content = content[:insert_pos] + portfolio_generate_code + content[insert_pos:]
    
    # Write back
    with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully added portfolio generate endpoints!")
    print(f"   Inserted {len(portfolio_generate_code)} characters after portfolio/list endpoint")
else:
    print("❌ Could not find insertion point. Looking for alternative...")
    # Try another pattern
    if "def list_portfolios" in content:
        print("   Found list_portfolios function")
        # Find end of list_portfolios
        idx = content.index("def list_portfolios")
        # Find the next "def " or "@app" after this
        rest = content[idx+100:]
        next_def = rest.find("\ndef ")
        next_app = rest.find("\n@app")
        
        if next_def > 0 and (next_app < 0 or next_def < next_app):
            insert_pos = idx + 100 + next_def
        elif next_app > 0:
            insert_pos = idx + 100 + next_app
        else:
            print("   Could not determine insert position")
            exit(1)
        
        new_content = content[:insert_pos] + portfolio_generate_code + content[insert_pos:]
        with open('main_simple_for_frontend.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Inserted code at position {insert_pos}")
    else:
        print("❌ Could not find list_portfolios function")
