import os

file_path = "main_simple_for_frontend.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_marker = "# Experience Translator Models (F7)"
end_marker = '@app.get("/api/v1/experience/styles")'

start_index = -1
end_index = -1

for i, line in enumerate(lines):
    if start_marker in line:
        start_index = i
        break

for i, line in enumerate(lines):
    if end_marker in line:
        end_index = i
        break

if start_index != -1 and end_index != -1:
    print(f"Found range: {start_index} to {end_index}")
    
    portfolio_code = """
# Portfolio Generation Models & Functions
@app.post("/api/v1/portfolio/generate", response_model=PortfolioGenerateResponse)
@app.post("/api/v1/generate-portfolio", response_model=PortfolioGenerateResponse)
async def generate_portfolio(request: PortfolioGenerateRequest):
    \"\"\"Generate a portfolio entry from existing CV analysis data.\"\"\"
    try:
        logger.info(" Portfolio generation requested")
        ensure_portfolio_seed_data()
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
                views=random.randint(80, 200),
                downloads=random.randint(10, 50),
                likes=random.randint(5, 40)
            )
        )
        portfolio_store[generated_id] = portfolio_item
        portfolio_export_store[generated_id] = html_content
        logger.info(f" Portfolio {generated_id} generated with template {template.id}")
        return PortfolioGenerateResponse(portfolio=portfolio_item, html_content=html_content)
    except Exception as e:
        logger.error(f" Portfolio generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio generation failed: {str(e)}")

@app.get("/api/v1/portfolio/export/{portfolio_id}", response_model=PortfolioExportResponse)
async def export_portfolio(portfolio_id: str, format: str = Query("html", pattern="^(html|pdf)$")):
    \"\"\"Return stored HTML (or placeholder PDF) content for a portfolio.\"\"\"
    ensure_portfolio_seed_data()
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

def generate_portfolio_html(cv_data: Dict[str, Any], template: str) -> str:
    \"\"\"Generate professional portfolio HTML with modern design\"\"\"
    try:
        # Extract CV data
        personal_info = cv_data.get('personal_info', {})
        name = personal_info.get('name', 'Professional')
        email = cv_data.get('contact_info', {}).get('email', personal_info.get('email', ''))
        phone = cv_data.get('contact_info', {}).get('phone', personal_info.get('phone', ''))
        location = personal_info.get('location', '')
        
        job_titles = cv_data.get('job_titles', [])
        title = job_titles[0] if job_titles else 'Software Professional'
        
        skills = cv_data.get('skills', [])
        # Ensure skills is a list of strings
        if skills and isinstance(skills[0], dict):
             # Handle case where skills might be objects (e.g. from enhanced parser)
             skills = [s.get('name', str(s)) for s in skills if isinstance(s, dict)]
        
        experience_years = cv_data.get('experience_years', 0)
        education = cv_data.get('education', [])
        summary = cv_data.get('summary', f'Experienced {title} with {experience_years}+ years of expertise in software development and technology.')
        
        # Organize skills by category
        skill_categories = {
            'Languages': [],
            'Frameworks': [],
            'Tools & Technologies': []
        }
        
        # Common tech keywords for categorization
        languages = ['python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'express', 'spring', 'laravel', 'rails', '.net']
        
        for skill in skills:
            if not isinstance(skill, str):
                continue # Skip non-string skills
            skill_lower = skill.lower()
            if any(lang in skill_lower for lang in languages):
                skill_categories['Languages'].append(skill)
            elif any(fw in skill_lower for fw in frameworks):
                skill_categories['Frameworks'].append(skill)
            else:
                skill_categories['Tools & Technologies'].append(skill)
        
        # Build skills HTML
        skills_html = ''
        for category, items in skill_categories.items():
            if items:
                skills_html += f'''
                <div class="skill-category">
                    <h4 class="category-title">{category}</h4>
                    <div class="skill-tags">
                        {''.join([f'<span class="skill-tag">{skill}</span>' for skill in items])}
                    </div>
                </div>
                '''
        
        # Build education HTML
        education_html = ''.join([f'<li class="education-item"><i class="icon"></i> {edu}</li>' for edu in education]) if education else '<li class="education-item"><i class="icon"></i> Advanced studies in Computer Science</li>'
        
        # Generate professional HTML
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Professional portfolio of {name} - {title}">
    <title>{name} | {title} Portfolio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --primary: #2563eb;
            --primary-dark: #1e40af;
            --secondary: #8b5cf6;
            --text-dark: #1f2937;
            --text-light: #6b7280;
            --bg-light: #f9fafb;
            --border: #e5e7eb;
            --success: #10b981;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: var(--text-dark);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 1rem;
        }}
        
        .portfolio-container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        /* Hero Section */
        .hero {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 4rem 3rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 15s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        
        .hero-content {{
            position: relative;
            z-index: 1;
        }}
        
        .profile-image {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 4rem;
            margin: 0 auto 1.5rem;
            border: 5px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}
        
        .hero h1 {{
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .hero .title {{
            font-size: 1.5rem;
            font-weight: 400;
            opacity: 0.95;
            margin-bottom: 1rem;
        }}
        
        .hero .experience-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 500;
            margin-top: 1rem;
        }}
        
        /* Contact Info */
        .contact-info {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 2rem;
            flex-wrap: wrap;
        }}
        
        .contact-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: rgba(255, 255, 255, 0.95);
            text-decoration: none;
            transition: transform 0.2s;
        }}
        
        .contact-item:hover {{
            transform: translateY(-2px);
        }}
        
        .contact-item .icon {{
            font-size: 1.2rem;
        }}
        
        /* Main Content */
        .content {{
            padding: 3rem;
        }}
        
        .section {{
            margin-bottom: 3rem;
        }}
        
        .section-title {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid var(--primary);
            display: inline-block;
        }}
        
        /* About Section */
        .about-text {{
            font-size: 1.1rem;
            line-height: 1.8;
            color: var(--text-light);
            background: var(--bg-light);
            padding: 2rem;
            border-radius: 12px;
            border-left: 4px solid var(--primary);
        }}
        
        /* Skills Section */
        .skills-grid {{
            display: grid;
            gap: 2rem;
        }}
        
        .skill-category {{
            background: var(--bg-light);
            padding: 1.5rem;
            border-radius: 12px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .skill-category:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .category-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .category-title::before {{
            content: '';
            font-size: 1.5rem;
        }}
        
        .skill-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        
        .skill-tag {{
            background: white;
            color: var(--primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.9rem;
            border: 2px solid var(--primary);
            transition: all 0.2s;
            cursor: default;
        }}
        
        .skill-tag:hover {{
            background: var(--primary);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
        }}
        
        /* Education Section */
        .education-list {{
            list-style: none;
        }}
        
        .education-item {{
            background: var(--bg-light);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            transition: transform 0.2s;
            border-left: 4px solid var(--success);
        }}
        
        .education-item:hover {{
            transform: translateX(8px);
        }}
        
        .education-item .icon {{
            font-size: 1.8rem;
            flex-shrink: 0;
        }}
        
        /* Stats Section */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-4px);
        }}
        
        .stat-number {{
            font-size: 3rem;
            font-weight: 700;
            display: block;
        }}
        
        .stat-label {{
            font-size: 1rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }}
        
        /* Footer */
        .footer {{
            background: var(--text-dark);
            color: white;
            text-align: center;
            padding: 2rem;
        }}
        
        .footer p {{
            opacity: 0.8;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            
            .hero {{
                padding: 3rem 1.5rem;
            }}
            
            .hero h1 {{
                font-size: 2rem;
            }}
            
            .hero .title {{
                font-size: 1.2rem;
            }}
            
            .content {{
                padding: 2rem 1.5rem;
            }}
            
            .contact-info {{
                flex-direction: column;
                gap: 1rem;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="portfolio-container">
        <!-- Hero Section -->
        <header class="hero">
            <div class="hero-content">
                <div class="profile-image">
                    
                </div>
                <h1>{name}</h1>
                <p class="title">{title}</p>
                <span class="experience-badge"> {experience_years}+ Years of Experience</span>
                
                <div class="contact-info">
                    {f'<a href="mailto:{email}" class="contact-item"><span class="icon"></span> {email}</a>' if email else ''}
                    {f'<div class="contact-item"><span class="icon"></span> {phone}</div>' if phone else ''}
                    {f'<div class="contact-item"><span class="icon"></span> {location}</div>' if location else ''}
                </div>
            </div>
        </header>
        
        <!-- Main Content -->
        <main class="content">
            <!-- About Section -->
            <section class="section">
                <h2 class="section-title">About Me</h2>
                <div class="about-text">
                    {summary}
                </div>
            </section>
            
            <!-- Stats Section -->
            <section class="section">
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">{experience_years}+</span>
                        <span class="stat-label">Years Experience</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{len(skills)}</span>
                        <span class="stat-label">Technical Skills</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{len(job_titles)}</span>
                        <span class="stat-label">Professional Roles</span>
                    </div>
                </div>
            </section>
            
            <!-- Skills Section -->
            <section class="section">
                <h2 class="section-title">Technical Skills</h2>
                <div class="skills-grid">
                    {skills_html if skills_html else '<div class="skill-category"><div class="skill-tags">' + ''.join([f'<span class="skill-tag">{skill}</span>' for skill in skills]) + '</div></div>'}
                </div>
            </section>
            
            <!-- Education Section -->
            <section class="section">
                <h2 class="section-title">Education</h2>
                <ul class="education-list">
                    {education_html}
                </ul>
            </section>
        </main>
        
        <!-- Footer -->
        <footer class="footer">
            <p> 2025 {name}. Generated with SkillSync - Professional Portfolio Generator</p>
        </footer>
    </div>
</body>
</html>'''
        
        return html_content
    except Exception as e:
        logger.error(f" Error generating portfolio HTML: {e}")
        logger.error(f"CV Data keys: {cv_data.keys() if cv_data else 'None'}")
        raise e

def ensure_portfolio_seed_data():
    \"\"\"Ensure demo CV data and portfolio entries exist for the UI.\"\"\"
    if portfolio_store:
        return
    demo_cv_id = "cv_demo_123"
    if demo_cv_id not in cv_analysis_storage:
        cv_analysis_storage[demo_cv_id] = {
            "analysis_id": demo_cv_id,
            "skills": ["Python", "React", "AWS", "Docker", "SQL"],
            "experience_years": 6,
            "job_titles": ["Full Stack Developer", "Frontend Engineer"],
            "education": [
                "B.Sc. Computer Science - University of Paris",
                "Cloud Computing Nanodegree"
            ],
            "summary": "Engineer delivering cloud-ready web products with measurable impact.",
            "personal_info": {
                "name": "Amina Haddad",
                "title": "Senior Full Stack Developer",
                "years_experience": 6
            },
            "contact_info": {
                "email": "amina.haddad@example.com",
                "phone": "+33 1 23 45 67 89"
            }
        }
    now_iso = datetime.utcnow().isoformat()
    default_customization = {
        "color_scheme": "indigo",
        "font_family": "Inter",
        "layout_style": "modern",
        "sections_visible": ["about", "experience", "skills", "projects", "education"]
    }
    template = portfolio_templates[0]
    html_preview = generate_portfolio_html(cv_analysis_storage[demo_cv_id], template.id)
    sample_item = PortfolioItem(
        id="portfolio-demo-1",
        name=f"{cv_analysis_storage[demo_cv_id]['personal_info']['name']} Portfolio",
        cv_id=demo_cv_id,
        template_id=template.id,
        customization=default_customization,
        generated_date=now_iso,
        last_modified=now_iso,
        status="published",
        metrics=PortfolioMetrics(
            views=random.randint(450, 900),
            downloads=random.randint(80, 180),
            likes=random.randint(60, 140)
        )
    )
    portfolio_store[sample_item.id] = sample_item
    portfolio_export_store[sample_item.id] = html_preview

def get_cv_data_for_portfolio(cv_id: str) -> Dict[str, Any]:
    \"\"\"Fetch CV analysis data or return a fallback profile.\"\"\"
    stored = cv_analysis_storage.get(cv_id)
    if stored:
        return stored
    fallback = {
        "analysis_id": cv_id,
        "skills": ["Python", "TypeScript", "Docker", "Kubernetes", "SQL"],
        "experience_years": 5,
        "job_titles": ["Software Engineer", "Platform Developer"],
        "education": ["MSc Computer Science - Stanford University"],
        "summary": "Builder of reliable backend services and polished frontends.",
        "personal_info": {
            "name": "Jordan Lee",
            "title": "Full Stack Engineer",
            "years_experience": 5
        },
        "contact_info": {
            "email": "jordan.lee@example.com",
            "phone": "+1 (555) 123-4567"
        }
    }
    cv_analysis_storage[cv_id] = fallback
    return fallback

def get_template_by_id(template_id: str) -> PortfolioTemplate:
    \"\"\"Return matching template or default to the first entry.\"\"\"
    for template in portfolio_templates:
        if template.id == template_id:
            return template
    return portfolio_templates[0]
"""

    experience_code = """
# Experience Translator Models (F7)
class ExperienceTranslationRequest(BaseModel):
    original_experience: str
    job_description: str
    style: Optional[str] = "professional"  # professional, technical, creative
    preserve_original: Optional[bool] = False

class ExperienceTranslationResponse(BaseModel):
    translation_id: str
    timestamp: str
    rewritten_text: str
    rewriting_style: str
    confidence_score: float
    keyword_matches: Dict[str, int]
    suggestions: List[str]
    enhancements_made: List[str]
    version_comparison: Dict[str, Any]
    export_formats: Dict[str, str]

@app.post("/api/v1/experience/translate", response_model=ExperienceTranslationResponse)
async def translate_experience(request: ExperienceTranslationRequest):
    \"\"\"Translate and enhance professional experience for specific job requirements\"\"\"
    try:
        logger.info(" Experience translation requested")
        
        # Check if Experience Translator is available
        if not translate_experience_api:
            raise HTTPException(status_code=503, detail="Experience Translator service is not available")
        
        # Validate inputs
        if not request.original_experience.strip():
            raise HTTPException(status_code=400, detail="Original experience text is required")
        
        if not request.job_description.strip():
            raise HTTPException(status_code=400, detail="Job description is required")
        
        # Perform experience translation
        translation_result = translate_experience_api(
            original_experience=request.original_experience,
            job_description=request.job_description,
            style=request.style
        )
        
        # Extract key information for response
        rewritten_experience = translation_result["rewritten_experience"]
        
        response = ExperienceTranslationResponse(
            translation_id=translation_result["translation_id"],
            timestamp=translation_result["timestamp"],
            rewritten_text=rewritten_experience["text"],
            rewriting_style=rewritten_experience["style"],
            confidence_score=rewritten_experience["confidence_score"],
            keyword_matches=rewritten_experience["keyword_matches"],
            suggestions=rewritten_experience["improvement_suggestions"],
            enhancements_made=rewritten_experience["enhancements_made"],
            version_comparison=rewritten_experience["version_comparison"],
            export_formats=rewritten_experience["export_formats"]
        )
        
        logger.info(f" Experience translation completed. Confidence: {response.confidence_score:.2f}")
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f" Experience translation failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Experience translation failed: {str(e)}")
"""

    # Construct the new file content
    # We replace everything from start_index to end_index (exclusive of end_index)
    # And we insert portfolio_code BEFORE experience_code
    
    final_content = lines[:start_index] + [portfolio_code + "\n"] + [experience_code + "\n"] + lines[end_index:]
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(final_content)
    
    print("Successfully repaired backend file.")
else:
    print(f"Failed to repair: could not locate code block. Start: {start_index}, End: {end_index}")
