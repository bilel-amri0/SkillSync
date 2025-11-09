"""
Portfolio Generator - Enhanced version integrated with SkillSync
Automatic professional portfolio generation with adaptive templates
"""

import json
import zipfile
import os
from pathlib import Path
import uuid
from typing import Dict, List, Any
from datetime import datetime
import logging
from jinja2 import Template

logger = logging.getLogger(__name__)

class PortfolioGenerator:
    """Enhanced portfolio generator for SkillSync platform"""
    
    def __init__(self):
        self.templates_dir = Path("templates")
        self.output_dir = Path("generated_portfolios")
        self.output_dir.mkdir(exist_ok=True)
        
        # Portfolio templates
        self.templates = {
            'modern': self._get_modern_template(),
            'classic': self._get_classic_template(),
            'creative': self._get_creative_template(),
            'minimal': self._get_minimal_template(),
            'tech': self._get_tech_template()
        }
        
        # Color schemes
        self.color_schemes = {
            'blue': {'primary': '#2563eb', 'secondary': '#64748b', 'accent': '#0ea5e9'},
            'green': {'primary': '#059669', 'secondary': '#6b7280', 'accent': '#10b981'},
            'purple': {'primary': '#7c3aed', 'secondary': '#6b7280', 'accent': '#8b5cf6'},
            'red': {'primary': '#dc2626', 'secondary': '#6b7280', 'accent': '#ef4444'},
            'orange': {'primary': '#ea580c', 'secondary': '#6b7280', 'accent': '#f97316'}
        }
    
    async def create_portfolio(
        self, 
        cv_data: Dict[str, Any], 
        template: str = "modern",
        customizations: Dict[str, Any] = None
    ) -> Any:
        """Create portfolio from CV analysis data"""
        
        try:
            portfolio_id = str(uuid.uuid4())
            logger.info(f"Generating portfolio {portfolio_id} with template {template}")
            
            # Extract and structure portfolio data
            portfolio_data = await self._prepare_portfolio_data(cv_data, customizations)
            
            # Generate portfolio files
            portfolio_files = await self._generate_portfolio_files(
                portfolio_data, template, portfolio_id
            )
            
            # Create ZIP file
            zip_path = await self._create_portfolio_zip(portfolio_files, portfolio_id)
            
            # Generate URLs (in production, these would be actual web URLs)
            download_url = f"/api/v1/portfolios/{portfolio_id}/download"
            preview_url = f"/api/v1/portfolios/{portfolio_id}/preview"
            
            # Return result object
            class PortfolioResult:
                def __init__(self, portfolio_id, download_url, preview_url, files):
                    self.portfolio_id = portfolio_id
                    self.download_url = download_url
                    self.preview_url = preview_url
                    self.files = files
            
            return PortfolioResult(
                portfolio_id=portfolio_id,
                download_url=download_url,
                preview_url=preview_url,
                files=list(portfolio_files.keys())
            )
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {str(e)}")
            raise
    
    async def _prepare_portfolio_data(
        self, 
        cv_data: Dict[str, Any], 
        customizations: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Prepare structured data for portfolio generation"""
        
        # Extract personal information
        personal_info = cv_data.get('cv_content', {}).get('personal_info', {})
        contact_info = cv_data.get('cv_content', {}).get('contact_info', {})
        sections = cv_data.get('cv_content', {}).get('sections', {})
        skills = cv_data.get('skills', [])
        
        # Structure portfolio data
        portfolio_data = {
            'personal': {
                'name': personal_info.get('name', 'Professional Name'),
                'title': personal_info.get('title', 'Professional Title'),
                'summary': sections.get('summary', 'Professional with diverse experience and strong technical skills.'),
                'location': customizations.get('location') if customizations else None,
                'website': customizations.get('website') if customizations else None
            },
            'contact': {
                'email': contact_info.get('email', ''),
                'phone': contact_info.get('phone', ''),
                'linkedin': contact_info.get('linkedin', ''),
                'github': contact_info.get('github', ''),
                'portfolio': contact_info.get('portfolio', '')
            },
            'skills': self._organize_skills(skills),
            'experience': self._extract_experience(sections.get('experience', '')),
            'education': self._extract_education(sections.get('education', '')),
            'projects': self._extract_projects(sections.get('projects', '')),
            'certifications': self._extract_certifications(sections.get('certifications', '')),
            'achievements': self._extract_achievements(sections.get('achievements', '')),
            'languages': self._extract_languages(sections.get('languages', '')),
            'interests': sections.get('interests', ''),
            'customizations': customizations or {},
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_id': cv_data.get('analysis_id', ''),
                'template': 'modern',
                'version': '2.0'
            }
        }
        
        return portfolio_data
    
    def _organize_skills(self, skills: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize skills by category for portfolio display"""
        
        organized_skills = {}
        
        for skill in skills:
            category = skill.get('category', 'other')
            if category not in organized_skills:
                organized_skills[category] = []
            
            organized_skills[category].append({
                'name': skill.get('normalized_name', skill.get('skill', '')),
                'level': skill.get('experience_level', 'intermediate'),
                'confidence': skill.get('confidence', 0.8),
                'importance': skill.get('importance_score', 0.5)
            })
        
        # Sort skills within each category by importance
        for category in organized_skills:
            organized_skills[category].sort(key=lambda x: x['importance'], reverse=True)
        
        return organized_skills
    
    def _extract_experience(self, experience_text: str) -> List[Dict[str, Any]]:
        """Extract structured experience data"""
        
        if not experience_text.strip():
            return []
        
        experiences = []
        
        # Simple parsing - in production, use more sophisticated NLP
        lines = experience_text.split('\n')
        current_job = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like a job title/company
            if any(keyword in line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'consultant']):
                if current_job:
                    experiences.append(current_job)
                
                current_job = {
                    'title': line,
                    'company': '',
                    'duration': '',
                    'description': [],
                    'technologies': []
                }
            elif current_job:
                current_job['description'].append(line)
        
        if current_job:
            experiences.append(current_job)
        
        return experiences
    
    def _extract_education(self, education_text: str) -> List[Dict[str, Any]]:
        """Extract structured education data"""
        
        if not education_text.strip():
            return []
        
        education = []
        lines = education_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Likely an education entry
                education.append({
                    'degree': line,
                    'institution': '',
                    'year': '',
                    'gpa': '',
                    'relevant_courses': []
                })
        
        return education
    
    def _extract_projects(self, projects_text: str) -> List[Dict[str, Any]]:
        """Extract structured project data"""
        
        if not projects_text.strip():
            return []
        
        projects = []
        lines = projects_text.split('\n')
        
        current_project = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristic: if line doesn't start with bullet, it's likely a project title
            if not line.startswith(('-', '*', '•')):
                if current_project:
                    projects.append(current_project)
                
                current_project = {
                    'title': line,
                    'description': '',
                    'technologies': [],
                    'url': '',
                    'github': '',
                    'highlights': []
                }
            elif current_project:
                current_project['highlights'].append(line.lstrip('-*• '))
        
        if current_project:
            projects.append(current_project)
        
        return projects
    
    def _extract_certifications(self, cert_text: str) -> List[Dict[str, Any]]:
        """Extract structured certification data"""
        
        if not cert_text.strip():
            return []
        
        certifications = []
        lines = cert_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                certifications.append({
                    'name': line,
                    'issuer': '',
                    'date': '',
                    'url': '',
                    'status': 'active'
                })
        
        return certifications
    
    def _extract_achievements(self, achievements_text: str) -> List[str]:
        """Extract achievements list"""
        
        if not achievements_text.strip():
            return []
        
        achievements = []
        lines = achievements_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line:
                achievements.append(line.lstrip('-*• '))
        
        return achievements
    
    def _extract_languages(self, languages_text: str) -> List[Dict[str, str]]:
        """Extract language skills"""
        
        if not languages_text.strip():
            return []
        
        languages = []
        lines = languages_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line:
                # Simple parsing
                parts = line.split('-')
                if len(parts) >= 2:
                    languages.append({
                        'language': parts[0].strip(),
                        'level': parts[1].strip()
                    })
                else:
                    languages.append({
                        'language': line,
                        'level': 'conversational'
                    })
        
        return languages
    
    async def _generate_portfolio_files(
        self, 
        portfolio_data: Dict[str, Any], 
        template: str, 
        portfolio_id: str
    ) -> Dict[str, str]:
        """Generate all portfolio files"""
        
        files = {}
        
        # Get color scheme
        color_scheme = self.color_schemes.get(
            portfolio_data.get('customizations', {}).get('color_scheme', 'blue'),
            self.color_schemes['blue']
        )
        
        # Generate HTML
        html_template = self.templates[template]['html']
        html_content = html_template.render(
            data=portfolio_data,
            colors=color_scheme
        )
        files['index.html'] = html_content
        
        # Generate CSS
        css_template = self.templates[template]['css']
        css_content = css_template.render(
            colors=color_scheme,
            customizations=portfolio_data.get('customizations', {})
        )
        files['styles.css'] = css_content
        
        # Generate JavaScript
        js_content = self.templates[template]['js']
        files['script.js'] = js_content
        
        # Generate JSON data file
        files['portfolio_data.json'] = json.dumps(portfolio_data, indent=2)
        
        # Generate README
        files['README.md'] = self._generate_readme(portfolio_data, portfolio_id)
        
        return files
    
    async def _create_portfolio_zip(
        self, 
        portfolio_files: Dict[str, str], 
        portfolio_id: str
    ) -> str:
        """Create ZIP file containing all portfolio files"""
        
        zip_path = self.output_dir / f"{portfolio_id}_portfolio.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename, content in portfolio_files.items():
                zipf.writestr(filename, content)
        
        return str(zip_path)
    
    def _generate_readme(self, portfolio_data: Dict[str, Any], portfolio_id: str) -> str:
        """Generate README for the portfolio"""
        
        name = portfolio_data.get('personal', {}).get('name', 'Professional')
        
        return f"""# {name}'s Portfolio

Generated by SkillSync - AI-Powered Career Platform

## Portfolio Information
- **Portfolio ID**: {portfolio_id}
- **Generated**: {portfolio_data.get('metadata', {}).get('generated_at', 'Unknown')}
- **Template**: {portfolio_data.get('metadata', {}).get('template', 'modern')}
- **Version**: {portfolio_data.get('metadata', {}).get('version', '2.0')}

## Files Included
- `index.html` - Main portfolio website
- `styles.css` - Styling and layout
- `script.js` - Interactive features
- `portfolio_data.json` - Structured portfolio data
- `README.md` - This file

## How to Use
1. Open `index.html` in any web browser
2. Upload all files to a web hosting service for online portfolio
3. Customize CSS and content as needed

## Features
- Responsive design (mobile-friendly)
- Professional layout
- Interactive elements
- SEO optimized
- Print-friendly styles

---

Generated by SkillSync © 2025
"""
    
    def _get_modern_template(self) -> Dict[str, Template]:
        """Get modern portfolio template"""
        
        html_template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ data.personal.name }} - Portfolio</title>
    <link rel="stylesheet" href="styles.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="profile-section">
                <div class="profile-info">
                    <h1 class="name">{{ data.personal.name }}</h1>
                    <h2 class="title">{{ data.personal.title }}</h2>
                    <p class="summary">{{ data.personal.summary }}</p>
                </div>
                <div class="contact-info">
                    {% if data.contact.email %}
                    <a href="mailto:{{ data.contact.email }}" class="contact-item">
                        <i class="fas fa-envelope"></i> {{ data.contact.email }}
                    </a>
                    {% endif %}
                    {% if data.contact.phone %}
                    <a href="tel:{{ data.contact.phone }}" class="contact-item">
                        <i class="fas fa-phone"></i> {{ data.contact.phone }}
                    </a>
                    {% endif %}
                    {% if data.contact.linkedin %}
                    <a href="{{ data.contact.linkedin }}" class="contact-item" target="_blank">
                        <i class="fab fa-linkedin"></i> LinkedIn
                    </a>
                    {% endif %}
                    {% if data.contact.github %}
                    <a href="{{ data.contact.github }}" class="contact-item" target="_blank">
                        <i class="fab fa-github"></i> GitHub
                    </a>
                    {% endif %}
                </div>
            </div>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <!-- Skills Section -->
            {% if data.skills %}
            <section class="section">
                <h2 class="section-title">Technical Skills</h2>
                <div class="skills-grid">
                    {% for category, skills in data.skills.items() %}
                    <div class="skill-category">
                        <h3 class="category-title">{{ category.replace('_', ' ').title() }}</h3>
                        <div class="skills-list">
                            {% for skill in skills[:6] %}
                            <span class="skill-tag">{{ skill.name }}</span>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </section>
            {% endif %}

            <!-- Experience Section -->
            {% if data.experience %}
            <section class="section">
                <h2 class="section-title">Professional Experience</h2>
                <div class="timeline">
                    {% for job in data.experience %}
                    <div class="timeline-item">
                        <div class="timeline-content">
                            <h3 class="job-title">{{ job.title }}</h3>
                            {% if job.company %}<p class="job-company">{{ job.company }}</p>{% endif %}
                            {% if job.duration %}<p class="job-duration">{{ job.duration }}</p>{% endif %}
                            {% if job.description %}
                            <ul class="job-description">
                                {% for desc in job.description[:3] %}
                                <li>{{ desc }}</li>
                                {% endfor %}
                            </ul>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </section>
            {% endif %}

            <!-- Projects Section -->
            {% if data.projects %}
            <section class="section">
                <h2 class="section-title">Featured Projects</h2>
                <div class="projects-grid">
                    {% for project in data.projects %}
                    <div class="project-card">
                        <h3 class="project-title">{{ project.title }}</h3>
                        {% if project.description %}<p class="project-description">{{ project.description }}</p>{% endif %}
                        {% if project.highlights %}
                        <ul class="project-highlights">
                            {% for highlight in project.highlights[:3] %}
                            <li>{{ highlight }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        <div class="project-links">
                            {% if project.url %}
                            <a href="{{ project.url }}" target="_blank" class="project-link">
                                <i class="fas fa-external-link-alt"></i> View Project
                            </a>
                            {% endif %}
                            {% if project.github %}
                            <a href="{{ project.github }}" target="_blank" class="project-link">
                                <i class="fab fa-github"></i> Code
                            </a>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </section>
            {% endif %}

            <!-- Education Section -->
            {% if data.education %}
            <section class="section">
                <h2 class="section-title">Education</h2>
                <div class="education-list">
                    {% for edu in data.education %}
                    <div class="education-item">
                        <h3 class="education-degree">{{ edu.degree }}</h3>
                        {% if edu.institution %}<p class="education-institution">{{ edu.institution }}</p>{% endif %}
                        {% if edu.year %}<p class="education-year">{{ edu.year }}</p>{% endif %}
                    </div>
                    {% endfor %}
                </div>
            </section>
            {% endif %}
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2025 {{ data.personal.name }}. Generated by SkillSync.</p>
        </div>
    </footer>

    <script src="script.js"></script>
</body>
</html>
        """)
        
        css_template = Template("""
/* Modern Portfolio Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8fafc;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header Styles */
.header {
    background: linear-gradient(135deg, {{ colors.primary }}, {{ colors.accent }});
    color: white;
    padding: 60px 0;
    margin-bottom: 40px;
}

.profile-section {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 40px;
    align-items: center;
}

.name {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.title {
    font-size: 1.5rem;
    font-weight: 400;
    opacity: 0.9;
    margin-bottom: 20px;
}

.summary {
    font-size: 1.1rem;
    opacity: 0.9;
    line-height: 1.7;
}

.contact-info {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.contact-item {
    color: white;
    text-decoration: none;
    padding: 12px 20px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 10px;
}

.contact-item:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
}

/* Main Content */
.main {
    padding-bottom: 60px;
}

.section {
    margin-bottom: 50px;
    background: white;
    border-radius: 12px;
    padding: 40px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.section-title {
    font-size: 2rem;
    color: {{ colors.primary }};
    margin-bottom: 30px;
    position: relative;
    padding-bottom: 15px;
}

.section-title::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 60px;
    height: 3px;
    background: {{ colors.accent }};
    border-radius: 2px;
}

/* Skills Styles */
.skills-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 30px;
}

.skill-category {
    background: #f8fafc;
    padding: 25px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

.category-title {
    color: {{ colors.primary }};
    font-size: 1.2rem;
    margin-bottom: 15px;
    font-weight: 600;
}

.skills-list {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.skill-tag {
    background: {{ colors.primary }};
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Timeline Styles */
.timeline {
    position: relative;
    padding-left: 30px;
}

.timeline::before {
    content: '';
    position: absolute;
    left: 15px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: {{ colors.accent }};
}

.timeline-item {
    position: relative;
    margin-bottom: 40px;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: -24px;
    top: 8px;
    width: 12px;
    height: 12px;
    background: {{ colors.primary }};
    border-radius: 50%;
    border: 3px solid white;
    box-shadow: 0 0 0 3px {{ colors.accent }};
}

.timeline-content {
    background: #f8fafc;
    padding: 25px;
    border-radius: 8px;
    border-left: 4px solid {{ colors.primary }};
}

.job-title {
    color: {{ colors.primary }};
    font-size: 1.3rem;
    margin-bottom: 8px;
}

.job-company {
    font-weight: 600;
    color: #64748b;
    margin-bottom: 5px;
}

.job-duration {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 15px;
}

.job-description {
    list-style: none;
    padding-left: 0;
}

.job-description li {
    position: relative;
    padding-left: 20px;
    margin-bottom: 8px;
    color: #475569;
}

.job-description li::before {
    content: '•';
    color: {{ colors.accent }};
    position: absolute;
    left: 0;
}

/* Projects Styles */
.projects-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 30px;
}

.project-card {
    background: #f8fafc;
    padding: 30px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.project-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
}

.project-title {
    color: {{ colors.primary }};
    font-size: 1.3rem;
    margin-bottom: 15px;
}

.project-description {
    color: #64748b;
    margin-bottom: 15px;
    line-height: 1.6;
}

.project-highlights {
    list-style: none;
    margin-bottom: 20px;
}

.project-highlights li {
    position: relative;
    padding-left: 20px;
    margin-bottom: 8px;
    color: #475569;
}

.project-highlights li::before {
    content: '✓';
    color: {{ colors.accent }};
    position: absolute;
    left: 0;
    font-weight: bold;
}

.project-links {
    display: flex;
    gap: 15px;
}

.project-link {
    color: {{ colors.primary }};
    text-decoration: none;
    padding: 10px 20px;
    border: 2px solid {{ colors.primary }};
    border-radius: 6px;
    transition: all 0.3s ease;
    font-size: 0.9rem;
}

.project-link:hover {
    background: {{ colors.primary }};
    color: white;
}

/* Education Styles */
.education-list {
    display: grid;
    gap: 25px;
}

.education-item {
    background: #f8fafc;
    padding: 25px;
    border-radius: 8px;
    border-left: 4px solid {{ colors.primary }};
}

.education-degree {
    color: {{ colors.primary }};
    font-size: 1.2rem;
    margin-bottom: 8px;
}

.education-institution {
    font-weight: 600;
    color: #64748b;
    margin-bottom: 5px;
}

.education-year {
    color: #64748b;
    font-size: 0.9rem;
}

/* Footer */
.footer {
    background: #1e293b;
    color: white;
    text-align: center;
    padding: 30px 0;
    margin-top: 60px;
}

/* Responsive Design */
@media (max-width: 768px) {
    .profile-section {
        grid-template-columns: 1fr;
        text-align: center;
        gap: 30px;
    }
    
    .name {
        font-size: 2.5rem;
    }
    
    .section {
        padding: 25px 20px;
        margin-bottom: 30px;
    }
    
    .skills-grid,
    .projects-grid {
        grid-template-columns: 1fr;
    }
    
    .timeline {
        padding-left: 20px;
    }
}

@media (max-width: 480px) {
    .container {
        padding: 0 15px;
    }
    
    .name {
        font-size: 2rem;
    }
    
    .section {
        padding: 20px 15px;
    }
}
        """)
        
        js_content = """
// Modern Portfolio JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add animation to elements when they come into view
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all sections
    document.querySelectorAll('.section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
    
    // Add hover effects to skill tags
    document.querySelectorAll('.skill-tag').forEach(tag => {
        tag.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
        });
        
        tag.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
    
    // Add click handlers for contact items
    document.querySelectorAll('.contact-item').forEach(item => {
        item.addEventListener('click', function(e) {
            // Add a small animation feedback
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 150);
        });
    });
    
    // Console message
    console.log('Portfolio generated by SkillSync - AI-Powered Career Platform');
});

// Print optimization
window.addEventListener('beforeprint', function() {
    document.body.classList.add('printing');
});

window.addEventListener('afterprint', function() {
    document.body.classList.remove('printing');
});
        """
        
        return {
            'html': html_template,
            'css': css_template,
            'js': js_content
        }
    
    def _get_classic_template(self) -> Dict[str, Template]:
        """Get classic portfolio template"""
        # Similar structure but with classic styling
        # Implementation would be similar to modern template
        return self._get_modern_template()  # Simplified for now
    
    def _get_creative_template(self) -> Dict[str, Template]:
        """Get creative portfolio template"""
        # Similar structure but with creative styling
        return self._get_modern_template()  # Simplified for now
    
    def _get_minimal_template(self) -> Dict[str, Template]:
        """Get minimal portfolio template"""
        return self._get_modern_template()  # Simplified for now
    
    def _get_tech_template(self) -> Dict[str, Template]:
        """Get tech-focused portfolio template"""
        return self._get_modern_template()  # Simplified for now