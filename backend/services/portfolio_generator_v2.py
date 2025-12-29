"""
Portfolio Generator Service (F6)
Generates professional portfolio websites from CV data using Jinja2 templates.
Creates complete HTML/CSS/JS packages ready for deployment.
"""

import logging
import io
import zipfile
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 not available. Install: pip install jinja2")

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Portfolio generation configuration"""
    template_name: str
    color_scheme: str
    include_photo: bool = True
    include_projects: bool = True
    include_contact_form: bool = True
    dark_mode: bool = False


@dataclass
class GeneratedPortfolio:
    """Result of portfolio generation"""
    portfolio_id: str
    template_used: str
    color_scheme: str
    zip_bytes: bytes
    file_size: int
    files_included: List[str]
    preview_url: Optional[str] = None
    download_url: Optional[str] = None


class PortfolioGenerator:
    """
    Production-ready Portfolio Generator.
    Maps CV data to HTML templates and creates deployable ZIP packages.
    """
    
    def __init__(self, templates_dir: str = "templates/portfolio"):
        """
        Initialize Portfolio Generator with Jinja2 environment.
        
        Args:
            templates_dir: Path to portfolio templates directory
        """
        self.templates_dir = Path(templates_dir)
        self.jinja_env: Optional[Environment] = None
        
        # Initialize Jinja2 environment
        if JINJA2_AVAILABLE:
            try:
                if self.templates_dir.exists():
                    self.jinja_env = Environment(
                        loader=FileSystemLoader(str(self.templates_dir)),
                        autoescape=select_autoescape(['html', 'xml']),
                        trim_blocks=True,
                        lstrip_blocks=True
                    )
                    logger.info(f"✅ Jinja2 environment initialized with templates from {templates_dir}")
                else:
                    logger.warning(f"Templates directory not found: {templates_dir}, using inline templates")
                    self.jinja_env = Environment(autoescape=True)
            except Exception as e:
                logger.error(f"Failed to initialize Jinja2: {str(e)}")
                self.jinja_env = None
        
        # Color schemes
        self.color_schemes = {
            'blue': {
                'primary': '#2563eb',
                'secondary': '#64748b',
                'accent': '#0ea5e9',
                'background': '#ffffff',
                'text': '#1e293b'
            },
            'green': {
                'primary': '#059669',
                'secondary': '#6b7280',
                'accent': '#10b981',
                'background': '#ffffff',
                'text': '#1f2937'
            },
            'purple': {
                'primary': '#7c3aed',
                'secondary': '#6b7280',
                'accent': '#8b5cf6',
                'background': '#ffffff',
                'text': '#1e1b4b'
            },
            'red': {
                'primary': '#dc2626',
                'secondary': '#6b7280',
                'accent': '#ef4444',
                'background': '#ffffff',
                'text': '#1f2937'
            },
            'orange': {
                'primary': '#ea580c',
                'secondary': '#6b7280',
                'accent': '#f97316',
                'background': '#ffffff',
                'text': '#1f2937'
            }
        }
        
        # Available templates
        self.templates = {
            'modern': 'Modern - Clean design with animations',
            'classic': 'Classic - Professional corporate style',
            'creative': 'Creative - Colorful and dynamic',
            'minimal': 'Minimal - Ultra-simple focus on content',
            'tech': 'Tech - Developer-friendly terminal theme'
        }
    
    async def generate_portfolio(
        self,
        cv_data: Dict[str, Any],
        config: PortfolioConfig
    ) -> GeneratedPortfolio:
        """
        Generate complete portfolio website package.
        
        Args:
            cv_data: Parsed CV data from CVProcessor
            config: Portfolio configuration
            
        Returns:
            GeneratedPortfolio with ZIP bytes
        """
        logger.info(f"🎨 Generating {config.template_name} portfolio with {config.color_scheme} colors")
        
        try:
            # Step 1: Prepare portfolio data
            portfolio_data = self._prepare_portfolio_data(cv_data, config)
            
            # Step 2: Render HTML template
            html_content = self._render_template(config.template_name, portfolio_data)
            
            # Step 3: Generate CSS with custom colors
            css_content = self._generate_css(config.color_scheme, config.template_name)
            
            # Step 4: Generate JavaScript
            js_content = self._generate_javascript(config.template_name)
            
            # Step 5: Create ZIP package in memory
            portfolio_id = str(uuid.uuid4())
            zip_bytes = self._create_zip_package(
                html_content=html_content,
                css_content=css_content,
                js_content=js_content,
                template_name=config.template_name
            )
            
            # Step 6: Create result object
            result = GeneratedPortfolio(
                portfolio_id=portfolio_id,
                template_used=config.template_name,
                color_scheme=config.color_scheme,
                zip_bytes=zip_bytes,
                file_size=len(zip_bytes),
                files_included=['index.html', 'style.css', 'script.js', 'README.md'],
                preview_url=f"/api/v1/portfolios/{portfolio_id}/preview",
                download_url=f"/api/v1/portfolios/{portfolio_id}/download"
            )
            
            logger.info(f"✅ Portfolio generated: {result.file_size} bytes, {len(result.files_included)} files")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating portfolio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Portfolio generation failed: {str(e)}") from e
    
    def _prepare_portfolio_data(
        self,
        cv_data: Dict[str, Any],
        config: PortfolioConfig
    ) -> Dict[str, Any]:
        """
        Prepare and structure data for template rendering.
        
        Args:
            cv_data: Raw CV data
            config: Portfolio configuration
            
        Returns:
            Structured data dictionary for template
        """
        # Extract personal info
        personal_info = cv_data.get('personal_info', {})
        
        # Extract sections
        sections = cv_data.get('sections', {})
        
        # Extract skills
        skills = cv_data.get('skills', [])
        skills_by_category = self._group_skills_by_category(skills)
        
        # Extract experience
        experience = cv_data.get('experience', [])
        formatted_experience = self._format_experience(experience)
        
        # Extract education
        education = cv_data.get('education', [])
        formatted_education = self._format_education(education)
        
        # Prepare template data
        portfolio_data = {
            'name': personal_info.get('name', 'Your Name'),
            'title': self._extract_title(sections.get('header', '')),
            'email': personal_info.get('email', ''),
            'phone': personal_info.get('phone', ''),
            'location': personal_info.get('location', ''),
            'summary': self._extract_summary(sections),
            'skills': skills_by_category,
            'experience': formatted_experience,
            'education': formatted_education,
            'projects': self._extract_projects(sections.get('projects', '')),
            'certifications': self._extract_certifications(sections.get('certifications', '')),
            'config': {
                'include_photo': config.include_photo,
                'include_projects': config.include_projects,
                'include_contact_form': config.include_contact_form,
                'dark_mode': config.dark_mode
            },
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'template': config.template_name,
                'color_scheme': config.color_scheme
            }
        }
        
        return portfolio_data
    
    def _group_skills_by_category(self, skills: List[Any]) -> Dict[str, List[str]]:
        """
        Group skills by category for better organization.
        
        Args:
            skills: List of skill objects or strings
            
        Returns:
            Dictionary mapping categories to skill lists
        """
        grouped = {}
        
        for skill in skills:
            if isinstance(skill, dict):
                category = skill.get('category', 'other')
                skill_name = skill.get('skill', str(skill))
            else:
                category = 'other'
                skill_name = str(skill)
            
            if category not in grouped:
                grouped[category] = []
            
            grouped[category].append(skill_name)
        
        # Sort categories
        category_order = [
            'programming_languages', 'web_frameworks', 'databases',
            'cloud_platforms', 'devops_tools', 'data_science', 'soft_skills', 'other'
        ]
        
        sorted_grouped = {}
        for cat in category_order:
            if cat in grouped:
                sorted_grouped[cat] = grouped[cat]
        
        # Add any remaining categories
        for cat, skills in grouped.items():
            if cat not in sorted_grouped:
                sorted_grouped[cat] = skills
        
        return sorted_grouped
    
    def _format_experience(self, experience: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format experience entries for template.
        
        Args:
            experience: Raw experience list
            
        Returns:
            Formatted experience list
        """
        formatted = []
        
        for exp in experience:
            formatted.append({
                'title': exp.get('title', 'Position Title'),
                'company': exp.get('company', 'Company Name'),
                'location': exp.get('location', ''),
                'start_date': exp.get('start_year', ''),
                'end_date': exp.get('end_year', 'Present'),
                'description': exp.get('context', ''),
                'highlights': self._extract_highlights(exp.get('context', ''))
            })
        
        return formatted
    
    def _format_education(self, education: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format education entries for template.
        
        Args:
            education: Raw education list
            
        Returns:
            Formatted education list
        """
        formatted = []
        
        for edu in education:
            formatted.append({
                'degree': edu.get('degree', 'Degree'),
                'institution': edu.get('institution', 'Institution'),
                'location': edu.get('location', ''),
                'graduation_year': edu.get('year', ''),
                'gpa': edu.get('gpa', ''),
                'honors': edu.get('honors', '')
            })
        
        return formatted
    
    def _extract_title(self, header_text: str) -> str:
        """Extract professional title from header."""
        # Simple heuristic: second line or first line after name
        lines = [l.strip() for l in header_text.split('\n') if l.strip()]
        if len(lines) >= 2:
            return lines[1]
        return "Professional"
    
    def _extract_summary(self, sections: Dict[str, str]) -> str:
        """Extract professional summary."""
        summary_section = sections.get('summary', sections.get('about', ''))
        if summary_section:
            # Take first paragraph
            paragraphs = summary_section.split('\n\n')
            return paragraphs[0] if paragraphs else summary_section[:500]
        return "Experienced professional with a passion for excellence."
    
    def _extract_projects(self, projects_text: str) -> List[Dict[str, str]]:
        """Extract project entries."""
        projects = []
        if not projects_text:
            return projects
        
        # Simple parsing: look for project titles (lines with certain patterns)
        lines = projects_text.split('\n')
        current_project = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Heuristic: if line starts with capital or has certain keywords
            if line[0].isupper() and len(line.split()) <= 6:
                if current_project:
                    projects.append(current_project)
                current_project = {'title': line, 'description': ''}
            elif current_project:
                current_project['description'] += line + ' '
        
        if current_project:
            projects.append(current_project)
        
        return projects
    
    def _extract_certifications(self, cert_text: str) -> List[Dict[str, str]]:
        """Extract certification entries."""
        certifications = []
        if not cert_text:
            return certifications
        
        lines = [l.strip() for l in cert_text.split('\n') if l.strip()]
        for line in lines:
            certifications.append({
                'name': line,
                'issuer': '',
                'date': ''
            })
        
        return certifications
    
    def _extract_highlights(self, text: str) -> List[str]:
        """Extract bullet points/highlights from text."""
        # Look for lines starting with bullets or dashes
        lines = text.split('\n')
        highlights = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '◦')):
                highlights.append(line.lstrip('•-*◦ '))
        
        return highlights if highlights else [text[:200]]
    
    def _render_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Render HTML template with data.
        
        Args:
            template_name: Template identifier
            data: Portfolio data
            
        Returns:
            Rendered HTML string
        """
        if self.jinja_env and self.templates_dir.exists():
            try:
                template = self.jinja_env.get_template(f"{template_name}.html")
                return template.render(**data)
            except Exception as e:
                logger.warning(f"Failed to load template {template_name}: {str(e)}, using inline")
        
        # Fallback to inline template
        return self._get_inline_template(template_name, data)
    
    def _get_inline_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Generate HTML from inline template (fallback).
        
        Args:
            template_name: Template identifier
            data: Portfolio data
            
        Returns:
            HTML string
        """
        # Modern template (inline version)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{data['name']} - Portfolio</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="name">{data['name']}</h1>
            <p class="title">{data['title']}</p>
            <div class="contact">
                {f'<a href="mailto:{data["email"]}">{data["email"]}</a>' if data.get('email') else ''}
                {f'<span>{data["phone"]}</span>' if data.get('phone') else ''}
            </div>
        </div>
    </header>
    
    <main class="main">
        <section class="section summary">
            <div class="container">
                <h2>About Me</h2>
                <p>{data['summary']}</p>
            </div>
        </section>
        
        <section class="section skills">
            <div class="container">
                <h2>Skills</h2>
                <div class="skills-grid">
                    {self._render_skills(data['skills'])}
                </div>
            </div>
        </section>
        
        <section class="section experience">
            <div class="container">
                <h2>Experience</h2>
                {self._render_experience(data['experience'])}
            </div>
        </section>
        
        <section class="section education">
            <div class="container">
                <h2>Education</h2>
                {self._render_education(data['education'])}
            </div>
        </section>
    </main>
    
    <footer class="footer">
        <div class="container">
            <p>Generated with SkillSync - {data['metadata']['generated_at'][:10]}</p>
        </div>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>"""
        return html
    
    def _render_skills(self, skills: Dict[str, List[str]]) -> str:
        """Render skills HTML."""
        html_parts = []
        for category, skill_list in skills.items():
            category_name = category.replace('_', ' ').title()
            skills_str = ', '.join(skill_list)
            html_parts.append(f"""
                <div class="skill-category">
                    <h3>{category_name}</h3>
                    <p>{skills_str}</p>
                </div>
            """)
        return ''.join(html_parts)
    
    def _render_experience(self, experience: List[Dict[str, Any]]) -> str:
        """Render experience HTML."""
        html_parts = []
        for exp in experience:
            html_parts.append(f"""
                <div class="experience-item">
                    <h3>{exp['title']}</h3>
                    <p class="company">{exp['company']} | {exp['start_date']} - {exp['end_date']}</p>
                    <p class="description">{exp['description']}</p>
                </div>
            """)
        return ''.join(html_parts)
    
    def _render_education(self, education: List[Dict[str, Any]]) -> str:
        """Render education HTML."""
        html_parts = []
        for edu in education:
            html_parts.append(f"""
                <div class="education-item">
                    <h3>{edu['degree']}</h3>
                    <p class="institution">{edu['institution']}</p>
                    {f'<p class="year">{edu["graduation_year"]}</p>' if edu.get('graduation_year') else ''}
                </div>
            """)
        return ''.join(html_parts)
    
    def _generate_css(self, color_scheme: str, template_name: str) -> str:
        """
        Generate CSS with custom colors.
        
        Args:
            color_scheme: Color scheme name
            template_name: Template identifier
            
        Returns:
            CSS string
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes['blue'])
        
        css = f"""
/* SkillSync Portfolio - {template_name.title()} Template */
/* Color Scheme: {color_scheme.title()} */

:root {{
    --color-primary: {colors['primary']};
    --color-secondary: {colors['secondary']};
    --color-accent: {colors['accent']};
    --color-background: {colors['background']};
    --color-text: {colors['text']};
    --font-main: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: var(--font-main);
    color: var(--color-text);
    background-color: var(--color-background);
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}}

.header {{
    background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
    color: white;
    padding: 4rem 0;
    text-align: center;
}}

.name {{
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}

.title {{
    font-size: 1.5rem;
    opacity: 0.9;
    margin-bottom: 1rem;
}}

.contact {{
    display: flex;
    gap: 2rem;
    justify-content: center;
    flex-wrap: wrap;
}}

.contact a {{
    color: white;
    text-decoration: none;
    opacity: 0.9;
}}

.contact a:hover {{
    opacity: 1;
    text-decoration: underline;
}}

.section {{
    padding: 4rem 0;
}}

.section h2 {{
    font-size: 2rem;
    color: var(--color-primary);
    margin-bottom: 2rem;
    border-bottom: 3px solid var(--color-accent);
    padding-bottom: 0.5rem;
}}

.skills-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
}}

.skill-category h3 {{
    color: var(--color-secondary);
    margin-bottom: 0.5rem;
}}

.experience-item,
.education-item {{
    margin-bottom: 2rem;
    padding: 1.5rem;
    border-left: 4px solid var(--color-accent);
    background-color: #f8f9fa;
}}

.experience-item h3,
.education-item h3 {{
    color: var(--color-primary);
    margin-bottom: 0.5rem;
}}

.company,
.institution {{
    color: var(--color-secondary);
    font-weight: 600;
    margin-bottom: 1rem;
}}

.footer {{
    background-color: #1e293b;
    color: white;
    text-align: center;
    padding: 2rem 0;
    margin-top: 4rem;
}}

/* Responsive */
@media (max-width: 768px) {{
    .name {{
        font-size: 2rem;
    }}
    
    .container {{
        padding: 1rem;
    }}
}}
"""
        return css
    
    def _generate_javascript(self, template_name: str) -> str:
        """
        Generate JavaScript for interactivity.
        
        Args:
            template_name: Template identifier
            
        Returns:
            JavaScript string
        """
        js = """
// SkillSync Portfolio - Interactive Elements

document.addEventListener('DOMContentLoaded', function() {
    console.log('Portfolio loaded successfully');
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add animation on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.section').forEach(section => {
        observer.observe(section);
    });
});
"""
        return js
    
    def _create_zip_package(
        self,
        html_content: str,
        css_content: str,
        js_content: str,
        template_name: str
    ) -> bytes:
        """
        Create ZIP package with all files in memory.
        
        Args:
            html_content: Rendered HTML
            css_content: Generated CSS
            js_content: Generated JavaScript
            template_name: Template identifier
            
        Returns:
            ZIP file bytes
        """
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add HTML
            zip_file.writestr('index.html', html_content)
            
            # Add CSS
            zip_file.writestr('style.css', css_content)
            
            # Add JavaScript
            zip_file.writestr('script.js', js_content)
            
            # Add README
            readme_content = f"""# Portfolio Website

Generated by SkillSync on {datetime.utcnow().strftime('%Y-%m-%d')}

## Template: {template_name.title()}

## How to Use

1. Extract this ZIP file
2. Open `index.html` in a web browser
3. To deploy online:
   - Upload to Netlify (drag & drop the folder)
   - Push to GitHub Pages
   - Use Vercel, GitHub Pages, or any static hosting

## Files Included

- `index.html` - Main portfolio page
- `style.css` - Styles and colors
- `script.js` - Interactive features
- `README.md` - This file

## Customization

Edit the HTML, CSS, and JS files to further customize your portfolio.

---
Powered by SkillSync - skillsync.ai
"""
            zip_file.writestr('README.md', readme_content)
        
        # Get ZIP bytes
        zip_bytes = zip_buffer.getvalue()
        logger.info(f"📦 Created ZIP package: {len(zip_bytes)} bytes")
        
        return zip_bytes
    
    def list_available_templates(self) -> Dict[str, str]:
        """
        Get list of available templates.
        
        Returns:
            Dictionary of template names and descriptions
        """
        return self.templates.copy()
    
    def list_color_schemes(self) -> List[str]:
        """
        Get list of available color schemes.
        
        Returns:
            List of color scheme names
        """
        return list(self.color_schemes.keys())
