"""
ADVANCED CV PARSER - ML-DRIVEN SYSTEM (95% ML, 5% rules)
Uses Advanced ML Modules with Production CV Parser base
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# ML
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Advanced ML Modules
from advanced_ml_modules import (
    SemanticSkillExtractor,
    MLJobTitleExtractor,
    SemanticResponsibilityExtractor,
    SemanticEducationExtractor,
    MLConfidenceScorer,
    IndustryClassifier,
    CareerTrajectoryAnalyzer,
    ProjectExtractor,
    extract_portfolio_links
)

# PDF
import PyPDF2
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== DATA MODELS ====================

@dataclass
class AdvancedCVParseResult:
    """Advanced CV parsing result with ML features"""
    # Personal
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    
    # Professional
    current_title: Optional[str] = None
    seniority_level: str = "Unknown"
    
    # Skills
    skills: List[str] = None
    skill_categories: Dict[str, List[str]] = None
    soft_skills: List[str] = None
    tech_stack_clusters: Dict[str, List[str]] = None
    
    # Experience
    total_years_experience: int = 0
    job_titles: List[str] = None
    companies: List[str] = None
    responsibilities: List[str] = None
    
    # Education
    degrees: List[str] = None
    institutions: List[str] = None
    degree_level: Optional[str] = None
    graduation_year: Optional[int] = None
    
    # Certifications & Languages
    certifications: List[Dict] = None
    languages: List[Dict] = None
    
    # NEW ML FEATURES
    industries: List[Tuple[str, float]] = None  # Top 3 industries with confidence
    career_trajectory: Dict = None  # Speed, gaps, predictions
    projects: List[Dict] = None  # Extracted projects with tech stack
    portfolio_links: Dict[str, str] = None  # GitHub, LinkedIn, Portfolio
    ml_confidence_breakdown: Dict = None  # Per-field confidence scores
    
    # Metadata
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    parser_version: str = "advanced-ml-v1.0"
    
    def __post_init__(self):
        if self.skills is None:
            self.skills = []
        if self.skill_categories is None:
            self.skill_categories = {}
        if self.soft_skills is None:
            self.soft_skills = []
        if self.tech_stack_clusters is None:
            self.tech_stack_clusters = {}
        if self.job_titles is None:
            self.job_titles = []
        if self.companies is None:
            self.companies = []
        if self.responsibilities is None:
            self.responsibilities = []
        if self.degrees is None:
            self.degrees = []
        if self.institutions is None:
            self.institutions = []
        if self.certifications is None:
            self.certifications = []
        if self.languages is None:
            self.languages = []
        if self.industries is None:
            self.industries = []
        if self.projects is None:
            self.projects = []
        if self.portfolio_links is None:
            self.portfolio_links = {}
    
    def to_dict(self):
        return asdict(self)


# ==================== ADVANCED CV PARSER ====================

class AdvancedCVParser:
    """
    Advanced CV Parser - 95% ML-driven
    - Pure semantic skill extraction
    - ML-based job title & seniority
    - Impact vs routine classification
    - Industry classification (25 categories)
    - Career trajectory analysis
    - Project extraction
    """
    
    def __init__(self):
        logger.info(" Initializing Advanced CV Parser (ML-driven)...")
        
        # Load embedding model
        logger.info(" Loading paraphrase-mpnet-base-v2...")
        self.embedder = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
        
        # Skill database
        self._init_skill_database()
        
        # Initialize Advanced ML Modules
        logger.info(" Initializing ML modules...")
        self.semantic_skill_extractor = SemanticSkillExtractor(self.embedder, self.all_skills)
        self.ml_job_extractor = MLJobTitleExtractor(self.embedder)
        self.responsibility_extractor = SemanticResponsibilityExtractor(self.embedder)
        self.education_extractor = SemanticEducationExtractor(self.embedder)
        self.confidence_scorer = MLConfidenceScorer(self.embedder)
        self.industry_classifier = IndustryClassifier(self.embedder)
        self.trajectory_analyzer = CareerTrajectoryAnalyzer(self.embedder)
        self.project_extractor = ProjectExtractor(self.embedder)
        
        logger.info(" Advanced CV Parser ready")
    
    def _init_skill_database(self):
        """Initialize skill categories and database"""
        self.skill_categories = {
            'Programming Languages': [
                'Python', 'JavaScript', 'Java', 'C++', 'C#', 'TypeScript', 'Go', 'Rust',
                'PHP', 'Ruby', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'Perl'
            ],
            'Web Technologies': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django', 'Flask',
                'FastAPI', 'Spring Boot', 'HTML', 'CSS', 'SASS', 'Bootstrap', 'Tailwind'
            ],
            'Databases': [
                'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Cassandra', 'DynamoDB',
                'Oracle', 'SQL Server', 'SQLite', 'Neo4j', 'Elasticsearch'
            ],
            'Cloud & DevOps': [
                'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins',
                'GitLab CI', 'GitHub Actions', 'Terraform', 'Ansible', 'CI/CD'
            ],
            'AI & Machine Learning': [
                'TensorFlow', 'PyTorch', 'scikit-learn', 'Keras', 'NLP', 'Computer Vision',
                'Deep Learning', 'Neural Networks', 'MLOps', 'LLM', 'Transformers'
            ],
            'Data Science': [
                'Pandas', 'NumPy', 'Jupyter', 'Data Analysis', 'Statistics',
                'Data Visualization', 'Power BI', 'Tableau', 'Apache Spark'
            ],
            'Soft Skills': [
                'Leadership', 'Communication', 'Problem Solving', 'Teamwork',
                'Project Management', 'Agile', 'Scrum', 'Critical Thinking'
            ]
        }
        
        self.all_skills = []
        for category, skills in self.skill_categories.items():
            self.all_skills.extend(skills)
        
        # Compute skill embeddings
        logger.info(" Computing skill embeddings...")
        self.skill_embeddings = self.embedder.encode(self.all_skills, show_progress_bar=False)
    
    def parse_cv(self, text: str) -> AdvancedCVParseResult:
        """Parse CV using advanced ML modules"""
        import time
        start = time.time()
        
        result = AdvancedCVParseResult()
        
        # Basic extraction (keep regex for high accuracy)
        result.email = self._extract_email(text)
        result.phone = self._extract_phone(text)
        result.name = self._extract_name(text)
        result.location = self._extract_location(text)
        
        # ML-powered extraction
        result.skills, result.skill_categories = self._extract_skills_ml(text)
        result.current_title, result.job_titles = self._extract_job_titles_ml(text)
        result.total_years_experience, result.companies, result.responsibilities = self._extract_experience_ml(text)
        result.degrees, result.institutions, result.degree_level, result.graduation_year = self._extract_education_ml(text)
        result.certifications = self._extract_certifications_ml(text)
        result.languages = self._extract_languages(text)
        
        # ML seniority prediction
        result.seniority_level = self._ml_predicted_seniority if hasattr(self, '_ml_predicted_seniority') else 'Unknown'
        
        # NEW ML FEATURES
        result.industries = self._extract_industry_classification(text)
        result.career_trajectory = self._analyze_career_trajectory()
        result.projects = self._extract_projects(text)
        result.portfolio_links = self._extract_portfolio_links(text)
        
        # ML confidence scoring
        cv_data = {
            'name': result.name,
            'skills': result.skills,
            'job_titles': result.job_titles,
            'responsibilities': result.responsibilities,
            'education': result.degrees
        }
        confidence_result = self.confidence_scorer.calculate_ml_confidence(cv_data)
        result.confidence_score = confidence_result['overall']
        result.ml_confidence_breakdown = confidence_result['per_field']
        
        result.processing_time_ms = int((time.time() - start) * 1000)
        
        return result
    
    # ==================== ML-POWERED EXTRACTION ====================
    
    def _extract_skills_ml(self, text: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Pure ML skill extraction"""
        skills_dict = self.semantic_skill_extractor.extract_skills_semantic(text, threshold=0.72)
        skills = sorted(skills_dict.keys(), key=lambda s: skills_dict[s][1], reverse=True)
        
        # Categorize
        categorized = {}
        for category, category_skills in self.skill_categories.items():
            found = [s for s in skills if s in category_skills]
            if found:
                categorized[category] = found
        
        return skills, categorized
    
    def _extract_job_titles_ml(self, text: str) -> Tuple[Optional[str], List[str]]:
        """ML-based job title extraction with seniority prediction"""
        current, titles, seniority, progression = self.ml_job_extractor.extract_job_titles_ml(text)
        
        # Store for later use
        self._career_progression = progression
        self._ml_predicted_seniority = seniority
        
        return current, titles
    
    def _extract_experience_ml(self, text: str) -> Tuple[int, List[str], List[str]]:
        """ML-based experience and responsibility extraction"""
        # Years of experience (keep regex)
        years_match = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)', text, re.IGNORECASE)
        total_years = int(years_match[0]) if years_match else 0
        
        # Companies (keep regex for accuracy)
        companies = []
        for match in re.finditer(r'(?:at|@)\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*[\|\(,]|\s+\d{4})', text):
            company = match.group(1).strip()
            if len(company) > 2 and company not in companies:
                companies.append(company)
        
        # ML-based responsibility extraction (impact-focused)
        responsibilities_dict = self.responsibility_extractor.extract_responsibilities_ml(text)
        responsibilities = [s['text'] for s in responsibilities_dict['impact'][:7]]
        
        return total_years, companies, responsibilities
    
    def _extract_education_ml(self, text: str) -> Tuple[List[str], List[str], Optional[str], Optional[int]]:
        """ML-based education extraction"""
        edu_data = self.education_extractor.extract_education_ml(text)
        
        degrees = [d['text'] for d in edu_data.get('degrees', [])]
        institutions = edu_data.get('institutions', [])
        degree_level = edu_data.get('level')
        graduation_year = edu_data.get('graduation_year')
        
        return degrees, institutions, degree_level, graduation_year
    
    def _extract_certifications_ml(self, text: str) -> List[Dict]:
        """ML-based certification extraction"""
        certs = self.education_extractor.extract_certifications_ml(text)
        return certs
    
    def _extract_industry_classification(self, text: str) -> List[Tuple[str, float]]:
        """Classify CV into industries"""
        return self.industry_classifier.classify_industry(text, top_k=3)
    
    def _analyze_career_trajectory(self) -> Dict:
        """Analyze career progression"""
        if hasattr(self, '_career_progression') and self._career_progression:
            return self.trajectory_analyzer.analyze_trajectory(self._career_progression)
        return {}
    
    def _extract_projects(self, text: str) -> List[Dict]:
        """Extract projects with tech stack"""
        return self.project_extractor.extract_projects(text)
    
    def _extract_portfolio_links(self, text: str) -> Dict[str, str]:
        """Extract portfolio links"""
        return extract_portfolio_links(text)
    
    # ==================== BASIC EXTRACTION (Keep from original) ====================
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email (regex - 99% accurate)"""
        match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        return match.group(0) if match else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone (regex - 99% accurate)"""
        patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{2,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name from first lines"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            first_line = lines[0]
            if len(first_line.split()) <= 4 and not re.search(r'[@\d]', first_line):
                return first_line
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location"""
        location_match = re.search(r'(?:Location|Address|City)[\s:]+([A-Za-z\s,]+)', text, re.IGNORECASE)
        return location_match.group(1).strip() if location_match else None
    
    def _extract_languages(self, text: str) -> List[Dict]:
        """Extract languages (keep original logic)"""
        languages = []
        language_section = re.search(r'(?:Languages|Language Skills)[\s:]+(.+?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        
        if language_section:
            lang_text = language_section.group(1)
            common_langs = ['English', 'French', 'Spanish', 'German', 'Chinese', 'Japanese', 
                          'Arabic', 'Portuguese', 'Russian', 'Italian', 'Korean']
            
            for lang in common_langs:
                if re.search(rf'\b{lang}\b', lang_text, re.IGNORECASE):
                    level_match = re.search(rf'{lang}[\s:-]*(\w+)', lang_text, re.IGNORECASE)
                    level = level_match.group(1) if level_match else 'Unknown'
                    languages.append({'language': lang, 'proficiency': level})
        
        return languages
    
    def parse_pdf(self, pdf_content: bytes) -> AdvancedCVParseResult:
        """Parse PDF CV"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return self.parse_cv(text)
        
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return AdvancedCVParseResult()
