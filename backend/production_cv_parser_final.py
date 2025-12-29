"""
PRODUCTION CV PARSER - ADVANCED ML INTEGRATED
Uses: paraphrase-mpnet-base-v2 (768-dim) + dslim/bert-base-NER + Advanced ML Modules
CPU-optimized, 4-8GB RAM, Fast, Accurate, 95% ML-driven
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
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

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
class CVParseResult:
    """Complete CV parsing result"""
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
    soft_skills: List[str] = None  # NEW
    tech_stack_clusters: Dict[str, List[str]] = None  # NEW
    
    # Experience
    total_years_experience: int = 0
    job_titles: List[str] = None
    companies: List[str] = None
    responsibilities: List[str] = None  # NEW
    
    # Education
    degrees: List[str] = None
    institutions: List[str] = None
    degree_level: Optional[str] = None  # NEW: Bachelor/Master/PhD
    graduation_year: Optional[int] = None  # NEW
    
    # Certifications & Languages (NEW)
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
    parser_version: str = "advanced-ml-integrated"
    
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
        if self.career_trajectory is None:
            self.career_trajectory = {}
        if self.projects is None:
            self.projects = []
        if self.portfolio_links is None:
            self.portfolio_links = {}
    
    def to_dict(self):
        return asdict(self)


# ==================== PRODUCTION CV PARSER ====================

class ProductionCVParser:
    """
    Production CV Parser - CPU optimized
    - One embedding model (mpnet-768)
    - One NER model (BERT-NER)
    - Hybrid extraction (ML + regex)
    - 4-8GB RAM compatible
    """
    
    def __init__(self):
        logger.info(" Initializing Production CV Parser (Advanced ML)...")
        
        # Load embedding model (420MB)
        logger.info(" Loading paraphrase-mpnet-base-v2...")
        self.embedder = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
        
        # Load NER model (420MB)
        logger.info(" Loading BERT-NER...")
        try:
            self.ner = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                device=-1,  # CPU
                aggregation_strategy="simple"
            )
            self.ner_available = True
        except:
            logger.warning(" NER not available")
            self.ner_available = False
        
        # Load skill database
        self._load_skills()
        
        # Precompute skill embeddings (once)
        logger.info(" Computing skill embeddings...")
        skill_names = [s for s, _ in self.all_skills]
        self.skill_embeddings = self.embedder.encode(skill_names, show_progress_bar=False)
        self.skill_names = skill_names
        
        # Initialize Advanced ML Modules
        logger.info(" Initializing Advanced ML modules...")
        self.semantic_skill_extractor = SemanticSkillExtractor(self.embedder, self.all_skills)
        self.ml_job_extractor = MLJobTitleExtractor(self.embedder)
        self.responsibility_extractor = SemanticResponsibilityExtractor(self.embedder)
        self.education_extractor = SemanticEducationExtractor(self.embedder)
        self.confidence_scorer = MLConfidenceScorer(self.embedder)
        self.industry_classifier = IndustryClassifier(self.embedder)
        self.trajectory_analyzer = CareerTrajectoryAnalyzer(self.embedder)
        self.project_extractor = ProjectExtractor(self.embedder)
        
        logger.info(" Parser ready with Advanced ML")
    
    def _load_skills(self):
        """Load comprehensive skill database"""
        self.skill_db = {
            'Programming': ['Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C', 'PHP', 'Ruby', 'Go', 'Rust', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'Shell', 'Bash'],
            'Frontend': ['HTML', 'CSS', 'React', 'Angular', 'Vue', 'Next.js', 'jQuery', 'Bootstrap', 'Tailwind', 'Redux', 'Webpack', 'Sass', 'TypeScript'],
            'Backend': ['Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 'Spring', 'Laravel', 'ASP.NET', '.NET', 'GraphQL', 'REST API', 'Microservices'],
            'Database': ['SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQLite', 'Elasticsearch', 'NoSQL'],
            'Cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'CI/CD', 'Terraform', 'Ansible'],
            'AI_ML': ['Machine Learning', 'Deep Learning', 'AI', 'NLP', 'Computer Vision', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy', 'Jupyter', 'MLOps'],
            'Business': ['Project Management', 'Product Management', 'Agile', 'Scrum', 'Business Analysis', 'Strategic Planning', 'Lean', 'Six Sigma'],
            'Marketing': ['Digital Marketing', 'SEO', 'SEM', 'Content Marketing', 'Social Media', 'Brand Management', 'Google Analytics', 'CRM', 'Salesforce'],
            'Design': ['UI Design', 'UX Design', 'Figma', 'Sketch', 'Adobe Photoshop', 'Illustrator', 'Adobe XD'],
            'Soft_Skills': ['Leadership', 'Communication', 'Teamwork', 'Problem Solving', 'Critical Thinking', 'Time Management', 'Presentation', 'Negotiation'],
            'Other': ['Git', 'GitHub', 'Linux', 'Testing', 'Selenium', 'Mobile', 'iOS', 'Android', 'React Native']
        }
        
        # Alias for ML method compatibility
        self.skill_categories = self.skill_db
        
        # Flatten
        self.all_skills = []
        for category, skills in self.skill_db.items():
            for skill in skills:
                self.all_skills.append((skill, category))
        
        # Skill synonyms for normalization
        self.skill_synonyms = {
            'JavaScript': ['JS', 'Javascript'], 'TypeScript': ['TS'], 'React': ['ReactJS', 'React.js'],
            'Node.js': ['NodeJS', 'Node'], 'PostgreSQL': ['Postgres'], 'MongoDB': ['Mongo'],
            'Kubernetes': ['K8s'], 'Machine Learning': ['ML'], 'Artificial Intelligence': ['AI'],
            'Natural Language Processing': ['NLP'], 'Amazon Web Services': ['AWS']
        }
        
        # Skill disambiguation contexts
        self.disambiguators = {
            'React': {'pos': ['frontend', 'ui', 'component', 'jsx'], 'neg': ['chemistry']},
            'Python': {'pos': ['programming', 'code', 'django'], 'neg': ['snake']},
            'Swift': {'pos': ['ios', 'apple', 'mobile'], 'neg': ['bird']}
        }
        
        # Soft skills database (expanded)
        self.soft_skills_db = [
            'Leadership', 'Communication', 'Teamwork', 'Problem Solving', 'Critical Thinking',
            'Time Management', 'Adaptability', 'Creativity', 'Work Ethic', 'Attention to Detail',
            'Interpersonal Skills', 'Decision Making', 'Conflict Resolution', 'Emotional Intelligence',
            'Negotiation', 'Presentation', 'Collaboration', 'Flexibility', 'Initiative'
        ]
        
        # Tech stack clusters
        self.tech_clusters = {
            'Frontend': ['HTML', 'CSS', 'JavaScript', 'TypeScript', 'React', 'Angular', 'Vue', 'Next.js'],
            'Backend': ['Node.js', 'Python', 'Java', 'Django', 'Flask', 'FastAPI', 'Spring'],
            'Mobile': ['React Native', 'Flutter', 'Swift', 'Kotlin', 'iOS', 'Android'],
            'Cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes'],
            'DevOps': ['Docker', 'Kubernetes', 'Jenkins', 'CI/CD', 'Terraform'],
            'Data_Science': ['Pandas', 'NumPy', 'Scikit-learn', 'TensorFlow', 'PyTorch'],
            'Database': ['SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis']
        }
        
        # Certifications database
        self.certifications_db = [
            ('AWS Certified', 'Cloud', 'AWS'), ('Azure Certified', 'Cloud', 'Microsoft'),
            ('GCP Professional', 'Cloud', 'Google'), ('PMP', 'Project Management', 'PMI'),
            ('Scrum Master', 'Agile', 'Scrum'), ('CSM', 'Agile', 'Scrum'),
            ('CCNA', 'Networking', 'Cisco'), ('CompTIA Security+', 'Security', 'CompTIA'),
            ('CISSP', 'Security', 'ISC2'), ('Coursera', 'Online Learning', 'Coursera'),
            ('Udemy', 'Online Learning', 'Udemy')
        ]
        
        # Languages database
        self.languages_db = [
            'English', 'Spanish', 'French', 'German', 'Chinese', 'Mandarin', 'Japanese',
            'Arabic', 'Portuguese', 'Russian', 'Italian', 'Korean', 'Hindi'
        ]
        self.proficiency_patterns = {
            'Native': ['native', 'mother tongue', 'bilingual'],
            'Fluent': ['fluent', 'fluency', 'c2', 'proficient'],
            'Advanced': ['advanced', 'c1'],
            'Intermediate': ['intermediate', 'b1', 'b2'],
            'Beginner': ['beginner', 'a1', 'a2', 'elementary']
        }
    
    def parse_cv(self, text: str) -> CVParseResult:
        """Main parsing function with Advanced ML"""
        start_time = datetime.now()
        logger.info("="*60)
        logger.info(" Starting CV analysis (Advanced ML)...")
        
        result = CVParseResult()
        
        # Basic extraction (keep regex for high accuracy)
        result.email = self._extract_email(text)
        result.phone = self._extract_phone(text)
        result.name, result.location = self._extract_name_location(text)
        
        # ML-POWERED EXTRACTION
        result.skills, result.skill_categories = self._extract_skills_ml(text)
        result.current_title, result.job_titles = self._extract_job_titles_ml(text)
        result.total_years_experience, result.companies, result.responsibilities = self._extract_experience_ml(text)
        result.degrees, result.institutions, result.degree_level, result.graduation_year = self._extract_education_ml(text)
        
        # ML seniority prediction
        result.seniority_level = self._ml_predicted_seniority if hasattr(self, '_ml_predicted_seniority') else self._calculate_seniority(result.total_years_experience, len(result.skills))
        
        # Extract certifications, languages, soft skills (ML)
        result.certifications = self._extract_certifications_ml(text)
        result.languages = self._extract_languages(text)
        result.soft_skills = self._extract_soft_skills(text)
        result.tech_stack_clusters = self._cluster_tech_stack(result.skills)
        
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
        
        elapsed = int((datetime.now() - start_time).total_seconds() * 1000)
        result.processing_time_ms = elapsed
        
        logger.info(f" Analysis complete ({elapsed}ms)")
        logger.info(f"   Name: {result.name}")
        logger.info(f"   Skills: {len(result.skills)} (+ {len(result.soft_skills)} soft)")
        logger.info(f"   Industries: {len(result.industries)}")
        logger.info(f"   Projects: {len(result.projects)}")
        logger.info(f"   Seniority: {result.seniority_level} (ML-predicted)")
        logger.info(f"   Certifications: {len(result.certifications)}")
        logger.info(f"   Languages: {len(result.languages)}")
        logger.info(f"   Experience: {result.total_years_experience} years")
        logger.info(f"   Confidence: {result.confidence_score:.2%}")
        logger.info("="*60)
        
        return result
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email with regex"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        return matches[0] if matches else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number"""
        patterns = [
            r'\+\d{1,3}[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}',
            r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _extract_name_location(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract name and location with NER + regex"""
        name = None
        location = None
        
        # NER extraction
        if self.ner_available:
            try:
                entities = self.ner(text[:1500])  # First 1500 chars
                for ent in entities:
                    if ent['entity_group'] == 'PER' and ent['score'] > 0.85 and not name:
                        name = ent['word'].strip()
                    elif ent['entity_group'] == 'LOC' and ent['score'] > 0.80 and not location:
                        location = ent['word'].strip()
            except:
                pass
        
        # Regex fallback for name
        if not name:
            lines = [l.strip() for l in text.split('\n')[:15] if l.strip()]
            patterns = [
                r'^([A-Z][a-z]+\s+[A-Z][a-z]+)$',
                r'^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)$',
                r'^([A-Z]+\s+[A-Z]+)$'
            ]
            
            for line in lines:
                # Skip headers
                if any(kw in line.lower() for kw in ['cv', 'resume', 'email', 'phone', 'experience', 'education']):
                    continue
                
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1)
                        break
                if name:
                    break
        
        return name, location
    
    def _extract_skills(self, text: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        IMPROVED: Hybrid skill extraction with disambiguation
        1. Keyword matching with synonyms
        2. Context disambiguation
        3. Semantic similarity
        """
        found_skills = {}  # skill: (category, confidence, method)
        text_lower = text.lower()
        
        # Stage 1: Enhanced keyword matching
        for skill, category in self.all_skills:
            # Check main skill
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                # Normalize synonyms
                normalized = skill
                for standard, synonyms in self.skill_synonyms.items():
                    if skill in synonyms:
                        normalized = standard
                        break
                
                # Disambiguate (React = frontend, not chemistry)
                if skill in self.disambiguators:
                    ctx = self.disambiguators[skill]
                    pos_count = sum(1 for w in ctx['pos'] if w in text_lower)
                    neg_count = sum(1 for w in ctx['neg'] if w in text_lower)
                    if neg_count > 0 and pos_count == 0:
                        continue  # Skip ambiguous match
                
                found_skills[normalized] = (category, 0.90, 'keyword')
                continue
            
            # Check synonyms
            if skill in self.skill_synonyms:
                for syn in self.skill_synonyms[skill]:
                    syn_pattern = r'\b' + re.escape(syn) + r'\b'
                    if re.search(syn_pattern, text, re.IGNORECASE):
                        found_skills[skill] = (category, 0.85, 'synonym')
                        break
        
        # Stage 2: Semantic similarity
        # Extract candidates
        words = re.findall(r'\b[A-Z][a-zA-Z0-9+#.]*\b', text)
        words += re.findall(r'\b[A-Z]{2,}\b', text)
        words += re.findall(r'\b\w+[.#+-]\w+\b', text)
        
        candidates = list(set(words))[:150]  # Limit for performance
        
        if candidates:
            # Compute embeddings
            candidate_emb = self.embedder.encode(candidates, show_progress_bar=False)
            
            # Calculate similarities
            sims = cosine_similarity(candidate_emb, self.skill_embeddings)
            
            # Find matches (threshold 0.78 - improved)
            for i, candidate in enumerate(candidates):
                best_idx = np.argmax(sims[i])
                best_sim = sims[i][best_idx]
                
                if best_sim > 0.78:  # Higher threshold for better precision
                    matched_skill = self.skill_names[best_idx]
                    category = next(cat for s, cat in self.all_skills if s == matched_skill)
                    
                    if matched_skill not in found_skills or found_skills[matched_skill][1] < best_sim:
                        found_skills[matched_skill] = (category, float(best_sim), 'semantic')
        
        # Build result
        skills = sorted(found_skills.keys(), key=lambda s: found_skills[s][1], reverse=True)
        
        # Categorize
        categories = {}
        for skill in skills:
            cat = found_skills[skill][0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(skill)
        
        return skills, categories
    
    def _extract_job_titles(self, text: str) -> Tuple[Optional[str], List[str]]:
        """Extract job titles with keyword matching"""
        title_keywords = [
            'engineer', 'developer', 'manager', 'analyst', 'scientist', 'designer',
            'architect', 'director', 'lead', 'senior', 'junior', 'specialist',
            'consultant', 'coordinator', 'administrator', 'ingnieur', 'dveloppeur'
        ]
        
        titles = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines[:30]):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Skip empty or very short
            if len(line_stripped) < 5:
                continue
            
            # Check for title keywords
            if any(kw in line_lower for kw in title_keywords):
                # Skip section headers
                if line_stripped.isupper() and len(line_stripped) < 30:
                    continue
                
                # Skip lines with years (likely experience entries)
                if re.search(r'\b(19|20)\d{2}\b', line):
                    continue
                
                # Clean the title
                title = re.sub(r'[\-\|]', '', line_stripped).strip()
                if title and len(title) > 5:
                    titles.append(title)
        
        # Get current title (first one found)
        current = titles[0] if titles else None
        
        # Deduplicate
        unique_titles = []
        seen = set()
        for t in titles:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique_titles.append(t)
        
        return current, unique_titles[:5]  # Max 5
    
    def _extract_experience(self, text: str) -> Tuple[int, List[str], List[str]]:
        """IMPROVED: Extract years, companies, and responsibilities"""
        years = []
        companies = []
        responsibilities = []
        
        # Enhanced date range detection
        patterns = [
            r'(\d{4})\s*[-]\s*(\d{4})',  # 2020 - 2023
            r'(\d{4})\s*[-]\s*(present|current|now)',  # 2020 - Present
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'present' in str(match).lower() or 'current' in str(match).lower():
                    years.append(int(match[0]))
                    years.append(datetime.now().year)
                else:
                    years.extend([int(match[0]), int(match[1])])
        
        # Calculate total experience
        if years:
            min_year = min(years)
            max_year = max(years)
            current_year = datetime.now().year
            
            if max_year >= current_year - 1:
                total_years = current_year - min_year
            else:
                total_years = max_year - min_year
            
            total_years = max(0, min(total_years, 50))
        else:
            total_years = 0
        
        # Extract companies with NER
        if self.ner_available:
            try:
                entities = self.ner(text[:3000])
                for ent in entities:
                    if ent['entity_group'] == 'ORG' and ent['score'] > 0.75:
                        company = ent['word'].strip()
                        if len(company) > 2 and company not in companies:
                            companies.append(company)
            except:
                pass
        
        # Extract responsibilities (bullet points)
        lines = text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if re.match(r'^[\-\*]\s*', line_stripped):
                resp = re.sub(r'^[\-\*]\s*', '', line_stripped)
                if 15 < len(resp) < 200:
                    responsibilities.append(resp)
        
        return total_years, companies[:5], responsibilities[:10]
    
    def _extract_education(self, text: str) -> Tuple[List[str], List[str], Optional[str], Optional[int]]:
        """IMPROVED: Extract degrees, institutions, degree level, graduation year"""
        degrees = []
        institutions = []
        degree_level = None
        graduation_year = None
        
        # Degree level patterns
        degree_levels = {
            'PhD': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'Master': ['master', 'msc', 'ma', 'mba', 'ms', 'me'],
            'Bachelor': ['bachelor', 'bsc', 'ba', 'bs', 'be', 'btech']
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect degree level
            detected_level = None
            for level, patterns in degree_levels.items():
                if any(p in line_lower for p in patterns):
                    detected_level = level
                    break
            
            # Check for degree keywords
            if detected_level:
                if re.search(r'\b(bachelor|master|phd|degree)\b', line_lower):
                    degree = line.strip()
                    if len(degree) > 5 and degree not in degrees:
                        degrees.append(degree)
                        if not degree_level:
                            degree_level = detected_level
            
            # Extract graduation year
            year_match = re.search(r'\b(19|20)\d{2}\b', line)
            if year_match and any(kw in line_lower for kw in ['graduation', 'graduated', 'degree']):
                year = int(year_match.group(0))
                if not graduation_year or year > graduation_year:
                    graduation_year = year
        
        # Extract institutions with NER
        if self.ner_available:
            try:
                # Focus on education section
                edu_section = ""
                in_edu = False
                for line in lines:
                    if any(kw in line.lower() for kw in ['education', 'academic', 'formation']):
                        in_edu = True
                    if in_edu:
                        edu_section += line + "\n"
                        if len(edu_section) > 1000:
                            break
                
                if edu_section:
                    entities = self.ner(edu_section)
                    for ent in entities:
                        if ent['entity_group'] == 'ORG' and ent['score'] > 0.70:
                            inst = ent['word'].strip()
                            if len(inst) > 3 and inst not in institutions:
                                institutions.append(inst)
            except:
                pass
        
        return degrees[:3], institutions[:3], degree_level, graduation_year
    
    def _extract_certifications(self, text: str) -> List[Dict]:
        """NEW: Extract certifications with category and issuer"""
        certifications = []
        seen = set()
        
        for cert_name, category, issuer in self.certifications_db:
            pattern = r'\b' + re.escape(cert_name) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                if cert_name not in seen:
                    certifications.append({
                        'name': cert_name,
                        'category': category,
                        'issuer': issuer,
                        'confidence': 0.90
                    })
                    seen.add(cert_name)
        
        return certifications
    
    def _extract_languages(self, text: str) -> List[Dict]:
        """NEW: Extract natural languages with proficiency"""
        languages = []
        seen = set()
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if len(line) < 5:
                continue
            
            for lang in self.languages_db:
                if lang.lower() in line_lower and lang not in seen:
                    # Find proficiency
                    proficiency = 'Intermediate'
                    confidence = 0.70
                    
                    for level, patterns in self.proficiency_patterns.items():
                        if any(p in line_lower for p in patterns):
                            proficiency = level
                            confidence = 0.90
                            break
                    
                    languages.append({
                        'language': lang,
                        'proficiency': proficiency,
                        'confidence': confidence
                    })
                    seen.add(lang)
        
        return languages
    
    def _extract_soft_skills(self, text: str) -> List[str]:
        """NEW: Extract soft skills using keyword + embedding hybrid"""
        found_soft_skills = set()
        text_lower = text.lower()
        
        # Keyword matching
        for skill in self.soft_skills_db:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_soft_skills.add(skill)
        
        # Semantic matching for variations
        sentences = re.split(r'[.!?]\s+', text)
        candidates = []
        
        for sent in sentences[:50]:  # Limit for performance
            if any(kw in sent.lower() for kw in ['skill', 'ability', 'strength']):
                words = sent.split()
                for i, word in enumerate(words):
                    if len(word) > 4 and i < len(words) - 1:
                        phrase = f"{word} {words[i+1]}"
                        candidates.append(phrase)
        
        if candidates:
            try:
                candidate_emb = self.embedder.encode(candidates[:20], show_progress_bar=False)
                soft_skill_emb = self.embedder.encode(self.soft_skills_db, show_progress_bar=False)
                sims = cosine_similarity(candidate_emb, soft_skill_emb)
                
                for i in range(len(candidates[:20])):
                    best_idx = np.argmax(sims[i])
                    if sims[i][best_idx] > 0.80:
                        found_soft_skills.add(self.soft_skills_db[best_idx])
            except:
                pass
        
        return sorted(list(found_soft_skills))
    
    def _cluster_tech_stack(self, skills: List[str]) -> Dict[str, List[str]]:
        """NEW: Group skills into tech stack clusters"""
        clusters = {}
        
        for skill in skills:
            for cluster_name, cluster_skills in self.tech_clusters.items():
                if skill in cluster_skills or skill.lower() in [s.lower() for s in cluster_skills]:
                    if cluster_name not in clusters:
                        clusters[cluster_name] = []
                    if skill not in clusters[cluster_name]:
                        clusters[cluster_name].append(skill)
        
        return clusters
    
    def _calculate_seniority(self, years: int, skill_count: int) -> str:
        """Calculate seniority level"""
        if years >= 8 or skill_count >= 25:
            return "Senior"
        elif years >= 4 or skill_count >= 15:
            return "Mid-Level"
        elif years >= 1 or skill_count >= 8:
            return "Junior"
        else:
            return "Entry-Level"
    
    def _calculate_confidence(self, result: CVParseResult) -> float:
        """Calculate overall confidence score (fallback)"""
        scores = []
        
        # Name (0.15 weight)
        if result.name:
            scores.append(0.15)
        
        # Contact (0.10 weight)
        if result.email:
            scores.append(0.05)
        if result.phone:
            scores.append(0.05)
        
        # Skills (0.35 weight)
        if result.skills:
            skill_score = min(len(result.skills) / 20, 1.0) * 0.35
            scores.append(skill_score)
        
        # Experience (0.20 weight)
        if result.total_years_experience > 0:
            scores.append(0.20)
        
        # Education (0.10 weight)
        if result.degrees:
            scores.append(0.10)
        
        # Job titles (0.10 weight)
        if result.job_titles:
            scores.append(0.10)
        
        return sum(scores)
    
    # ==================== ADVANCED ML EXTRACTION METHODS ====================
    
    def _extract_skills_ml(self, text: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Pure ML skill extraction using SemanticSkillExtractor"""
        skills_dict = self.semantic_skill_extractor.extract_skills_semantic(text, threshold=0.72)
        skills = sorted(skills_dict.keys(), key=lambda s: skills_dict[s][1], reverse=True)
        
        # Categorize skills
        categorized = {}
        for category, category_skills in self.skill_categories.items():
            found = [s for s in skills if any(cs.lower() in s.lower() or s.lower() in cs.lower() for cs in category_skills)]
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
        return self.education_extractor.extract_certifications_ml(text)
    
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


# ==================== HELPER FUNCTIONS ====================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        if isinstance(pdf_file, bytes):
            pdf_file = io.BytesIO(pdf_file)
        
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean text
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Initialize parser (loads models once)
    parser = ProductionCVParser()
    
    # Example CV text
    sample_cv = """
    JOHN DOE
    Software Engineer
    john.doe@email.com | +1-555-123-4567 | San Francisco, CA
    
    PROFESSIONAL SUMMARY
    Senior software engineer with 8 years of experience in full-stack development.
    
    SKILLS
    Python, JavaScript, React, Node.js, Docker, AWS, PostgreSQL, Git, Agile
    
    EXPERIENCE
    2020 - Present: Senior Software Engineer at Tech Corp
    - Led team of 5 developers
    - Built microservices with Docker and Kubernetes
    - Improved performance by 40%
    
    2016 - 2020: Software Engineer at StartupXYZ
    - Developed web applications with React and Node.js
    - Implemented CI/CD pipelines
    
    EDUCATION
    Bachelor of Science in Computer Science
    Stanford University, 2016
    """
    
    # Parse CV
    result = parser.parse_cv(sample_cv)
    
    # Output as JSON
    import json
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(json.dumps(result.to_dict(), indent=2))
