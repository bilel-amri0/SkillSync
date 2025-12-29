"""
Advanced AI CV Analyzer - Intgration avec SkillSync
Version adapte pour le systme SkillSync existant
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings
import PyPDF2
import io
import logging

from utils.hf_compat import ensure_hf_cached_download

ensure_hf_cached_download()

# Suppress TensorFlow/Transformers warnings
warnings.filterwarnings('ignore')
import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
_os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

logger = logging.getLogger(__name__)

# Fallback imports - use basic functionality if advanced packages not available
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Try importing sentence_transformers with better error handling
    try:
        from sentence_transformers import SentenceTransformer
        ADVANCED_AI_AVAILABLE = True
        logger.info(" Advanced AI packages available (NumPy + SentenceTransformer)")
    except Exception as e:
        logger.warning(f" SentenceTransformer failed: {str(e)[:100]}")
        # NumPy and sklearn are available, just not sentence-transformers
        ADVANCED_AI_AVAILABLE = False
        SentenceTransformer = None
        
except ImportError as e:
    ADVANCED_AI_AVAILABLE = False
    logger.warning(" Advanced AI packages not available, using basic extraction")
    np = None
    cosine_similarity = None
    SentenceTransformer = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# ==================== CONFIGURATION ====================
class Config:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SPACY_MODEL = "fr_core_news_md"
    MIN_CONFIDENCE = 0.65

# ==================== STRUCTURES ====================
@dataclass
class CVAnalysisData:
    raw_text: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    experience: List[Dict] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    confidence_scores: Dict = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'raw_text': self.raw_text,
            'personal_info': {
                'name': self.name,
                'title': self.title
            },
            'contact_info': {
                'email': self.email,
                'phone': self.phone
            },
            'skills': self.skills,
            'experience': self.experience,
            'education': self.education,
            'languages': self.languages,
            'confidence_scores': self.confidence_scores
        }

@dataclass
class JobOffer:
    title: str
    company: str
    description: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    experience_required: int = 0
    location: str = "Remote"

@dataclass
class CVAnalysis:
    match_score: float
    ats_score: float
    matched_skills: List[Dict]
    missing_skills: List[Dict]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    explanation: str
    confidence: float
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'match_score': self.match_score,
            'ats_score': self.ats_score,
            'matched_skills': self.matched_skills,
            'missing_skills': self.missing_skills,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'recommendations': self.recommendations,
            'explanation': self.explanation,
            'confidence': self.confidence
        }

# ==================== ADVANCED AI CV EXTRACTOR ====================
class AdvancedCVExtractor:
    def __init__(self):
        logger.info(" Initialisation Advanced CV Extractor...")
        self.nlp = None
        self.embedder = None
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(Config.SPACY_MODEL)
                logger.info(f" spaCy {Config.SPACY_MODEL} loaded")
            except OSError:
                logger.warning(" spaCy model not found, using basic extraction")
                self.nlp = None
        
        # Initialize embedder if available
        if ADVANCED_AI_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
                logger.info(" Sentence transformer loaded")
            except Exception as e:
                logger.warning(f" Could not load embedder: {e}")
                self.embedder = None
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            else:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def parse_cv_advanced(self, text: str) -> CVAnalysisData:
        """Advanced CV parsing with AI"""
        logger.info("="*80)
        logger.info(" STARTING ADVANCED ML CV ANALYSIS")
        logger.info("="*80)
        logger.info(f" CV Text Length: {len(text)} characters")
        logger.info(f" First 300 characters:")
        logger.info(f"   {text[:300]}")
        logger.info(f" Last 200 characters:")
        logger.info(f"   {text[-200:]}")
        logger.info("-"*80)
        
        cv_data = CVAnalysisData(raw_text=text)
        
        # Extract basic information
        cv_data.name, cv_data.confidence_scores['name'] = self._extract_name(text)
        cv_data.email, cv_data.confidence_scores['email'] = self._extract_email(text)
        cv_data.phone, cv_data.confidence_scores['phone'] = self._extract_phone(text)
        cv_data.title, cv_data.confidence_scores['title'] = self._extract_title(text)
        cv_data.skills, cv_data.confidence_scores['skills'] = self._extract_skills_advanced(text)
        cv_data.experience, cv_data.confidence_scores['experience'] = self._extract_experience_advanced(text)
        cv_data.education, cv_data.confidence_scores['education'] = self._extract_education_advanced(text)
        cv_data.languages, cv_data.confidence_scores['languages'] = self._extract_languages(text)
        
        logger.info("="*80)
        logger.info(" ML ANALYSIS COMPLETE")
        logger.info(f"   Name: {cv_data.name}")
        logger.info(f"   Title: {cv_data.title}")
        logger.info(f"   Email: {cv_data.email}")
        logger.info(f"   Skills: {len(cv_data.skills)} found - {', '.join(cv_data.skills[:10])}")
        logger.info(f"   Experience: {len(cv_data.experience)} entries")
        logger.info(f"   Education: {len(cv_data.education)} entries")
        logger.info("="*80)
        
        return cv_data
    
    def _extract_name(self, text: str) -> Tuple[Optional[str], float]:
        """Extract name with intelligent pattern matching (works without spaCy)"""
        logger.info(" Extracting name...")
        
        # Try spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text[:1000])
                persons = [ent.text for ent in doc.ents if ent.label_ == "PER"]
                if persons:
                    logger.info(f"    Name found (spaCy): {persons[0]}")
                    return persons[0], 0.95
            except:
                pass
        
        # Pattern matching approach (works without spaCy)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Strategy 1: Look for typical name patterns in first 15 lines
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)$',  # "John Doe" or "Jean Pierre Martin"
            r'^([A-Z][A-Z]+\s+[A-Z][a-z]+)$',  # "JOHN Doe"
            r'^([A-Z][a-z]+\s+[A-Z]+)$',  # "John DOE"
            r'^([A-Z]+\s+[A-Z]+)$',  # "JOHN DOE"
            r'^([A-Z]{2,}[A-Z]{2,})$',  # "JOHNDOE" or "RICHARDSANCHEZ" (no space)
        ]
        
        for line in lines[:15]:
            # Skip lines with common CV keywords
            line_lower = line.lower()
            skip_keywords = [
                'cv', 'resume', 'curriculum', 'vitae', 'tel', 'phone', 'email',
                'address', 'adresse', 'age', 'date', 'birth', 'n', 'experience',
                'education', 'skills', 'comptences', 'formation', '@', 'contact',
                'profile', 'summary', 'objective', 'manager', 'engineer', 'developer'
            ]
            
            if any(keyword in line_lower for keyword in skip_keywords):
                continue
            
            # Skip very short or long lines
            if len(line) < 4 or len(line) > 50:
                continue
            
            # Try each pattern
            for pattern in name_patterns:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1)
                    
                    # Handle concatenated names like "RICHARDSANCHEZ"
                    if len(name) > 15 and name.isupper() and ' ' not in name:
                        # Try to split by capital letters (camelCase-like)
                        # Find position where second name likely starts (middle of string)
                        mid = len(name) // 2
                        for i in range(mid - 2, mid + 3):
                            if i > 0 and i < len(name):
                                first = name[:i].capitalize()
                                last = name[i:].capitalize()
                                formatted_name = f"{first} {last}"
                                logger.info(f"    Name found (split): {formatted_name} (from {name})")
                                return formatted_name, 0.75
                    
                    # Additional validation: should have 2-4 words OR be long concatenated name
                    words = name.split()
                    if 2 <= len(words) <= 4:
                        logger.info(f"    Name found (pattern): {name}")
                        return name, 0.80
        
        # Strategy 2: Look for capitalized words in first line
        first_line = lines[0] if lines else ""
        if first_line and not any(kw in first_line.lower() for kw in ['cv', 'resume', 'curriculum']):
            words = first_line.split()
            cap_words = [w for w in words if w and w[0].isupper() and w.isalpha()]
            if 2 <= len(cap_words) <= 4:
                name = ' '.join(cap_words)
                logger.info(f"    Name found (first line): {name}")
                return name, 0.70
        
        logger.info("    No name found")
        return None, 0.0
    
    def _extract_email(self, text: str) -> Tuple[Optional[str], float]:
        """Extract email using regex"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        return (matches[0], 0.95) if matches else (None, 0.0)
    
    def _extract_phone(self, text: str) -> Tuple[Optional[str], float]:
        """Extract phone number"""
        patterns = [
            r'\+33\s?\d{1,2}\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}',
            r'0\d{1}\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}',
            r'\+\d{10,15}',
            r'\d{10,15}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0), 0.90
        
        return None, 0.0
    
    def _extract_title(self, text: str) -> Tuple[Optional[str], float]:
        """Extract professional title with ML-enhanced matching"""
        logger.info(" Extracting job title...")
        
        # Comprehensive title keywords
        title_keywords = [
            # English
            'engineer', 'developer', 'programmer', 'analyst', 'scientist', 'manager', 
            'consultant', 'specialist', 'architect', 'designer', 'administrator',
            'lead', 'senior', 'junior', 'staff', 'principal', 'director',
            'software', 'data', 'full stack', 'backend', 'frontend', 'devops',
            'machine learning', 'ai', 'web', 'mobile', 'cloud', 'security',
            # French
            'ingnieur', 'dveloppeur', 'analyste', 'responsable', 'chef',
            'consultant', 'architecte', 'concepteur', 'administrateur',
            'data scientist', 'tudiant', 'stagiaire', 'technicien'
        ]
        
        lines = text.split('\n')
        found_titles = []
        
        # Search in first 15 lines (after name)
        for i, line in enumerate(lines[:15]):
            line_lower = line.lower().strip()
            
            # Skip empty lines
            if not line_lower:
                continue
            
            # Skip lines with contact info
            if any(skip in line_lower for skip in ['@', 'tel', 'phone', 'email', 'address', '+33', '+1']):
                continue
            
            # Skip very short or very long lines
            if len(line) < 5 or len(line) > 100:
                continue
            
            # Check if line contains title keyword
            for keyword in title_keywords:
                if keyword in line_lower:
                    # Extract the whole line as title
                    title = line.strip()
                    
                    # Clean up common prefixes
                    title = re.sub(r'^(poste|position|titre|title):\s*', '', title, flags=re.IGNORECASE)
                    
                    found_titles.append((title, 0.90))
                    logger.info(f"    Title found: {title}")
                    break
            
            # Stop after finding 2 titles
            if len(found_titles) >= 2:
                break
        
        # Return first found title or None
        if found_titles:
            return found_titles[0]
        
        logger.info("    No job title found")
        return None, 0.0
    
    def _extract_skills_advanced(self, text: str) -> Tuple[List[str], float]:
        """REAL ML-powered skill extraction with semantic understanding"""
        
        # Comprehensive skill database (Technical + Business + Soft Skills)
        tech_skills = [
            # Programming Languages
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "PHP", "Ruby", "Go", 
            "Rust", "Swift", "Kotlin", "Scala", "R", "MATLAB", "Perl", "Shell", "Bash",
            
            # Web Frontend
            "HTML", "HTML5", "CSS", "CSS3", "React", "React.js", "Angular", "Vue", "Vue.js", 
            "Next.js", "Nuxt.js", "Svelte", "jQuery", "Bootstrap", "Tailwind", "Material-UI",
            "Redux", "Vuex", "Webpack", "Vite", "Sass", "SCSS", "LESS",
            
            # Web Backend
            "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring", "Spring Boot",
            "Laravel", "Ruby on Rails", "ASP.NET", ".NET", "Symfony", "NestJS",
            
            # Databases
            "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle", "SQLite",
            "Cassandra", "DynamoDB", "Firebase", "Firestore", "MariaDB", "MS SQL Server",
            "Elasticsearch", "Neo4j", "GraphQL", "NoSQL",
            
            # Cloud & DevOps
            "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes", "K8s",
            "Jenkins", "GitLab CI", "GitHub Actions", "CircleCI", "Travis CI",
            "Terraform", "Ansible", "Puppet", "Chef", "CloudFormation",
            
            # Version Control
            "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Mercurial",
            
            # AI/ML/Data Science
            "Machine Learning", "Deep Learning", "AI", "Artificial Intelligence",
            "NLP", "Natural Language Processing", "Computer Vision", "Data Science",
            "Big Data", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "sklearn",
            "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Jupyter",
            "Spark", "Hadoop", "Kafka", "MLflow", "MLOps",
            
            # Mobile
            "iOS", "Android", "React Native", "Flutter", "Xamarin", "Ionic",
            
            # Testing & QA
            "Testing", "Unit Testing", "Integration Testing", "Jest", "Pytest",
            "Selenium", "Cypress", "JUnit", "Mocha", "Chai", "TestNG",
            
            # Methodologies
            "Agile", "Scrum", "Kanban", "DevOps", "CI/CD", "TDD", "BDD",
            "Microservices", "REST API", "RESTful", "SOAP", "gRPC",
            
            # Operating Systems
            "Linux", "Unix", "Windows", "MacOS", "Ubuntu", "CentOS", "Debian",
            
            # Blockchain
            "Blockchain", "Solidity", "Smart Contracts", "Web3", "Crypto",
            
            # Business & Management Skills
            "Project Management", "Product Management", "Program Management",
            "Team Management", "People Management", "Stakeholder Management",
            "Budget Management", "Risk Management", "Change Management",
            "Strategic Planning", "Business Strategy", "Business Development",
            "Business Analysis", "Process Improvement", "Lean", "Six Sigma",
            
            # Marketing & Sales
            "Digital Marketing", "Content Marketing", "Social Media Marketing",
            "SEO", "SEM", "Email Marketing", "Marketing Strategy",
            "Brand Management", "Public Relations", "PR", "Media Relations",
            "Campaign Management", "Market Research", "Marketing Analytics",
            "Customer Acquisition", "Lead Generation", "Sales",
            "CRM", "Salesforce", "HubSpot", "Google Analytics",
            
            # Design & Creative
            "UI Design", "UX Design", "Graphic Design", "Web Design",
            "Adobe Photoshop", "Illustrator", "Figma", "Sketch",
            "Adobe XD", "InDesign", "Premiere Pro", "After Effects",
            
            # Communication & Soft Skills
            "Communication", "Leadership", "Teamwork", "Collaboration",
            "Problem Solving", "Critical Thinking", "Analytical Skills",
            "Presentation", "Public Speaking", "Negotiation",
            "Time Management", "Organization", "Attention to Detail",
            "Adaptability", "Creativity", "Innovation",
            
            # Finance & Accounting
            "Financial Analysis", "Accounting", "Budgeting", "Forecasting",
            "Excel", "Financial Modeling", "QuickBooks", "SAP",
            
            # HR & Recruitment
            "Recruiting", "Talent Acquisition", "HR Management",
            "Employee Relations", "Training", "Onboarding"
        ]
        
        found_skills = set()  # Use set for automatic deduplication
        text_lower = text.lower()
        
        logger.info(f" ML Skill Extraction Started")
        logger.info(f"   CV Length: {len(text)} characters")
        logger.info(f"   Preview: {text[:150]}...")
        
        # PHASE 1: Exact keyword matching (baseline)
        logger.info(" Phase 1: Keyword Matching...")
        for skill in tech_skills:
            # Check multiple variations
            variations = [
                skill,
                skill.lower(),
                skill.replace('.', ''),
                skill.replace('-', ''),
                skill.replace(' ', '')
            ]
            
            for var in variations:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(var) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    found_skills.add(skill)
                    logger.info(f"    Keyword match: {skill}")
                    break
        
        logger.info(f"   Keywords found: {len(found_skills)} skills")
        
        # PHASE 2: ML Semantic Matching (only if embedder available)
        if self.embedder and ADVANCED_AI_AVAILABLE:
            try:
                logger.info(" Phase 2: ML Semantic Matching...")
                
                # Extract potential skill candidates from CV
                # Strategy: Look for capitalized words, acronyms, technical terms
                words = re.findall(r'\b[A-Z][a-zA-Z0-9+#.]*\b', text)  # Capitalized words
                words += re.findall(r'\b[A-Z]{2,}\b', text)  # Acronyms (AWS, API, etc)
                words += re.findall(r'\b\w+[.#+-]\w+\b', text)  # Tech terms (Node.js, C++, etc)
                
                # Clean and deduplicate
                candidates = []
                for word in words:
                    cleaned = word.strip()
                    if len(cleaned) >= 2 and cleaned not in tech_skills:
                        candidates.append(cleaned)
                
                candidates = list(set(candidates))[:200]  # Limit to 200 for performance
                
                if candidates:
                    logger.info(f"   Found {len(candidates)} candidate terms")
                    logger.info(f"   Candidates: {candidates[:20]}...")  # Show first 20
                    
                    # Batch encode for efficiency
                    import numpy as np
                    candidate_embeddings = self.embedder.encode(
                        candidates, 
                        batch_size=32,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    skill_embeddings = self.embedder.encode(
                        tech_skills,
                        batch_size=32, 
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    
                    # Calculate cosine similarities
                    similarities = cosine_similarity(candidate_embeddings, skill_embeddings)
                    
                    # Find best matches (threshold: 0.70 = high similarity)
                    ml_found = 0
                    for i, candidate in enumerate(candidates):
                        best_match_idx = np.argmax(similarities[i])
                        best_similarity = similarities[i][best_match_idx]
                        
                        if best_similarity > 0.70:
                            matched_skill = tech_skills[best_match_idx]
                            if matched_skill not in found_skills:
                                found_skills.add(matched_skill)
                                ml_found += 1
                                logger.info(f"    ML match: '{candidate}'  {matched_skill} (sim: {best_similarity:.3f})")
                    
                    logger.info(f"   ML found {ml_found} additional skills")
                    
            except Exception as e:
                logger.error(f"    ML extraction failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("    ML embeddings not available, using keywords only")
        
        # Convert to sorted list
        final_skills = sorted(list(found_skills))
        confidence = min(0.95, 0.50 + (len(final_skills) * 0.05))  # Higher confidence with more skills
        
        logger.info(f" FINAL RESULT: {len(final_skills)} skills extracted")
        logger.info(f"   Skills: {', '.join(final_skills)}")
        logger.info(f"   Confidence: {confidence:.2f}")
        
        return final_skills, confidence
    
    def _extract_experience_advanced(self, text: str) -> Tuple[List[Dict], float]:
        """Extract work experience with ML-powered analysis"""
        logger.info(" Extracting work experience...")
        experiences = []
        
        # Date patterns
        year_pattern = r'\b(19|20)\d{2}\b'
        date_range_pattern = r'(19|20)\d{2}\s*[-]\s*((19|20)\d{2}|present|current|aujourd\'hui|actuellement)'
        
        lines = text.split('\n')
        
        # Job title indicators
        title_keywords = [
            'developer', 'engineer', 'manager', 'designer', 'architect', 'analyst',
            'consultant', 'specialist', 'lead', 'senior', 'junior', 'intern',
            'dveloppeur', 'ingnieur', 'chef', 'responsable', 'analyste', 'consultant',
            'directeur', 'coordinateur', 'technicien', 'stagiaire'
        ]
        
        # Education keywords to skip
        education_keywords = [
            'universit', 'university', 'cole', 'school', 'college', 'master',
            'bachelor', 'licence', 'diplme', 'degree', 'phd', 'doctorate'
        ]
        
        years_found = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Find lines with date ranges (likely job entries)
            if re.search(date_range_pattern, line, re.IGNORECASE):
                # Skip if education-related
                if any(edu_kw in line_lower for edu_kw in education_keywords):
                    continue
                
                # Extract years
                years = re.findall(year_pattern, line)
                if years:
                    years_found.extend(years)
                
                # Check if line or nearby lines contain job title
                context_lines = [line]
                if i > 0:
                    context_lines.append(lines[i-1])
                if i < len(lines) - 1:
                    context_lines.append(lines[i+1])
                
                context_text = ' '.join(context_lines).lower()
                has_job_keyword = any(keyword in context_text for keyword in title_keywords)
                
                if has_job_keyword or len(line.split()) >= 3:
                    # Extract description from surrounding lines
                    description_lines = []
                    for j in range(i+1, min(len(lines), i+4)):
                        if lines[j].strip() and not re.search(date_range_pattern, lines[j]):
                            description_lines.append(lines[j].strip())
                    
                    experience = {
                        'title': line.strip(),
                        'company': 'Company',
                        'duration': '-'.join(years) if len(years) >= 2 else (years[0] if years else 'N/A'),
                        'description': ' '.join(description_lines[:2])
                    }
                    experiences.append(experience)
                    logger.info(f"    Found experience: {line.strip()[:50]}...")
        
        # Calculate total years of experience
        total_years = 0
        if years_found:
            years_int = [int(y) for y in years_found]
            if len(years_int) >= 2:
                total_years = max(years_int) - min(years_int)
                logger.info(f"    Estimated {total_years} years experience (from {min(years_int)} to {max(years_int)})")
        
        confidence = 0.85 if experiences else 0.3
        logger.info(f"   Found {len(experiences)} work experiences")
        
        return experiences, confidence
    
    def _extract_education_advanced(self, text: str) -> Tuple[List[Dict], float]:
        """Extract education information"""
        educations = []
        
        # Look for education keywords
        education_keywords = ['universit', 'university', 'cole', 'school', 'master', 'bachelor', 'licence', 'diplme']
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in education_keywords):
                # Extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', line)
                year = year_match.group(0) if year_match else 'N/A'
                
                education = {
                    'degree': line.strip(),
                    'institution': 'Institution',
                    'year': year,
                    'description': ''
                }
                educations.append(education)
        
        return educations, 0.80 if educations else 0.0
    
    def _extract_languages(self, text: str) -> Tuple[List[str], float]:
        """Extract language skills"""
        languages = []
        common_languages = ['franais', 'anglais', 'espagnol', 'allemand', 'italien', 'arabe', 'chinois']
        
        text_lower = text.lower()
        for lang in common_languages:
            if lang in text_lower:
                languages.append(lang.capitalize())
        
        return languages, 0.80 if languages else 0.0

# ==================== CV ANALYZER FOR GAP ANALYSIS ====================
class CVGapAnalyzer:
    def __init__(self):
        self.embedder = None
        if ADVANCED_AI_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            except Exception as e:
                logger.warning(f"Could not load embedder for gap analysis: {e}")
    
    def analyze_cv_job_match(self, cv_data: CVAnalysisData, job_offer: JobOffer = None) -> CVAnalysis:
        """Analyze CV against job requirements"""
        
        # Default job offer if none provided
        if job_offer is None:
            job_offer = JobOffer(
                title="Software Developer",
                company="Tech Company",
                description="Software development position",
                required_skills=["Python", "JavaScript", "SQL"],
                preferred_skills=["React", "Docker", "Git"]
            )
        
        # Calculate match score
        all_job_skills = job_offer.required_skills + job_offer.preferred_skills
        matched_skills = []
        missing_skills = []
        
        if self.embedder and ADVANCED_AI_AVAILABLE:
            match_results = self._advanced_skill_matching(cv_data.skills, all_job_skills)
            matched_skills = match_results['matched']
            missing_skills = match_results['missing']
        else:
            # Basic matching
            for skill in all_job_skills:
                if skill.lower() in [s.lower() for s in cv_data.skills]:
                    matched_skills.append({'skill': skill, 'similarity': 1.0})
                else:
                    missing_skills.append({'skill': skill, 'priority': 'High'})
        
        # Calculate scores
        match_score = (len(matched_skills) / len(all_job_skills) * 100) if all_job_skills else 50
        ats_score = self._calculate_ats_score(cv_data, job_offer)
        
        # Generate insights
        strengths = [f" {skill['skill']}" for skill in matched_skills[:3]]
        weaknesses = [f" Missing: {skill['skill']}" for skill in missing_skills[:3]]
        
        recommendations = self._generate_recommendations(cv_data, missing_skills, match_score)
        
        explanation = f"CV analysis complete. Match score: {match_score:.1f}%, ATS score: {ats_score:.1f}%"
        
        return CVAnalysis(
            match_score=round(match_score, 1),
            ats_score=round(ats_score, 1),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            explanation=explanation,
            confidence=0.85
        )
    
    def _advanced_skill_matching(self, cv_skills: List[str], job_skills: List[str]) -> Dict:
        """Advanced skill matching using embeddings"""
        try:
            if not cv_skills or not job_skills:
                return {'matched': [], 'missing': [{'skill': s, 'priority': 'High'} for s in job_skills]}
            
            cv_emb = self.embedder.encode(cv_skills)
            job_emb = self.embedder.encode(job_skills)
            sims = cosine_similarity(job_emb, cv_emb)
            
            matched = []
            missing = []
            threshold = 0.6
            
            for i, job_skill in enumerate(job_skills):
                max_idx = np.argmax(sims[i])
                max_sim = sims[i][max_idx]
                
                if max_sim > threshold:
                    matched.append({
                        'skill': job_skill,
                        'cv_skill': cv_skills[max_idx],
                        'similarity': float(max_sim)
                    })
                else:
                    missing.append({'skill': job_skill, 'priority': 'High'})
            
            return {'matched': matched, 'missing': missing}
        
        except Exception as e:
            logger.warning(f"Advanced matching failed: {e}")
            # Fallback to basic matching
            matched = []
            missing = []
            for skill in job_skills:
                if skill.lower() in [s.lower() for s in cv_skills]:
                    matched.append({'skill': skill, 'similarity': 1.0})
                else:
                    missing.append({'skill': skill, 'priority': 'High'})
            return {'matched': matched, 'missing': missing}
    
    def _calculate_ats_score(self, cv_data: CVAnalysisData, job_offer: JobOffer) -> float:
        """Calculate ATS optimization score"""
        job_keywords = job_offer.required_skills + job_offer.preferred_skills
        if not job_keywords:
            return 75.0
        
        text_lower = cv_data.raw_text.lower()
        found_keywords = sum(1 for keyword in job_keywords if keyword.lower() in text_lower)
        
        return min(100, (found_keywords / len(job_keywords) * 100) + 20)
    
    def _generate_recommendations(self, cv_data: CVAnalysisData, missing_skills: List[Dict], match_score: float) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Skill recommendations
        for skill in missing_skills[:3]:
            recommendations.append(f" Consider learning {skill['skill']} to improve your profile")
        
        # Experience recommendations
        if len(cv_data.experience) < 2:
            recommendations.append(" Add more work experience or projects to strengthen your profile")
        
        # ATS recommendations
        if match_score < 70:
            recommendations.append(" Optimize your CV with relevant keywords for better ATS scoring")
        
        # Education recommendations
        if not cv_data.education:
            recommendations.append(" Consider adding relevant certifications or education details")
        
        return recommendations[:4]

# ==================== MAIN INTEGRATION CLASS ====================
class AdvancedCVAnalyzer:
    """Main class for advanced CV analysis integration"""
    
    def __init__(self):
        self.extractor = AdvancedCVExtractor()
        self.analyzer = CVGapAnalyzer()
        logger.info(" Advanced CV Analyzer initialized")
    
    def analyze_cv_file(self, file_content, filename: str = None) -> Dict:
        """Analyze CV from file content"""
        try:
            # Extract text
            if filename and filename.lower().endswith('.pdf'):
                text = self.extractor.extract_text_from_pdf(io.BytesIO(file_content))
            else:
                text = file_content.decode('utf-8') if isinstance(file_content, bytes) else file_content
            
            if not text.strip():
                raise ValueError("Could not extract text from file")
            
            # Parse CV
            cv_data = self.extractor.parse_cv_advanced(text)
            
            # Analyze
            analysis = self.analyzer.analyze_cv_job_match(cv_data)
            
            return {
                'success': True,
                'cv_data': cv_data.to_dict(),
                'analysis': analysis.to_dict(),
                'extracted_text': text[:500] + "..." if len(text) > 500 else text
            }
            
        except Exception as e:
            logger.error(f"CV analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'cv_data': None,
                'analysis': None
            }
    
    def generate_dashboard_data(self, cv_data: Dict, analysis: Dict) -> Dict:
        """Generate enhanced dashboard data"""
        try:
            return {
                'total_analyses': 1,
                'skill_progress': [
                    {'skill': skill, 'progress': min(100, 60 + (i * 10))} 
                    for i, skill in enumerate(cv_data.get('skills', [])[:6])
                ],
                'match_score': analysis.get('match_score', 0),
                'ats_score': analysis.get('ats_score', 0),
                'strengths': analysis.get('strengths', []),
                'weaknesses': analysis.get('weaknesses', []),
                'recommendations': analysis.get('recommendations', []),
                'confidence_scores': cv_data.get('confidence_scores', {}),
                'personal_info': cv_data.get('personal_info', {}),
                'contact_info': cv_data.get('contact_info', {}),
                'experience_count': len(cv_data.get('experience', [])),
                'education_count': len(cv_data.get('education', [])),
                'skills_count': len(cv_data.get('skills', [])),
                'languages': cv_data.get('languages', [])
            }
        except Exception as e:
            logger.error(f"Dashboard data generation failed: {str(e)}")
            return {}
    
    def enhance_portfolio_data(self, cv_data: Dict, analysis: Dict) -> Dict:
        """Enhance portfolio with analysis insights"""
        try:
            enhanced_cv = cv_data.copy()
            
            # Add analysis insights to skills
            if analysis and 'matched_skills' in analysis:
                skill_insights = {}
                for matched in analysis['matched_skills']:
                    skill_insights[matched['skill']] = {
                        'confidence': matched.get('similarity', 1.0),
                        'status': 'strong'
                    }
                enhanced_cv['skill_insights'] = skill_insights
            
            # Add recommendations as a new section
            if analysis and 'recommendations' in analysis:
                enhanced_cv['recommendations'] = analysis['recommendations']
            
            # Add match scores
            if analysis:
                enhanced_cv['analysis_scores'] = {
                    'match_score': analysis.get('match_score', 0),
                    'ats_score': analysis.get('ats_score', 0),
                    'confidence': analysis.get('confidence', 0)
                }
            
            return enhanced_cv
            
        except Exception as e:
            logger.error(f"Portfolio enhancement failed: {str(e)}")
            return cv_data

# Global instance for use in FastAPI
advanced_analyzer = AdvancedCVAnalyzer()
