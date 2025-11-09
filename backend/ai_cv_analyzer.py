"""
Advanced AI CV Analyzer - Intégration avec SkillSync
Version adaptée pour le système SkillSync existant
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

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Fallback imports - use basic functionality if advanced packages not available
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    ADVANCED_AI_AVAILABLE = True
    logger.info("✅ Advanced AI packages available")
except ImportError:
    ADVANCED_AI_AVAILABLE = False
    logger.warning("⚠️ Advanced AI packages not available, using basic extraction")
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
        logger.info("🧠 Initialisation Advanced CV Extractor...")
        self.nlp = None
        self.embedder = None
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(Config.SPACY_MODEL)
                logger.info(f"✅ spaCy {Config.SPACY_MODEL} loaded")
            except OSError:
                logger.warning("⚠️ spaCy model not found, using basic extraction")
                self.nlp = None
        
        # Initialize embedder if available
        if ADVANCED_AI_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
                logger.info("✅ Sentence transformer loaded")
            except Exception as e:
                logger.warning(f"⚠️ Could not load embedder: {e}")
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
        logger.info("🤖 Advanced CV analysis...")
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
        
        return cv_data
    
    def _extract_name(self, text: str) -> Tuple[Optional[str], float]:
        """Extract name using NLP or pattern matching"""
        if self.nlp:
            doc = self.nlp(text[:1000])
            persons = [ent.text for ent in doc.ents if ent.label_ == "PER"]
            if persons:
                return persons[0], 0.95
        
        # Fallback to pattern matching
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[:5]:
            if re.match(r'^[A-Z][a-z]+\s[A-Z][a-z]+$', line):
                return line, 0.75
        
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
        """Extract professional title"""
        title_keywords = [
            'engineer', 'developer', 'analyst', 'scientist', 'manager', 
            'consultant', 'specialist', 'ingénieur', 'développeur', 
            'analyste', 'data scientist', 'étudiant'
        ]
        
        lines = text.split('\n')
        for line in lines[:10]:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in title_keywords):
                return line.strip(), 0.85
        
        return None, 0.0
    
    def _extract_skills_advanced(self, text: str) -> Tuple[List[str], float]:
        """Advanced skill extraction using embeddings or pattern matching"""
        tech_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "C", "PHP", "Ruby", "Go", "Rust", "Swift", "Kotlin",
            "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Express", "Django", "Flask", "Spring",
            "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle", "SQLite",
            "Git", "GitHub", "GitLab", "Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "GCP",
            "Linux", "Unix", "Windows", "MacOS",
            "Machine Learning", "Deep Learning", "AI", "NLP", "Computer Vision", "Data Science", "Big Data",
            "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "MATLAB", "R",
            "Agile", "Scrum", "DevOps", "CI/CD", "Testing"
        ]
        
        found_skills = []
        text_upper = text.upper()
        
        # Basic pattern matching
        for skill in tech_skills:
            if skill.upper() in text_upper:
                found_skills.append(skill)
        
        # Advanced extraction with embeddings if available
        if self.embedder and self.nlp:
            try:
                doc = self.nlp(text)
                candidates = [token.text for token in doc if token.pos_ in ["PROPN", "NOUN"] and len(token.text) > 2]
                
                if candidates:
                    candidate_emb = self.embedder.encode(candidates)
                    tech_emb = self.embedder.encode(tech_skills)
                    sims = cosine_similarity(candidate_emb, tech_emb)
                    
                    for i, cand in enumerate(candidates):
                        max_sim_idx = np.argmax(sims[i])
                        if sims[i][max_sim_idx] > 0.7:
                            found_skills.append(tech_skills[max_sim_idx])
            except Exception as e:
                logger.warning(f"Advanced skill extraction failed: {e}")
        
        return list(set(found_skills)), 0.85 if found_skills else 0.0
    
    def _extract_experience_advanced(self, text: str) -> Tuple[List[Dict], float]:
        """Extract work experience"""
        experiences = []
        
        # Look for date patterns indicating experience
        date_pattern = r'\b(19|20)\d{2}\b'
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if re.search(date_pattern, line) and len(line.split()) > 2:
                # Skip education lines
                if not any(edu_word in line.lower() for edu_word in ['université', 'university', 'école', 'school', 'master', 'bachelor']):
                    # Extract dates
                    dates = re.findall(date_pattern, line)
                    duration = '-'.join(dates) if len(dates) > 1 else dates[0] if dates else 'N/A'
                    
                    # Get description from next lines
                    description_lines = []
                    for j in range(i+1, min(len(lines), i+3)):
                        if lines[j].strip() and not re.search(date_pattern, lines[j]):
                            description_lines.append(lines[j].strip())
                    
                    experience = {
                        'title': line.strip(),
                        'company': 'Entreprise',
                        'duration': duration,
                        'description': ' '.join(description_lines[:2])
                    }
                    experiences.append(experience)
        
        return experiences, 0.80 if experiences else 0.0
    
    def _extract_education_advanced(self, text: str) -> Tuple[List[Dict], float]:
        """Extract education information"""
        educations = []
        
        # Look for education keywords
        education_keywords = ['université', 'university', 'école', 'school', 'master', 'bachelor', 'licence', 'diplôme']
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
        common_languages = ['français', 'anglais', 'espagnol', 'allemand', 'italien', 'arabe', 'chinois']
        
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
        strengths = [f"✅ {skill['skill']}" for skill in matched_skills[:3]]
        weaknesses = [f"⚠️ Missing: {skill['skill']}" for skill in missing_skills[:3]]
        
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
            recommendations.append(f"🎯 Consider learning {skill['skill']} to improve your profile")
        
        # Experience recommendations
        if len(cv_data.experience) < 2:
            recommendations.append("💼 Add more work experience or projects to strengthen your profile")
        
        # ATS recommendations
        if match_score < 70:
            recommendations.append("📄 Optimize your CV with relevant keywords for better ATS scoring")
        
        # Education recommendations
        if not cv_data.education:
            recommendations.append("🎓 Consider adding relevant certifications or education details")
        
        return recommendations[:4]

# ==================== MAIN INTEGRATION CLASS ====================
class AdvancedCVAnalyzer:
    """Main class for advanced CV analysis integration"""
    
    def __init__(self):
        self.extractor = AdvancedCVExtractor()
        self.analyzer = CVGapAnalyzer()
        logger.info("🚀 Advanced CV Analyzer initialized")
    
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
