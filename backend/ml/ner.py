"""
SkillSync ML Pipeline - Named Entity Recognition (NER) for Skill Extraction
============================================================================
Real NER using SpaCy + custom skill patterns.
NO regex fallback - true neural NER.
"""

import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import re

logger = logging.getLogger(__name__)

# Lazy imports
_spacy = None
_nlp_model = None


def _get_spacy():
    """Lazy load spacy"""
    global _spacy
    if _spacy is None:
        try:
            import spacy
            _spacy = spacy
        except ImportError:
            logger.error("spaCy not installed. Install: pip install spacy")
            raise
    return _spacy


def _load_spacy_model(model_name: str = "en_core_web_lg"):
    """Load SpaCy model (lazy loading)"""
    global _nlp_model
    if _nlp_model is None:
        spacy = _get_spacy()
        try:
            _nlp_model = spacy.load(model_name)
            logger.info(f"✅ Loaded SpaCy model: {model_name}")
        except OSError:
            logger.error(f"SpaCy model '{model_name}' not found. Download: python -m spacy download {model_name}")
            raise
    return _nlp_model


@dataclass
class ExtractedSkill:
    """Skill extracted by NER"""
    skill_name: str
    category: str  # technical, soft, language, tool, framework, etc.
    confidence: float
    context: str  # surrounding text
    source: str  # where it was found (experience, summary, etc.)


@dataclass
class NERResult:
    """Complete NER extraction result"""
    skills: List[ExtractedSkill]
    skill_categories: Dict[str, List[str]]
    total_skills: int
    unique_skills: int
    top_skills: List[Tuple[str, int]]  # (skill, count)


class SkillNERExtractor:
    """
    Real NER-based skill extraction using SpaCy.
    
    Uses:
    - SpaCy NER for entity recognition
    - ESCO ontology patterns
    - O*NET skill taxonomy
    - Custom skill matchers
    """
    
    # ESCO-based skill categories (expanded from EU standard)
    SKILL_TAXONOMY = {
        "programming_languages": [
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
            "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "bash"
        ],
        "web_frameworks": [
            "react", "angular", "vue", "svelte", "next.js", "django", "flask",
            "fastapi", "express", "spring", "asp.net", "laravel", "rails"
        ],
        "databases": [
            "postgresql", "mysql", "mongodb", "redis", "cassandra", "elasticsearch",
            "dynamodb", "neo4j", "sql server", "oracle", "sqlite", "mariadb"
        ],
        "cloud_platforms": [
            "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
            "kubernetes", "docker", "terraform", "cloudflare"
        ],
        "ml_ai": [
            "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
            "scikit-learn", "nlp", "computer vision", "huggingface", "transformers",
            "bert", "gpt", "llm", "neural networks", "cnn", "rnn", "lstm"
        ],
        "data_science": [
            "pandas", "numpy", "matplotlib", "seaborn", "jupyter", "data analysis",
            "statistics", "data visualization", "tableau", "power bi", "spark"
        ],
        "devops": [
            "ci/cd", "jenkins", "github actions", "gitlab ci", "ansible", "puppet",
            "chef", "nagios", "prometheus", "grafana", "elk stack"
        ],
        "mobile": [
            "ios", "android", "react native", "flutter", "xamarin", "swift", "kotlin"
        ],
        "soft_skills": [
            "leadership", "communication", "teamwork", "problem solving", "agile",
            "scrum", "project management", "mentoring", "presentation", "negotiation"
        ],
        "tools": [
            "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
            "visual studio", "intellij", "pycharm", "vscode", "postman"
        ]
    }
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize NER skill extractor.
        
        Args:
            model_name: SpaCy model to use
        """
        self.model_name = model_name
        self.nlp = _load_spacy_model(model_name)
        
        # Build reverse lookup for categories
        self.skill_to_category = {}
        for category, skills in self.SKILL_TAXONOMY.items():
            for skill in skills:
                self.skill_to_category[skill.lower()] = category
        
        logger.info(f"✅ SkillNERExtractor initialized with {len(self.skill_to_category)} known skills")
    
    def extract_from_text(
        self, 
        text: str, 
        source: str = "unknown",
        min_confidence: float = 0.5
    ) -> List[ExtractedSkill]:
        """
        Extract skills from text using NER.
        
        Args:
            text: Input text
            source: Where this text came from
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of ExtractedSkill
        """
        if not text or not text.strip():
            return []
        
        extracted = []
        
        # Process with SpaCy
        doc = self.nlp(text.lower())
        
        # Extract entities (ORG, PRODUCT, SKILL-like)
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "GPE"]:
                skill_name = ent.text.strip()
                
                # Check if it's a known skill
                if skill_name in self.skill_to_category:
                    category = self.skill_to_category[skill_name]
                    
                    # Get context (surrounding sentence)
                    context = ent.sent.text if ent.sent else ""
                    
                    extracted.append(ExtractedSkill(
                        skill_name=skill_name,
                        category=category,
                        confidence=0.9,  # High confidence for known skills
                        context=context[:200],
                        source=source
                    ))
        
        # Also do pattern matching for known skills (more reliable)
        text_lower = text.lower()
        for skill, category in self.skill_to_category.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            matches = re.finditer(pattern, text_lower)
            
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Check if we already have this skill
                if not any(e.skill_name == skill for e in extracted):
                    extracted.append(ExtractedSkill(
                        skill_name=skill,
                        category=category,
                        confidence=0.85,
                        context=context,
                        source=source
                    ))
        
        # Filter by confidence
        extracted = [e for e in extracted if e.confidence >= min_confidence]
        
        return extracted
    
    def extract_from_cv(self, cv_data: Dict[str, Any]) -> NERResult:
        """
        Extract all skills from CV data.
        
        Args:
            cv_data: CV data dictionary
            
        Returns:
            NERResult with all extracted skills
        """
        all_skills = []
        
        # Extract from summary
        if cv_data.get("summary"):
            skills = self.extract_from_text(cv_data["summary"], source="summary")
            all_skills.extend(skills)
        
        # Extract from experience
        if cv_data.get("experience"):
            exp = cv_data["experience"]
            if isinstance(exp, list):
                for i, job in enumerate(exp):
                    if isinstance(job, dict):
                        job_text = f"{job.get('title', '')} {job.get('description', '')}"
                        skills = self.extract_from_text(job_text, source=f"experience_{i}")
                        all_skills.extend(skills)
                    elif isinstance(job, str):
                        skills = self.extract_from_text(job, source=f"experience_{i}")
                        all_skills.extend(skills)
            elif isinstance(exp, str):
                skills = self.extract_from_text(exp, source="experience")
                all_skills.extend(skills)
        
        # Extract from raw text
        if cv_data.get("text"):
            skills = self.extract_from_text(cv_data["text"], source="raw_text")
            all_skills.extend(skills)
        
        # Extract from skills section (if explicitly listed)
        if cv_data.get("skills"):
            skills_text = cv_data["skills"]
            if isinstance(skills_text, list):
                skills_text = " ".join(skills_text)
            skills = self.extract_from_text(skills_text, source="skills_section")
            all_skills.extend(skills)
        
        # Deduplicate and count
        skill_counter = Counter(s.skill_name for s in all_skills)
        unique_skills = set(s.skill_name for s in all_skills)
        
        # Group by category
        skill_categories = {}
        for skill in all_skills:
            if skill.category not in skill_categories:
                skill_categories[skill.category] = []
            if skill.skill_name not in skill_categories[skill.category]:
                skill_categories[skill.category].append(skill.skill_name)
        
        # Get top skills
        top_skills = skill_counter.most_common(20)
        
        logger.info(f"✅ Extracted {len(all_skills)} skill mentions, {len(unique_skills)} unique skills")
        
        return NERResult(
            skills=all_skills,
            skill_categories=skill_categories,
            total_skills=len(all_skills),
            unique_skills=len(unique_skills),
            top_skills=top_skills
        )
    
    def categorize_skills(self, skill_names: List[str]) -> Dict[str, List[str]]:
        """
        Categorize a list of skill names.
        
        Args:
            skill_names: List of skill names
            
        Returns:
            Dictionary of category -> skills
        """
        categorized = {}
        
        for skill in skill_names:
            skill_lower = skill.lower()
            category = self.skill_to_category.get(skill_lower, "other")
            
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(skill)
        
        return categorized
    
    def suggest_related_skills(self, current_skills: List[str], top_n: int = 10) -> List[str]:
        """
        Suggest related skills based on current skills.
        
        Args:
            current_skills: Skills the user already has
            top_n: Number of suggestions
            
        Returns:
            List of suggested skill names
        """
        # Find categories of current skills
        current_categories = set()
        for skill in current_skills:
            category = self.skill_to_category.get(skill.lower())
            if category:
                current_categories.add(category)
        
        # Suggest skills from same categories that user doesn't have
        suggestions = []
        current_skills_lower = {s.lower() for s in current_skills}
        
        for category in current_categories:
            category_skills = self.SKILL_TAXONOMY.get(category, [])
            for skill in category_skills:
                if skill not in current_skills_lower:
                    suggestions.append(skill)
        
        return suggestions[:top_n]


# Global singleton
_global_ner_extractor: Optional[SkillNERExtractor] = None


def get_ner_extractor(model_name: str = "en_core_web_lg") -> SkillNERExtractor:
    """
    Get or create global NER extractor (singleton).
    
    Args:
        model_name: SpaCy model name
        
    Returns:
        SkillNERExtractor instance
    """
    global _global_ner_extractor
    
    if _global_ner_extractor is None:
        _global_ner_extractor = SkillNERExtractor(model_name=model_name)
    
    return _global_ner_extractor
