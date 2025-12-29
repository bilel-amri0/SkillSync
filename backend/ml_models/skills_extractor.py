"""
BERT-based Skills Extraction Model
Adapted from notebook for production use
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import re
import logging

# PyTorch imports with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    torch = None
    TORCH_AVAILABLE = False

# Transformers imports with error handling
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_RESUME_MODEL_NAME = os.getenv("SKILLSYNC_RESUME_MODEL_NAME", "dslim/bert-base-NER")
DEFAULT_RESUME_MODEL_PATH = os.getenv("SKILLSYNC_RESUME_MODEL_PATH")
DEFAULT_SKILL_CONFIDENCE = float(os.getenv("SKILLSYNC_SKILL_CONFIDENCE", "0.58"))

# NER Configuration
LABEL_NAMES = ["O", "B-SKILL", "I-SKILL"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Skills database
SKILLS_DATABASE = {
    "data_science": ["Python", "R", "SQL", "Machine Learning", "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn", "Jupyter"],
    "web_development": ["JavaScript", "React", "Node.js", "HTML", "CSS", "MongoDB", "Express", "Vue.js", "Angular", "TypeScript"],
    "mobile": ["Swift", "Kotlin", "React Native", "Flutter", "iOS", "Android", "Java", "Xamarin"],
    "devops": ["Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "Terraform", "Linux", "Git", "CI/CD"],
    "backend": ["Java", "C#", "Go", "Rust", "Spring", "Django", "FastAPI", "MySQL", "PostgreSQL"],
    "frontend": ["React", "Vue.js", "Angular", "CSS", "Sass", "Webpack", "Redux", "Next.js"],
    "cloud": ["AWS", "Azure", "GCP", "Lambda", "EC2", "S3", "CloudFormation", "Serverless"]
}

SAFE_SKILL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+#/\- ]{1,40}$")


class SkillsExtractorModel:
    """Production-ready transformer model configured for CV skill extraction."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        min_confidence: float = DEFAULT_SKILL_CONFIDENCE
    ):
        self.device = None
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_name = model_name or DEFAULT_RESUME_MODEL_NAME
        self.model_path = model_path or DEFAULT_RESUME_MODEL_PATH
        self.min_confidence = min_confidence
        
        self.tokenizer = None
        self.model = None
        self.pipeline_extractor = None
        self.known_skill_lookup = self._build_skill_lookup()
        self.skill_regex = self._build_skill_regex()
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Transformers not available, falling back to regex extraction")
            
    def _initialize_model(self):
        """Initialize Hugging Face NER pipeline"""
        try:
            load_target = self.model_path if self.model_path and Path(self.model_path).exists() else self.model_name
            logger.info("Loading NER pipeline: %s", load_target)
            device = 0 if TORCH_AVAILABLE and torch.cuda.is_available() else -1
            self.pipeline_extractor = pipeline(
                task="ner",
                model=load_target,
                tokenizer=load_target,
                aggregation_strategy="simple",
                device=device
            )
            logger.info("NER pipeline ready on %s", "cuda" if device == 0 else "cpu")
        except Exception as e:
            logger.error("Error loading NER pipeline: %s", e)
            self.pipeline_extractor = None
    
    def extract_skills_bert(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract skills using the transformer NER model with per-skill confidence."""
        if not text.strip():
            return {"skills": [], "discarded": []}
        if self.pipeline_extractor is None:
            fallback = self.extract_skills_fallback(text)
            return {"skills": [{"skill": skill, "confidence": 0.65} for skill in fallback], "discarded": []}

        try:
            ner_input = text[:4000]
            ner_results = self.pipeline_extractor(ner_input)
            accepted: List[Dict[str, Any]] = []
            discarded: List[Dict[str, Any]] = []

            for chunk in ner_results:
                candidate = self._normalize_skill(chunk.get("word"))
                if not candidate:
                    continue
                confidence = float(chunk.get("score") or 0.0)
                record = {"skill": candidate, "confidence": round(confidence, 4)}
                if confidence >= self.min_confidence:
                    accepted.append(record)
                else:
                    discarded.append(record)

            deduped = self._dedupe_by_confidence(accepted)
            if discarded:
                logger.debug("Discarded %s low-confidence spans", len(discarded))

            return {"skills": deduped, "discarded": discarded}

        except Exception as e:
            logger.error("Error in NER extraction: %s", e)
            fallback = self.extract_skills_fallback(text)
            return {"skills": [{"skill": skill, "confidence": 0.62} for skill in fallback], "discarded": []}

    def _finalize_skill(
        self,
        tokens: List[str],
        confidences: List[float],
        accepted: List[Dict[str, Any]],
        discarded: List[Dict[str, Any]]
    ) -> None:
        """Normalize collected tokens into a skill span and route by confidence."""
        if not tokens:
            return
        merged = " ".join(tokens).replace(' ##', '').replace('  ', ' ').strip()
        if not merged or len(merged) <= 1:
            return
        confidence = float(np.mean(confidences)) if confidences else 0.5
        record = {"skill": merged.title(), "confidence": round(confidence, 4)}
        if confidence >= self.min_confidence:
            accepted.append(record)
        else:
            discarded.append(record)
    
    def extract_skills_fallback(self, text: str) -> List[str]:
        """Regex-based fallback that safely matches known skills."""
        if not text or not self.skill_regex:
            return []
        matches = set()
        for match in self.skill_regex.finditer(text.lower()):
            token = match.group(0).strip()
            canonical = self.known_skill_lookup.get(token.lower(), token.title())
            normalized = self._normalize_skill(canonical)
            if normalized:
                matches.add(normalized)
        return sorted(matches)
    
    def extract_skills(self, text: str, use_bert: bool = True) -> Dict[str, Any]:
        """Main skills extraction method with fallback and confidence calibration."""
        if use_bert and self.pipeline_extractor is not None:
            bert_payload = self.extract_skills_bert(text)
            bert_details = bert_payload.get("skills", [])
            discarded = bert_payload.get("discarded", [])
            fallback_skills = self.extract_skills_fallback(text)

            skill_map: Dict[str, Dict[str, Any]] = {
                detail["skill"]: {"confidence": detail["confidence"], "source": "ml"}
                for detail in bert_details
            }

            for skill in fallback_skills:
                title_case = skill.title()
                if title_case not in skill_map:
                    skill_map[title_case] = {"confidence": 0.62, "source": "fallback"}
                else:
                    skill_map[title_case]["confidence"] = max(0.62, skill_map[title_case]["confidence"])

            skill_details = [
                {"skill": name, "confidence": round(meta["confidence"], 3), "source": meta["source"]}
                for name, meta in sorted(skill_map.items(), key=lambda item: item[0].lower())
            ]
            avg_conf = np.mean([detail["confidence"] for detail in skill_details]) if skill_details else 0.0

            return {
                "skills": [detail["skill"] for detail in skill_details],
                "skill_details": skill_details,
                "bert_skills": [detail["skill"] for detail in bert_details],
                "fallback_skills": fallback_skills,
                "low_confidence_skills": discarded,
                "method": "bert_with_fallback",
                "confidence": self._label_confidence(avg_conf),
                "confidence_score": round(float(avg_conf), 3)
            }

        fallback_skills = self.extract_skills_fallback(text)
        fallback_details = [
            {"skill": skill.title(), "confidence": 0.58, "source": "fallback"}
            for skill in fallback_skills
        ]
        return {
            "skills": [detail["skill"] for detail in fallback_details],
            "skill_details": fallback_details,
            "fallback_skills": fallback_skills,
            "method": "rule_based",
            "confidence": "medium",
            "confidence_score": 0.58
        }

    def _label_confidence(self, score: float) -> str:
        if score >= 0.82:
            return "high"
        if score >= 0.65:
            return "medium"
        return "low"
    
    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize extracted skills by domain
        """
        categorized = {category: [] for category in SKILLS_DATABASE.keys()}
        categorized["other"] = []
        
        for skill in skills:
            found_category = False
            for category, category_skills in SKILLS_DATABASE.items():
                if any(skill.lower() == cat_skill.lower() for cat_skill in category_skills):
                    categorized[category].append(skill)
                    found_category = True
                    break
            
            if not found_category:
                categorized["other"].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def get_skill_suggestions(self, current_skills: List[str], target_level: str = "intermediate") -> List[Dict]:
        """
        Suggest related skills based on current skills
        """
        suggestions = []
        
        # Find domains of current skills
        current_domains = set()
        for skill in current_skills:
            for domain, domain_skills in SKILLS_DATABASE.items():
                if any(skill.lower() == ds.lower() for ds in domain_skills):
                    current_domains.add(domain)
        
        # Suggest skills from related domains
        for domain in current_domains:
            domain_skills = SKILLS_DATABASE[domain]
            missing_skills = [s for s in domain_skills if s not in current_skills]
            
            for skill in missing_skills[:3]:  # Top 3 suggestions per domain
                suggestions.append({
                    "skill": skill,
                    "domain": domain,
                    "reason": f"Complements your {domain} skills",
                    "priority": "high" if len(missing_skills) <= 5 else "medium"
                })
        
        return suggestions[:10]  # Limit to top 10

    def _build_skill_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for skills in SKILLS_DATABASE.values():
            for skill in skills:
                lookup[skill.lower()] = skill
        return lookup

    def _build_skill_regex(self) -> Optional[re.Pattern]:
        if not SKILLS_DATABASE:
            return None
        unique_skills = sorted({skill.lower() for skills in SKILLS_DATABASE.values() for skill in skills})
        if not unique_skills:
            return None
        pattern = r"\\b(?:" + "|".join(re.escape(skill) for skill in unique_skills) + r")\\b"
        return re.compile(pattern, re.IGNORECASE)

    def _normalize_skill(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        cleaned = raw.replace('##', '').replace('Ġ', ' ').strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return None
        canonical = self.known_skill_lookup.get(cleaned.lower())
        candidate = canonical or cleaned
        candidate = candidate.strip().title()
        if not SAFE_SKILL_PATTERN.match(candidate):
            return None
        return candidate

    def _dedupe_by_confidence(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            name = entry["skill"]
            existing = deduped.get(name)
            if not existing or existing["confidence"] < entry["confidence"]:
                deduped[name] = entry
        return list(deduped.values())
