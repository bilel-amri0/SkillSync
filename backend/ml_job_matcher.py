"""
 ML-Based Job Recommendation Engine
Uses semantic similarity and ML models instead of static job databases
"""
import logging
from typing import Dict, List, Any, Tuple, Optional, Sequence
import numpy as np
from dataclasses import dataclass
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logger = logging.getLogger(__name__)


@dataclass
class MLJobRecommendation:
    """ML-predicted job recommendation"""
    title: str
    similarity_score: float
    predicted_salary_min: int
    predicted_salary_max: int
    confidence: float
    matching_skills: List[str]
    skill_gaps: List[str]
    reasons: List[str]
    growth_potential: str


@dataclass
class MLCertRecommendation:
    """ML-ranked certification recommendation"""
    name: str
    relevance_score: float
    predicted_roi: str
    estimated_time: str
    skill_alignment: float
    reasons: List[str]
    career_boost: float


class MLJobMatcher:
    """ML job matcher that scores *live* job postings instead of static templates."""

    def __init__(self, model):
        self.model = model
        logger.info(" ML Job Matcher initialized with dynamic job source")

    def predict_job_matches(
        self,
        cv_skills: List[str],
        cv_text: str,
        industries: List[Tuple[str, float]],
        experience_years: int,
        seniority: str,
        hard_skills: Optional[List[str]] = None,
        soft_skills: Optional[List[str]] = None,
        projects: Optional[List[Any]] = None,
        work_history: Optional[List[Any]] = None,
        certifications: Optional[List[Any]] = None,
        education: Optional[List[Any]] = None,
        languages: Optional[List[str]] = None,
        tech_stack_clusters: Optional[Dict[str, List[str]]] = None,
        job_postings: Optional[Sequence[Dict[str, Any]]] = None
    ) -> List[MLJobRecommendation]:
        """Use ML to predict best job matches against live job postings."""

        job_postings = job_postings or []
        if not job_postings:
            logger.warning(" No live job postings supplied to ML matcher; returning empty set")
            return []

        cv_skills = cv_skills or []
        hard_skills = hard_skills or []
        soft_skills = soft_skills or []
        projects = projects or []
        work_history = work_history or []
        certifications = certifications or []
        education = education or []
        languages = languages or []
        tech_stack_clusters = tech_stack_clusters or {}

        all_skills = list(dict.fromkeys(cv_skills + hard_skills))
        if not all_skills and soft_skills:
            all_skills = soft_skills
        logger.info(" Predicting job matches for %d candidate skills using %d postings", len(all_skills), len(job_postings))

        cv_profile = self._compose_cv_profile(
            all_skills,
            soft_skills,
            projects,
            work_history,
            certifications,
            education,
            languages,
            tech_stack_clusters,
            cv_text
        )
        cv_embedding = self.model.encode(cv_profile, convert_to_numpy=True)
        cv_skill_lookup = {skill.lower(): skill for skill in all_skills}

        recommendations: List[MLJobRecommendation] = []
        fallback_recs: List[MLJobRecommendation] = []
        derived_experience = experience_years or len(work_history)

        for posting in job_postings:
            normalized_job = self._normalize_posting(posting)
            job_embedding = self.model.encode(normalized_job["profile_text"], convert_to_numpy=True)
            similarity = self._cosine_similarity(cv_embedding, job_embedding)

            matching_skills = self._find_matching_skills(normalized_job["profile_text"], cv_skill_lookup)
            job_keywords = self._extract_keywords(normalized_job["profile_text"], top_n=12)
            skill_gaps = [kw for kw in job_keywords if kw.lower() not in {m.lower() for m in matching_skills}]

            predicted_min, predicted_max = self._estimate_salary_range(
                normalized_job.get("salary"),
                similarity,
                derived_experience,
                len(matching_skills)
            )

            skill_coverage = len(matching_skills) / max(len(job_keywords), 1)
            experience_factor = min(max(derived_experience, 1) / 5.0, 2.0)
            confidence = float(np.clip(similarity * 0.6 + skill_coverage * 0.25 + (experience_factor * 0.15), 0, 1))

            reasons = self._build_reasons(
                normalized_job,
                similarity,
                matching_skills,
                skill_gaps,
                industries,
                projects,
                work_history,
                certifications,
                languages
            )

            recommendation = MLJobRecommendation(
                title=normalized_job["title"],
                similarity_score=similarity,
                predicted_salary_min=predicted_min,
                predicted_salary_max=predicted_max,
                confidence=confidence,
                matching_skills=matching_skills,
                skill_gaps=skill_gaps,
                reasons=reasons,
                growth_potential=self._infer_growth(normalized_job)
            )

            if similarity >= 0.6:
                recommendations.append(recommendation)
            elif similarity >= 0.45:
                fallback_recs.append(recommendation)

        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        if not recommendations and fallback_recs:
            logger.warning(" No postings passed main similarity threshold; returning fallback matches")
            fallback_recs.sort(key=lambda x: x.similarity_score, reverse=True)
            recommendations = fallback_recs

        logger.info(" ML predicted %d job matches from live data", len(recommendations))
        return recommendations

    def _compose_cv_profile(
        self,
        all_skills: List[str],
        soft_skills: List[str],
        projects: List[Any],
        work_history: List[Any],
        certifications: List[Any],
        education: List[Any],
        languages: List[str],
        tech_stack_clusters: Dict[str, List[str]],
        cv_text: str
    ) -> str:
        sections = [
            f"Skills: {', '.join(all_skills)}" if all_skills else None,
            f"Soft skills: {', '.join(soft_skills)}" if soft_skills else None,
            self._stringify_section("Projects", projects),
            self._stringify_section("Work history", work_history),
            self._stringify_section("Certifications", certifications),
            self._stringify_section("Education", education),
            f"Languages: {', '.join(languages)}" if languages else None,
            f"Tech stack clusters: {', '.join(sum((v for v in tech_stack_clusters.values()), []))}" if tech_stack_clusters else None,
            cv_text
        ]
        return '\n'.join(section for section in sections if section)

    def _stringify_section(self, label: str, items: List[Any]) -> Optional[str]:
        if not items:
            return None
        rendered: List[str] = []
        for item in items:
            if isinstance(item, dict):
                rendered.append(' - '.join(str(value) for value in item.values() if value))
            else:
                rendered.append(str(item))
        return f"{label}: {' | '.join(rendered)}" if rendered else None

    def _normalize_posting(self, posting: Dict[str, Any]) -> Dict[str, Any]:
        description = posting.get("description") or posting.get("summary") or ""
        title = posting.get("title") or "Unknown role"
        company = posting.get("company") or posting.get("employer") or "Unknown company"
        profile_text = f"{title} at {company}. {description}"
        return {
            **posting,
            "title": title,
            "company": company,
            "profile_text": profile_text,
            "salary": posting.get("salary"),
        }

    def _extract_keywords(self, text: str, top_n: int = 12) -> List[str]:
        if not text:
            return []
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\+\-#\.]{1,}", text.lower())
        stop_words = ENGLISH_STOP_WORDS.union({
            "and", "with", "for", "the", "will", "role", "team", "company", "experience", "skills", "required"
        })
        filtered = [token for token in tokens if len(token) > 2 and token not in stop_words]
        counts = Counter(filtered)
        return [word.title() for word, _ in counts.most_common(top_n)]

    def _find_matching_skills(self, job_text: str, cv_skill_lookup: Dict[str, str]) -> List[str]:
        if not cv_skill_lookup:
            return []
        text_lower = job_text.lower()
        matches: List[str] = []
        for skill_lower, original in cv_skill_lookup.items():
            if skill_lower and skill_lower in text_lower:
                matches.append(original)
        return matches[:15]

    def _estimate_salary_range(
        self,
        salary_text: Optional[str],
        similarity: float,
        experience_years: int,
        matched_skills: int
    ) -> Tuple[int, int]:
        parsed = self._parse_salary(salary_text) if salary_text else None
        if parsed:
            return parsed
        base = 60000 + int(similarity * 40000) + matched_skills * 1200
        base = int(base * max(1.0, experience_years * 0.15))
        return base, int(base * (1.2 + similarity * 0.6))

    def _parse_salary(self, salary_text: str) -> Optional[Tuple[int, int]]:
        numbers = [int(num) for num in re.findall(r"\d+", salary_text.replace(',', ''))]
        if not numbers:
            return None
        if len(numbers) == 1:
            value = numbers[0]
            return value, int(value * 1.2)
        low, high = numbers[0], numbers[1]
        if low > high:
            low, high = high, low
        return low, high

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_reasons(
        self,
        job: Dict[str, Any],
        similarity: float,
        matching_skills: List[str],
        skill_gaps: List[str],
        industries: List[Tuple[str, float]],
        projects: List[Any],
        work_history: List[Any],
        certifications: List[Any],
        languages: List[str]
    ) -> List[str]:
        reasons = [f" {similarity*100:.1f}% semantic similarity (live job embedding)"]
        if matching_skills:
            reasons.append(f" {len(matching_skills)} direct skills mentioned in the posting")
        if skill_gaps:
            reasons.append(f" Skill gaps surfaced: {', '.join(skill_gaps[:3])}")
        if industries:
            top_inds = ', '.join(str(ind[0] if isinstance(ind, (list, tuple)) else ind) for ind in industries[:2])
            reasons.append(f" Aligns with industries detected in CV: {top_inds}")
        if projects:
            reasons.append(f" Referenced {len(projects)} projects for context")
        if work_history:
            reasons.append(f" Used {len(work_history)} work experiences for temporal context")
        if certifications:
            reasons.append(f" Backed by {len(certifications)} certification signals")
        if languages:
            reasons.append(f" Multilingual advantage: {', '.join(languages)}")
        source = job.get("source")
        if source:
            reasons.append(f" Sourced from {source} in real time")
        return reasons

    def _infer_growth(self, job: Dict[str, Any]) -> str:
        title = (job.get("title") or "").lower()
        if any(keyword in title for keyword in ["lead", "principal", "head", "architect"]):
            return "Very High"
        if "senior" in title or job.get("remote"):
            return "High"
        return "Medium"
    
    


class MLCertRanker:
    """Rank certifications mentioned in *real* job data using semantic similarity."""

    CERT_PATTERN = re.compile(
        r"\b([A-Z0-9][A-Za-z0-9\+&/\- ]{3,80}?(?:Certification|Certificate|Certified(?:\sProfessional|\sAssociate|\sExpert|\sSpecialist)?))",
        re.IGNORECASE
    )

    def __init__(self, model):
        self.model = model
        logger.info(" ML Cert Ranker initialized for dynamic extraction")

    def rank_certifications(
        self,
        cv_skills: List[str],
        job_recommendations: List[MLJobRecommendation],
        cv_text: str,
        hard_skills: Optional[List[str]] = None,
        soft_skills: Optional[List[str]] = None,
        projects: Optional[List[Any]] = None,
        work_history: Optional[List[Any]] = None,
        industries: Optional[List[Any]] = None,
        education: Optional[List[Any]] = None,
        job_postings: Optional[Sequence[Dict[str, Any]]] = None
    ) -> List[MLCertRecommendation]:
        logger.info(" Ranking certifications using live job requirements")

        hard_skills = hard_skills or []
        soft_skills = soft_skills or []
        projects = projects or []
        work_history = work_history or []
        industries = industries or []
        education = education or []
        job_postings = job_postings or []

        combined_skills = list(dict.fromkeys(cv_skills + hard_skills))

        goal_embedding = self._build_goal_embedding(job_recommendations, industries, projects)
        cv_embedding = self._build_cv_embedding(combined_skills, soft_skills, education, cv_text)

        cert_candidates = self._collect_cert_mentions(job_postings)
        if not cert_candidates:
            logger.warning(" No certification mentions detected in job data")
            return []

        recommendations: List[MLCertRecommendation] = []
        skill_gaps = job_recommendations[0].skill_gaps if job_recommendations else []

        for cert_name, info in cert_candidates.items():
            context_text = ' '.join(info["descriptions"])[:1200]
            cert_embedding = self.model.encode(f"{cert_name}. {context_text}", convert_to_numpy=True)
            goal_similarity = self._cosine(goal_embedding, cert_embedding)
            skill_alignment = self._cosine(cv_embedding, cert_embedding)
            gap_coverage = self._gap_coverage(skill_gaps, context_text)
            skill_novelty = self._skill_novelty(combined_skills, context_text)

            relevance_score = float(
                np.clip(goal_similarity * 0.45 + gap_coverage * 0.35 + skill_novelty * 0.20, 0, 1)
            )
            if relevance_score < 0.45:
                continue

            estimated_time = f"{max(3, 8 - min(info['count'], 5))} weeks"
            career_boost = min(0.2 + info['count'] * 0.05 + gap_coverage * 0.25, 0.65)
            roi_pred = self._roi_description(goal_similarity, gap_coverage)

            reasons = [
                f" Mentioned in {info['count']} live job postings",
                f" {goal_similarity*100:.1f}% alignment with target role context",
                f" {gap_coverage*100:.1f}% of critical gaps addressed" if skill_gaps else " Derived from real employer requirements"
            ]
            if info["jobs"]:
                sample_job = info["jobs"][0]
                reasons.append(f" Required for {sample_job.get('title', 'target role')} at {sample_job.get('company', 'unknown company')}")

            recommendations.append(MLCertRecommendation(
                name=cert_name,
                relevance_score=relevance_score,
                predicted_roi=roi_pred,
                estimated_time=estimated_time,
                skill_alignment=float(skill_alignment),
                reasons=reasons,
                career_boost=career_boost
            ))

        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        logger.info(" ML ranked %d certifications from employer data", len(recommendations))
        return recommendations

    def _collect_cert_mentions(self, job_postings: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        certs: Dict[str, Dict[str, Any]] = {}
        for posting in job_postings:
            description = (posting.get("description") or posting.get("summary") or "") + " " + (posting.get("title") or "")
            matches = self.CERT_PATTERN.findall(description)
            for match in matches:
                name = match.strip().title()
                cert_entry = certs.setdefault(name, {"count": 0, "descriptions": [], "jobs": []})
                cert_entry["count"] += 1
                cert_entry["descriptions"].append(description)
                cert_entry["jobs"].append({"title": posting.get("title"), "company": posting.get("company")})
        return certs

    def _build_goal_embedding(self, job_recommendations, industries, projects) -> np.ndarray:
        goal_text_parts: List[str] = []
        if job_recommendations:
            top_job = job_recommendations[0]
            goal_text_parts.append(f"Target role: {top_job.title}")
            if top_job.skill_gaps:
                goal_text_parts.append(f"Skill gaps: {', '.join(top_job.skill_gaps[:6])}")
        if industries:
            industry_text = ', '.join(str(ind[0] if isinstance(ind, (list, tuple)) else ind) for ind in industries)
            goal_text_parts.append(f"Industries: {industry_text}")
        if projects:
            goal_text_parts.append(f"Projects delivered: {len(projects)}")
        goal_text = ' | '.join(goal_text_parts) or "General upskilling"
        return self.model.encode(goal_text, convert_to_numpy=True)

    def _build_cv_embedding(self, combined_skills, soft_skills, education, cv_text) -> np.ndarray:
        sections = [
            f"Skills: {', '.join(combined_skills)}" if combined_skills else None,
            f"Soft skills: {', '.join(soft_skills)}" if soft_skills else None,
            f"Education: {'; '.join(str(entry) for entry in education)}" if education else None,
            cv_text
        ]
        profile = '\n'.join(section for section in sections if section)
        return self.model.encode(profile, convert_to_numpy=True)

    def _gap_coverage(self, skill_gaps: List[str], context_text: str) -> float:
        if not skill_gaps:
            return 0.5
        text_lower = context_text.lower()
        covered = sum(1 for gap in skill_gaps if gap.lower() in text_lower)
        return covered / max(len(skill_gaps), 1)

    def _skill_novelty(self, combined_skills: List[str], context_text: str) -> float:
        if not combined_skills:
            return 1.0
        text_lower = context_text.lower()
        overlap = sum(1 for skill in combined_skills if skill.lower() in text_lower)
        return 1 - (overlap / len(combined_skills))

    def _roi_description(self, goal_similarity: float, gap_coverage: float) -> str:
        composite = (goal_similarity + gap_coverage) / 2
        if composite >= 0.7:
            return "Very High (35%+ salary impact)"
        if composite >= 0.55:
            return "High (25%+ salary impact)"
        return "Medium (15%+ salary impact)"

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
