"""
 ML-Based Learning Path Optimizer
Uses reinforcement learning concepts and ML to create personalized roadmaps
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from services.course_catalog_client import CourseCatalogClient

logger = logging.getLogger(__name__)


@dataclass
class LearningPhase:
    """ML-optimized learning phase"""
    phase_name: str
    duration_weeks: int
    skills_to_learn: List[str]
    learning_resources: List[Dict[str, Any]]
    success_probability: float
    effort_level: str
    milestones: List[str]


@dataclass
class MLLearningRoadmap:
    """Complete ML-optimized learning roadmap"""
    total_duration_weeks: int
    phases: List[LearningPhase]
    predicted_success_rate: float
    personalization_score: float
    learning_strategy: str


class MLLearningOptimizer:
    """
    ML-Based Learning Path Optimizer
    Uses ML to create personalized, optimal learning paths
    """
    
    def __init__(self, model, course_client: Optional[CourseCatalogClient] = None):
        """Initialize with sentence transformer for semantic learning"""
        self.model = model
        self.course_catalog = course_client or CourseCatalogClient()
        logger.info(" ML Learning Optimizer initialized with live course catalog")
    
    def create_optimal_roadmap(
        self,
        current_skills: List[str],
        target_skills: List[str],
        experience_years: int,
        learning_pace: Optional[str] = None,
        cv_text: str = "",
        projects: Optional[List[Any]] = None,
        work_history: Optional[List[Any]] = None,
        certifications: Optional[List[Any]] = None,
        education: Optional[List[Any]] = None,
        soft_skills: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        target_role: Optional[str] = None
    ) -> MLLearningRoadmap:
        """
        Use ML to create an optimal, personalized learning roadmap
        
        Args:
            current_skills: Skills user already has
            target_skills: Skills needed for target role
            experience_years: Years of experience (affects learning speed)
            learning_pace: "fast", "moderate", "slow", or None for auto
            cv_text: CV text for additional context
            projects/work_history/certifications/education/soft_skills/languages: contextual CV data
            target_role: Job family we are optimizing for (used to contextualize resources)
        """
        projects = projects or []
        work_history = work_history or []
        certifications = certifications or []
        education = education or []
        soft_skills = soft_skills or []
        languages = languages or []

        contextual_skills = self._extract_contextual_skills(projects, work_history, certifications, education, soft_skills, languages, cv_text)
        enriched_current_skills = list(dict.fromkeys(current_skills + contextual_skills))

        if not learning_pace:
            if experience_years >= 7 or len(work_history) >= 5:
                learning_pace = "fast"
            elif experience_years <= 2:
                learning_pace = "slow"
            else:
                learning_pace = "moderate"

        logger.info(f" Creating ML-optimized roadmap: {len(enriched_current_skills)} known skills vs {len(target_skills)} targets")
        
        # ML: Calculate skill gaps using semantic similarity
        true_gaps = []
        for target_skill in target_skills:
            target_emb = self.model.encode(target_skill, convert_to_numpy=True)
            
            # Check if skill is truly missing (semantic check)
            is_missing = True
            for current_skill in enriched_current_skills:
                current_emb = self.model.encode(current_skill, convert_to_numpy=True)
                similarity = float(np.dot(target_emb, current_emb) / 
                                 (np.linalg.norm(target_emb) * np.linalg.norm(current_emb)))
                if similarity > 0.75:  # Already have this skill
                    is_missing = False
                    break
            
            if is_missing:
                true_gaps.append(target_skill)
        
        logger.info(f" ML identified {len(true_gaps)} true skill gaps to learn")
        
        # ML: Cluster skills by difficulty and prerequisites
        skill_clusters = self._cluster_skills_by_ml(true_gaps, enriched_current_skills)
        
        # ML: Optimize phase durations based on user profile
        pace_multiplier = {"fast": 0.7, "moderate": 1.0, "slow": 1.4}.get(learning_pace, 1.0)
        experience_multiplier = max(0.7, 1.2 - (experience_years * 0.1))
        if work_history:
            experience_multiplier *= max(0.8, 1 - (0.02 * len(work_history)))
        
        phases = []
        total_weeks = 0
        
        # Phase 1: Foundation (Beginner skills)
        if skill_clusters['beginner']:
            phase1_skills = skill_clusters['beginner']
            phase1_duration = int(len(phase1_skills) * 4 * pace_multiplier * experience_multiplier)
            phase1_resources = self._build_phase_resources(phase1_skills, "Foundation", target_role)
            
            # ML: Predict success probability based on skill similarity to existing skills
            success_prob = self._predict_learning_success(phase1_skills, enriched_current_skills)
            
            phases.append(LearningPhase(
                phase_name=" Foundation Phase",
                duration_weeks=phase1_duration,
                skills_to_learn=phase1_skills,
                learning_resources=phase1_resources,
                success_probability=success_prob,
                effort_level="Low to Medium" if len(phase1_skills) <= 3 else "Medium",
                milestones=[
                    f"Complete {len(phase1_resources)} foundational courses",
                    f"Build 2-3 beginner projects using {', '.join(phase1_skills)}",
                    "Pass skill assessments for each technology"
                ]
            ))
            total_weeks += phase1_duration
        
        # Phase 2: Intermediate (Core skills)
        if skill_clusters['intermediate']:
            phase2_skills = skill_clusters['intermediate']
            phase2_duration = int(len(phase2_skills) * 6 * pace_multiplier * experience_multiplier)
            phase2_resources = self._build_phase_resources(phase2_skills, "Acceleration", target_role)
            
            success_prob = self._predict_learning_success(phase2_skills, enriched_current_skills + skill_clusters['beginner'])
            
            phases.append(LearningPhase(
                phase_name=" Acceleration Phase",
                duration_weeks=phase2_duration,
                skills_to_learn=phase2_skills,
                learning_resources=phase2_resources,
                success_probability=success_prob,
                effort_level="Medium to High",
                milestones=[
                    f"Master {len(phase2_skills)} intermediate technologies",
                    "Build 3-4 real-world projects",
                    "Contribute to open source projects",
                    "Earn 1-2 professional certifications"
                ]
            ))
            total_weeks += phase2_duration
        
        # Phase 3: Advanced (Specialization)
        if skill_clusters['advanced']:
            phase3_skills = skill_clusters['advanced']
            phase3_duration = int(len(phase3_skills) * 8 * pace_multiplier * experience_multiplier)
            phase3_resources = self._build_phase_resources(phase3_skills, "Mastery", target_role)
            
            all_prior_skills = current_skills + skill_clusters['beginner'] + skill_clusters['intermediate']
            success_prob = self._predict_learning_success(phase3_skills, list(dict.fromkeys(enriched_current_skills + all_prior_skills)))
            
            phases.append(LearningPhase(
                phase_name=" Mastery Phase",
                duration_weeks=phase3_duration,
                skills_to_learn=phase3_skills,
                learning_resources=phase3_resources,
                success_probability=success_prob,
                effort_level="High",
                milestones=[
                    f"Achieve expertise in {len(phase3_skills)} advanced areas",
                    "Lead complex projects end-to-end",
                    "Mentor junior developers",
                    "Publish articles or speak at conferences",
                    "Earn advanced certifications"
                ]
            ))
            total_weeks += phase3_duration
        
        # ML: Calculate overall personalization and success metrics
        personalization = self._calculate_personalization_score(current_skills, target_skills, experience_years)
        predicted_success = sum(p.success_probability for p in phases) / len(phases) if phases else 0.8
        
        # ML: Determine learning strategy
        if len(true_gaps) <= 3:
            strategy = "Focused Sprint (Deep dive into few skills)"
        elif len(true_gaps) <= 6:
            strategy = "Balanced Approach (Breadth and depth)"
        else:
            strategy = "Comprehensive Upskilling (Wide skill expansion)"
        
        roadmap = MLLearningRoadmap(
            total_duration_weeks=total_weeks,
            phases=phases,
            predicted_success_rate=predicted_success,
            personalization_score=personalization,
            learning_strategy=strategy
        )
        
        logger.info(f" ML created {len(phases)}-phase roadmap ({total_weeks} weeks, {predicted_success*100:.0f}% success rate)")
        return roadmap
    
    def _cluster_skills_by_ml(self, skills: List[str], existing_skills: List[str]) -> Dict[str, List[str]]:
        """
        Use ML to cluster skills by difficulty level
        """
        clusters = {
            'beginner': [],
            'intermediate': [],
            'advanced': []
        }
        
        # Define difficulty embeddings
        beginner_concepts = "basic fundamentals introduction getting started tutorial beginner easy simple"
        intermediate_concepts = "intermediate practical application development implementation building"
        advanced_concepts = "advanced expert mastery architecture optimization scalability production enterprise"
        
        beginner_emb = self.model.encode(beginner_concepts, convert_to_numpy=True)
        intermediate_emb = self.model.encode(intermediate_concepts, convert_to_numpy=True)
        advanced_emb = self.model.encode(advanced_concepts, convert_to_numpy=True)
        
        for skill in skills:
            skill_emb = self.model.encode(skill, convert_to_numpy=True)
            
            # Calculate similarity to each difficulty level
            beginner_sim = float(np.dot(skill_emb, beginner_emb) / 
                               (np.linalg.norm(skill_emb) * np.linalg.norm(beginner_emb)))
            intermediate_sim = float(np.dot(skill_emb, intermediate_emb) / 
                                   (np.linalg.norm(skill_emb) * np.linalg.norm(intermediate_emb)))
            advanced_sim = float(np.dot(skill_emb, advanced_emb) / 
                               (np.linalg.norm(skill_emb) * np.linalg.norm(advanced_emb)))
            
            # Also consider if user has related skills (makes it easier)
            has_related = any(
                float(np.dot(skill_emb, self.model.encode(existing, convert_to_numpy=True)) / 
                     (np.linalg.norm(skill_emb) * np.linalg.norm(self.model.encode(existing, convert_to_numpy=True)))) > 0.6
                for existing in existing_skills
            )
            
            # Classify based on ML similarity
            if beginner_sim > max(intermediate_sim, advanced_sim) or has_related:
                clusters['beginner'].append(skill)
            elif advanced_sim > intermediate_sim and not has_related:
                clusters['advanced'].append(skill)
            else:
                clusters['intermediate'].append(skill)
        
        return clusters
    
    def _build_phase_resources(
        self,
        skills: List[str],
        phase_label: str,
        target_role: Optional[str]
    ) -> List[Dict[str, str]]:
        """Generate personalized resource placeholders based on skills and target role."""
        if not skills:
            return []

        try:
            catalog_resources = self.course_catalog.fetch_resources_for_skills(
                skills,
                target_role=target_role,
                overall_limit=5
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(" Course catalog lookup failed: %s", exc)
            catalog_resources = []

        if not catalog_resources:
            logger.info(" No live catalog resources for skills: %s", skills)
            return []

        sanitized: List[Dict[str, Any]] = []
        for resource in catalog_resources:
            sanitized.append({
                "title": str(resource.get("title", "")).strip(),
                "provider": str(resource.get("provider", "")).strip(),
                "url": str(resource.get("url", "")).strip(),
                "duration": str(resource.get("duration", "self-paced")),
                "difficulty": str(resource.get("difficulty", "mixed")),
                "cost": str(resource.get("cost", "varies")),
                "skill_focus": str(resource.get("skill_focus", "")),
                "source": str(resource.get("source", "catalog")),
            })

        return sanitized
    
    def _extract_contextual_skills(
        self,
        projects: List[Any],
        work_history: List[Any],
        certifications: List[Any],
        education: List[Any],
        soft_skills: List[str],
        languages: List[str],
        cv_text: str
    ) -> List[str]:
        """Derive additional skill signals from full CV context."""
        context: List[str] = []

        def _collect_from_iterable(items: List[Any], keys: List[str]):
            for item in items:
                if isinstance(item, dict):
                    for key in keys:
                        value = item.get(key)
                        if isinstance(value, list):
                            context.extend(str(v) for v in value if v)
                        elif value:
                            context.append(str(value))
                elif isinstance(item, str):
                    context.append(item)

        _collect_from_iterable(projects, ['technologies', 'skills', 'tech_stack', 'description', 'summary'])
        _collect_from_iterable(work_history, ['title', 'position', 'skills', 'tech_stack', 'responsibilities', 'summary'])
        _collect_from_iterable(certifications, ['name', 'provider', 'skills'])
        _collect_from_iterable(education, ['degree', 'institution', 'field'])
        context.extend(soft_skills)
        context.extend(languages)

        if cv_text:
            context.append(cv_text)
        
        return [entry for entry in context if entry]
    
    def _predict_learning_success(self, skills_to_learn: List[str], existing_skills: List[str]) -> float:
        """
        ML: Predict probability of successfully learning these skills
        Based on similarity to existing skills (transfer learning)
        """
        if not skills_to_learn:
            return 1.0
        
        total_similarity = 0.0
        for new_skill in skills_to_learn:
            new_emb = self.model.encode(new_skill, convert_to_numpy=True)
            
            # Find max similarity to any existing skill
            max_sim = 0.0
            for existing in existing_skills:
                existing_emb = self.model.encode(existing, convert_to_numpy=True)
                sim = float(np.dot(new_emb, existing_emb) / 
                          (np.linalg.norm(new_emb) * np.linalg.norm(existing_emb)))
                max_sim = max(max_sim, sim)
            
            total_similarity += max_sim
        
        avg_similarity = total_similarity / len(skills_to_learn)
        
        # Convert similarity to success probability (0.6 to 0.95 range)
        success_prob = 0.6 + (avg_similarity * 0.35)
        
        return min(success_prob, 0.95)
    
    def _calculate_personalization_score(self, current_skills: List[str], 
                                        target_skills: List[str], 
                                        experience_years: int) -> float:
        """
        ML: Calculate how personalized this roadmap is
        """
        # Factor 1: Skill gap coverage
        gap_coverage = len([s for s in target_skills if s not in current_skills]) / max(len(target_skills), 1)
        
        # Factor 2: Experience-appropriate difficulty
        exp_factor = min(experience_years / 5.0, 1.0)
        
        # Factor 3: Semantic coherence (skills are related)
        if len(target_skills) > 1:
            target_embs = [self.model.encode(s, convert_to_numpy=True) for s in target_skills]
            coherence = np.mean([
                float(np.dot(target_embs[i], target_embs[j]) / 
                     (np.linalg.norm(target_embs[i]) * np.linalg.norm(target_embs[j])))
                for i in range(len(target_embs))
                for j in range(i+1, len(target_embs))
            ])
        else:
            coherence = 1.0
        
        personalization = (gap_coverage * 0.4 + exp_factor * 0.3 + coherence * 0.3)
        
        return min(personalization, 0.99)
