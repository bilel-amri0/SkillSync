"""Recommandeur spécialisé pour les certifications"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models import (
    UserFeatures, CertificationRecommendation, Certification,
    DifficultyLevel, RecommendationType
)

logger = logging.getLogger(__name__)


class CertificationRecommender:
    """
    Recommandeur spécialisé pour les certifications professionnelles
    """
    
    def __init__(self):
        self.certifications_db = self._initialize_certifications_database()
        logger.info("CertificationRecommender initialized")
    
    async def recommend(
        self, 
        user_features: UserFeatures, 
        preferences: Dict[str, Any] = None
    ) -> List[CertificationRecommendation]:
        """
        Génère des recommandations de certifications
        """
        try:
            if preferences is None:
                preferences = {}
            
            # Filtrage des certifications
            candidate_certs = self._filter_certifications(user_features, preferences)
            
            # Scoring et ranking
            scored_certs = []
            for cert in candidate_certs:
                score = self._score_certification(cert, user_features)
                match_reason = self._generate_match_reason(cert, user_features)
                preparation_estimate = self._estimate_preparation_time(cert, user_features)
                roi_estimate = self._calculate_roi(cert, user_features)
                
                recommendation = CertificationRecommendation(
                    id=f"cert_rec_{cert.id}",
                    recommendation_type=RecommendationType.CERTIFICATION,
                    title=cert.title,
                    description=f"Certification {cert.provider} - {cert.domain}",
                    scores={'base_score': score},
                    explanation={'match_reason': match_reason},
                    confidence=min(score + 0.1, 1.0),
                    created_at=datetime.now(),
                    certification=cert,
                    match_reason=match_reason,
                    preparation_estimate=preparation_estimate,
                    roi_estimate=roi_estimate
                )
                
                scored_certs.append(recommendation)
            
            # Tri et limitation
            scored_certs.sort(key=lambda x: x.scores['base_score'], reverse=True)
            return scored_certs[:5]
            
        except Exception as e:
            logger.error(f"Error generating certification recommendations: {e}")
            return []
    
    def _filter_certifications(self, user_features: UserFeatures, preferences: Dict[str, Any]) -> List[Certification]:
        """Filtre les certifications selon les critères"""
        filtered = []
        user_skills = user_features.career_features.get('skills', [])
        budget_score = user_features.preference_features.get('budget_score', 0.5)
        
        for cert in self.certifications_db:
            # Filtre par prérequis
            if self._meets_prerequisites(cert, user_skills):
                # Filtre par budget
                if self._fits_budget(cert, budget_score):
                    filtered.append(cert)
        
        return filtered
    
    def _score_certification(self, cert: Certification, user_features: UserFeatures) -> float:
        """Calcule le score d'une certification"""
        score = 0.0
        
        # Valeur marché (30%)
        score += cert.market_value * 0.3
        
        # Impact salaire (25%)
        score += cert.salary_impact * 0.25
        
        # Reconnaissance industrie (20%)
        score += cert.industry_recognition * 0.2
        
        # Alignement compétences (15%)
        skill_alignment = self._calculate_skill_alignment(cert, user_features)
        score += skill_alignment * 0.15
        
        # Facilité d'obtention (10%)
        ease_score = cert.pass_rate / 100.0 if cert.pass_rate > 0 else 0.5
        score += ease_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_skill_alignment(self, cert: Certification, user_features: UserFeatures) -> float:
        """Calcule l'alignement avec les compétences utilisateur"""
        user_skills = user_features.career_features.get('skills', [])
        required_skills = cert.required_skills
        
        if not required_skills:
            return 0.7
        
        matching = set([s.lower() for s in user_skills]).intersection(
            set([s.lower() for s in required_skills])
        )
        return len(matching) / len(required_skills)
    
    def _meets_prerequisites(self, cert: Certification, user_skills: List[str]) -> bool:
        """Vérifie les prérequis"""
        if not cert.prerequisites:
            return True
        
        user_skills_lower = [s.lower() for s in user_skills]
        prerequisites_lower = [p.lower() for p in cert.prerequisites]
        
        # Au moins 50% des prérequis remplis
        matching = sum(1 for prereq in prerequisites_lower if any(prereq in skill for skill in user_skills_lower))
        return matching >= len(cert.prerequisites) * 0.5
    
    def _fits_budget(self, cert: Certification, budget_score: float) -> bool:
        """Vérifie l'adéquation avec le budget"""
        if budget_score > 0.8:  # Budget élevé
            return True
        elif budget_score > 0.5:  # Budget moyen
            return cert.cost < 1000
        else:  # Budget limité
            return cert.cost < 300
    
    def _generate_match_reason(self, cert: Certification, user_features: UserFeatures) -> str:
        """Génère la raison de correspondance"""
        reasons = []
        
        # Raison marché
        if cert.market_value > 0.8:
            reasons.append("Très demandée sur le marché")
        
        # Raison salaire
        if cert.salary_impact > 0.7:
            reasons.append(f"Impact salaire important (+{cert.salary_impact:.0%})")
        
        # Raison compétences
        user_skills = user_features.career_features.get('skills', [])
        matching_skills = set([s.lower() for s in user_skills]).intersection(
            set([s.lower() for s in cert.required_skills])
        )
        if matching_skills:
            reasons.append(f"Vous avez déjà {len(matching_skills)} compétences requises")
        
        return "; ".join(reasons) if reasons else "Certification recommandée pour votre profil"
    
    def _estimate_preparation_time(self, cert: Certification, user_features: UserFeatures) -> str:
        """Estime le temps de préparation"""
        base_duration = cert.duration_prep
        user_skills = user_features.career_features.get('skills', [])
        
        # Réduction si compétences existantes
        skill_coverage = self._calculate_skill_alignment(cert, user_features)
        if skill_coverage > 0.7:
            return f"Réduit: {base_duration} (vous avez déjà les bases)"
        elif skill_coverage > 0.4:
            return f"Standard: {base_duration}"
        else:
            return f"Étendu: {base_duration} + préparation supplémentaire"
    
    def _calculate_roi(self, cert: Certification, user_features: UserFeatures) -> Dict[str, Any]:
        """Calcule le ROI de la certification"""
        # Simulation du ROI
        experience_years = user_features.experience_features.get('years', 2)
        current_salary_estimate = 50000 + (experience_years * 5000)  # Estimation
        
        salary_increase = current_salary_estimate * cert.salary_impact
        annual_benefit = salary_increase
        investment = cert.cost
        
        roi_years = investment / annual_benefit if annual_benefit > 0 else 10
        annual_roi_percentage = (annual_benefit / investment * 100) if investment > 0 else 0
        
        return {
            'investment_cost': investment,
            'annual_salary_increase': annual_benefit,
            'roi_period_years': min(roi_years, 10),
            'annual_roi_percentage': annual_roi_percentage,
            'career_opportunities': cert.job_opportunities
        }
    
    def _initialize_certifications_database(self) -> List[Certification]:
        """Initialise la base de certifications"""
        certifications = []
        
        # AWS Certifications
        certifications.append(Certification(
            id="aws_cloud_practitioner",
            title="AWS Certified Cloud Practitioner",
            provider="Amazon Web Services",
            domain="cloud",
            level="foundational",
            cost=100,
            duration_prep="1-2 mois",
            exam_format="multiple_choice",
            validity_period="3 ans",
            required_skills=["Cloud basics", "AWS services"],
            skills_gained=["AWS Cloud", "Cloud architecture", "Security"],
            prerequisites=[],
            market_value=0.8,
            salary_impact=0.15,
            job_opportunities=5000,
            industry_recognition=0.9,
            pass_rate=85.0,
            difficulty_rating=3.0
        ))
        
        certifications.append(Certification(
            id="aws_solutions_architect",
            title="AWS Certified Solutions Architect - Associate",
            provider="Amazon Web Services",
            domain="cloud",
            level="associate",
            cost=150,
            duration_prep="2-3 mois",
            exam_format="multiple_choice",
            validity_period="3 ans",
            required_skills=["AWS", "Cloud architecture", "Networking"],
            skills_gained=["Solution design", "AWS advanced", "Architecture"],
            prerequisites=["AWS Cloud Practitioner"],
            market_value=0.9,
            salary_impact=0.25,
            job_opportunities=8000,
            industry_recognition=0.95,
            pass_rate=72.0,
            difficulty_rating=4.0
        ))
        
        # Google Cloud
        certifications.append(Certification(
            id="gcp_cloud_engineer",
            title="Google Cloud Professional Cloud Architect",
            provider="Google Cloud",
            domain="cloud",
            level="professional",
            cost=200,
            duration_prep="3-4 mois",
            exam_format="multiple_choice",
            validity_period="2 ans",
            required_skills=["GCP", "Cloud architecture", "Kubernetes"],
            skills_gained=["GCP advanced", "Microservices", "DevOps"],
            prerequisites=["Cloud experience"],
            market_value=0.85,
            salary_impact=0.22,
            job_opportunities=4000,
            industry_recognition=0.8,
            pass_rate=68.0,
            difficulty_rating=4.5
        ))
        
        # Data Science
        certifications.append(Certification(
            id="google_data_engineer",
            title="Google Cloud Professional Data Engineer",
            provider="Google Cloud",
            domain="data_science",
            level="professional",
            cost=200,
            duration_prep="3-4 mois",
            exam_format="multiple_choice",
            validity_period="2 ans",
            required_skills=["Python", "SQL", "BigQuery", "Machine Learning"],
            skills_gained=["Data pipeline", "BigQuery", "ML engineering"],
            prerequisites=["Data experience"],
            market_value=0.9,
            salary_impact=0.3,
            job_opportunities=6000,
            industry_recognition=0.85,
            pass_rate=65.0,
            difficulty_rating=4.2
        ))
        
        # Microsoft
        certifications.append(Certification(
            id="azure_fundamentals",
            title="Microsoft Azure Fundamentals",
            provider="Microsoft",
            domain="cloud",
            level="foundational",
            cost=99,
            duration_prep="1-2 mois",
            exam_format="multiple_choice",
            validity_period="Pas d'expiration",
            required_skills=["Cloud basics"],
            skills_gained=["Azure services", "Cloud concepts"],
            prerequisites=[],
            market_value=0.75,
            salary_impact=0.12,
            job_opportunities=4500,
            industry_recognition=0.85,
            pass_rate=88.0,
            difficulty_rating=2.5
        ))
        
        return certifications
