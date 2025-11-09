"""Recommandeur spécialisé pour les compétences"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models import (
    UserFeatures, SkillRecommendation, Skill,
    RecommendationType
)

logger = logging.getLogger(__name__)


class SkillsRecommender:
    """
    Recommandeur spécialisé pour les compétences à développer
    """
    
    def __init__(self):
        self.skills_db = self._initialize_skills_database()
        logger.info("SkillsRecommender initialized")
    
    async def recommend(
        self, 
        user_features: UserFeatures, 
        preferences: Dict[str, Any] = None
    ) -> List[SkillRecommendation]:
        """
        Génère des recommandations de compétences
        """
        try:
            if preferences is None:
                preferences = {}
            
            # Filtrage des compétences
            candidate_skills = self._filter_skills(user_features, preferences)
            
            # Scoring et ranking
            scored_skills = []
            for skill in candidate_skills:
                score = self._score_skill(skill, user_features)
                priority = self._determine_priority(skill, user_features)
                learning_path = self._generate_learning_path(skill, user_features)
                immediate_benefits = self._identify_immediate_benefits(skill, user_features)
                
                recommendation = SkillRecommendation(
                    id=f"skill_rec_{skill.id}",
                    recommendation_type=RecommendationType.SKILL,
                    title=skill.name,
                    description=f"Compétence {skill.category} - {skill.subcategory}",
                    scores={'importance_score': score},
                    explanation={'priority_reason': self._generate_priority_reason(skill, user_features)},
                    confidence=min(score + 0.15, 1.0),
                    created_at=datetime.now(),
                    skill=skill,
                    priority=priority,
                    learning_path=learning_path,
                    immediate_benefits=immediate_benefits
                )
                
                scored_skills.append(recommendation)
            
            # Tri et limitation
            scored_skills.sort(key=lambda x: x.scores['importance_score'], reverse=True)
            return scored_skills[:8]  # Top 8 compétences
            
        except Exception as e:
            logger.error(f"Error generating skill recommendations: {e}")
            return []
    
    def _filter_skills(self, user_features: UserFeatures, preferences: Dict[str, Any]) -> List[Skill]:
        """Filtre les compétences selon les critères"""
        filtered = []
        user_skills = [s.lower() for s in user_features.career_features.get('skills', [])]
        
        for skill in self.skills_db:
            # Exclure les compétences déjà maîtrisées
            if skill.name.lower() not in user_skills:
                # Vérifier les prérequis
                if self._meets_prerequisites(skill, user_skills):
                    filtered.append(skill)
        
        return filtered
    
    def _score_skill(self, skill: Skill, user_features: UserFeatures) -> float:
        """Calcule le score d'importance d'une compétence"""
        score = 0.0
        
        # Demande du marché (35%)
        score += skill.market_demand * 0.35
        
        # Impact salaire (30%)
        score += skill.average_salary_impact * 0.30
        
        # Pertinence future (20%)
        score += skill.future_relevance * 0.20
        
        # Facilité d'apprentissage (15%)
        learning_ease = 1.0 - skill.difficulty_to_learn
        score += learning_ease * 0.15
        
        # Bonus pour complémentarité avec compétences existantes
        complementarity_bonus = self._calculate_complementarity(skill, user_features)
        score += complementarity_bonus * 0.1
        
        return min(score, 1.0)
    
    def _calculate_complementarity(self, skill: Skill, user_features: UserFeatures) -> float:
        """Calcule la complémentarité avec les compétences existantes"""
        user_skills = [s.lower() for s in user_features.career_features.get('skills', [])]
        complementary_skills = [s.lower() for s in skill.complementary_skills]
        
        # Compte les compétences complémentaires déjà possédées
        matching = sum(1 for comp_skill in complementary_skills if any(comp_skill in user_skill for user_skill in user_skills))
        
        if complementary_skills:
            return matching / len(complementary_skills)
        return 0.0
    
    def _meets_prerequisites(self, skill: Skill, user_skills: List[str]) -> bool:
        """Vérifie si les prérequis sont remplis"""
        if not skill.prerequisites:
            return True
        
        # Vérifie que l'utilisateur a au moins 70% des prérequis
        prerequisites_lower = [p.lower() for p in skill.prerequisites]
        matching = sum(1 for prereq in prerequisites_lower if any(prereq in user_skill for user_skill in user_skills))
        
        return matching >= len(skill.prerequisites) * 0.7
    
    def _determine_priority(self, skill: Skill, user_features: UserFeatures) -> str:
        """Détermine la priorité d'apprentissage"""
        score = self._score_skill(skill, user_features)
        
        if score > 0.8:
            return "high"
        elif score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_learning_path(self, skill: Skill, user_features: UserFeatures) -> List[str]:
        """Génère un chemin d'apprentissage"""
        path = []
        
        # Étape 1: Fondamentaux
        path.append(f"1. Apprendre les bases de {skill.name}")
        
        # Étape 2: Pratique
        if skill.practical_projects:
            path.append(f"2. Projet pratique: {skill.practical_projects[0]}")
        else:
            path.append("2. Exercices pratiques et mini-projets")
        
        # Étape 3: Approfondissement
        if skill.complementary_skills:
            complementary = skill.complementary_skills[0]
            path.append(f"3. Maîtriser {complementary} pour compléter")
        else:
            path.append(f"3. Approfondir les concepts avancés de {skill.name}")
        
        # Étape 4: Certification ou validation
        if skill.certifications:
            path.append(f"4. Obtenir la certification: {skill.certifications[0]}")
        else:
            path.append("4. Valider avec un projet complexe")
        
        return path
    
    def _identify_immediate_benefits(self, skill: Skill, user_features: UserFeatures) -> List[str]:
        """Identifie les bénéfices immédiats"""
        benefits = []
        
        # Bénéfice salaire
        if skill.average_salary_impact > 0.15:
            benefits.append(f"Augmentation salariale potentielle: +{skill.average_salary_impact:.0%}")
        
        # Bénéfice marché
        if skill.market_demand > 0.8:
            benefits.append("Compétence très demandée sur le marché")
        
        # Bénéfice carrière
        user_goals = user_features.career_features.get('goals', [])
        if any(goal.lower() in path.lower() for goal in user_goals for path in skill.career_paths):
            benefits.append("Directement aligné avec vos objectifs de carrière")
        
        # Bénéfice futur
        if skill.future_relevance > 0.8:
            benefits.append("Compétence d'avenir avec forte croissance")
        
        if not benefits:
            benefits.append("Développement professionnel et élargissement des compétences")
        
        return benefits
    
    def _generate_priority_reason(self, skill: Skill, user_features: UserFeatures) -> str:
        """Génère la raison de priorité"""
        reasons = []
        
        if skill.market_demand > 0.8:
            reasons.append("forte demande marché")
        
        if skill.average_salary_impact > 0.2:
            reasons.append("impact salaire important")
        
        if skill.future_relevance > 0.8:
            reasons.append("tendance future")
        
        complementarity = self._calculate_complementarity(skill, user_features)
        if complementarity > 0.5:
            reasons.append("complémente vos compétences")
        
        if not reasons:
            reasons.append("développement professionnel")
        
        return "Prioritaire pour: " + ", ".join(reasons)
    
    def _initialize_skills_database(self) -> List[Skill]:
        """Initialise la base de compétences"""
        skills = []
        
        # Compétences techniques - Programmation
        skills.append(Skill(
            id="python_advanced",
            name="Python Avancé",
            category="technical",
            subcategory="programming",
            difficulty_to_learn=0.4,
            market_demand=0.9,
            future_relevance=0.95,
            average_salary_impact=0.25,
            prerequisites=["Python"],
            complementary_skills=["Data Science", "Web Development", "Machine Learning"],
            career_paths=["Data Scientist", "Backend Developer", "Full-Stack Developer"],
            learning_resources=[{"type": "course", "name": "Advanced Python", "url": "#"}],
            estimated_learning_time="6-8 semaines",
            practical_projects=["API complexe", "Automatisation"],
            certifications=["Python Institute PCEP"]
        ))
        
        skills.append(Skill(
            id="react_advanced",
            name="React.js Avancé",
            category="technical",
            subcategory="frontend",
            difficulty_to_learn=0.5,
            market_demand=0.85,
            future_relevance=0.8,
            average_salary_impact=0.22,
            prerequisites=["JavaScript", "HTML", "CSS"],
            complementary_skills=["Node.js", "TypeScript", "Redux"],
            career_paths=["Frontend Developer", "Full-Stack Developer"],
            learning_resources=[{"type": "course", "name": "React Mastery", "url": "#"}],
            estimated_learning_time="4-6 semaines",
            practical_projects=["SPA complexe", "E-commerce frontend"],
            certifications=["React Developer Certification"]
        ))
        
        # Compétences Cloud
        skills.append(Skill(
            id="aws_architecture",
            name="Architecture AWS",
            category="technical",
            subcategory="cloud",
            difficulty_to_learn=0.6,
            market_demand=0.9,
            future_relevance=0.9,
            average_salary_impact=0.3,
            prerequisites=["Cloud basics", "Networking"],
            complementary_skills=["DevOps", "Kubernetes", "Terraform"],
            career_paths=["Cloud Architect", "DevOps Engineer", "Solutions Architect"],
            learning_resources=[{"type": "course", "name": "AWS Architecture", "url": "#"}],
            estimated_learning_time="8-12 semaines",
            practical_projects=["Infrastructure as Code", "Multi-tier architecture"],
            certifications=["AWS Solutions Architect"]
        ))
        
        skills.append(Skill(
            id="kubernetes",
            name="Kubernetes",
            category="technical",
            subcategory="devops",
            difficulty_to_learn=0.7,
            market_demand=0.85,
            future_relevance=0.9,
            average_salary_impact=0.28,
            prerequisites=["Docker", "Linux", "Networking"],
            complementary_skills=["AWS", "Terraform", "Monitoring"],
            career_paths=["DevOps Engineer", "Platform Engineer", "SRE"],
            learning_resources=[{"type": "hands-on", "name": "Kubernetes Lab", "url": "#"}],
            estimated_learning_time="10-14 semaines",
            practical_projects=["Microservices deployment", "Auto-scaling setup"],
            certifications=["CKA", "CKAD"]
        ))
        
        # Compétences Data Science
        skills.append(Skill(
            id="machine_learning",
            name="Machine Learning",
            category="technical",
            subcategory="data_science",
            difficulty_to_learn=0.6,
            market_demand=0.95,
            future_relevance=0.95,
            average_salary_impact=0.35,
            prerequisites=["Python", "Statistics", "Mathematics"],
            complementary_skills=["Deep Learning", "Data Engineering", "MLOps"],
            career_paths=["Data Scientist", "ML Engineer", "AI Specialist"],
            learning_resources=[{"type": "course", "name": "ML Specialization", "url": "#"}],
            estimated_learning_time="12-16 semaines",
            practical_projects=["Predictive model", "Recommendation system"],
            certifications=["Google ML Engineer", "AWS ML Specialty"]
        ))
        
        skills.append(Skill(
            id="data_engineering",
            name="Data Engineering",
            category="technical",
            subcategory="data_science",
            difficulty_to_learn=0.7,
            market_demand=0.9,
            future_relevance=0.9,
            average_salary_impact=0.32,
            prerequisites=["SQL", "Python", "Database systems"],
            complementary_skills=["Apache Spark", "Kafka", "Airflow"],
            career_paths=["Data Engineer", "Analytics Engineer", "Platform Engineer"],
            learning_resources=[{"type": "project", "name": "Data Pipeline Project", "url": "#"}],
            estimated_learning_time="10-14 semaines",
            practical_projects=["ETL pipeline", "Real-time data processing"],
            certifications=["Google Data Engineer", "Databricks"]
        ))
        
        # Compétences Soft Skills
        skills.append(Skill(
            id="technical_leadership",
            name="Leadership Technique",
            category="soft",
            subcategory="leadership",
            difficulty_to_learn=0.8,
            market_demand=0.8,
            future_relevance=0.85,
            average_salary_impact=0.4,
            prerequisites=["Experience technique", "Communication"],
            complementary_skills=["Project Management", "Mentoring", "Architecture"],
            career_paths=["Tech Lead", "Engineering Manager", "CTO"],
            learning_resources=[{"type": "book", "name": "Technical Leadership", "url": "#"}],
            estimated_learning_time="Continuous",
            practical_projects=["Team leadership", "Technical mentoring"],
            certifications=["Leadership certificates"]
        ))
        
        skills.append(Skill(
            id="system_design",
            name="System Design",
            category="technical",
            subcategory="architecture",
            difficulty_to_learn=0.8,
            market_demand=0.85,
            future_relevance=0.9,
            average_salary_impact=0.35,
            prerequisites=["Programming", "Database", "Networking"],
            complementary_skills=["Microservices", "Distributed Systems", "Scalability"],
            career_paths=["Senior Engineer", "Architect", "Tech Lead"],
            learning_resources=[{"type": "book", "name": "System Design Interview", "url": "#"}],
            estimated_learning_time="8-12 semaines",
            practical_projects=["Scalable system design", "Architecture review"],
            certifications=["Architecture certifications"]
        ))
        
        return skills
