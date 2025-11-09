"""Recommandeur spécialisé pour les roadmaps de carrière"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models import (
    UserFeatures, RoadmapRecommendation, CareerRoadmap, RoadmapStep,
    DifficultyLevel, ExperienceLevel, RecommendationType
)

logger = logging.getLogger(__name__)


class RoadmapRecommender:
    """
    Recommandeur spécialisé pour les roadmaps de carrière personnalisés
    """
    
    def __init__(self):
        # Base de connaissances des roadmaps
        self.roadmaps_db = self._initialize_roadmaps_database()
        logger.info("Roadmap Recommender initialized")
    
    async def recommend(
        self, 
        user_features: UserFeatures, 
        preferences: Dict[str, Any] = None
    ) -> List[RoadmapRecommendation]:
        """
        Génère des recommandations de roadmaps personnalisées
        """
        try:
            if preferences is None:
                preferences = {}
            
            # Filtrage initial des roadmaps
            candidate_roadmaps = self._filter_roadmaps(user_features, preferences)
            
            # Scoring et ranking
            scored_roadmaps = []
            for roadmap in candidate_roadmaps:
                score = self._score_roadmap_fit(roadmap, user_features)
                match_reason = self._generate_match_reason(roadmap, user_features)
                progression_fit = self._calculate_progression_fit(roadmap, user_features)
                next_steps = self._generate_next_steps(roadmap, user_features)
                
                recommendation = RoadmapRecommendation(
                    id=f"roadmap_rec_{roadmap.id}",
                    recommendation_type=RecommendationType.ROADMAP,
                    title=roadmap.title,
                    description=f"Roadmap de carrière {roadmap.domain} - {roadmap.estimated_duration}",
                    scores={'base_score': score},
                    explanation={'match_reason': match_reason},
                    confidence=min(score + 0.1, 1.0),
                    created_at=datetime.now(),
                    roadmap=roadmap,
                    match_reason=match_reason,
                    progression_fit=progression_fit,
                    next_steps=next_steps
                )
                
                scored_roadmaps.append(recommendation)
            
            # Tri par score et limitation
            scored_roadmaps.sort(key=lambda x: x.scores['base_score'], reverse=True)
            return scored_roadmaps[:5]  # Top 5 recommandations
            
        except Exception as e:
            logger.error(f"Error generating roadmap recommendations: {e}")
            return []
    
    def _filter_roadmaps(self, user_features: UserFeatures, preferences: Dict[str, Any]) -> List[CareerRoadmap]:
        """
        Filtre les roadmaps selon les critères utilisateur
        """
        filtered = []
        
        user_experience = user_features.experience_features.get('years', 0)
        user_skills = user_features.career_features.get('skills', [])
        
        for roadmap in self.roadmaps_db:
            # Filtre par niveau d'expérience
            if self._is_experience_appropriate(roadmap, user_experience):
                # Filtre par prérequis
                if self._meets_prerequisites(roadmap, user_skills):
                    # Filtre par préférences de domaine
                    if self._matches_domain_preference(roadmap, preferences, user_features):
                        filtered.append(roadmap)
        
        return filtered
    
    def _score_roadmap_fit(self, roadmap: CareerRoadmap, user_features: UserFeatures) -> float:
        """
        Calcule le score d'adéquation d'un roadmap
        """
        score = 0.0
        
        # 1. Correspondance avec les compétences actuelles (30%)
        skill_match = self._calculate_skill_match(roadmap, user_features)
        score += skill_match * 0.3
        
        # 2. Alignement avec les objectifs de carrière (25%)
        career_alignment = self._calculate_career_alignment(roadmap, user_features)
        score += career_alignment * 0.25
        
        # 3. Niveau de difficulté approprié (20%)
        difficulty_fit = self._calculate_difficulty_fit(roadmap, user_features)
        score += difficulty_fit * 0.2
        
        # 4. Demande du marché (15%)
        market_demand = roadmap.market_demand
        score += market_demand * 0.15
        
        # 5. Impact potentiel sur le salaire (10%)
        salary_impact = roadmap.avg_salary_impact
        score += salary_impact * 0.1
        
        return min(score, 1.0)
    
    def _generate_match_reason(self, roadmap: CareerRoadmap, user_features: UserFeatures) -> str:
        """
        Génère une raison pour la correspondance du roadmap
        """
        user_skills = user_features.career_features.get('skills', [])
        career_goals = user_features.career_features.get('goals', [])
        
        reasons = []
        
        # Raison basée sur les compétences
        matching_skills = set(user_skills).intersection(set(roadmap.required_base_skills))
        if matching_skills:
            reasons.append(f"Vous maîtrisez déjà {len(matching_skills)} compétences clés")
        
        # Raison basée sur les objectifs
        if career_goals:
            matching_goals = [goal for goal in career_goals if goal.lower() in roadmap.domain.lower()]
            if matching_goals:
                reasons.append(f"Aligné avec votre objectif: {matching_goals[0]}")
        
        # Raison basée sur la progression
        user_level = user_features.experience_features.get('level_numeric', 0.4)
        if user_level < 0.5 and roadmap.difficulty == DifficultyLevel.BEGINNER:
            reasons.append("Parfait pour démarrer votre progression")
        elif user_level > 0.6 and roadmap.difficulty in [DifficultyLevel.INTERMEDIATE, DifficultyLevel.ADVANCED]:
            reasons.append("Idéal pour faire évoluer votre expertise")
        
        if not reasons:
            reasons.append(f"Excellente opportunité dans le domaine {roadmap.domain}")
        
        return "; ".join(reasons)
    
    def _calculate_progression_fit(self, roadmap: CareerRoadmap, user_features: UserFeatures) -> float:
        """
        Calcule l'adéquation avec la progression de carrière
        """
        user_experience = user_features.experience_features.get('years', 0)
        
        # Mapping expérience -> niveau approprié
        if user_experience < 2 and roadmap.level_progression[0] == ExperienceLevel.JUNIOR:
            return 0.9
        elif 2 <= user_experience < 5 and ExperienceLevel.MID in roadmap.level_progression:
            return 0.9
        elif user_experience >= 5 and ExperienceLevel.SENIOR in roadmap.level_progression:
            return 0.8
        else:
            return 0.6
    
    def _generate_next_steps(self, roadmap: CareerRoadmap, user_features: UserFeatures) -> List[str]:
        """
        Génère les prochaines étapes recommandées
        """
        if not roadmap.steps:
            return ["Commencer l'exploration du roadmap"]
        
        user_skills = user_features.career_features.get('skills', [])
        
        # Trouve la première étape où l'utilisateur n'a pas les compétences
        for step in roadmap.steps:
            step_skills = step.skills_to_learn
            if not any(skill in user_skills for skill in step_skills):
                return [
                    f"Étape {step.step_number}: {step.title}",
                    f"Focus: {', '.join(step_skills[:2])}",
                    f"Durée estimée: {step.duration}"
                ]
        
        # Si toutes les compétences de base sont acquises
        return [
            "Vous avez déjà les bases solides!",
            "Focus sur les étapes avancées du roadmap",
            "Considérer les certifications recommandées"
        ]
    
    # Méthodes utilitaires
    def _is_experience_appropriate(self, roadmap: CareerRoadmap, user_experience: int) -> bool:
        """Vérifie si le niveau d'expérience est approprié"""
        if roadmap.difficulty == DifficultyLevel.BEGINNER:
            return user_experience <= 3
        elif roadmap.difficulty == DifficultyLevel.INTERMEDIATE:
            return 1 <= user_experience <= 6
        elif roadmap.difficulty == DifficultyLevel.ADVANCED:
            return user_experience >= 3
        return True
    
    def _meets_prerequisites(self, roadmap: CareerRoadmap, user_skills: List[str]) -> bool:
        """Vérifie si les prérequis sont remplis"""
        if not roadmap.required_base_skills:
            return True
        
        # Au moins 30% des compétences de base requises
        matching = set(user_skills).intersection(set(roadmap.required_base_skills))
        return len(matching) >= len(roadmap.required_base_skills) * 0.3
    
    def _matches_domain_preference(self, roadmap: CareerRoadmap, preferences: Dict[str, Any], user_features: UserFeatures) -> bool:
        """Vérifie la correspondance avec les préférences de domaine"""
        preferred_domains = preferences.get('focus_areas', [])
        if not preferred_domains:
            return True
        
        return any(domain.lower() in roadmap.domain.lower() for domain in preferred_domains)
    
    def _calculate_skill_match(self, roadmap: CareerRoadmap, user_features: UserFeatures) -> float:
        """Calcule la correspondance des compétences"""
        user_skills = user_features.career_features.get('skills', [])
        required_skills = roadmap.required_base_skills
        
        if not required_skills:
            return 0.7  # Score neutre
        
        matching = set(user_skills).intersection(set(required_skills))
        return len(matching) / len(required_skills)
    
    def _calculate_career_alignment(self, roadmap: CareerRoadmap, user_features: UserFeatures) -> float:
        """Calcule l'alignement avec les objectifs de carrière"""
        career_goals = user_features.career_features.get('goals', [])
        
        if not career_goals:
            return 0.6  # Score par défaut
        
        # Correspondance avec les rôles cibles
        for goal in career_goals:
            for target_role in roadmap.target_roles:
                if goal.lower() in target_role.lower() or target_role.lower() in goal.lower():
                    return 0.9
        
        return 0.5
    
    def _calculate_difficulty_fit(self, roadmap: CareerRoadmap, user_features: UserFeatures) -> float:
        """Calcule l'adéquation de la difficulté"""
        user_level = user_features.experience_features.get('level_numeric', 0.4)
        
        if roadmap.difficulty == DifficultyLevel.BEGINNER:
            return 1.0 - abs(user_level - 0.2)
        elif roadmap.difficulty == DifficultyLevel.INTERMEDIATE:
            return 1.0 - abs(user_level - 0.5)
        elif roadmap.difficulty == DifficultyLevel.ADVANCED:
            return 1.0 - abs(user_level - 0.8)
        
        return 0.7
    
    def _initialize_roadmaps_database(self) -> List[CareerRoadmap]:
        """
        Initialise la base de données des roadmaps avec des exemples
        CORRECTION: Ajout du paramètre certifications manquant dans tous les RoadmapStep
        """
        roadmaps = []
        
        # Roadmap 1: Data Science pour débutants
        ds_steps = [
            RoadmapStep(
                step_number=1,
                title="Fondamentaux Python & Statistiques",
                description="Maîtrise des bases Python et concepts statistiques",
                duration="4-6 semaines",
                skills_to_learn=["Python", "Pandas", "NumPy", "Statistiques"],
                resources=[
                    {"type": "course", "name": "Python pour Data Science", "url": "#"},
                    {"type": "book", "name": "Statistiques pour les nuls", "url": "#"}
                ],
                projects=["ds_project_1"],
                certifications=["python_basics"]  # CORRECTION: paramètre ajouté
            ),
            RoadmapStep(
                step_number=2,
                title="Visualisation & Exploration de Données",
                description="Création de visualisations et analyse exploratoire",
                duration="3-4 semaines",
                skills_to_learn=["Matplotlib", "Seaborn", "Plotly", "EDA"],
                resources=[
                    {"type": "tutorial", "name": "Visualisation avec Python", "url": "#"}
                ],
                projects=["ds_project_2"],
                certifications=["data_visualization"],  # CORRECTION: paramètre ajouté
                prerequisites=["Python", "Pandas"]
            ),
            RoadmapStep(
                step_number=3,
                title="Machine Learning Fondamentaux",
                description="Algorithmes ML de base et évaluation",
                duration="6-8 semaines",
                skills_to_learn=["Scikit-learn", "Regression", "Classification", "Clustering"],
                resources=[
                    {"type": "course", "name": "Machine Learning A-Z", "url": "#"}
                ],
                projects=["ds_project_3"],
                certifications=["ml_basics"]  # CORRECTION: paramètre ajouté
            )
        ]
        
        roadmaps.append(CareerRoadmap(
            id="roadmap_data_science_beginner",
            title="Data Scientist - Parcours Débutant",
            domain="data_science",
            level_progression=[ExperienceLevel.JUNIOR, ExperienceLevel.MID],
            estimated_duration="4-6 mois",
            difficulty=DifficultyLevel.BEGINNER,
            steps=ds_steps,
            required_base_skills=["Mathématiques", "Logique"],
            target_roles=["Data Analyst", "Junior Data Scientist", "Business Analyst"],
            industry_relevance=["Tech", "Finance", "Healthcare", "E-commerce"],
            popularity_score=0.9,
            market_demand=0.85,
            avg_salary_impact=0.8
        ))
        
        # Roadmap 2: Full-Stack Web Development
        fullstack_steps = [
            RoadmapStep(
                step_number=1,
                title="Frontend Fundamentals",
                description="HTML, CSS, JavaScript et responsive design",
                duration="6-8 semaines",
                skills_to_learn=["HTML5", "CSS3", "JavaScript", "Responsive Design"],
                resources=[
                    {"type": "course", "name": "Web Development Bootcamp", "url": "#"}
                ],
                projects=["portfolio_website"],
                certifications=["frontend_basics"]  # CORRECTION: paramètre ajouté
            ),
            RoadmapStep(
                step_number=2,
                title="Framework Frontend Modern",
                description="React.js et écosystème moderne",
                duration="4-6 semaines",
                skills_to_learn=["React", "JSX", "State Management", "Hooks"],
                resources=[
                    {"type": "tutorial", "name": "React Complete Guide", "url": "#"}
                ],
                projects=["react_todo_app"],
                certifications=["react_fundamentals"],  # CORRECTION: paramètre ajouté
                prerequisites=["JavaScript"]
            ),
            RoadmapStep(
                step_number=3,
                title="Backend Development",
                description="Node.js, APIs et bases de données",
                duration="6-8 semaines",
                skills_to_learn=["Node.js", "Express", "MongoDB", "REST APIs"],
                resources=[
                    {"type": "course", "name": "Backend avec Node.js", "url": "#"}
                ],
                projects=["fullstack_ecommerce"],
                certifications=["backend_dev"]  # CORRECTION: paramètre ajouté
            )
        ]
        
        roadmaps.append(CareerRoadmap(
            id="roadmap_fullstack_web",
            title="Développeur Full-Stack Web",
            domain="web_development",
            level_progression=[ExperienceLevel.JUNIOR, ExperienceLevel.MID, ExperienceLevel.SENIOR],
            estimated_duration="5-8 mois",
            difficulty=DifficultyLevel.INTERMEDIATE,
            steps=fullstack_steps,
            required_base_skills=["Programmation", "Logique"],
            target_roles=["Full-Stack Developer", "Web Developer", "Frontend Developer"],
            industry_relevance=["Tech", "Startups", "E-commerce", "Media"],
            popularity_score=0.95,
            market_demand=0.9,
            avg_salary_impact=0.75
        ))
        
        # Roadmap 3: DevOps Engineer
        devops_steps = [
            RoadmapStep(
                step_number=1,
                title="Infrastructure & Linux",
                description="Administration système et cloud basics",
                duration="4-6 semaines",
                skills_to_learn=["Linux", "Bash", "Networking", "Cloud Basics"],
                resources=[
                    {"type": "course", "name": "Linux Administration", "url": "#"}
                ],
                projects=["linux_server_setup"],
                certifications=["linux_essentials"]  # CORRECTION: paramètre ajouté
            ),
            RoadmapStep(
                step_number=2,
                title="Containerisation & Orchestration",
                description="Docker et Kubernetes pour le déploiement",
                duration="5-7 semaines",
                skills_to_learn=["Docker", "Kubernetes", "Container Security"],
                resources=[
                    {"type": "hands-on", "name": "Docker & Kubernetes Lab", "url": "#"}
                ],
                projects=["microservices_deployment"],
                certifications=["docker_kubernetes"],  # CORRECTION: paramètre ajouté
                prerequisites=["Linux"]
            ),
            RoadmapStep(
                step_number=3,
                title="CI/CD & Automation",
                description="Pipelines de déploiement automatisé",
                duration="4-6 semaines",
                skills_to_learn=["Jenkins", "GitLab CI", "Terraform", "Ansible"],
                resources=[
                    {"type": "project", "name": "CI/CD Pipeline Setup", "url": "#"}
                ],
                projects=["automated_deployment"],
                certifications=["devops_pro"]  # CORRECTION: paramètre ajouté
            )
        ]
        
        roadmaps.append(CareerRoadmap(
            id="roadmap_devops_engineer",
            title="DevOps Engineer - Infrastructure Moderne",
            domain="devops",
            level_progression=[ExperienceLevel.MID, ExperienceLevel.SENIOR, ExperienceLevel.LEAD],
            estimated_duration="6-9 mois",
            difficulty=DifficultyLevel.ADVANCED,
            steps=devops_steps,
            required_base_skills=["Programmation", "Systèmes", "Réseaux"],
            target_roles=["DevOps Engineer", "Site Reliability Engineer", "Cloud Architect"],
            industry_relevance=["Tech", "Finance", "Healthcare", "Government"],
            popularity_score=0.85,
            market_demand=0.9,
            avg_salary_impact=0.85
        ))
        
        return roadmaps
