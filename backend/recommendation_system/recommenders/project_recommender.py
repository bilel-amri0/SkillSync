"""Recommandeur spécialisé pour les projets pratiques"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models import (
    UserFeatures, ProjectRecommendation, PracticalProject,
    DifficultyLevel, RecommendationType
)

logger = logging.getLogger(__name__)


class ProjectRecommender:
    """
    Recommandeur spécialisé pour les projets pratiques de développement de compétences
    """
    
    def __init__(self):
        self.projects_db = self._initialize_projects_database()
        logger.info("ProjectRecommender initialized")
    
    async def recommend(
        self, 
        user_features: UserFeatures, 
        preferences: Dict[str, Any] = None
    ) -> List[ProjectRecommendation]:
        """
        Génère des recommandations de projets pratiques
        """
        try:
            if preferences is None:
                preferences = {}
            
            # Filtrage des projets
            candidate_projects = self._filter_projects(user_features, preferences)
            
            # Scoring et ranking
            scored_projects = []
            for project in candidate_projects:
                score = self._score_project(project, user_features)
                learning_value = self._calculate_learning_value(project, user_features)
                portfolio_impact = self._calculate_portfolio_impact(project, user_features)
                skill_development = self._identify_skill_development(project, user_features)
                
                recommendation = ProjectRecommendation(
                    id=f"project_rec_{project.id}",
                    recommendation_type=RecommendationType.PROJECT,
                    title=project.title,
                    description=project.description,
                    scores={'base_score': score},
                    explanation={'learning_benefit': self._generate_learning_explanation(project, user_features)},
                    confidence=min(score + 0.1, 1.0),
                    created_at=datetime.now(),
                    project=project,
                    learning_value=learning_value,
                    portfolio_impact=portfolio_impact,
                    skill_development=skill_development
                )
                
                scored_projects.append(recommendation)
            
            # Tri et limitation
            scored_projects.sort(key=lambda x: x.scores['base_score'], reverse=True)
            return scored_projects[:6]  # Top 6 projets
            
        except Exception as e:
            logger.error(f"Error generating project recommendations: {e}")
            return []
    
    def _filter_projects(self, user_features: UserFeatures, preferences: Dict[str, Any]) -> List[PracticalProject]:
        """Filtre les projets selon les critères utilisateur"""
        filtered = []
        user_skills = [s.lower() for s in user_features.career_features.get('skills', [])]
        experience_level = user_features.experience_features.get('level_numeric', 0.5)
        
        for project in self.projects_db:
            # Filtre par niveau de difficulté approprié
            if self._is_difficulty_appropriate(project, experience_level):
                # Filtre par compétences prérequises
                if self._has_required_skills(project, user_skills):
                    # Filtre par pertinence
                    if self._is_relevant_for_user(project, user_features):
                        filtered.append(project)
        
        return filtered
    
    def _score_project(self, project: PracticalProject, user_features: UserFeatures) -> float:
        """Calcule le score d'un projet"""
        score = 0.0
        
        # Valeur d'apprentissage (30%)
        learning_score = self._calculate_learning_score(project, user_features)
        score += learning_score * 0.3
        
        # Impact portfolio (25%)
        portfolio_score = project.portfolio_value
        score += portfolio_score * 0.25
        
        # Pertinence industrie (20%)
        industry_relevance = self._calculate_industry_relevance(project, user_features)
        score += industry_relevance * 0.2
        
        # Faisabilité technique (15%)
        feasibility = self._calculate_technical_feasibility(project, user_features)
        score += feasibility * 0.15
        
        # Innovation/Modernité (10%)
        innovation_score = self._calculate_innovation_score(project)
        score += innovation_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_learning_score(self, project: PracticalProject, user_features: UserFeatures) -> float:
        """Calcule la valeur d'apprentissage"""
        user_skills = [s.lower() for s in user_features.career_features.get('skills', [])]
        
        # Compétences nouvelles apprises
        new_skills = [s for s in project.skills_gained if s.lower() not in user_skills]
        new_skills_ratio = len(new_skills) / max(len(project.skills_gained), 1)
        
        # Compétences pratiquées (renforcement)
        practiced_skills = [s for s in project.skills_practiced if s.lower() in user_skills]
        practice_ratio = len(practiced_skills) / max(len(project.skills_practiced), 1)
        
        # Score combiné
        learning_score = (new_skills_ratio * 0.7 + practice_ratio * 0.3)
        return min(learning_score, 1.0)
    
    def _calculate_portfolio_impact(self, project: PracticalProject, user_features: UserFeatures) -> float:
        """Calcule l'impact sur le portfolio"""
        base_impact = project.portfolio_value
        
        # Bonus pour l'alignement avec les objectifs de carrière
        career_goals = user_features.career_features.get('goals', [])
        if any(goal.lower() in project.domain.lower() for goal in career_goals):
            base_impact += 0.2
        
        # Bonus pour les technologies modernes
        modern_techs = ['react', 'docker', 'kubernetes', 'aws', 'machine learning', 'ai']
        if any(tech in tech_used.lower() for tech_used in project.technologies for tech in modern_techs):
            base_impact += 0.15
        
        return min(base_impact, 1.0)
    
    def _calculate_industry_relevance(self, project: PracticalProject, user_features: UserFeatures) -> float:
        """Calcule la pertinence pour l'industrie utilisateur"""
        user_industry = user_features.experience_features.get('industry_score', 0.6)
        
        # Pertinence par défaut
        relevance = 0.6
        
        # Bonus si aligné avec l'industrie
        if project.industry_relevance:
            # Simulation - à améliorer avec vraie logique métier
            relevance = user_industry * 0.8 + 0.2
        
        return min(relevance, 1.0)
    
    def _calculate_technical_feasibility(self, project: PracticalProject, user_features: UserFeatures) -> float:
        """Calcule la faisabilité technique"""
        user_skills = [s.lower() for s in user_features.career_features.get('skills', [])]
        
        # Vérifie les technologies requises
        required_techs = project.technologies + project.frameworks + project.tools
        known_techs = sum(1 for tech in required_techs if any(tech.lower() in skill for skill in user_skills))
        
        if required_techs:
            feasibility = known_techs / len(required_techs)
        else:
            feasibility = 0.8  # Par défaut si pas de technologies spécifiées
        
        return min(feasibility + 0.2, 1.0)  # Bonus pour l'apprentissage
    
    def _calculate_innovation_score(self, project: PracticalProject) -> float:
        """Calcule le score d'innovation"""
        innovation_keywords = [
            'ai', 'machine learning', 'blockchain', 'iot', 'cloud native',
            'microservices', 'serverless', 'edge computing', 'quantum'
        ]
        
        # Vérifie les mots-clés d'innovation
        project_text = (project.title + " " + project.description).lower()
        innovation_matches = sum(1 for keyword in innovation_keywords if keyword in project_text)
        
        return min(innovation_matches * 0.3, 1.0)
    
    def _calculate_learning_value(self, project: PracticalProject, user_features: UserFeatures) -> float:
        """Calcule la valeur d'apprentissage détaillée"""
        learning_score = self._calculate_learning_score(project, user_features)
        
        # Bonus pour la complexité appropriée
        complexity_bonus = min(project.complexity_score * 0.3, 0.2)
        
        return min(learning_score + complexity_bonus, 1.0)
    
    def _identify_skill_development(self, project: PracticalProject, user_features: UserFeatures) -> List[str]:
        """Identifie les compétences développées"""
        skills_developed = []
        
        # Compétences gagnées
        for skill in project.skills_gained:
            skills_developed.append(f"Nouvelle compétence: {skill}")
        
        # Compétences renforcées
        user_skills = [s.lower() for s in user_features.career_features.get('skills', [])]
        for skill in project.skills_practiced:
            if skill.lower() in [us.lower() for us in user_skills]:
                skills_developed.append(f"Renforcement: {skill}")
        
        # Technologies apprises
        for tech in project.technologies[:2]:  # Top 2 technologies
            skills_developed.append(f"Maîtrise technologique: {tech}")
        
        return skills_developed[:5]  # Limite à 5 éléments
    
    def _generate_learning_explanation(self, project: PracticalProject, user_features: UserFeatures) -> str:
        """Génère l'explication des bénéfices d'apprentissage"""
        explanations = []
        
        # Nouvelles compétences
        user_skills = [s.lower() for s in user_features.career_features.get('skills', [])]
        new_skills = [s for s in project.skills_gained if s.lower() not in user_skills]
        if new_skills:
            explanations.append(f"Apprendre {len(new_skills)} nouvelles compétences")
        
        # Technologies modernes
        modern_techs = [t for t in project.technologies if t.lower() in ['react', 'docker', 'aws', 'kubernetes']]
        if modern_techs:
            explanations.append(f"Maîtriser {modern_techs[0]}")
        
        # Impact portfolio
        if project.portfolio_value > 0.7:
            explanations.append("Excellent ajout au portfolio")
        
        if not explanations:
            explanations.append("Développement pratique des compétences")
        
        return "; ".join(explanations)
    
    def _is_difficulty_appropriate(self, project: PracticalProject, experience_level: float) -> bool:
        """Vérifie si la difficulté est appropriée"""
        difficulty_ranges = {
            DifficultyLevel.BEGINNER: (0.0, 0.4),
            DifficultyLevel.INTERMEDIATE: (0.2, 0.7),
            DifficultyLevel.ADVANCED: (0.5, 0.9),
            DifficultyLevel.EXPERT: (0.7, 1.0)
        }
        
        if project.difficulty in difficulty_ranges:
            min_level, max_level = difficulty_ranges[project.difficulty]
            return min_level <= experience_level <= max_level
        
        return True
    
    def _has_required_skills(self, project: PracticalProject, user_skills: List[str]) -> bool:
        """Vérifie si l'utilisateur a les compétences de base requises"""
        # Au moins 40% des compétences pratiquées doivent être connues
        if project.skills_practiced:
            known_skills = sum(1 for skill in project.skills_practiced 
                             if any(skill.lower() in user_skill for user_skill in user_skills))
            return known_skills >= len(project.skills_practiced) * 0.4
        
        return True
    
    def _is_relevant_for_user(self, project: PracticalProject, user_features: UserFeatures) -> bool:
        """Vérifie la pertinence générale du projet"""
        # Vérifie l'alignement avec les objectifs de carrière
        career_goals = user_features.career_features.get('goals', [])
        if career_goals:
            return any(goal.lower() in project.domain.lower() or 
                      project.domain.lower() in goal.lower() 
                      for goal in career_goals)
        
        return True  # Par défaut, considérer comme pertinent
    
    def _initialize_projects_database(self) -> List[PracticalProject]:
        """Initialise la base de projets pratiques"""
        projects = []
        
        # Projet 1: E-commerce Full-Stack
        projects.append(PracticalProject(
            id="ecommerce_fullstack",
            title="Plateforme E-commerce Full-Stack",
            description="Développement d'une plateforme e-commerce complète avec panier, paiement et gestion admin",
            domain="web_development",
            difficulty=DifficultyLevel.INTERMEDIATE,
            technologies=["React", "Node.js", "PostgreSQL", "Stripe API"],
            frameworks=["Express.js", "React Router", "Material-UI"],
            tools=["Git", "Docker", "Jest"],
            skills_practiced=["JavaScript", "React", "Node.js", "Database Design"],
            skills_gained=["E-commerce Architecture", "Payment Integration", "State Management"],
            learning_objectives=[
                "Maîtriser l'architecture full-stack",
                "Intégrer des APIs de paiement",
                "Gérer l'état complexe d'une application"
            ],
            estimated_time="8-12 semaines",
            complexity_score=0.7,
            portfolio_value=0.9,
            industry_relevance=["E-commerce", "Retail", "Startups"],
            tutorial_links=["https://example.com/ecommerce-tutorial"],
            github_templates=["https://github.com/example/ecommerce-template"]
        ))
        
        # Projet 2: Système de Recommandation ML
        projects.append(PracticalProject(
            id="ml_recommendation_system",
            title="Système de Recommandation avec ML",
            description="Construction d'un système de recommandation utilisant des algorithmes de machine learning",
            domain="data_science",
            difficulty=DifficultyLevel.ADVANCED,
            technologies=["Python", "Scikit-learn", "Pandas", "Flask", "Redis"],
            frameworks=["TensorFlow", "Flask", "NumPy"],
            tools=["Jupyter", "Docker", "MLflow"],
            skills_practiced=["Python", "Machine Learning", "Data Analysis"],
            skills_gained=["Recommendation Algorithms", "Model Deployment", "Real-time ML"],
            learning_objectives=[
                "Implémenter des algorithmes de recommandation",
                "Déployer des modèles ML en production",
                "Optimiser les performances en temps réel"
            ],
            estimated_time="10-14 semaines",
            complexity_score=0.8,
            portfolio_value=0.95,
            industry_relevance=["Tech", "Media", "E-commerce"],
            tutorial_links=["https://example.com/ml-recommendation"],
            github_templates=["https://github.com/example/recommendation-system"]
        ))
        
        # Projet 3: Infrastructure Cloud avec Kubernetes
        projects.append(PracticalProject(
            id="cloud_kubernetes_infrastructure",
            title="Infrastructure Cloud avec Kubernetes",
            description="Déploiement d'une infrastructure scalable sur AWS avec Kubernetes et CI/CD",
            domain="devops",
            difficulty=DifficultyLevel.ADVANCED,
            technologies=["Kubernetes", "Docker", "AWS", "Terraform"],
            frameworks=["Helm", "Istio"],
            tools=["Jenkins", "Prometheus", "Grafana", "ArgoCD"],
            skills_practiced=["Docker", "AWS", "Linux"],
            skills_gained=["Kubernetes Orchestration", "Infrastructure as Code", "Monitoring"],
            learning_objectives=[
                "Maîtriser l'orchestration de containers",
                "Automatiser le déploiement avec CI/CD",
                "Implémenter la surveillance et l'observabilité"
            ],
            estimated_time="12-16 semaines",
            complexity_score=0.9,
            portfolio_value=0.85,
            industry_relevance=["Tech", "Finance", "Healthcare"],
            tutorial_links=["https://example.com/k8s-infrastructure"],
            github_templates=["https://github.com/example/k8s-setup"]
        ))
        
        # Projet 4: Application Mobile React Native
        projects.append(PracticalProject(
            id="react_native_mobile_app",
            title="App Mobile avec React Native",
            description="Développement d'une application mobile cross-platform avec fonctionnalités natives",
            domain="mobile_development",
            difficulty=DifficultyLevel.INTERMEDIATE,
            technologies=["React Native", "TypeScript", "Firebase", "Redux"],
            frameworks=["Expo", "React Navigation"],
            tools=["Xcode", "Android Studio", "Flipper"],
            skills_practiced=["JavaScript", "React", "Mobile UI/UX"],
            skills_gained=["Mobile Development", "Cross-platform", "Native Integration"],
            learning_objectives=[
                "Développer des apps cross-platform",
                "Intégrer des fonctionnalités natives",
                "Gérer l'état avec Redux"
            ],
            estimated_time="6-10 semaines",
            complexity_score=0.6,
            portfolio_value=0.8,
            industry_relevance=["Mobile", "Startups", "Consumer Apps"],
            tutorial_links=["https://example.com/react-native-tutorial"],
            github_templates=["https://github.com/example/rn-template"]
        ))
        
        # Projet 5: Blockchain DApp
        projects.append(PracticalProject(
            id="blockchain_dapp",
            title="Application Décentralisée (DApp)",
            description="Développement d'une DApp sur Ethereum avec smart contracts et interface web",
            domain="blockchain",
            difficulty=DifficultyLevel.EXPERT,
            technologies=["Solidity", "Web3.js", "React", "Ethereum"],
            frameworks=["Truffle", "Hardhat", "OpenZeppelin"],
            tools=["MetaMask", "Ganache", "IPFS"],
            skills_practiced=["JavaScript", "React", "Cryptography"],
            skills_gained=["Blockchain Development", "Smart Contracts", "DeFi Concepts"],
            learning_objectives=[
                "Développer des smart contracts sécurisés",
                "Créer des interfaces Web3",
                "Comprendre l'écosystème DeFi"
            ],
            estimated_time="14-18 semaines",
            complexity_score=0.95,
            portfolio_value=0.9,
            industry_relevance=["Fintech", "Crypto", "Innovation"],
            tutorial_links=["https://example.com/dapp-tutorial"],
            github_templates=["https://github.com/example/dapp-template"]
        ))
        
        # Projet 6: API REST avec Microservices
        projects.append(PracticalProject(
            id="microservices_api",
            title="Architecture Microservices",
            description="Développement d'une API REST avec architecture microservices et communication inter-services",
            domain="backend_development",
            difficulty=DifficultyLevel.INTERMEDIATE,
            technologies=["Node.js", "Docker", "MongoDB", "Redis"],
            frameworks=["Express.js", "Mongoose"],
            tools=["Docker Compose", "Postman", "Jest"],
            skills_practiced=["Node.js", "API Design", "Database"],
            skills_gained=["Microservices Architecture", "Service Communication", "API Gateway"],
            learning_objectives=[
                "Architecturer des microservices",
                "Implémenter la communication inter-services",
                "Gérer la persistence distribuée"
            ],
            estimated_time="8-12 semaines",
            complexity_score=0.7,
            portfolio_value=0.85,
            industry_relevance=["Tech", "Enterprise", "Startups"],
            tutorial_links=["https://example.com/microservices-tutorial"],
            github_templates=["https://github.com/example/microservices-template"]
        ))
        
        return projects
