#!/usr/bin/env python3
"""
Script de démonstration du système de recommandations multicritères SkillSync
Met en avant toutes les fonctionnalités du nouveau système étendu
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from enum import Enum

# Import du système de recommandations
from recommendation_system.models import (
    UserProfile, RecommendationPreferences, ExperienceLevel, DifficultyLevel
)
from recommendation_system.core.recommendation_orchestrator import RecommendationOrchestrator


def make_json_serializable(obj):
    """Convertit un objet en format JSON sérialisable"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def create_sample_user_profiles():
    """
    Crée des profils utilisateur d'exemple pour la démonstration
    """
    profiles = []
    
    # Profil 1: Développeur Junior Backend
    profiles.append(UserProfile(
        user_id="user_001_junior_backend",
        cv_data={
            "summary": "Développeur backend junior passionné par Python et les APIs",
            "experience_text": "Stage de 6 mois chez une startup, développement d'APIs REST"
        },
        current_skills=["Python", "Flask", "SQL", "Git", "REST APIs"],
        experience_years=1,
        current_role="Junior Backend Developer",
        industry="Tech",
        level=ExperienceLevel.JUNIOR,
        career_goals=["Full-Stack Developer", "Senior Backend", "Tech Lead"],
        learning_preferences={"style": ["hands-on", "structured"], "difficulty_preference": "intermediate"},
        time_availability="10-15h/week",
        budget_constraints={"monthly_budget": 100, "certification_budget": 500}
    ))
    
    # Profil 2: Analyste Data transition vers Data Science
    profiles.append(UserProfile(
        user_id="user_002_data_analyst",
        cv_data={
            "summary": "Analyste data avec 3 ans d'expérience, transition vers data science",
            "experience_text": "Analyse de données business, reporting, dashboards avec Excel et SQL"
        },
        current_skills=["Excel", "SQL", "PowerBI", "Statistics", "Python"],
        experience_years=3,
        current_role="Data Analyst",
        industry="Finance",
        level=ExperienceLevel.MID,
        career_goals=["Data Scientist", "ML Engineer", "AI Specialist"],
        learning_preferences={"style": ["self-paced", "project-based"], "difficulty_preference": "advanced"},
        time_availability="8-12h/week",
        budget_constraints={"monthly_budget": 200, "certification_budget": 1000}
    ))
    
    # Profil 3: Développeur Senior vers Architecture
    profiles.append(UserProfile(
        user_id="user_003_senior_dev",
        cv_data={
            "summary": "Développeur senior full-stack avec expertise cloud, transition vers architecture",
            "experience_text": "7 ans de développement, lead technique sur projets complexes"
        },
        current_skills=["JavaScript", "React", "Node.js", "AWS", "Docker", "Kubernetes", "PostgreSQL", "MongoDB"],
        experience_years=7,
        current_role="Senior Full-Stack Developer",
        industry="FinTech",
        level=ExperienceLevel.SENIOR,
        career_goals=["Solutions Architect", "Tech Lead", "CTO"],
        learning_preferences={"style": ["strategic", "hands-on"], "difficulty_preference": "expert"},
        time_availability="5-8h/week",
        budget_constraints={"monthly_budget": 300, "certification_budget": 2000}
    ))
    
    return profiles


def create_sample_preferences():
    """
    Crée des préférences d'exemple
    """
    return [
        # Préférences focusées sur le développement web
        RecommendationPreferences(
            max_recommendations=15,
            focus_areas=["web_development", "full_stack"],
            difficulty_preference=DifficultyLevel.INTERMEDIATE,
            time_commitment="moderate",
            budget_range="free-premium",
            learning_style=["hands-on", "project-based"]
        ),
        
        # Préférences orientées data science
        RecommendationPreferences(
            max_recommendations=20,
            focus_areas=["data_science", "machine_learning", "ai"],
            difficulty_preference=DifficultyLevel.ADVANCED,
            time_commitment="high",
            budget_range="premium",
            learning_style=["self-paced", "theoretical"]
        ),
        
        # Préférences leadership et architecture
        RecommendationPreferences(
            max_recommendations=12,
            focus_areas=["leadership", "architecture", "cloud"],
            difficulty_preference=DifficultyLevel.EXPERT,
            time_commitment="low",
            budget_range="enterprise",
            learning_style=["strategic", "mentorship"]
        )
    ]


async def demonstrate_comprehensive_recommendations():
    """
    Démontre le système de recommandations complet
    """
    print("\n" + "="*80)
    print("🚀 DÉMONSTRATION SYSTÈME DE RECOMMANDATIONS SKILLSYNC v2.0")
    print("="*80)
    
    # Initialisation du système
    print("\n🔧 Initialisation du système...")
    orchestrator = RecommendationOrchestrator()
    
    # Profils et préférences d'exemple
    user_profiles = create_sample_user_profiles()
    preferences_list = create_sample_preferences()
    
    results = []
    
    # Test avec chaque profil utilisateur
    for i, (user_profile, preferences) in enumerate(zip(user_profiles, preferences_list)):
        print(f"\n\n👤 UTILISATEUR {i+1}: {user_profile.current_role}")
        print("-" * 60)
        print(f"Expérience: {user_profile.experience_years} ans | Niveau: {user_profile.level.value}")
        print(f"Compétences actuelles: {', '.join(user_profile.current_skills[:5])}...")
        print(f"Objectifs: {', '.join(user_profile.career_goals)}")
        
        try:
            # Génération des recommandations complètes
            print("\n⏳ Génération des recommandations...")
            recommendations = await orchestrator.generate_comprehensive_recommendations(
                user_profile, preferences
            )
            
            # Affichage des résultats
            print(f"\n✅ {len(recommendations.recommendations)} types de recommandations générés")
            print(f"Confiance globale: {recommendations.confidence:.1%}")
            
            # Détail par type
            for rec_type, rec_list in recommendations.recommendations.items():
                if rec_list:
                    print(f"\n🎯 {rec_type.upper()} ({len(rec_list)} recommandations):")
                    
                    for j, rec in enumerate(rec_list[:3], 1):  # Top 3 par type
                        score = rec.scores.get('unified', rec.scores.get('base_score', 0))
                        print(f"  {j}. {rec.title}")
                        print(f"     Score: {score:.1%} | Confiance: {rec.confidence:.1%}")
                        
                        # Explication spécifique selon le type
                        if rec_type == 'roadmaps' and hasattr(rec, 'match_reason'):
                            print(f"     Raison: {rec.match_reason}")
                        elif rec_type == 'certifications' and hasattr(rec, 'preparation_estimate'):
                            print(f"     Préparation: {rec.preparation_estimate}")
                        elif rec_type == 'skills' and hasattr(rec, 'priority'):
                            print(f"     Priorité: {rec.priority}")
                        elif rec_type == 'projects' and hasattr(rec, 'learning_value'):
                            print(f"     Valeur apprentissage: {rec.learning_value:.1%}")
            
            # Explication globale
            if recommendations.global_explanation:
                print(f"\n📝 EXPLICATION GLOBALE:")
                print(f"   {recommendations.global_explanation.get('summary', 'Recommandations personnalisées')}")
                
                if 'next_steps' in recommendations.global_explanation:
                    print("\n🎯 PROCHAINES ÉTAPES RECOMMANDÉES:")
                    for step in recommendations.global_explanation['next_steps']:
                        print(f"   • {step}")
            
            # Sauvegarde pour analyse
            results.append({
                'user_profile': asdict(user_profile),
                'recommendations_count': {k: len(v) for k, v in recommendations.recommendations.items()},
                'confidence': recommendations.confidence,
                'generated_at': recommendations.generated_at.isoformat()
            })
            
        except Exception as e:
            print(f"❌ Erreur pour l'utilisateur {user_profile.user_id}: {e}")
            continue
    
    return results


async def demonstrate_specific_recommenders():
    """
    Démontre chaque recommandeur individuellement
    """
    print("\n\n" + "="*80)
    print("🔍 TEST DES RECOMMANDEURS SPÉCIALISÉS")
    print("="*80)
    
    orchestrator = RecommendationOrchestrator()
    user_profile = create_sample_user_profiles()[1]  # Profil data analyst
    
    # Extraction des features utilisateur
    profile_analysis = {'extracted_skills': {'skills': user_profile.current_skills}}
    user_features = orchestrator.personalization_engine.extract_user_features(
        user_profile, profile_analysis
    )
    
    # Test du recommandeur de roadmaps
    print("\n🗺️ TEST RECOMMANDEUR ROADMAPS")
    print("-" * 40)
    roadmap_recs = await orchestrator.roadmap_recommender.recommend(user_features)
    for i, rec in enumerate(roadmap_recs[:3], 1):
        print(f"{i}. {rec.roadmap.title}")
        print(f"   Domaine: {rec.roadmap.domain} | Durée: {rec.roadmap.estimated_duration}")
        print(f"   Étapes: {len(rec.roadmap.steps)} | Score: {rec.scores['base_score']:.1%}")
    
    # Test du recommandeur de certifications
    print("\n🏅 TEST RECOMMANDEUR CERTIFICATIONS")
    print("-" * 40)
    cert_recs = await orchestrator.certification_recommender.recommend(user_features)
    for i, rec in enumerate(cert_recs[:3], 1):
        print(f"{i}. {rec.certification.title}")
        print(f"   Fournisseur: {rec.certification.provider} | Coût: {rec.certification.cost}€")
        print(f"   ROI: {rec.roi_estimate.get('annual_roi_percentage', 0):.0f}% | Score: {rec.scores['base_score']:.1%}")
    
    # Test du recommandeur de compétences
    print("\n💡 TEST RECOMMANDEUR COMPÉTENCES")
    print("-" * 40)
    skill_recs = await orchestrator.skills_recommender.recommend(user_features)
    for i, rec in enumerate(skill_recs[:5], 1):
        print(f"{i}. {rec.skill.name}")
        print(f"   Catégorie: {rec.skill.category} | Priorité: {rec.priority}")
        print(f"   Impact salaire: {rec.skill.average_salary_impact:.1%} | Score: {rec.scores['importance_score']:.1%}")
    
    # Test du recommandeur de projets
    print("\n🛠️ TEST RECOMMANDEUR PROJETS")
    print("-" * 40)
    project_recs = await orchestrator.project_recommender.recommend(user_features)
    for i, rec in enumerate(project_recs[:3], 1):
        print(f"{i}. {rec.project.title}")
        print(f"   Difficulté: {rec.project.difficulty.value} | Temps: {rec.project.estimated_time}")
        print(f"   Technologies: {', '.join(rec.project.technologies[:3])}...")
        print(f"   Impact portfolio: {rec.portfolio_impact:.1%} | Score: {rec.scores['base_score']:.1%}")


async def demonstrate_scoring_engine():
    """
    Démontre le moteur de scoring unifié
    """
    print("\n\n" + "="*80)
    print("🎯 TEST DU MOTEUR DE SCORING UNIFIÉ")
    print("="*80)
    
    orchestrator = RecommendationOrchestrator()
    user_profile = create_sample_user_profiles()[0]  # Profil junior
    
    profile_analysis = {'extracted_skills': {'skills': user_profile.current_skills}}
    user_features = orchestrator.personalization_engine.extract_user_features(
        user_profile, profile_analysis
    )
    
    # Test avec une recommandation de roadmap
    roadmap_recs = await orchestrator.roadmap_recommender.recommend(user_features)
    if roadmap_recs:
        rec = roadmap_recs[0]
        print(f"\n🗺️ ANALYSE SCORING - ROADMAP: {rec.title}")
        print("-" * 50)
        
        # Application du scoring unifié
        scoring_result = orchestrator.scoring_engine.calculate_unified_score(rec, user_features)
        
        print("SCORES INDIVIDUELS:")
        for factor, score in scoring_result.individual_scores.items():
            print(f"  • {factor}: {score:.1%}")
        
        print(f"\nSCORES FINAUX:")
        print(f"  • Score pondéré: {scoring_result.weighted_score:.1%}")
        print(f"  • Score neural: {scoring_result.neural_score:.1%}")
        print(f"  • Score combiné: {scoring_result.combined_score:.1%}")
        print(f"  • Confiance: {scoring_result.confidence:.1%}")
        
        print(f"\nEXPLICATION:")
        explanation = scoring_result.explanation
        if 'strengths' in explanation:
            print("  Points forts:")
            for strength in explanation['strengths']:
                print(f"    ✓ {strength}")
        if 'considerations' in explanation:
            print("  Points d'attention:")
            for consideration in explanation['considerations']:
                print(f"    ⚠ {consideration}")


async def generate_demo_report(results):
    """
    Génère un rapport de démonstration
    CORRECTION: Fix ZeroDivisionError potentiel
    """
    print("\n\n" + "="*80)
    print("📊 RAPPORT DE DÉMONSTRATION")
    print("="*80)
    
    # Statistiques globales
    total_recommendations = sum(
        sum(result['recommendations_count'].values()) 
        for result in results
    )
    
    # CORRECTION: Éviter division par zéro
    if len(results) > 0:
        avg_confidence = sum(result['confidence'] for result in results) / len(results)
    else:
        avg_confidence = 0.0
    
    print(f"\n📈 STATISTIQUES GLOBALES:")
    print(f"  • Utilisateurs testés: {len(results)}")
    print(f"  • Recommandations générées: {total_recommendations}")
    print(f"  • Confiance moyenne: {avg_confidence:.1%}")
    
    # Répartition par type
    type_counts = {}
    for result in results:
        for rec_type, count in result['recommendations_count'].items():
            type_counts[rec_type] = type_counts.get(rec_type, 0) + count
    
    print(f"\n📊 RÉPARTITION PAR TYPE:")
    for rec_type, count in type_counts.items():
        # CORRECTION: Éviter division par zéro
        if total_recommendations > 0:
            percentage = count / total_recommendations * 100
        else:
            percentage = 0.0
        print(f"  • {rec_type}: {count} ({percentage:.1f}%)")
    
    # Sauvegarde du rapport
    report_path = Path("recommendation_demo_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        report_data = {
            'demo_results': results,
            'statistics': {
                'total_users': len(results),
                'total_recommendations': total_recommendations,
                'average_confidence': avg_confidence,
                'type_distribution': type_counts
            },
            'generated_at': datetime.now().isoformat()
        }
        # Conversion en format JSON sérialisable
        serializable_data = make_json_serializable(report_data)
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Rapport sauvegardé: {report_path}")


async def main():
    """
    Fonction principale de démonstration
    """
    try:
        print("🎆 LANCEMENT DE LA DÉMONSTRATION SKILLSYNC v2.0")
        print("Système de recommandations multicritères avec IA avancée\n")
        
        # Démonstration complète
        results = await demonstrate_comprehensive_recommendations()
        
        # Tests des recommandeurs individuels
        await demonstrate_specific_recommenders()
        
        # Test du moteur de scoring
        await demonstrate_scoring_engine()
        
        # Génération du rapport
        await generate_demo_report(results)
        
        print("\n" + "="*80)
        print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
        print("\nLe système SkillSync v2.0 est opérationnel avec:")
        print("• Recommandations multicritères (roadmaps, certifications, compétences, projets)")
        print("• Scoring unifié avec IA et heuristiques")
        print("• Personnalisation avancée")
        print("• Équilibrage et diversification intelligente")
        print("• API complète pour intégration frontend")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Erreur durant la démonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Exécution de la démonstration
    asyncio.run(main())