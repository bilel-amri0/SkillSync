from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio
from datetime import datetime
import base64
import uuid
import re
import traceback
import os

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    # Fix: Point to the correct .env file path
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print("🔑 Environment variables loaded from .env file")
    else:
        print("⚠️ .env file not found at", dotenv_path)
except ImportError:
    # If python-dotenv is not installed, try to load manually
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("🔑 Environment variables loaded manually from .env file")

# Configure logging AFTER env loading
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Show that APIs will be enabled
print(f"🔍 JSEARCH_RAPIDAPI_KEY: {'✅ LOADED' if os.getenv('JSEARCH_RAPIDAPI_KEY') else '❌ MISSING'}")
print(f"🔍 ADZUNA_APP_ID: {'✅ LOADED' if os.getenv('ADZUNA_APP_ID') else '❌ MISSING'}")

# Import our services AFTER env loading
from services.multi_job_api_service import get_job_service, JobResult

# Import Experience Translator (F7)
try:
    from experience_translator import translate_experience_api, ExperienceTranslator
    logger.info("✅ Experience Translator (F7) loaded successfully")
except ImportError as exp_error:
    logger.warning(f"⚠️ Experience Translator not available: {exp_error}")
    translate_experience_api = None
    ExperienceTranslator = None

# Enhanced recommendation engine import with ML support and fallback
ML_MODE_ENABLED = False
ML_MODE_TYPE = "hybrid"  # rules, lite, full, ou hybrid
USE_ENHANCED_ENGINE = False
recommendation_engine = None
ml_recommendation_engine = None
ml_backend = None
ml_models = {}

# Initialisation Backend ML Hybride
try:
    from ml_backend_hybrid import get_ml_backend
    ml_backend = get_ml_backend()
    ML_MODE_ENABLED = True
    ML_MODE_TYPE = "hybrid"
    print("🚀 Backend ML Hybride initialisé avec succès")
    print(f"📊 Status: {ml_backend.get_system_status()}")
except Exception as e:
    print(f"⚠️ Backend ML Hybride non disponible: {e}")
    ML_MODE_ENABLED = False
    ml_backend = None

def detect_ml_configuration():
    """Détecte automatiquement la configuration ML disponible"""
    global ML_MODE_ENABLED, ML_MODE_TYPE
    
    # Vérifier les flags de configuration
    import os
    from pathlib import Path
    
    if Path('ml_mode_enabled.flag').exists():
        logger.info("🚀 Flag ML détecté, activation du mode ML...")
        
        # Vérifier le type de config dans .env.ml
        env_config = {}
        if Path('.env.ml').exists():
            try:
                with open('.env.ml', 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            env_config[key] = value.lower() == 'true'
            except:
                pass
        
        use_tensorflow = env_config.get('USE_TENSORFLOW', True)
        
        if use_tensorflow:
            return try_full_ml_mode()
        else:
            return try_lite_ml_mode()
    
    return False

def try_lite_ml_mode():
    """Essaie d'activer le mode ML léger (sans TensorFlow)"""
    global ML_MODE_ENABLED, ML_MODE_TYPE, ml_models
    
    try:
        logger.info("🔥 Tentative mode ML LÉGER...")
        
        # Import des bibliothèques essentielles uniquement
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
        import torch
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("   📦 Chargement modèles légers...")
        
        # Modèles plus légers et plus rapides
        ml_models['tokenizer'] = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        ml_models['model'] = AutoModel.from_pretrained('distilbert-base-uncased')
        ml_models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Scorer basé sur PyTorch (plus léger que TensorFlow)
        class LiteScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(384, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 1),
                    torch.nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        ml_models['lite_scorer'] = LiteScorer()
        
        ML_MODE_ENABLED = True
        ML_MODE_TYPE = "lite"
        
        logger.info("✅ Mode ML LÉGER activé avec succès!")
        logger.info(f"   🧠 Modèle: DistilBERT (léger)")
        logger.info(f"   🔗 Sentence-BERT: MiniLM-L6-v2")
        logger.info(f"   🎯 Scorer: PyTorch (lite)")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Bibliothèques ML léger manquantes: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur mode ML léger: {e}")
        return False

def try_full_ml_mode():
    """Essaie d'activer le mode ML complet (avec TensorFlow)"""
    global ML_MODE_ENABLED, ML_MODE_TYPE, ml_models
    
    try:
        logger.info("🚀 Tentative mode ML COMPLET avec TensorFlow...")
        
        # Import complet avec TensorFlow
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
        import tensorflow as tf
        import torch
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("   📦 Chargement modèles ML complets...")
        
        # Modèles complets avec TensorFlow
        ml_models['bert_tokenizer'] = AutoTokenizer.from_pretrained('bert-base-uncased')
        ml_models['bert_model'] = AutoModel.from_pretrained('bert-base-uncased')
        ml_models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Neural Scorer TensorFlow (plus puissant)
        logger.info("   🧠 Création Neural Scorer TensorFlow...")
        ml_models['neural_scorer'] = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(384,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compiler le modèle pour optimiser les performances
        ml_models['neural_scorer'].compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Test du modèle TensorFlow
        test_input = np.random.random((1, 384))
        test_prediction = ml_models['neural_scorer'].predict(test_input, verbose=0)
        logger.info(f"   ✅ Test Neural Scorer: {test_prediction[0][0]:.4f}")
        
        ML_MODE_ENABLED = True
        ML_MODE_TYPE = "full"
        
        logger.info("🎉 Mode ML COMPLET avec TensorFlow activé!")
        logger.info(f"   🧠 BERT: {ml_models['bert_model'].__class__.__name__}")
        logger.info(f"   🔗 Sentence-BERT: {ml_models['sentence_model'].__class__.__name__}")
        logger.info(f"   🎯 Neural Scorer: TensorFlow {tf.__version__}")
        logger.info(f"   ⚡ Architecture: {len(ml_models['neural_scorer'].layers)} couches")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Bibliothèques ML complètes manquantes: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur mode ML complet: {e}")
        return False

# Ajout des fonctions ML pour les recommandations
def extract_skills_with_ml(text):
    """Extrait les compétences avec ML"""
    if not ML_MODE_ENABLED or 'sentence_model' not in ml_models:
        return []
    
    try:
        # Liste de compétences de référence
        skill_database = [
            "Python", "JavaScript", "Java", "C++", "SQL", "HTML", "CSS",
            "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
            "Machine Learning", "Data Science", "AI", "Deep Learning",
            "Docker", "Kubernetes", "AWS", "Azure", "Git", "DevOps"
        ]
        
        # Utiliser sentence-transformers pour la similarité
        text_embedding = ml_models['sentence_model'].encode([text])
        skill_embeddings = ml_models['sentence_model'].encode(skill_database)
        
        # Calculer similarités
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(text_embedding, skill_embeddings)[0]
        
        # Retourner les compétences avec similarité > 0.5
        detected_skills = []
        for i, similarity in enumerate(similarities):
            if similarity > 0.5:
                detected_skills.append(skill_database[i])
        
        return detected_skills[:5]  # Top 5
        
    except Exception as e:
        logger.error(f"Erreur extraction ML: {e}")
        return []

def score_with_ml(profile, recommendation):
    """Score une recommandation avec ML (lite, full ou hybrid)"""
    if not ML_MODE_ENABLED:
        return 0.5
    
    try:
        # Encoder le profil et la recommandation
        profile_text = f"{profile.get('current_role', '')} {' '.join(profile.get('skills', []))}"
        rec_text = recommendation.get('title', '')
        
        # Mode ML HYBRID avec Backend ML Hybride (utilise son propre système)
        if ML_MODE_TYPE == "hybrid" and ml_backend:
            try:
                # Utiliser le backend ML hybride pour le scoring direct
                skills_text = ' '.join(profile.get('skills', []))
                hybrid_score = ml_backend.score_skill_match(skills_text, rec_text)
                
                # Ajouter un bonus basé sur l'expérience et le rôle
                role_bonus = 0.1 if profile.get('current_role', '').lower() in rec_text.lower() else 0
                experience_bonus = min(0.15, profile.get('experience_years', 0) * 0.02)
                
                final_score = hybrid_score + role_bonus + experience_bonus
                final_score = max(0.1, min(1.0, final_score))
                
                logger.debug(f"Hybrid ML scoring - Base: {hybrid_score:.3f}, Role bonus: {role_bonus:.3f}, Exp bonus: {experience_bonus:.3f}, Final: {final_score:.3f}")
                return final_score
                
            except Exception as hybrid_error:
                logger.warning(f"Hybrid ML scoring failed, using fallback: {hybrid_error}")
                # Fallback simple pour le mode hybrid
                return 0.6 + (len(profile.get('skills', [])) * 0.02)
        
        # Pour les autres modes (full/lite), utiliser le système d'embeddings existant
        if 'sentence_model' not in ml_models:
            logger.warning("Sentence model not available, using simple scoring")
            return 0.5 + (len(profile.get('skills', [])) * 0.03)
            
        # Utiliser sentence-transformers pour l'embedding de base
        embeddings = ml_models['sentence_model'].encode([profile_text, rec_text])
        
        # Calculer similarité comme score base
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Mode ML COMPLET avec TensorFlow Neural Scorer
        if ML_MODE_TYPE == "full" and 'neural_scorer' in ml_models:
            try:
                import numpy as np
                # Utiliser le Neural Scorer TensorFlow pour un scoring avancé
                profile_embedding = embeddings[0].reshape(1, -1)
                neural_score = ml_models['neural_scorer'].predict(profile_embedding, verbose=0)[0][0]
                
                # Combiner similarity et neural score avec pondération
                # 60% neural score (plus sophistiqué) + 40% similarity
                final_score = (neural_score * 0.6) + (similarity * 0.4)
                
                logger.debug(f"TensorFlow scoring - Similarity: {similarity:.3f}, Neural: {neural_score:.3f}, Final: {final_score:.3f}")
                
            except Exception as tf_error:
                logger.warning(f"TensorFlow scoring failed, fallback to similarity: {tf_error}")
                final_score = similarity
        
        # Mode ML LITE avec PyTorch Lite Scorer
        elif ML_MODE_TYPE == "lite" and 'lite_scorer' in ml_models:
            try:
                import torch
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(embeddings[0]).unsqueeze(0)
                    pytorch_score = ml_models['lite_scorer'](input_tensor).item()
                    
                    # Combiner similarity et pytorch score
                    final_score = (similarity + pytorch_score) / 2
                    
                    logger.debug(f"PyTorch scoring - Similarity: {similarity:.3f}, PyTorch: {pytorch_score:.3f}, Final: {final_score:.3f}")
                    
            except Exception as pt_error:
                logger.warning(f"PyTorch scoring failed, fallback to similarity: {pt_error}")
                final_score = similarity
        
        # Fallback: utiliser seulement la similarité
        else:
            final_score = similarity
            logger.debug(f"Similarity-only scoring: {final_score:.3f}")
        
        # Normaliser le score entre 0.1 et 1.0
        final_score = max(0.1, min(1.0, final_score))
        
        return final_score
        
    except Exception as e:
        logger.error(f"Erreur scoring ML: {e}")
        return 0.5

async def generate_ml_enhanced_recommendations(skills_data, cv_data):
    """Génère des recommandations avec le mode ML (lite ou full) - DYNAMIC VERSION"""
    try:
        # Profile utilisateur enrichi
        user_profile = {
            'current_role': ' '.join(cv_data.get('job_titles', ['Developer'])),
            'skills': [skill.get('skill', '') for skill in skills_data],
            'experience_years': cv_data.get('experience_years', 2),
            'education': cv_data.get('education', []),
            'industry': cv_data.get('industry', 'Tech')
        }
        
        # 🎯 NEW: Use DynamicRecommendationEngine for personalized recommendations
        from dynamic_recommendations import DynamicRecommendationEngine
        dynamic_engine = DynamicRecommendationEngine()
        
        # Prepare CV text for analysis (simulate text from profile data)
        cv_text = f"""
        Job Title: {user_profile['current_role']}
        Skills: {', '.join(user_profile['skills'])}
        Experience: {user_profile['experience_years']} years
        Industry: {user_profile['industry']}
        Education: {', '.join(user_profile['education']) if user_profile['education'] else 'Not specified'}
        """.lower()
        
        # Generate dynamic analysis and recommendations
        logger.info(f"🎯 Analyzing CV with {len(user_profile['skills'])} skills for {user_profile['current_role']}")
        
        try:
            cv_analysis = dynamic_engine.analyze_cv_content(cv_text)
            if not isinstance(cv_analysis, dict) or 'primary_domains' not in cv_analysis:
                logger.warning("⚠️ CV analysis returned invalid format, using fallback")
                cv_analysis = dynamic_engine._get_default_analysis()
            
            logger.info(f"📊 CV Analysis result: {cv_analysis['experience_level']} level, domains: {cv_analysis['primary_domains']}")
            
            # Generate personalized skill recommendations
            skill_recommendations = dynamic_engine.generate_skill_recommendations(cv_analysis)
            logger.info(f"🎯 Generated {len(skill_recommendations)} personalized skill recommendations")
            
        except Exception as e:
            logger.error(f"❌ Error in CV analysis: {e}")
            logger.info("🔄 Using fallback analysis")
            cv_analysis = {
                "experience_level": "mid",
                "primary_domains": ["frontend", "backend"],
                "current_skills": user_profile['skills'],
                "role_focus": "fullstack",
                "text_length": len(cv_text),
                "has_technical_content": True
            }
            skill_recommendations = dynamic_engine.generate_skill_recommendations(cv_analysis)
        
        # Convert to backend format and enhance with ML scoring
        base_recommendations = []
        
        # 1. Immediate Actions (personalized based on analysis)
        immediate_actions = []
        primary_domain = cv_analysis['primary_domains'][0] if cv_analysis['primary_domains'] else 'fullstack'
        role_focus = cv_analysis.get('role_focus', 'fullstack')
        
        if cv_analysis['experience_level'] == 'junior':
            immediate_actions = [
                {"title": f"Créer portfolio {primary_domain} débutant", "type": "immediate", "category": "portfolio"},
                {"title": "Optimiser profil LinkedIn junior", "type": "immediate", "category": "personal_branding"},
                {"title": f"Rejoindre communautés {role_focus}", "type": "immediate", "category": "networking"},
                {"title": "Mettre à jour CV avec projets récents", "type": "immediate", "category": "cv_optimization"},
                {"title": f"Configurer environnement {primary_domain}", "type": "immediate", "category": "technical_setup"}
            ]
        elif cv_analysis['experience_level'] == 'senior':
            immediate_actions = [
                {"title": f"Leadership technique en {role_focus}", "type": "immediate", "category": "leadership"},
                {"title": "Mentoring et formation équipe", "type": "immediate", "category": "mentoring"},
                {"title": f"Architecture avancée {primary_domain}", "type": "immediate", "category": "architecture"},
                {"title": "Publication articles techniques", "type": "immediate", "category": "thought_leadership"},
                {"title": "Contribution projets open source", "type": "immediate", "category": "networking"}
            ]
        else:  # mid-level
            immediate_actions = [
                {"title": f"Portfolio avancé {role_focus}", "type": "immediate", "category": "portfolio"},
                {"title": f"Contribution projets {primary_domain}", "type": "immediate", "category": "networking"},
                {"title": "Optimisation LinkedIn avec mots-clés", "type": "immediate", "category": "personal_branding"},
                {"title": f"Démo interactive {role_focus}", "type": "immediate", "category": "demonstration"},
                {"title": f"Configuration environnement {primary_domain} pro", "type": "immediate", "category": "technical_setup"}
            ]
        
        base_recommendations.extend(immediate_actions)
        
        # 2. Add personalized skill recommendations from dynamic engine
        for skill_rec in skill_recommendations:
            base_recommendations.append({
                "title": skill_rec['skill'],
                "type": "skill", 
                "category": skill_rec['category'].lower(),
                "reason": skill_rec['reason'],
                "priority": skill_rec['priority'],
                "estimated_time": skill_rec['estimated_time'],
                "difficulty": skill_rec['difficulty']
            })
        
        # 3. Domain-specific certifications
        cert_recommendations = []
        primary_domains = cv_analysis.get('primary_domains', ['frontend', 'backend'])
        for domain in primary_domains:
            if domain == 'data_science':
                cert_recommendations.extend([
                    {"title": "AWS Machine Learning Specialty", "type": "certification", "category": "aws_ml"},
                    {"title": "Google Professional ML Engineer", "type": "certification", "category": "gcp_ml"}
                ])
            elif domain == 'frontend':
                cert_recommendations.extend([
                    {"title": "React Developer Certification", "type": "certification", "category": "react"},
                    {"title": "AWS Certified Solutions Architect", "type": "certification", "category": "aws"}
                ])
            elif domain == 'backend':
                cert_recommendations.extend([
                    {"title": "AWS Certified Developer", "type": "certification", "category": "aws"},
                    {"title": "Node.js Professional Certification", "type": "certification", "category": "nodejs"}
                ])
            elif domain == 'devops':
                cert_recommendations.extend([
                    {"title": "AWS DevOps Engineer Professional", "type": "certification", "category": "aws_devops"},
                    {"title": "Kubernetes Administrator", "type": "certification", "category": "k8s"}
                ])
        
        base_recommendations.extend(cert_recommendations[:4])  # Limit to 4 certifications
        
        # 4. Add learning resources
        base_recommendations.extend([
            {"title": f"Formation {role_focus} avancée", "type": "resource", "category": "online_course"},
            {"title": f"Documentation officielle {primary_domain}", "type": "resource", "category": "documentation"}
        ])
        
        # Continue with ML scoring and processing
        logger.info(f"🎯 Generated {len(base_recommendations)} personalized recommendations for {ML_MODE_TYPE} mode")
        
        # Scorer chaque recommandation avec ML approprié
        scored_recommendations = []
        for rec in base_recommendations:
            score = score_with_ml(user_profile, rec)
            
            # Enrichir avec descriptions spécifiques au mode ML
            if ML_MODE_TYPE == "full":
                user_skills_text = ', '.join(user_profile['skills'][:3])
                rec['description'] = f"Recommandation TensorFlow personnalisée pour {user_skills_text}. Score Neural: {score:.1%}"
                rec['ml_engine'] = "TensorFlow Neural Network"
            elif ML_MODE_TYPE == "hybrid":
                user_skills_text = ', '.join(user_profile['skills'][:4])
                rec['description'] = f"Recommandation ML hybride optimisée pour {user_skills_text}. Score IA: {score:.1%}"
                rec['ml_engine'] = "PyTorch + Scikit-learn Hybrid"
            else:
                user_skills_text = ', '.join(user_profile['skills'][:5])
                rec['description'] = f"Recommandation ML pour votre profil ({user_skills_text}). Score PyTorch: {score:.1%}"
                rec['ml_engine'] = "PyTorch Lite Model"
            
            rec['priority_score'] = round(score * 100, 1)
            rec['confidence'] = min(0.95, score + 0.1)
            rec['ml_mode_used'] = ML_MODE_TYPE.upper()
            
            scored_recommendations.append(rec)
        
        # Trier par score et grouper par type
        scored_recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Organiser par catégories avec quantités adaptées au mode
        recommendations_count = {
            "immediate": 6 if ML_MODE_TYPE == "full" else (5 if ML_MODE_TYPE == "hybrid" else 5),
            "skill": 8 if ML_MODE_TYPE == "full" else (7 if ML_MODE_TYPE == "hybrid" else 6),
            "certification": 5 if ML_MODE_TYPE == "full" else (4 if ML_MODE_TYPE == "hybrid" else 4),
            "career": 5 if ML_MODE_TYPE == "full" else (4 if ML_MODE_TYPE == "hybrid" else 4)
        }
        
        formatted_recommendations = {
            "IMMEDIATE_ACTIONS": [r for r in scored_recommendations if r['type'] == 'immediate'][:recommendations_count["immediate"]],
            "SKILL_DEVELOPMENT": [r for r in scored_recommendations if r['type'] == 'skill'][:recommendations_count["skill"]],
            "CERTIFICATIONS": [r for r in scored_recommendations if r['type'] == 'certification'][:recommendations_count["certification"]],
            "CAREER_OPPORTUNITIES": [r for r in scored_recommendations if r['type'] == 'career'][:recommendations_count["career"]],
            "LEARNING_RESOURCES": [],
            "CAREER_ROADMAP": {}  # Initialize career roadmap
        }
        
        # 5. Generate Career Roadmap based on CV analysis
        roadmap_milestones = []
        
        if cv_analysis['experience_level'] == 'junior':
            roadmap_milestones = [
                {
                    "month": 3,
                    "title": "Foundation Skills",
                    "description": f"Master core {primary_domain} fundamentals and build first projects",
                    "focus_areas": ["Basic Programming", "Version Control", "First Portfolio"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Technical Proficiency", 
                    "description": f"Develop intermediate {role_focus} skills and complete real projects",
                    "focus_areas": ["Framework Knowledge", "Database Basics", "Code Quality"],
                    "status": "upcoming"
                },
                {
                    "month": 9,
                    "title": "Industry Integration",
                    "description": "Join tech communities, contribute to open source, network actively",
                    "focus_areas": ["Open Source", "Networking", "Industry Knowledge"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Job Ready",
                    "description": "Complete portfolio, practice interviews, apply for entry-level positions",
                    "focus_areas": ["Interview Prep", "Portfolio Polish", "Job Applications"],
                    "status": "upcoming"
                }
            ]
        elif cv_analysis['experience_level'] == 'senior':
            roadmap_milestones = [
                {
                    "month": 3,
                    "title": "Leadership Development",
                    "description": f"Enhance technical leadership and {primary_domain} architecture skills",
                    "focus_areas": ["Team Leadership", "Architecture Design", "Mentoring"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Strategic Impact",
                    "description": "Drive technical strategy and cross-team collaboration",
                    "focus_areas": ["Technical Strategy", "Cross-team Collaboration", "Business Alignment"],
                    "status": "upcoming"
                },
                {
                    "month": 9,
                    "title": "Industry Influence",
                    "description": "Establish thought leadership and external presence",
                    "focus_areas": ["Public Speaking", "Technical Writing", "Community Building"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Executive Readiness",
                    "description": "Position for executive or principal technical roles",
                    "focus_areas": ["Technical Leadership", "Business Impact", "Strategy Input"],
                    "status": "upcoming"
                }
            ]
        else:  # mid-level
            roadmap_milestones = [
                {
                    "month": 3,
                    "title": "Skill Specialization",
                    "description": f"Deepen expertise in {primary_domain} and related technologies",
                    "focus_areas": ["Advanced Technical Skills", "Best Practices", "Problem Solving"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Project Leadership",
                    "description": "Lead technical projects and mentor junior developers",
                    "focus_areas": ["Project Management", "Code Reviews", "Junior Mentoring"],
                    "status": "upcoming"
                },
                {
                    "month": 9,
                    "title": "Cross-Domain Growth",
                    "description": "Expand into adjacent technologies and gain broader perspective",
                    "focus_areas": ["Technology Breadth", "System Design", "DevOps Knowledge"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Senior Readiness",
                    "description": "Demonstrate senior-level impact and prepare for promotion",
                    "focus_areas": ["Technical Leadership", "Business Impact", "Strategy Input"],
                    "status": "upcoming"
                }
            ]
        
        # Create career roadmap data structure
        career_roadmap_data = {
            "target_role": f"Senior {role_focus.title()} Developer" if cv_analysis['experience_level'] != 'senior' else f"Lead {role_focus.title()} Architect",
            "timeline_months": 12,
            "milestones": roadmap_milestones,
            "current_level": cv_analysis['experience_level'],
            "focus_domains": cv_analysis['primary_domains'],
            "personalized": True
        }
        
        formatted_recommendations["CAREER_ROADMAP"] = career_roadmap_data
        
        # Debug logging for roadmap
        logger.info(f"🗺️ Career roadmap created: {len(roadmap_milestones)} milestones for {cv_analysis['experience_level']} level")
        logger.info(f"🎯 Target role: {career_roadmap_data['target_role']}")
        
        total_recommendations = sum(len(v) for v in formatted_recommendations.values() if isinstance(v, list))
        logger.info(f"🎯 Generated {total_recommendations} ML-{ML_MODE_TYPE} enhanced recommendations")
        
        return formatted_recommendations
        
    except Exception as e:
        logger.error(f"Erreur génération ML recommendations: {e}")
        # Fallback vers recommandations basiques
        return generate_fallback_recommendations(skills_data)

try:
    import sys
    import os
    # Add workspace to path for enhanced engine
    workspace_path = '/workspace'
    if workspace_path not in sys.path:
        sys.path.append(workspace_path)
    
    # Tentative d'activation ML automatique
    if not detect_ml_configuration():
        logger.info("💡 Mode fallback: recommandations basées sur des règles")
    
    # Try ML mode first (Advanced AI) - Check existing ML models
    if not ML_MODE_ENABLED:
        try:
            from SkillSync_Project.backend.ml_models.advanced_recommendation_engine import AdvancedRecommendationEngine
            ml_recommendation_engine = AdvancedRecommendationEngine()
            ML_MODE_ENABLED = True
            ML_MODE_TYPE = "full"
            logger.info("🧠 MODE ML AVANCÉ EXISTANT ACTIVÉ")
        except ImportError as ml_error:
            logger.info(f"🔄 Mode ML avancé non disponible: {ml_error}")
        
    # Fallback to enhanced engine (Rule-based)
    if not ML_MODE_ENABLED:
        try:
            from enhanced_recommendation_engine import EnhancedRecommendationEngine
            recommendation_engine = EnhancedRecommendationEngine()
            USE_ENHANCED_ENGINE = True
            logger.info("✅ Moteur de recommandations amélioré chargé")
        except ImportError as enhanced_error:
            logger.warning(f"⚠️ Moteur amélioré non disponible: {enhanced_error}")
            
            # Final fallback to original engine
            try:
                from recommendation_engine import RecommendationEngine
                recommendation_engine = RecommendationEngine()
                logger.info("✅ Moteur de base chargé")
            except ImportError as original_error:
                logger.error(f"❌ Aucun moteur de recommandations disponible: {original_error}")
                recommendation_engine = None

except Exception as e:
    logger.error(f"❌ Erreur critique lors du chargement des moteurs: {e}")
    recommendation_engine = None
    ml_recommendation_engine = None
    ML_MODE_ENABLED = False
    USE_ENHANCED_ENGINE = False

app = FastAPI(
    title="SkillSync Multi-API Backend",
    description="Professional job matching service with multiple API integrations",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:8080",  # Vue dev server
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response models
class JobSearchRequest(BaseModel):
    query: str
    location: Optional[str] = ""
    skills: Optional[List[str]] = []
    max_results: Optional[int] = 50

class JobResponse(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    url: str
    salary: Optional[str]
    posted_date: Optional[str]
    source: str
    skills_match: float
    remote: bool

class SearchResponse(BaseModel):
    jobs: List[JobResponse]
    total_count: int
    search_query: str
    location: str
    sources_used: List[str]
    search_time_ms: int
    timestamp: str

class StatusResponse(BaseModel):
    apis: Dict[str, Dict[str, Any]]
    total_enabled: int
    system_status: str
    timestamp: str

class CVAnalysisRequest(BaseModel):
    cv_content: str
    format: Optional[str] = "text"

class CVAnalysisResponse(BaseModel):
    analysis_id: str
    skills: List[str]
    experience_years: Optional[int]
    job_titles: List[str]
    education: List[str]
    summary: str
    confidence_score: float
    timestamp: str
    sections: Optional[Dict[str, Any]] = None
    personal_info: Optional[Dict[str, Any]] = None
    contact_info: Optional[Dict[str, Any]] = None

# Initialize services
job_service = get_job_service()
# recommendation_engine is already initialized in the try-catch block above

# In-memory storage for CV analysis results (for recommendations)
cv_analysis_storage = {}

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SkillSync Multi-API Backend",
        "version": "2.0.1-ML" if ML_MODE_ENABLED else "2.0.0",
        "status": "operational",
        "ml_mode": ML_MODE_ENABLED,
        "enhanced_mode": USE_ENHANCED_ENGINE,
        "apis_available": list((await job_service.get_api_status()).keys()),
        "endpoints": {
            "job_search_get": "GET /api/v1/jobs/search",
            "job_search_post": "POST /api/v1/jobs/search", 
            "api_status": "GET /api/v1/jobs/status",
            "ml_status": "GET /api/v1/ml/status",
            "upload_cv": "POST /api/v1/upload-cv",
            "analyze_cv": "POST /api/v1/analyze-cv",
            "dashboard_latest": "GET /api/v1/dashboard/latest",
            "dashboard_by_id": "GET /api/v1/dashboard/{analysis_id}",
            "recommendations": "GET /api/v1/recommendations/{analysis_id}",
            "recommendations_post": "POST /api/v1/recommendations",
            "generate_portfolio": "POST /api/v1/generate-portfolio",
            "experience_translate": "POST /api/v1/experience/translate",
            "experience_styles": "GET /api/v1/experience/styles",
            "experience_analysis": "GET /api/v1/experience/analysis/{translation_id}"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/ml/status", response_model=dict)
async def get_ml_status():
    """Get ML system status and capabilities"""
    
    # Déterminer le type d'engine actuel
    engine_type = "Basic"
    version = "2.0.0"
    
    if ML_MODE_ENABLED:
        if ML_MODE_TYPE == "lite":
            engine_type = "ML-Lite"
            version = "2.1.0-ML-Lite"
        elif ML_MODE_TYPE == "full":
            engine_type = "ML-Full" 
            version = "2.1.0-ML-Full"
        else:
            engine_type = "ML-AI"
            version = "2.0.1-ML"
    elif USE_ENHANCED_ENGINE:
        engine_type = "Enhanced"
        version = "2.0.0-Enhanced"
    
    return {
        "ml_enabled": ML_MODE_ENABLED,
        "ml_mode_type": ML_MODE_TYPE,
        "enhanced_mode_enabled": USE_ENHANCED_ENGINE,
        "engine_type": engine_type,
        "version": version,
        "capabilities": {
            "ai_powered_recommendations": ML_MODE_ENABLED,
            "lite_ml_scoring": ML_MODE_ENABLED and ML_MODE_TYPE == "lite",
            "full_ml_scoring": ML_MODE_ENABLED and ML_MODE_TYPE == "full",
            "bert_skills_extraction": ML_MODE_ENABLED,
            "neural_scoring": ML_MODE_ENABLED,
            "semantic_similarity": ML_MODE_ENABLED,
            "pytorch_models": ML_MODE_ENABLED and ML_MODE_TYPE in ["lite", "full"],
            "tensorflow_models": ML_MODE_ENABLED and ML_MODE_TYPE == "full",
            "rule_based_recommendations": USE_ENHANCED_ENGINE or not ML_MODE_ENABLED,
            "basic_recommendations": True,
            "fallback_available": True
        },
        "models_loaded": {
            "sentence_transformer": "sentence_model" in ml_models,
            "distilbert": "tokenizer" in ml_models and ML_MODE_TYPE == "lite",
            "bert": "bert_tokenizer" in ml_models and ML_MODE_TYPE == "full",
            "lite_scorer": "lite_scorer" in ml_models and ML_MODE_TYPE == "lite",
            "neural_scorer": "neural_scorer" in ml_models and ML_MODE_TYPE == "full"
        },
        "components": {
            "ml_recommendation_engine": ml_recommendation_engine is not None,
            "enhanced_recommendation_engine": recommendation_engine is not None,
            "ml_models": len(ml_models) > 0,
            "fallback_engine": True
        },
        "performance": {
            "mode": "lite" if ML_MODE_TYPE == "lite" else ("full" if ML_MODE_TYPE == "full" else "rules"),
            "speed": "fast" if ML_MODE_TYPE == "lite" else ("medium" if ML_MODE_TYPE == "full" else "fastest"),
            "accuracy": "high" if ML_MODE_ENABLED else "good"
        },
        "timestamp": datetime.now().isoformat()
    }

# Request model for POST endpoint
class JobSearchBody(BaseModel):
    query: str
    location: Optional[str] = ""
    skills: Optional[List[str]] = []
    max_results: Optional[int] = 50

@app.get("/api/v1/jobs/search", response_model=SearchResponse)
async def search_jobs_get(
    query: str = Query(..., description="Job search query"),
    location: str = Query("", description="Location filter"),
    skills: Optional[str] = Query(None, description="Comma-separated skills"),
    max_results: int = Query(50, ge=1, le=100, description="Maximum results")
):
    """Search jobs across all configured APIs (GET method)"""
    # Parse skills for GET request
    skills_list = []
    if skills:
        skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
    
    return await _perform_job_search(query, location, skills_list, max_results)

@app.post("/api/v1/jobs/search", response_model=SearchResponse)
async def search_jobs_post(search_request: JobSearchBody):
    """Search jobs across all configured APIs (POST method)"""
    return await _perform_job_search(
        search_request.query, 
        search_request.location, 
        search_request.skills or [], 
        search_request.max_results
    )

async def _perform_job_search(query: str, location: str, skills_list: List[str], max_results: int):
    """Search jobs across all configured APIs"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"🔍 Job search request: query='{query}', location='{location}', skills={skills_list}")
        
        # Search jobs
        jobs = await job_service.search_jobs(query, location, skills_list)
        
        # Calculate search time
        search_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Convert to response format
        job_responses = []
        sources_used = set()
        
        for job in jobs[:max_results]:
            sources_used.add(job.source)
            
            # Fix: Convert posted_date to string if it's an integer
            posted_date_str = str(job.posted_date) if job.posted_date is not None else None
            
            job_responses.append(JobResponse(
                id=job.id,
                title=job.title,
                company=job.company,
                location=job.location,
                description=job.description,
                url=job.url,
                salary=job.salary,
                posted_date=posted_date_str,
                source=job.source,
                skills_match=job.skills_match,
                remote=job.remote
            ))
        
        response = SearchResponse(
            jobs=job_responses,
            total_count=len(job_responses),
            search_query=query,
            location=location,
            sources_used=list(sources_used),
            search_time_ms=search_time_ms,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Search completed: {len(job_responses)} jobs from {len(sources_used)} sources in {search_time_ms}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Job search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")

@app.get("/api/v1/jobs/status", response_model=StatusResponse)
async def get_api_status():
    """Get status of all configured job APIs"""
    try:
        api_status = await job_service.get_api_status()
        enabled_count = sum(1 for config in api_status.values() if config['enabled'])
        
        system_status = "operational" if enabled_count > 0 else "no_apis_configured"
        
        return StatusResponse(
            apis=api_status,
            total_enabled=enabled_count,
            system_status=system_status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.post("/api/v1/upload-cv", response_model=CVAnalysisResponse)
async def upload_cv(file: UploadFile = File(...)):
    """Upload and analyze CV file - VERSION CORRIGÉE"""
    try:
        logger.info(f"📄 CV upload request: {file.filename} ({file.content_type})")
        
        file_content = await file.read()
        
        if file.content_type == "text/plain":
            cv_text = file_content.decode('utf-8')
        else:
            cv_text = base64.b64encode(file_content).decode('utf-8')[:1000]
        
        # Analyse CV - nouvelle version
        cv_result = create_cv_analysis(cv_text)
        
        # Store CV analysis result for recommendations
        cv_analysis_storage[cv_result.analysis_id] = cv_result.model_dump()
        
        skills_found = cv_result.skills
        total_skills = len(skills_found)
        logger.info(f"✅ CV analysis completed: {total_skills} skills found, analysis_id: {cv_result.analysis_id}")
        
        return cv_result
        
    except Exception as e:
        logger.error(f"❌ CV upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"CV upload failed: {str(e)}")

@app.post("/api/v1/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv_text(request: CVAnalysisRequest):
    """Analyze CV content directly - VERSION CORRIGÉE"""
    try:
        content_length = len(request.cv_content)
        logger.info(f"📄 CV text analysis request: {content_length} characters")
        
        # Analyse CV - nouvelle version
        cv_result = create_cv_analysis(request.cv_content)
        
        # Store CV analysis result for recommendations
        cv_analysis_storage[cv_result.analysis_id] = cv_result.model_dump()
        
        skills_found = cv_result.skills
        total_skills = len(skills_found)
        logger.info(f"✅ CV analysis completed: {total_skills} skills found, analysis_id: {cv_result.analysis_id}")
        
        return cv_result
        
    except Exception as e:
        logger.error(f"❌ CV analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"CV analysis failed: {str(e)}")

def create_cv_analysis(cv_text: str) -> CVAnalysisResponse:
    """Enhanced CV analysis with complete data structure"""
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Extract skills with better detection
    detected_skills = []
    skill_keywords = {
        'python': ['python', 'py', 'django', 'flask', 'fastapi'],
        'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'typescript'],
        'java': ['java', 'spring', 'hibernate'],
        'react': ['react', 'reactjs', 'react.js'],
        'vue': ['vue', 'vuejs', 'vue.js'],
        'angular': ['angular', 'angularjs'],
        'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'oracle'],
        'html': ['html', 'html5'],
        'css': ['css', 'css3', 'sass', 'scss', 'less'],
        'git': ['git', 'github', 'gitlab'],
        'docker': ['docker', 'containerization'],
        'aws': ['aws', 'amazon web services'],
        'azure': ['azure', 'microsoft azure'],
        'mongodb': ['mongodb', 'mongo'],
        'machine learning': ['machine learning', 'ml', 'ai', 'artificial intelligence'],
        'data science': ['data science', 'data analysis', 'analytics'],
        'kubernetes': ['kubernetes', 'k8s'],
        'tensorflow': ['tensorflow', 'tf'],
        'pytorch': ['pytorch', 'torch']
    }
    
    cv_lower = cv_text.lower()
    for main_skill, variations in skill_keywords.items():
        for variation in variations:
            if variation in cv_lower:
                detected_skills.append(main_skill.title())
                break
    
    # If no skills detected, provide fallback skills for demo
    if not detected_skills:
        detected_skills = ['Python', 'JavaScript', 'React', 'SQL', 'Git']
    
    # Extract job titles with better patterns
    detected_titles = []
    title_patterns = [
        r'(senior|junior|lead|principal)?\s*(software|web|full stack|backend|frontend|data)\s*(developer|engineer|analyst)',
        r'(project|product|technical)\s*manager',
        r'(devops|system)\s*engineer',
        r'(data|business)\s*analyst',
        r'(software|solution)\s*architect'
    ]
    
    lines = cv_text.split('\n')
    for line in lines[:20]:  # Check more lines
        line_clean = line.strip()
        if len(line_clean) > 5 and len(line_clean) < 100:
            for pattern in title_patterns:
                if re.search(pattern, line_clean, re.IGNORECASE):
                    detected_titles.append(line_clean)
                    break
    
    # Fallback titles if none detected
    if not detected_titles:
        detected_titles = ['Senior Full Stack Developer']
    
    # Calculate experience years with better estimation
    experience_indicators = re.findall(r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', cv_text, re.IGNORECASE)
    if experience_indicators:
        estimated_experience = max([int(x) for x in experience_indicators])
    else:
        estimated_experience = max(3, len(detected_skills) // 2)  # Minimum 3 years
    
    # Extract education
    education = []
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'diploma']
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in education_keywords) and len(line.strip()) > 10:
            education.append(line.strip())
            if len(education) >= 3:
                break
    
    # Extract personal information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(\+?\d{1,3}[-.\ s]?)\(?\d{3}\)?[-.\ s]?\d{3}[-.\ s]?\d{4}'
    
    email_match = re.search(email_pattern, cv_text)
    phone_match = re.search(phone_pattern, cv_text)
    
    # Extract name (first meaningful line that looks like a name)
    name = "Professional"
    first_lines = cv_text.split('\n')[:5]
    for line in first_lines:
        line_clean = line.strip()
        if (len(line_clean.split()) >= 2 and 
            len(line_clean) < 50 and 
            not '@' in line_clean and 
            not any(char.isdigit() for char in line_clean)):
            name = line_clean
            break
    
    # Create comprehensive response
    analysis_result = CVAnalysisResponse(
        analysis_id=analysis_id,
        skills=detected_skills[:10],
        experience_years=min(estimated_experience, 25),
        job_titles=detected_titles[:3],
        education=education[:3],
        summary=f"CV analysis found {len(detected_skills)} technical skills and {len(detected_titles)} relevant job titles. Professional with {min(estimated_experience, 25)} years of experience.",
        confidence_score=0.85,
        timestamp=datetime.now().isoformat(),
        sections={
            "summary": f"Experienced professional with {min(estimated_experience, 25)} years in technology",
            "technical_skills": detected_skills[:10],
            "professional_experience": detected_titles[:3],
            "education_background": education[:3]
        },
        personal_info={
            "name": name,
            "title": detected_titles[0] if detected_titles else "Professional",
            "years_experience": min(estimated_experience, 25)
        },
        contact_info={
            "email": email_match.group() if email_match else None,
            "phone": phone_match.group() if phone_match else None
        }
    )
    
    return analysis_result

# New endpoints for frontend integration
@app.get("/api/v1/dashboard/latest")
async def get_dashboard_latest():
    """Get latest dashboard data"""
    try:
        # Return sample dashboard data
        dashboard_data = {
            "recent_analyses": [],
            "job_match_count": 0,
            "skills_summary": [],
            "recommendations": [],
            "portfolio_status": "ready",
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info("📊 Dashboard data requested")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Dashboard data failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

@app.get("/api/v1/dashboard/{analysis_id}")
async def get_dashboard_by_analysis_id(analysis_id: str):
    """Get dashboard data for specific analysis ID"""
    try:
        # Return dashboard data for specific analysis
        dashboard_data = {
            "analysis_id": analysis_id,
            "recent_analyses": [
                {
                    "id": analysis_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "skills_count": 5,
                    "job_matches": 12
                }
            ],
            "job_match_count": 12,
            "skills_summary": ["Python", "JavaScript", "React", "SQL", "Git"],
            "recommendations": [
                {
                    "type": "skill",
                    "title": "Improve Python Skills",
                    "description": "Consider learning advanced Python frameworks",
                    "priority": "high"
                },
                {
                    "type": "job",
                    "title": "Full Stack Developer",
                    "description": "12 matching positions found",
                    "priority": "medium"
                }
            ],
            "portfolio_status": "ready",
            "last_updated": datetime.now().isoformat(),
            "analytics": {
                "profile_strength": 85,
                "market_demand": 92,
                "skill_relevance": 88
            }
        }
        
        logger.info(f"📊 Dashboard data requested for analysis_id: {analysis_id}")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Dashboard data failed for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

def format_enhanced_recommendations(enhanced_data):
    """
    Format enhanced recommendations to match expected API structure
    Ensures scores are properly formatted and structure is consistent
    """
    try:
        logger.info("🔧 Formatting enhanced recommendations...")
        
        # Convert lowercase keys to uppercase to match API expectations
        formatted_recommendations = {}
        
        key_mapping = {
            'immediate_actions': 'IMMEDIATE_ACTIONS',
            'skill_development': 'SKILL_DEVELOPMENT', 
            'project_suggestions': 'PROJECT_SUGGESTIONS',
            'certification_roadmap': 'CERTIFICATION_ROADMAP',
            'learning_resources': 'LEARNING_RESOURCES',
            'career_roadmap': 'CAREER_ROADMAP',
            'networking_opportunities': 'NETWORKING_OPPORTUNITIES',
            'timeline': 'TIMELINE'
        }
        
        for original_key, formatted_key in key_mapping.items():
            if original_key in enhanced_data:
                data = enhanced_data[original_key]
                
                # Format list recommendations (immediate_actions, skill_development, etc.)
                if isinstance(data, list):
                    formatted_list = []
                    for item in data:
                        if isinstance(item, dict):
                            formatted_item = {
                                'title': item.get('title', 'N/A'),
                                'description': item.get('description', ''),
                                'score': float(item.get('score', 0.0)),  # Ensure score is float
                                'priority': item.get('priority', 'medium'),
                                'estimated_time': item.get('estimated_time', ''),
                                'category': item.get('category', 'general')
                            }
                            # Add any additional fields that might exist
                            for key, value in item.items():
                                if key not in formatted_item:
                                    formatted_item[key] = value
                            formatted_list.append(formatted_item)
                        else:
                            formatted_list.append(item)
                    formatted_recommendations[formatted_key] = formatted_list
                else:
                    # Keep complex structures as-is
                    formatted_recommendations[formatted_key] = data
            else:
                # Provide empty structure if missing
                formatted_recommendations[formatted_key] = []
        
        logger.info("✅ Enhanced recommendations formatted successfully")
        return formatted_recommendations
        
    except Exception as e:
        logger.error(f"❌ Error formatting enhanced recommendations: {e}")
        # Return original data if formatting fails
        return enhanced_data

def generate_fallback_recommendations(skills_list):
    """
    Generate fallback recommendations when enhanced engine is not available
    """
    return {
        "IMMEDIATE_ACTIONS": [
            {
                "title": "Update Your Portfolio",
                "description": "Showcase your skills with recent projects",
                "score": 0.8,
                "priority": "high",
                "estimated_time": "1-2 weeks"
            }
        ],
        "SKILL_DEVELOPMENT": [
            {
                "title": "Advance Current Skills",
                "description": "Deepen your expertise in your strongest areas",
                "score": 0.75,
                "priority": "medium", 
                "estimated_time": "4-6 weeks"
            }
        ],
        "PROJECT_SUGGESTIONS": [
            {
                "title": "Personal Project",
                "description": "Build a project showcasing your skills",
                "score": 0.7,
                "estimated_time": "2-4 weeks"
            }
        ],
        "CERTIFICATION_ROADMAP": [],
        "LEARNING_RESOURCES": {
            "free_resources": [],
            "paid_courses": [],
            "books": [],
            "practice_platforms": []
        },
        "CAREER_ROADMAP": {
            "target_role": "Tech Professional",
            "timeline_months": 12,
            "milestones": [
                {
                    "month": 3,
                    "title": "Skill Enhancement",
                    "description": "Focus on improving core technical skills",
                    "focus_areas": ["Technical Skills", "Best Practices", "Code Quality"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Project Development",
                    "description": "Build and showcase meaningful projects",
                    "focus_areas": ["Portfolio Projects", "Problem Solving", "Implementation"],
                    "status": "upcoming"
                },
                {
                    "month": 9,
                    "title": "Industry Engagement",
                    "description": "Connect with the professional community",
                    "focus_areas": ["Networking", "Open Source", "Industry Knowledge"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Career Advancement",
                    "description": "Position yourself for next career step",
                    "focus_areas": ["Career Growth", "Leadership", "Advanced Skills"],
                    "status": "upcoming"
                }
            ],
            "current_level": "intermediate",
            "focus_domains": ["general"],
            "personalized": False
        },
        "NETWORKING_OPPORTUNITIES": [],
        "TIMELINE": {
            "month_1_3": [],
            "month_4_6": [],
            "month_7_9": [],
            "month_10_12": []
        }
    }

def sanitize_recommendations_data(data):
    """
    Sanitize recommendations data to ensure JSON serialization
    and prevent slice/hashable type errors
    """
    if isinstance(data, dict):
        return {k: sanitize_recommendations_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_recommendations_data(item) for item in data]
    elif hasattr(data, '__dict__'):  # Handle custom objects
        return sanitize_recommendations_data(data.__dict__)
    elif isinstance(data, slice):  # Handle slice objects
        return f"slice({data.start}, {data.stop}, {data.step})"
    elif hasattr(data, 'value'):  # Handle Enum objects
        return data.value
    elif callable(data):  # Skip functions
        return str(data)
    else:
        try:
            # Test if object is JSON serializable
            import json
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)

@app.get("/api/v1/recommendations/{analysis_id}")
async def get_recommendations(analysis_id: str):
    """
    Get personalized recommendations based on CV analysis ID.
    Generates skill, career, and learning recommendations.
    """
    try:
        logger.info(f"🎯 Recommendations requested for analysis_id: {analysis_id}")
        
        # Get CV data from storage
        cv_data = cv_analysis_storage.get(analysis_id)
        
        if not cv_data:
            logger.warning(f"Analysis ID {analysis_id} not found in storage.")
            raise HTTPException(status_code=404, detail=f"Analysis ID {analysis_id} not found. Please analyze your CV first.")

        # Convert CV data to skills format for recommendation engine
        skills_for_recommendations = []
        
        # Extract skills from CV data
        cv_skills = cv_data.get('skills', [])
        for skill in cv_skills:
            skills_for_recommendations.append({
                'skill': skill,
                'normalized_name': skill.lower(),
                'experience_level': 'intermediate',  # Default level
                'confidence': 0.8,
                'importance_score': 0.7,
                'category': 'technical'  # Default category
            })
        
        # Generate comprehensive recommendations using the best available engine
        if ML_MODE_ENABLED and ML_MODE_TYPE in ["lite", "full", "hybrid"]:
            # Use new ML lite/full/hybrid mode (most advanced)
            logger.info(f"🧠 Using ML mode: {ML_MODE_TYPE.upper()}")
            try:
                # Generate ML-enhanced recommendations
                safe_recommendations = await generate_ml_enhanced_recommendations(
                    skills_for_recommendations, cv_data
                )
                logger.info(f"✅ ML {ML_MODE_TYPE} recommendations generated successfully")
            except Exception as ml_error:
                logger.error(f"❌ ML {ML_MODE_TYPE} engine failed: {ml_error}")
                # Fallback to enhanced engine
                if USE_ENHANCED_ENGINE and recommendation_engine:
                    raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                        skills=skills_for_recommendations,
                        gap_analysis=None,
                        career_goals=None
                    )
                    safe_recommendations = format_enhanced_recommendations(raw_recommendations)
                    logger.info("✅ Enhanced recommendations as fallback")
                else:
                    safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
                    logger.warning("⚠️ Using basic fallback recommendations")
        elif ML_MODE_ENABLED and ml_recommendation_engine:
            # Use existing ML engine (legacy)
            logger.info("🧠 Using legacy ML recommendation engine (AI-powered)")
            try:
                raw_recommendations = await ml_recommendation_engine.generate_comprehensive_recommendations(
                    skills=skills_for_recommendations,
                    gap_analysis=None,
                    career_goals=None
                )
                safe_recommendations = format_enhanced_recommendations(raw_recommendations)
                logger.info("✅ Legacy ML recommendations generated successfully")
            except Exception as ml_error:
                logger.error(f"❌ Legacy ML engine failed: {ml_error}")
                # Fallback to enhanced engine
                if USE_ENHANCED_ENGINE and recommendation_engine:
                    raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                        skills=skills_for_recommendations,
                        gap_analysis=None,
                        career_goals=None
                    )
                    safe_recommendations = format_enhanced_recommendations(raw_recommendations)
                    logger.info("✅ Enhanced recommendations as fallback")
                else:
                    safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
                    logger.warning("⚠️ Using basic fallback recommendations")
                    
        elif USE_ENHANCED_ENGINE and recommendation_engine:
            # Use enhanced engine (rule-based)
            logger.info("⚡ Using enhanced recommendation engine (rule-based)")
            raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                skills=skills_for_recommendations,
                gap_analysis=None,
                career_goals=None
            )
            safe_recommendations = format_enhanced_recommendations(raw_recommendations)
            logger.info("✅ Enhanced recommendations formatted successfully")
        else:
            # Fallback to basic recommendations
            logger.info("🔧 Using fallback recommendation engine (basic)")
            safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
            logger.warning("⚠️ Using basic fallback recommendations")
        
        # Final sanitization to prevent serialization errors
        safe_recommendations = sanitize_recommendations_data(safe_recommendations)
        
        # Format response to match frontend expectations
        response = {
            "analysis_id": analysis_id,
            "recommendations": safe_recommendations,
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "skills_count": len(cv_skills),
                "experience_years": cv_data.get('experience_years', 0),
                "job_titles": cv_data.get('job_titles', []),
                "education": cv_data.get('education', [])
            },
            "global_confidence": 0.85,  # Add missing confidence score
            "engine_info": {
                "ml_mode_enabled": ML_MODE_ENABLED,
                "ml_mode_type": ML_MODE_TYPE,
                "enhanced_mode_enabled": USE_ENHANCED_ENGINE,
                "engine_type": f"ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("Enhanced" if USE_ENHANCED_ENGINE else "Basic"),
                "version": f"2.1.0-ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("2.0.0-Enhanced" if USE_ENHANCED_ENGINE else "2.0.0"),
                "capabilities": {
                    "ai_scoring": ML_MODE_ENABLED,
                    "semantic_similarity": ML_MODE_ENABLED,
                    "personalized_recommendations": True
                }
            }
        }
        
        logger.info(f"✅ Generated recommendations for analysis_id: {analysis_id}")
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"❌ Recommendations generation failed for {analysis_id}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a safe fallback response instead of crashing
        fallback_response = {
            "analysis_id": analysis_id,
            "recommendations": {
                "immediate_actions": [
                    {
                        "title": "Update LinkedIn Profile",
                        "description": "Optimize your professional profile",
                        "score": 0.75,
                        "priority": "high"
                    },
                    {
                        "title": "Create GitHub Portfolio",
                        "description": "Showcase your coding projects",
                        "score": 0.70,
                        "priority": "medium"
                    }
                ],
                "skill_development": [
                    {
                        "title": "Advanced Python Techniques",
                        "description": "Master advanced Python concepts",
                        "score": 0.80,
                        "category": "programming"
                    },
                    {
                        "title": "Cloud Architecture",
                        "description": "Learn cloud platform fundamentals",
                        "score": 0.75,
                        "category": "infrastructure"
                    }
                ],
                "career_roadmap": [
                    {
                        "title": "Senior Developer Path",
                        "description": "Roadmap to senior technical roles",
                        "score": 0.85,
                        "timeline": "12-18 months"
                    },
                    {
                        "title": "Tech Lead Transition",
                        "description": "Leadership and technical growth",
                        "score": 0.70,
                        "timeline": "18-24 months"
                    },
                    {
                        "title": "Architecture Specialist",
                        "description": "Deep technical architecture focus",
                        "score": 0.65,
                        "timeline": "24-36 months"
                    }
                ]
            },
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "skills_count": len(cv_data.get('skills', [])) if cv_data else 0,
                "experience_years": cv_data.get('experience_years', 0) if cv_data else 0,
                "job_titles": cv_data.get('job_titles', []) if cv_data else [],
                "education": cv_data.get('education', []) if cv_data else []
            },
            "global_confidence": 0.75,
            "error_handled": True,
            "original_error": str(e)
        }
        
        logger.info("✅ Returned fallback recommendations due to engine error")
        return fallback_response

# Modèle pour les recommandations POST
class RecommendationRequest(BaseModel):
    current_role: Optional[str] = "Developer"
    target_role: Optional[str] = "Senior Developer"
    skills: Optional[List[str]] = []
    experience_years: Optional[int] = 2
    industry: Optional[str] = "Tech"

@app.post("/api/v1/recommendations")
async def generate_recommendations_direct(request: RecommendationRequest):
    """
    Generate recommendations directly from profile data (for testing and direct use)
    """
    try:
        logger.info(f"🎯 Direct recommendations requested for {request.current_role} -> {request.target_role}")
        
        # Convert request to CV data format
        cv_data = {
            'job_titles': [request.current_role],
            'skills': request.skills,
            'experience_years': request.experience_years,
            'education': [],
            'industry': request.industry
        }
        
        # Convert to skills format for recommendation engine
        skills_for_recommendations = []
        for skill in request.skills:
            skills_for_recommendations.append({
                'skill': skill,
                'normalized_name': skill.lower(),
                'experience_level': 'intermediate',
                'confidence': 0.8,
                'importance_score': 0.7,
                'category': 'technical'
            })
        
        # Generate recommendations using the same logic as GET endpoint
        if ML_MODE_ENABLED and ML_MODE_TYPE in ["lite", "full"]:
            logger.info(f"🧠 Using ML mode: {ML_MODE_TYPE.upper()}")
            safe_recommendations = await generate_ml_enhanced_recommendations(
                skills_for_recommendations, cv_data
            )
        elif ML_MODE_ENABLED and ml_recommendation_engine:
            logger.info("🧠 Using legacy ML recommendation engine")
            raw_recommendations = await ml_recommendation_engine.generate_comprehensive_recommendations(
                skills=skills_for_recommendations,
                gap_analysis=None,
                career_goals=None
            )
            safe_recommendations = format_enhanced_recommendations(raw_recommendations)
        elif USE_ENHANCED_ENGINE and recommendation_engine:
            logger.info("⚡ Using enhanced recommendation engine")
            raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                skills=skills_for_recommendations,
                gap_analysis=None,
                career_goals=None
            )
            safe_recommendations = format_enhanced_recommendations(raw_recommendations)
        else:
            logger.info("🔧 Using fallback recommendation engine")
            safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
        
        # Final sanitization
        safe_recommendations = sanitize_recommendations_data(safe_recommendations)
        
        # Generate a unique analysis ID for this session
        import uuid
        analysis_id = str(uuid.uuid4())
        
        # Format response
        response = {
            "analysis_id": analysis_id,
            "recommendations": safe_recommendations,
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "current_role": request.current_role,
                "target_role": request.target_role,
                "skills_count": len(request.skills),
                "experience_years": request.experience_years,
                "industry": request.industry
            },
            "global_confidence": 0.90,
            "engine_info": {
                "ml_mode_enabled": ML_MODE_ENABLED,
                "ml_mode_type": ML_MODE_TYPE,
                "enhanced_mode_enabled": USE_ENHANCED_ENGINE,
                "engine_type": f"ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("Enhanced" if USE_ENHANCED_ENGINE else "Basic"),
                "version": f"2.1.0-ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("2.0.0-Enhanced" if USE_ENHANCED_ENGINE else "2.0.0"),
                "capabilities": {
                    "ai_scoring": ML_MODE_ENABLED,
                    "semantic_similarity": ML_MODE_ENABLED,
                    "personalized_recommendations": True
                }
            }
        }
        
        logger.info(f"✅ Generated direct recommendations for profile: {request.current_role}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Direct recommendations generation failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return safe fallback
        fallback_response = {
            "analysis_id": "fallback",
            "recommendations": generate_fallback_recommendations([]),
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "current_role": request.current_role if request else "Developer",
                "target_role": request.target_role if request else "Senior Developer",
                "skills_count": len(request.skills) if request else 0,
                "experience_years": request.experience_years if request else 0,
                "industry": request.industry if request else "Tech"
            },
            "global_confidence": 0.70,
            "error_handled": True,
            "original_error": str(e)
        }
        
        return fallback_response

class PortfolioGenerateRequest(BaseModel):
    cv_data: Dict[str, Any]
    template: Optional[str] = "modern"
    style: Optional[str] = "professional"

# Experience Translator Models (F7)
class ExperienceTranslationRequest(BaseModel):
    original_experience: str
    job_description: str
    style: Optional[str] = "professional"  # professional, technical, creative
    preserve_original: Optional[bool] = False

class ExperienceTranslationResponse(BaseModel):
    translation_id: str
    timestamp: str
    rewritten_text: str
    rewriting_style: str
    confidence_score: float
    keyword_matches: Dict[str, int]
    suggestions: List[str]
    enhancements_made: List[str]
    version_comparison: Dict[str, Any]
    export_formats: Dict[str, str]

@app.post("/api/v1/generate-portfolio")
async def generate_portfolio(request: PortfolioGenerateRequest):
    """Generate portfolio from CV data"""
    try:
        logger.info("🎨 Portfolio generation requested")
        
        # Extract data from request
        cv_data = request.cv_data
        template = request.template or "modern"
        
        # Generate portfolio HTML
        portfolio_html = generate_portfolio_html(cv_data, template)
        
        # Return response
        response = {
            "portfolio_id": str(uuid.uuid4()),
            "html_content": portfolio_html,
            "template": template,
            "generated_at": datetime.now().isoformat(),
            "status": "success"
        }
        
        logger.info("✅ Portfolio generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Portfolio generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio generation failed: {str(e)}")

def generate_portfolio_html(cv_data: Dict[str, Any], template: str) -> str:
    """Generate HTML portfolio from CV data"""
    
    # Extract basic info
    name = cv_data.get('personal_info', {}).get('name', 'Professional')
    title = cv_data.get('job_titles', ['Developer'])[0] if cv_data.get('job_titles') else 'Professional'
    skills = cv_data.get('skills', [])
    experience = cv_data.get('experience_years', 0)
    
    # Generate HTML
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{name} - Portfolio</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background: #f4f4f4;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #333;
                padding-bottom: 20px;
                margin-bottom: 20px;
            }}
            .skills {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .skill {{
                background: #007bff;
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{name}</h1>
                <h2>{title}</h2>
                <p>Experience: {experience} years</p>
            </div>
            
            <div class="section">
                <h3>Skills</h3>
                <div class="skills">
                    {''.join([f'<span class="skill">{skill}</span>' for skill in skills[:10]])}
                </div>
            </div>
            
            <div class="section">
                <h3>Summary</h3>
                <p>{cv_data.get('summary', 'Professional with expertise in various technologies.')}</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return html_content

# Experience Translator Endpoints (F7)
@app.post("/api/v1/experience/translate", response_model=ExperienceTranslationResponse)
async def translate_experience(request: ExperienceTranslationRequest):
    """Translate and enhance professional experience for specific job requirements"""
    try:
        logger.info("🔄 Experience translation requested")
        
        # Check if Experience Translator is available
        if not translate_experience_api:
            raise HTTPException(status_code=503, detail="Experience Translator service is not available")
        
        # Validate inputs
        if not request.original_experience.strip():
            raise HTTPException(status_code=400, detail="Original experience text is required")
        
        if not request.job_description.strip():
            raise HTTPException(status_code=400, detail="Job description is required")
        
        # Perform experience translation
        translation_result = translate_experience_api(
            original_experience=request.original_experience,
            job_description=request.job_description,
            style=request.style
        )
        
        # Extract key information for response
        rewritten_experience = translation_result["rewritten_experience"]
        
        response = ExperienceTranslationResponse(
            translation_id=translation_result["translation_id"],
            timestamp=translation_result["timestamp"],
            rewritten_text=rewritten_experience["text"],
            rewriting_style=rewritten_experience["style"],
            confidence_score=rewritten_experience["confidence_score"],
            keyword_matches=rewritten_experience["keyword_matches"],
            suggestions=rewritten_experience["improvement_suggestions"],
            enhancements_made=rewritten_experience["enhancements_made"],
            version_comparison=rewritten_experience["version_comparison"],
            export_formats=rewritten_experience["export_formats"]
        )
        
        logger.info(f"✅ Experience translation completed. Confidence: {response.confidence_score:.2f}")
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"❌ Experience translation failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Experience translation failed: {str(e)}")

@app.get("/api/v1/experience/styles")
async def get_translation_styles():
    """Get available translation styles and their descriptions"""
    try:
        if not ExperienceTranslator:
            raise HTTPException(status_code=503, detail="Experience Translator service is not available")
        
        styles_info = {
            "available_styles": [
                {
                    "style": "professional",
                    "description": "Formal, achievement-focused language with bullet-point structure",
                    "best_for": "Corporate environments, formal applications"
                },
                {
                    "style": "technical", 
                    "description": "Precise, skills-focused language highlighting technical competencies",
                    "best_for": "Technical roles, engineering positions"
                },
                {
                    "style": "creative",
                    "description": "Engaging, innovation-focused narrative highlighting impact",
                    "best_for": "Creative roles, startup environments"
                }
            ],
            "default_style": "professional",
            "supported_formats": ["text", "markdown", "html", "json"]
        }
        
        logger.info("📋 Translation styles information requested")
        return styles_info
        
    except Exception as e:
        logger.error(f"❌ Failed to get translation styles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get styles: {str(e)}")

@app.get("/api/v1/experience/analysis/{translation_id}")
async def get_translation_analysis(translation_id: str):
    """Get detailed analysis of a previous translation"""
    try:
        # In a real implementation, this would fetch from a database
        # For now, return a placeholder response
        analysis_data = {
            "translation_id": translation_id,
            "analysis_type": "detailed",
            "status": "placeholder",
            "message": "Detailed analysis storage not yet implemented"
        }
        
        logger.info(f"📊 Translation analysis requested for ID: {translation_id}")
        return analysis_data
        
    except Exception as e:
        logger.error(f"❌ Failed to get translation analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "SkillSync Multi-API Backend",
        "features": {
            "experience_translator": translate_experience_api is not None,
            "job_search": True,
            "cv_analysis": True,
            "recommendations": True,
            "portfolio_generation": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting SkillSync Multi-API Backend Server...")
    logger.info("📊 Endpoints available:")
    logger.info("   • GET/POST /api/v1/jobs/search - Search jobs across all APIs")
    logger.info("   • GET      /api/v1/jobs/status - Check API status")
    logger.info("   • POST     /api/v1/upload-cv - Upload and analyze CV")
    logger.info("   • POST     /api/v1/analyze-cv - Analyze CV text")
    logger.info("   • GET      /api/v1/dashboard/latest - Get latest dashboard data")
    logger.info("   • GET      /api/v1/dashboard/{analysis_id} - Get dashboard by analysis ID")
    logger.info("   • GET      /api/v1/recommendations/{analysis_id} - Get personalized recommendations")
    logger.info("   • POST     /api/v1/generate-portfolio - Generate portfolio")
    logger.info("   • POST     /api/v1/experience/translate - Translate experience (F7)")
    logger.info("   • GET      /api/v1/experience/styles - Get translation styles (F7)")
    logger.info("   • GET      /api/v1/experience/analysis/{translation_id} - Get translation analysis (F7)")
    logger.info("   • GET      /health - Health check")
    
    uvicorn.run(
        "main_simple_for_frontend:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )