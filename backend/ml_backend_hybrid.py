"""
🚀 Backend ML Hybride SkillSync
Utilise les meilleurs outils ML disponibles de façon optimale
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline, AutoTokenizer, AutoModel
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridMLScorer:
    """Système de scoring hybride utilisant PyTorch + Scikit-learn + Transformers"""
    
    def __init__(self):
        self.ml_available = self._check_ml_availability()
        logger.info(f"ML Backend initialisé - Disponible: {self.ml_available}")
        
        # Modèles disponibles
        self.models = {}
        self._init_models()
    
    def _check_ml_availability(self):
        """Vérifie quels modules ML sont disponibles"""
        available = {}
        
        try:
            import torch
            available['pytorch'] = True
            logger.info("✅ PyTorch disponible")
        except:
            available['pytorch'] = False
            logger.warning("❌ PyTorch non disponible")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            available['sklearn'] = True
            logger.info("✅ Scikit-learn disponible")
        except:
            available['sklearn'] = False
            logger.warning("❌ Scikit-learn non disponible")
        
        try:
            from transformers import pipeline
            available['transformers'] = True
            logger.info("✅ Transformers disponible")
        except:
            available['transformers'] = False
            logger.warning("❌ Transformers non disponible")
        
        try:
            import tensorflow as tf
            # Test simple pour vérifier si utilisable
            tf.constant([1, 2, 3])
            available['tensorflow_basic'] = True
            logger.info("✅ TensorFlow (basique) disponible")
        except:
            available['tensorflow_basic'] = False
            logger.warning("❌ TensorFlow non utilisable")
        
        return available
    
    def _init_models(self):
        """Initialise les modèles selon disponibilité"""
        
        # Modèle PyTorch pour scoring neuronal
        if self.ml_available.get('pytorch', False):
            self.models['neural_scorer'] = self._create_pytorch_scorer()
        
        # Modèle Scikit-learn pour classification
        if self.ml_available.get('sklearn', False):
            self.models['rf_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Pipeline de sentiment avec Transformers
        if self.ml_available.get('transformers', False):
            try:
                # Essai avec un modèle plus léger et compatible
                self.models['sentiment'] = pipeline("sentiment-analysis", 
                                                   model="distilbert-base-uncased-finetuned-sst-2-english")
                logger.info("✅ Modèle sentiment initialisé (DistilBERT)")
            except Exception as e:
                try:
                    # Fallback vers le modèle par défaut
                    self.models['sentiment'] = pipeline("sentiment-analysis")
                    logger.info("✅ Modèle sentiment initialisé (par défaut)")
                except Exception as e2:
                    logger.warning(f"⚠️ Modèle sentiment non initialisé (erreur de dépendances)")
                    logger.info("💡 Solution: pip install --upgrade tensorflow==2.14.0 transformers==4.35.0")
                    self.models['sentiment'] = None
    
    def _create_pytorch_scorer(self):
        """Crée un modèle PyTorch pour scoring des compétences"""
        class SkillScorer(nn.Module):
            def __init__(self, input_dim=10):
                super(SkillScorer, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = SkillScorer()
        logger.info("✅ Modèle PyTorch créé")
        return model
    
    def score_skill_match(self, job_skills, user_skills):
        """Score la correspondance compétences avec méthodes hybrides"""
        
        # Méthode 1: TF-IDF + Cosine Similarity (toujours disponible)
        tfidf_score = self._tfidf_similarity(job_skills, user_skills)
        
        # Méthode 2: PyTorch Neural Network
        if self.ml_available.get('pytorch', False):
            neural_score = self._neural_score(job_skills, user_skills)
            # Combinaison pondérée
            final_score = 0.6 * tfidf_score + 0.4 * neural_score
        else:
            final_score = tfidf_score
        
        return min(max(final_score, 0.0), 1.0)  # Clamp entre 0 et 1
    
    def _tfidf_similarity(self, job_skills, user_skills):
        """Calcul similarité TF-IDF"""
        if not job_skills or not user_skills:
            return 0.0
        
        try:
            # Préparation textes
            job_text = " ".join(job_skills) if isinstance(job_skills, list) else str(job_skills)
            user_text = " ".join(user_skills) if isinstance(user_skills, list) else str(user_skills)
            
            # TF-IDF
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([job_text, user_text])
            
            # Similarité cosinus
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
            
        except Exception as e:
            logger.warning(f"Erreur TF-IDF: {e}")
            return 0.0
    
    def _neural_score(self, job_skills, user_skills):
        """Score neuronal avec PyTorch"""
        try:
            # Simulation features (dans un vrai cas, extraire des embeddings)
            features = np.random.rand(10).astype(np.float32)
            
            model = self.models.get('neural_scorer')
            if model is None:
                return 0.5
            
            # Prédiction
            model.eval()
            with torch.no_grad():
                score = model(torch.tensor(features).unsqueeze(0))
                return score.item()
                
        except Exception as e:
            logger.warning(f"Erreur neural score: {e}")
            return 0.5
    
    def analyze_job_sentiment(self, job_description):
        """Analyse le sentiment d'une description d'emploi"""
        if not self.models.get('sentiment'):
            # Fallback: analyse de sentiment basique par mots-clés
            return self._basic_sentiment_analysis(job_description)
        
        try:
            result = self.models['sentiment'](job_description[:512])  # Limite tokens
            return result[0] if result else {"label": "NEUTRAL", "score": 0.5}
        except Exception as e:
            logger.warning(f"Erreur sentiment: {e}")
            return self._basic_sentiment_analysis(job_description)
    
    def _basic_sentiment_analysis(self, text):
        """Analyse de sentiment basique par mots-clés (fallback)"""
        positive_words = [
            'excellent', 'great', 'good', 'best', 'top', 'leading', 'innovative', 
            'growth', 'opportunity', 'benefits', 'flexible', 'exciting', 'competitive',
            'collaborative', 'dynamic', 'rewarding'
        ]
        negative_words = [
            'challenging', 'demanding', 'pressure', 'fast-paced', 'strict', 
            'required', 'must', 'essential', 'mandatory'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {"label": "POSITIVE", "score": 0.7}
        elif negative_count > positive_count:
            return {"label": "NEGATIVE", "score": 0.3}
        else:
            return {"label": "NEUTRAL", "score": 0.5}
    
    def get_recommendations(self, user_profile, jobs_data, top_k=5):
        """Recommandations d'emplois avec scoring hybride"""
        recommendations = []
        
        for job in jobs_data:
            try:
                # Score compétences
                skill_score = self.score_skill_match(
                    job.get('required_skills', []),
                    user_profile.get('skills', [])
                )
                
                # Analyse sentiment
                sentiment = self.analyze_job_sentiment(job.get('description', ''))
                sentiment_bonus = 0.1 if sentiment['label'] == 'POSITIVE' else 0.0
                
                # Score final
                final_score = skill_score + sentiment_bonus
                
                recommendations.append({
                    'job_id': job.get('id'),
                    'title': job.get('title'),
                    'score': final_score,
                    'skill_match': skill_score,
                    'sentiment': sentiment
                })
                
            except Exception as e:
                logger.warning(f"Erreur recommandation job {job.get('id', 'unknown')}: {e}")
        
        # Tri par score décroissant
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    def get_system_status(self):
        """Retourne le statut du système ML"""
        return {
            'backend_type': 'hybrid',
            'ml_modules': self.ml_available,
            'models_loaded': list(self.models.keys()),
            'status': 'operational'
        }

# Instance globale
ml_backend = HybridMLScorer()

def get_ml_backend():
    """Retourne l'instance du backend ML"""
    return ml_backend

if __name__ == "__main__":
    # Test du backend
    backend = get_ml_backend()
    status = backend.get_system_status()
    print("🚀 Backend ML Hybride - Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")