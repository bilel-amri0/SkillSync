# 🚀 Guide d'Intégration - Système de Recommandations Multicritères SkillSync v2.0

## 🎯 Vue d'Ensemble

Le nouveau système de recommandations SkillSync v2.0 transforme votre plateforme en un conseiller de carrière IA complet, capable de recommander :

- **🗺️ Roadmaps de Carrière** - Chemins personnalisés de progression professionnelle
- **🏅 Certifications** - Certifications adaptées au profil et aux objectifs
- **💡 Compétences** - Skills prioritaires à développer
- **🛠️ Projets Pratiques** - Projets pour appliquer et développer les compétences
- **💼 Opportunités Emploi** - Jobs correspondant au profil (existant amélioré)

## ⚡ Installation et Configuration

### 1. Installation des Dépendances

```bash
# Navigation vers le backend
cd SkillSync_Project/backend

# Installation des nouvelles dépendances (déjà dans requirements.txt)
pip install -r requirements.txt

# Vérification que tout est installé
python -c "import recommendation_system; print('Système de recommandations chargé avec succès!')"
```

### 2. Configuration du Système Principal

```python
# Dans votre main.py existant, ajoutez :
from recommendation_system.api_v2 import router as recommendations_v2_router

# Ajout du nouveau router
app.include_router(
    recommendations_v2_router,
    prefix="/api",
    tags=["Recommendations v2"]
)

print("🚀 Système de recommandations v2.0 activé!")
```

### 3. Test de Base

```bash
# Exécution du script de démonstration
python demo_comprehensive_recommendations.py

# Démarrage du serveur avec les nouvelles APIs
python main.py
```

## 🔌 Nouveaux Endpoints API

### Endpoint Principal - Recommandations Complètes

```http
POST /api/v2/recommendations/comprehensive
Content-Type: application/json

{
  "user_id": "user_123",
  "cv_data": {
    "summary": "Développeur backend avec 3 ans d'expérience",
    "skills_text": "Python, FastAPI, PostgreSQL, Docker"
  },
  "current_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
  "experience_years": 3,
  "current_role": "Backend Developer",
  "industry": "FinTech",
  "level": "mid",
  "career_goals": ["Senior Backend", "Full-Stack", "Tech Lead"],
  "learning_preferences": {
    "style": ["hands-on", "project-based"],
    "difficulty_preference": "intermediate"
  },
  "time_availability": "8-12h/week"
}
```

**Réponse:**
```json
{
  "user_profile": { /* profil utilisateur */ },
  "recommendations": {
    "roadmaps": [
      {
        "id": "roadmap_rec_fullstack_web",
        "title": "Développeur Full-Stack Web",
        "description": "Roadmap de carrière web_development - 5-8 mois",
        "scores": {
          "base_score": 0.89,
          "unified": 0.87,
          "personalized": 0.91
        },
        "confidence": 0.88,
        "roadmap": {
          "id": "roadmap_fullstack_web",
          "title": "Développeur Full-Stack Web",
          "domain": "web_development",
          "estimated_duration": "5-8 mois",
          "steps": [ /* étapes détaillées */ ]
        },
        "match_reason": "Vous maîtrisez déjà 2 compétences clés; Aligné avec votre objectif: Full-Stack",
        "progression_fit": 0.9,
        "next_steps": [
          "Étape 1: Frontend Fundamentals",
          "Focus: HTML5, CSS3",
          "Durée estimée: 6-8 semaines"
        ]
      }
    ],
    "certifications": [ /* certifications recommandées */ ],
    "skills": [ /* compétences à développer */ ],
    "projects": [ /* projets pratiques */ ]
  },
  "global_explanation": {
    "summary": "Nous avons généré 18 recommandations personnalisées pour votre profil.",
    "approach": "Nos recommandations combinent l'IA avancée et votre profil unique.",
    "next_steps": [
      "🎯 Priorité : Développer 'React' pour maximiser vos opportunités",
      "🛠️ Action : Démarrer le projet 'Plateforme E-commerce Full-Stack' pour appliquer vos compétences"
    ]
  },
  "confidence": 0.87,
  "generated_at": "2025-01-17T12:22:05Z"
}
```

### Endpoints Spécialisés

```http
# Roadmaps uniquement
POST /api/v2/recommendations/roadmaps

# Certifications avec filtres
POST /api/v2/recommendations/certifications?budget_range=premium

# Compétences par priorité
POST /api/v2/recommendations/skills?priority_filter=high

# Projets par difficulté
POST /api/v2/recommendations/projects?difficulty_level=intermediate

# Feedback utilisateur
POST /api/v2/recommendations/feedback

# Analytics utilisateur
GET /api/v2/recommendations/analytics/{user_id}?time_period=30d

# Santé du système
GET /api/v2/recommendations/health
```

## 🔧 Intégration Frontend

### 1. Service de Recommandations

```javascript
// services/recommendationsV2Service.js
class RecommendationsV2Service {
  constructor(baseURL = '/api/v2/recommendations') {
    this.baseURL = baseURL;
  }

  async getComprehensiveRecommendations(userProfile, preferences = null) {
    const response = await fetch(`${this.baseURL}/comprehensive`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...userProfile,
        preferences
      })
    });
    
    if (!response.ok) {
      throw new Error(`Erreur ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  }

  async getRoadmaps(userProfile, options = {}) {
    const queryParams = new URLSearchParams(options).toString();
    const response = await fetch(`${this.baseURL}/roadmaps?${queryParams}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userProfile)
    });
    return await response.json();
  }

  async submitFeedback(feedback) {
    const response = await fetch(`${this.baseURL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedback)
    });
    return await response.json();
  }

  async getAnalytics(userId, timePeriod = '30d') {
    const response = await fetch(`${this.baseURL}/analytics/${userId}?time_period=${timePeriod}`);
    return await response.json();
  }
}

export default new RecommendationsV2Service();
```

### 2. Composant React Principal

```jsx
// components/ComprehensiveRecommendations.jsx
import React, { useState, useEffect } from 'react';
import recommendationsV2Service from '../services/recommendationsV2Service';

function ComprehensiveRecommendations({ userProfile }) {
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    generateRecommendations();
  }, [userProfile]);

  const generateRecommendations = async () => {
    setLoading(true);
    try {
      const result = await recommendationsV2Service.getComprehensiveRecommendations(
        userProfile
      );
      setRecommendations(result);
    } catch (error) {
      console.error('Erreur génération recommandations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (recommendationId, feedbackType) => {
    try {
      await recommendationsV2Service.submitFeedback({
        user_id: userProfile.user_id,
        recommendation_id: recommendationId,
        feedback_type: feedbackType,
        created_at: new Date().toISOString()
      });
      // Mise à jour de l'UI
    } catch (error) {
      console.error('Erreur feedback:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-lg">Génération des recommandations...</span>
      </div>
    );
  }

  if (!recommendations) {
    return <div>Aucune recommandation disponible</div>;
  }

  return (
    <div className="comprehensive-recommendations">
      {/* Header avec résumé */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-6 rounded-lg mb-6">
        <h2 className="text-2xl font-bold mb-2">
          🚀 Vos Recommandations Personnalisées
        </h2>
        <p className="text-blue-100">
          {recommendations.global_explanation?.summary}
        </p>
        <div className="mt-4 flex items-center">
          <span className="bg-white bg-opacity-20 px-3 py-1 rounded-full text-sm">
            Confiance: {(recommendations.confidence * 100).toFixed(0)}%
          </span>
          <span className="ml-3 text-sm opacity-75">
            Généré le {new Date(recommendations.generated_at).toLocaleDateString()}
          </span>
        </div>
      </div>

      {/* Navigation par onglets */}
      <div className="flex border-b border-gray-200 mb-6">
        {[
          { id: 'overview', label: '🎯 Vue d\'ensemble', count: null },
          { id: 'roadmaps', label: '🗺️ Roadmaps', count: recommendations.recommendations.roadmaps?.length },
          { id: 'certifications', label: '🏅 Certifications', count: recommendations.recommendations.certifications?.length },
          { id: 'skills', label: '💡 Compétences', count: recommendations.recommendations.skills?.length },
          { id: 'projects', label: '🛠️ Projets', count: recommendations.recommendations.projects?.length }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 border-b-2 font-medium text-sm ${
              activeTab === tab.id
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.label}
            {tab.count && (
              <span className="ml-2 bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full">
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Contenu des onglets */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <OverviewTab 
            recommendations={recommendations} 
            onFeedback={handleFeedback}
          />
        )}
        {activeTab === 'roadmaps' && (
          <RoadmapsTab 
            roadmaps={recommendations.recommendations.roadmaps || []} 
            onFeedback={handleFeedback}
          />
        )}
        {activeTab === 'certifications' && (
          <CertificationsTab 
            certifications={recommendations.recommendations.certifications || []} 
            onFeedback={handleFeedback}
          />
        )}
        {activeTab === 'skills' && (
          <SkillsTab 
            skills={recommendations.recommendations.skills || []} 
            onFeedback={handleFeedback}
          />
        )}
        {activeTab === 'projects' && (
          <ProjectsTab 
            projects={recommendations.recommendations.projects || []} 
            onFeedback={handleFeedback}
          />
        )}
      </div>
    </div>
  );
}

export default ComprehensiveRecommendations;
```

### 3. Composants Spécialisés par Type

```jsx
// components/RoadmapCard.jsx
function RoadmapCard({ roadmap, onFeedback }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            {roadmap.title}
          </h3>
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <span>🕰️ {roadmap.roadmap.estimated_duration}</span>
            <span>📈 {roadmap.roadmap.steps.length} étapes</span>
            <span>🎯 Score: {(roadmap.scores.unified * 100).toFixed(0)}%</span>
          </div>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => onFeedback(roadmap.id, 'like')}
            className="p-2 text-green-600 hover:bg-green-50 rounded"
          >
            👍
          </button>
          <button
            onClick={() => onFeedback(roadmap.id, 'dislike')}
            className="p-2 text-red-600 hover:bg-red-50 rounded"
          >
            👎
          </button>
        </div>
      </div>

      <p className="text-gray-700 mb-4">{roadmap.match_reason}</p>

      <div className="space-y-2">
        <h4 className="font-medium text-gray-900">Prochaines étapes:</h4>
        {roadmap.next_steps.map((step, index) => (
          <div key={index} className="flex items-center text-sm text-gray-600">
            <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
            {step}
          </div>
        ))}
      </div>

      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-4 text-blue-600 hover:text-blue-800 text-sm font-medium"
      >
        {expanded ? 'Moins de détails' : 'Plus de détails'} →
      </button>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <h4 className="font-medium mb-3">Étapes détaillées du roadmap:</h4>
          <div className="space-y-3">
            {roadmap.roadmap.steps.map((step, index) => (
              <div key={index} className="bg-gray-50 p-3 rounded">
                <div className="flex items-center mb-2">
                  <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded">
                    Étape {step.step_number}
                  </span>
                  <span className="ml-2 font-medium">{step.title}</span>
                </div>
                <p className="text-sm text-gray-600 mb-2">{step.description}</p>
                <div className="text-xs text-gray-500">
                  <span>Durée: {step.duration}</span>
                  <span className="ml-4">Compétences: {step.skills_to_learn.join(', ')}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

## 📈 Monitoring et Analytics

### 1. Dashboard Analytics

```jsx
// components/RecommendationsAnalytics.jsx
function RecommendationsAnalytics({ userId }) {
  const [analytics, setAnalytics] = useState(null);
  const [timePeriod, setTimePeriod] = useState('30d');

  useEffect(() => {
    loadAnalytics();
  }, [userId, timePeriod]);

  const loadAnalytics = async () => {
    try {
      const data = await recommendationsV2Service.getAnalytics(userId, timePeriod);
      setAnalytics(data);
    } catch (error) {
      console.error('Erreur chargement analytics:', error);
    }
  };

  if (!analytics) return <div>Chargement analytics...</div>;

  return (
    <div className="analytics-dashboard">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Carte Total Recommandations */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Recommandations Générées</h3>
          <div className="text-3xl font-bold text-blue-600">
            {analytics.recommendations_generated.total}
          </div>
          <div className="text-sm text-gray-600">Dernières {timePeriod}</div>
        </div>

        {/* Carte Taux d'Adoption */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Taux d'Adoption Moyen</h3>
          <div className="text-3xl font-bold text-green-600">
            {(Object.values(analytics.adoption_rates).reduce((a, b) => a + b, 0) / Object.keys(analytics.adoption_rates).length * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-gray-600">Recommandations suivies</div>
        </div>

        {/* Carte Progression */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Compétences Apprises</h3>
          <div className="text-3xl font-bold text-purple-600">
            {analytics.skill_progression.skills_learned}
          </div>
          <div className="text-sm text-gray-600">Nouvelles compétences</div>
        </div>
      </div>

      {/* Graphiques détaillés */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Répartition par type */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Répartition par Type</h3>
          {Object.entries(analytics.recommendations_generated).map(([type, count]) => {
            if (type === 'total') return null;
            const percentage = (count / analytics.recommendations_generated.total * 100).toFixed(1);
            return (
              <div key={type} className="mb-3">
                <div className="flex justify-between items-center mb-1">
                  <span className="capitalize">{type}</span>
                  <span className="text-sm text-gray-600">{count} ({percentage}%)</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Taux d'adoption par type */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Taux d'Adoption par Type</h3>
          {Object.entries(analytics.adoption_rates).map(([type, rate]) => (
            <div key={type} className="mb-3">
              <div className="flex justify-between items-center mb-1">
                <span className="capitalize">{type}</span>
                <span className="text-sm text-gray-600">{(rate * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    rate > 0.7 ? 'bg-green-500' : rate > 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${rate * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

## 🔊 Feedback et Amélioration Continue

### 1. Système de Feedback Intégré

```jsx
// components/FeedbackSystem.jsx
function FeedbackSystem({ recommendationId, onFeedbackSubmitted }) {
  const [feedbackType, setFeedbackType] = useState('');
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const submitFeedback = async () => {
    try {
      await recommendationsV2Service.submitFeedback({
        recommendation_id: recommendationId,
        feedback_type: feedbackType,
        rating: rating,
        comment: comment
      });
      setSubmitted(true);
      onFeedbackSubmitted?.();
    } catch (error) {
      console.error('Erreur soumission feedback:', error);
    }
  };

  if (submitted) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center">
          <span className="text-green-600 mr-2">✓</span>
          <span className="text-green-800">Merci pour votre feedback!</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
      <h4 className="font-medium mb-3">Cette recommandation vous aide-t-elle?</h4>
      
      {/* Boutons de feedback rapide */}
      <div className="flex space-x-2 mb-4">
        {[
          { type: 'like', emoji: '👍', label: 'Utile' },
          { type: 'applied', emoji: '✅', label: 'Je commence' },
          { type: 'completed', emoji: '🎆', label: 'Terminé' },
          { type: 'not_relevant', emoji: '❌', label: 'Pas pertinent' }
        ].map(option => (
          <button
            key={option.type}
            onClick={() => {
              setFeedbackType(option.type);
              submitFeedback();
            }}
            className="flex items-center px-3 py-2 bg-white border border-gray-300 rounded hover:bg-gray-50 text-sm"
          >
            <span className="mr-1">{option.emoji}</span>
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
```

## 🛠️ Configuration Avancée

### 1. Personnalisation des Poids de Scoring

```python
# Dans votre configuration personnalisée
from recommendation_system.core.scoring_engine import ScoringEngine

# Personnalisation des poids selon votre domaine
custom_weights = {
    'skill_match': 0.30,        # Augmenté pour plus de focus compétences
    'experience_fit': 0.25,     # Augmenté pour entreprise senior-friendly
    'career_alignment': 0.20,
    'market_demand': 0.15,
    'learning_feasibility': 0.05,  # Réduit si utilisateurs motivés
    'user_preferences': 0.05
}

# Application des poids personnalisés
scoring_engine = ScoringEngine()
scoring_engine.weights = custom_weights
```

### 2. Ajout de Données Personnalisées

```python
# Extension de la base de connaissances
from recommendation_system.recommenders.roadmap_recommender import RoadmapRecommender
from recommendation_system.models import CareerRoadmap, RoadmapStep, DifficultyLevel

# Ajout de roadmaps spécifiques à votre entreprise
custom_roadmap = CareerRoadmap(
    id="custom_enterprise_architect",
    title="Architecte Solutions Entreprise",
    domain="enterprise_architecture",
    difficulty=DifficultyLevel.EXPERT,
    steps=[
        RoadmapStep(
            step_number=1,
            title="Maîtrise Architecture Patterns",
            description="Patterns d'architecture enterprise",
            duration="8-10 semaines",
            skills_to_learn=["Enterprise Patterns", "Microservices", "Event Sourcing"]
        )
    ],
    # ... autres propriétés
)

# Ajout à la base de connaissances
roadmap_recommender = RoadmapRecommender()
roadmap_recommender.roadmaps_db.append(custom_roadmap)
```

## 📊 Métriques et KPIs

### Métriques de Performance

```python
# Métriques automatiquement trackées
metrics = {
    "recommendation_generation_time": "< 2 secondes",
    "user_satisfaction_rate": "> 85%",
    "recommendation_adoption_rate": "> 60%",
    "skill_progression_completion": "> 40%",
    "certification_success_rate": "> 70%",
    "project_completion_rate": "> 55%"
}
```

### Dashboard de Monitoring

```http
GET /api/v2/recommendations/health
# Retourne l'état de santé complet du système

GET /api/v2/recommendations/analytics/global
# Analytics globales de la plateforme
```

## 🚀 Déploiement en Production

### 1. Variables d'Environnement

```bash
# Configuration production
export RECOMMENDATION_SYSTEM_VERSION="2.0.0"
export ML_MODELS_PATH="/app/models"
export RECOMMENDATION_CACHE_TTL="3600"  # 1 heure
export FEEDBACK_PROCESSING_ENABLED="true"
export ANALYTICS_TRACKING_ENABLED="true"
```

### 2. Optimisations Performance

```python
# Cache Redis pour les recommandations
import redis
from functools import wraps

def cache_recommendations(ttl=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Logique de cache Redis
            cache_key = f"rec:{hash(str(args))}"
            cached = redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### 3. Monitoring Production

```python
# Logging et monitoring
import logging
from prometheus_client import Counter, Histogram

# Métriques Prometheus
RECOMMENDATION_REQUESTS = Counter('recommendations_total', 'Total recommendations generated')
RECOMMENDATION_LATENCY = Histogram('recommendation_duration_seconds', 'Time spent generating recommendations')

# Dans votre endpoint
@RECOMMENDATION_LATENCY.time()
async def get_comprehensive_recommendations(...):
    RECOMMENDATION_REQUESTS.inc()
    # ... logique existante
```

## 🎆 Félicitations!

Votre système SkillSync v2.0 est maintenant prêt avec :

✅ **Recommandations multicritères** (roadmaps, certifications, compétences, projets)  
✅ **IA avancée** avec scoring unifié et personnalisation  
✅ **API complète** pour intégration frontend  
✅ **Monitoring et analytics** temps réel  
✅ **Feedback et amélioration continue**  
✅ **Scalabilité et performance** optimisées  

Votre plateforme est désormais un **conseiller de carrière IA complet** ! 🚀