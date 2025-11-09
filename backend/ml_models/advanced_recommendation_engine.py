"""
Advanced ML-Powered Recommendation Engine
Integrates BERT, Sentence-Transformers, and Neural Scoring
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# Import ML components
from .skills_extractor import SkillsExtractorModel
from .similarity_engine import SimilarityEngine
from .neural_scorer import NeuralScorer

logger = logging.getLogger(__name__)

class AdvancedRecommendationEngine:
    """
    Advanced ML-powered recommendation engine combining multiple AI models
    """
    
    def __init__(self, models_path: Optional[str] = None):
        self.models_path = models_path
        
        # Initialize ML components
        self.skills_extractor = SkillsExtractorModel(
            model_path=f"{models_path}/bert-skills-ner-final" if models_path else None
        )
        
        self.similarity_engine = SimilarityEngine(
            model_path=f"{models_path}/similarity_model" if models_path else None
        )
        
        self.neural_scorer = NeuralScorer(
            model_path=f"{models_path}/neural_scorer/neural_scorer.h5" if models_path else None
        )
        
        logger.info("Advanced Recommendation Engine initialized")
    
    def analyze_cv_profile(self, cv_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive CV analysis using all ML models
        """
        try:
            # Extract skills using BERT NER
            cv_text = self._extract_text_from_cv(cv_data)
            skills_analysis = self.skills_extractor.extract_skills(cv_text)
            
            # Categorize skills
            categorized_skills = self.skills_extractor.categorize_skills(
                skills_analysis['skills']
            )
            
            # Get skill suggestions
            skill_suggestions = self.skills_extractor.get_skill_suggestions(
                skills_analysis['skills']
            )
            
            # Analyze experience level and domain
            profile_analysis = self._analyze_profile_characteristics(cv_data)
            
            return {
                'extracted_skills': skills_analysis,
                'categorized_skills': categorized_skills,
                'skill_suggestions': skill_suggestions,
                'profile_analysis': profile_analysis,
                'ml_confidence': skills_analysis.get('confidence', 'medium'),
                'analysis_method': skills_analysis.get('method', 'hybrid')
            }
            
        except Exception as e:
            logger.error(f"Error in CV analysis: {e}")
            return {
                'extracted_skills': {'skills': [], 'method': 'error'},
                'error': str(e)
            }
    
    def get_personalized_recommendations(self, cv_data: Dict, recommendation_types: List[str] = None) -> Dict[str, Any]:
        """
        Generate personalized recommendations using ML models
        """
        if recommendation_types is None:
            recommendation_types = ['jobs', 'courses', 'certifications', 'projects']
        
        try:
            # Analyze CV profile
            cv_analysis = self.analyze_cv_profile(cv_data)
            
            recommendations = {
                'profile_summary': cv_analysis,
                'recommendations': {},
                'explanation': self._generate_recommendation_explanation(cv_analysis)
            }
            
            # Generate job recommendations
            if 'jobs' in recommendation_types:
                job_recs = self._generate_job_recommendations(cv_data, cv_analysis)
                recommendations['recommendations']['jobs'] = job_recs
            
            # Generate course recommendations
            if 'courses' in recommendation_types:
                course_recs = self._generate_course_recommendations(cv_analysis)
                recommendations['recommendations']['courses'] = course_recs
            
            # Generate certification recommendations
            if 'certifications' in recommendation_types:
                cert_recs = self._generate_certification_recommendations(cv_analysis)
                recommendations['recommendations']['certifications'] = cert_recs
            
            # Generate project recommendations
            if 'projects' in recommendation_types:
                project_recs = self._generate_project_recommendations(cv_analysis)
                recommendations['recommendations']['projects'] = project_recs
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                'error': str(e),
                'recommendations': {}
            }
    
    def score_job_matches(self, cv_data: Dict, job_list: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Score and rank job matches using all ML models
        """
        try:
            results = []
            
            for job in job_list:
                # Calculate semantic similarity
                similarity_details = self.similarity_engine.calculate_cv_job_similarity(cv_data, job)
                overall_similarity = similarity_details.get('overall', 0.0)
                
                # Get neural network score
                neural_result = self.neural_scorer.predict_score(cv_data, job, overall_similarity)
                neural_score = neural_result['neural_score']
                
                # Calculate combined score
                combined_score = (overall_similarity * 0.4 + neural_score * 0.6)
                
                # Generate explanation
                explanation = self.neural_scorer.explain_prediction(cv_data, job, overall_similarity)
                
                result = {
                    'job': job,
                    'scores': {
                        'similarity': overall_similarity,
                        'neural': neural_score,
                        'combined': combined_score
                    },
                    'similarity_details': similarity_details,
                    'explanation': explanation,
                    'match_quality': self._determine_match_quality(combined_score),
                    'recommendation_reason': self._generate_match_reason(cv_data, job, explanation)
                }
                
                results.append(result)
            
            # Sort by combined score
            results.sort(key=lambda x: x['scores']['combined'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error scoring job matches: {e}")
            return []
    
    def explain_recommendation(self, cv_data: Dict, recommendation: Dict) -> Dict[str, Any]:
        """
        Provide detailed explanation for a recommendation
        """
        try:
            explanation = {
                'recommendation_type': recommendation.get('type', 'unknown'),
                'ml_analysis': {},
                'key_factors': [],
                'learning_opportunities': [],
                'confidence_level': 'medium'
            }
            
            # Skills-based explanation
            cv_skills = cv_data.get('skills', [])
            rec_skills = recommendation.get('required_skills', [])
            
            matching_skills = set(cv_skills).intersection(set(rec_skills))
            missing_skills = set(rec_skills) - set(cv_skills)
            
            if matching_skills:
                explanation['key_factors'].append(
                    f"✅ Your skills match: {', '.join(list(matching_skills)[:3])}"
                )
            
            if missing_skills:
                explanation['learning_opportunities'].append(
                    f"📚 Skills to develop: {', '.join(list(missing_skills)[:3])}"
                )
            
            # Experience-based explanation
            cv_experience = cv_data.get('experience_years', 0)
            rec_experience = recommendation.get('min_experience', 0)
            
            if cv_experience >= rec_experience:
                explanation['key_factors'].append(
                    f"✅ Experience level matches ({cv_experience} years)"
                )
            else:
                gap = rec_experience - cv_experience
                explanation['learning_opportunities'].append(
                    f"📈 Gain {gap} more years of experience"
                )
            
            # Domain alignment
            cv_domain = cv_data.get('domain', '')
            rec_domain = recommendation.get('domain', '')
            
            if cv_domain == rec_domain:
                explanation['key_factors'].append(
                    f"✅ Perfect domain match: {cv_domain}"
                )
                explanation['confidence_level'] = 'high'
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining recommendation: {e}")
            return {'error': str(e)}
    
    def _extract_text_from_cv(self, cv_data: Dict) -> str:
        """Extract comprehensive text from CV data for analysis"""
        text_parts = []
        
        # Add all relevant text fields
        for field in ['summary', 'objective', 'skills_text', 'experience_text']:
            if cv_data.get(field):
                text_parts.append(cv_data[field])
        
        # Add skills as text
        skills = cv_data.get('skills', [])
        if skills:
            text_parts.append(' '.join(skills))
        
        # Add experience descriptions
        experience = cv_data.get('experience', [])
        for exp in experience:
            if isinstance(exp, dict):
                for field in ['description', 'role', 'company']:
                    if exp.get(field):
                        text_parts.append(exp[field])
        
        return ' '.join(filter(None, text_parts))
    
    def _analyze_profile_characteristics(self, cv_data: Dict) -> Dict[str, Any]:
        """Analyze CV profile characteristics"""
        characteristics = {
            'experience_level': self._determine_experience_level(cv_data),
            'primary_domain': self._determine_primary_domain(cv_data),
            'skill_diversity': self._calculate_skill_diversity(cv_data),
            'career_progression': self._analyze_career_progression(cv_data)
        }
        
        return characteristics
    
    def _determine_experience_level(self, cv_data: Dict) -> str:
        """Determine experience level from CV data"""
        years = cv_data.get('experience_years', 0)
        
        if years <= 2:
            return 'junior'
        elif years <= 5:
            return 'mid'
        elif years <= 10:
            return 'senior'
        else:
            return 'expert'
    
    def _determine_primary_domain(self, cv_data: Dict) -> str:
        """Determine primary domain from skills and experience"""
        # This could use ML clustering in the future
        skills = cv_data.get('skills', [])
        domain_scores = {
            'web_development': 0,
            'data_science': 0,
            'mobile': 0,
            'devops': 0,
            'backend': 0
        }
        
        # Simple rule-based domain detection
        skill_keywords = {
            'web_development': ['javascript', 'react', 'html', 'css', 'vue', 'angular'],
            'data_science': ['python', 'r', 'machine learning', 'pandas', 'numpy'],
            'mobile': ['swift', 'kotlin', 'react native', 'flutter', 'ios', 'android'],
            'devops': ['docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'terraform'],
            'backend': ['java', 'c#', 'go', 'rust', 'spring', 'django']
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            for domain, keywords in skill_keywords.items():
                if any(keyword in skill_lower for keyword in keywords):
                    domain_scores[domain] += 1
        
        if max(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def _calculate_skill_diversity(self, cv_data: Dict) -> float:
        """Calculate how diverse the skill set is"""
        skills = cv_data.get('skills', [])
        if not skills:
            return 0.0
        
        # Simple diversity metric based on number of different skill categories
        categorized = self.skills_extractor.categorize_skills(skills)
        num_categories = len([k for k, v in categorized.items() if v])
        
        return min(1.0, num_categories / 5)  # Normalize to 0-1
    
    def _analyze_career_progression(self, cv_data: Dict) -> str:
        """Analyze career progression pattern"""
        experience = cv_data.get('experience', [])
        
        if len(experience) <= 1:
            return 'early_career'
        elif len(experience) <= 3:
            return 'building'
        else:
            return 'established'
    
    def _generate_job_recommendations(self, cv_data: Dict, cv_analysis: Dict) -> List[Dict]:
        """Generate job-specific recommendations"""
        # This would typically query a job database
        # For now, return template recommendations based on analysis
        
        primary_domain = cv_analysis['profile_analysis']['primary_domain']
        experience_level = cv_analysis['profile_analysis']['experience_level']
        skills = cv_analysis['extracted_skills']['skills']
        
        job_templates = {
            'web_development': [
                {
                    'title': f'{experience_level.title()} Frontend Developer',
                    'domain': 'web_development',
                    'required_skills': ['JavaScript', 'React', 'CSS', 'HTML'],
                    'description': f'Frontend development role suitable for {experience_level} level',
                    'type': 'job'
                },
                {
                    'title': f'{experience_level.title()} Full Stack Developer',
                    'domain': 'web_development', 
                    'required_skills': ['JavaScript', 'React', 'Node.js', 'MongoDB'],
                    'description': f'Full stack development role for {experience_level} developer',
                    'type': 'job'
                }
            ],
            'data_science': [
                {
                    'title': f'{experience_level.title()} Data Scientist',
                    'domain': 'data_science',
                    'required_skills': ['Python', 'Machine Learning', 'Pandas', 'SQL'],
                    'description': f'Data science position for {experience_level} professional',
                    'type': 'job'
                }
            ]
        }
        
        return job_templates.get(primary_domain, [])[:3]
    
    def _generate_course_recommendations(self, cv_analysis: Dict) -> List[Dict]:
        """Generate course recommendations based on skills gaps"""
        skill_suggestions = cv_analysis.get('skill_suggestions', [])
        
        courses = []
        for suggestion in skill_suggestions[:3]:
            skill = suggestion['skill']
            domain = suggestion['domain']
            
            courses.append({
                'title': f'{skill} Fundamentals',
                'domain': domain,
                'skill_focus': skill,
                'description': f'Learn {skill} to enhance your {domain} skills',
                'type': 'course',
                'duration': '4-6 weeks',
                'level': 'intermediate'
            })
        
        return courses
    
    def _generate_certification_recommendations(self, cv_analysis: Dict) -> List[Dict]:
        """Generate certification recommendations"""
        primary_domain = cv_analysis['profile_analysis']['primary_domain']
        experience_level = cv_analysis['profile_analysis']['experience_level']
        
        cert_map = {
            'web_development': [
                {
                    'title': 'AWS Certified Developer',
                    'provider': 'AWS',
                    'domain': 'cloud',
                    'description': 'Cloud development certification',
                    'type': 'certification'
                }
            ],
            'data_science': [
                {
                    'title': 'TensorFlow Developer Certificate',
                    'provider': 'Google',
                    'domain': 'machine_learning',
                    'description': 'ML framework certification',
                    'type': 'certification'
                }
            ]
        }
        
        return cert_map.get(primary_domain, [])[:2]
    
    def _generate_project_recommendations(self, cv_analysis: Dict) -> List[Dict]:
        """Generate project recommendations"""
        primary_domain = cv_analysis['profile_analysis']['primary_domain']
        skills = cv_analysis['extracted_skills']['skills']
        
        project_templates = {
            'web_development': [
                {
                    'title': 'E-commerce Platform',
                    'description': 'Build a full-stack e-commerce application',
                    'skills_used': ['React', 'Node.js', 'MongoDB'],
                    'type': 'project',
                    'difficulty': 'intermediate'
                }
            ],
            'data_science': [
                {
                    'title': 'Predictive Analytics Dashboard',
                    'description': 'Create a data visualization and prediction system',
                    'skills_used': ['Python', 'Pandas', 'Machine Learning'],
                    'type': 'project',
                    'difficulty': 'intermediate'
                }
            ]
        }
        
        return project_templates.get(primary_domain, [])[:2]
    
    def _generate_recommendation_explanation(self, cv_analysis: Dict) -> Dict[str, Any]:
        """Generate explanation for recommendations"""
        return {
            'based_on': {
                'skills_extracted': len(cv_analysis.get('extracted_skills', {}).get('skills', [])),
                'primary_domain': cv_analysis.get('profile_analysis', {}).get('primary_domain'),
                'experience_level': cv_analysis.get('profile_analysis', {}).get('experience_level'),
                'ml_confidence': cv_analysis.get('ml_confidence', 'medium')
            },
            'methodology': 'Advanced ML analysis using BERT NER, Sentence-Transformers, and Neural Scoring'
        }
    
    def _determine_match_quality(self, score: float) -> str:
        """Determine match quality based on score"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_match_reason(self, cv_data: Dict, job_data: Dict, explanation: Dict) -> str:
        """Generate human-readable match reason"""
        reasons = []
        
        # Skills match
        cv_skills = set(cv_data.get('skills', []))
        job_skills = set(job_data.get('required_skills', []))
        matching_skills = cv_skills.intersection(job_skills)
        
        if matching_skills:
            reasons.append(f"Skills match: {', '.join(list(matching_skills)[:2])}")
        
        # Experience
        if explanation.get('key_factors'):
            reasons.extend(explanation['key_factors'][:2])
        
        return '; '.join(reasons) if reasons else 'General profile compatibility'
