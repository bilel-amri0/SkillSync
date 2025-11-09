"""
F7: Personalized Recommendations - Custom development paths
Suggestion of trainings, projects and certifications adapted
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Intelligent recommendation system for career development"""
    
    def __init__(self):
        self.recommendation_data = self._load_recommendation_database()
        self.learning_platforms = self._load_learning_platforms()
        self.certification_providers = self._load_certification_providers()
        
    def _load_recommendation_database(self) -> Dict[str, Any]:
        """Load recommendation database (in production, this would be a real database)"""
        
        return {
            'skills_progression': {
                'programming_languages': {
                    'beginner': ['python', 'javascript', 'html', 'css'],
                    'intermediate': ['java', 'react', 'node.js', 'sql'],
                    'advanced': ['typescript', 'go', 'rust', 'kubernetes']
                },
                'data_science': {
                    'beginner': ['python', 'pandas', 'numpy', 'excel'],
                    'intermediate': ['scikit-learn', 'matplotlib', 'sql', 'tableau'],
                    'advanced': ['tensorflow', 'pytorch', 'spark', 'docker']
                },
                'cloud_platforms': {
                    'beginner': ['aws basics', 'azure basics', 'cloud concepts'],
                    'intermediate': ['aws solutions architect', 'azure developer', 'terraform'],
                    'advanced': ['aws devops', 'azure architect', 'kubernetes']
                }
            },
            'career_paths': {
                'software_engineer': {
                    'skills': ['programming_languages', 'web_technologies', 'databases'],
                    'progression': ['junior developer', 'software engineer', 'senior engineer', 'tech lead'],
                    'timeline_months': [0, 18, 36, 60]
                },
                'data_scientist': {
                    'skills': ['data_science', 'programming_languages', 'databases'],
                    'progression': ['data analyst', 'data scientist', 'senior data scientist', 'data science manager'],
                    'timeline_months': [0, 24, 48, 72]
                },
                'cloud_architect': {
                    'skills': ['cloud_platforms', 'programming_languages', 'methodologies'],
                    'progression': ['cloud engineer', 'cloud architect', 'senior architect', 'cloud director'],
                    'timeline_months': [0, 30, 54, 84]
                }
            },
            'project_templates': {
                'beginner': [
                    {
                        'title': 'Personal Portfolio Website',
                        'description': 'Create a responsive portfolio website using HTML, CSS, and JavaScript',
                        'skills': ['html', 'css', 'javascript'],
                        'duration': '1-2 weeks',
                        'difficulty': 'beginner',
                        'category': 'web_development'
                    },
                    {
                        'title': 'Data Analysis Dashboard',
                        'description': 'Build an interactive dashboard to visualize data using Python and Plotly',
                        'skills': ['python', 'pandas', 'plotly'],
                        'duration': '2-3 weeks',
                        'difficulty': 'beginner',
                        'category': 'data_science'
                    }
                ],
                'intermediate': [
                    {
                        'title': 'E-commerce API',
                        'description': 'Develop a RESTful API for an e-commerce platform with authentication',
                        'skills': ['node.js', 'express', 'mongodb', 'jwt'],
                        'duration': '3-4 weeks',
                        'difficulty': 'intermediate',
                        'category': 'backend_development'
                    },
                    {
                        'title': 'Machine Learning Classifier',
                        'description': 'Build and deploy a machine learning model for classification tasks',
                        'skills': ['python', 'scikit-learn', 'flask', 'docker'],
                        'duration': '4-6 weeks',
                        'difficulty': 'intermediate',
                        'category': 'machine_learning'
                    }
                ],
                'advanced': [
                    {
                        'title': 'Microservices Architecture',
                        'description': 'Design and implement a scalable microservices system',
                        'skills': ['docker', 'kubernetes', 'api_gateway', 'monitoring'],
                        'duration': '8-12 weeks',
                        'difficulty': 'advanced',
                        'category': 'system_architecture'
                    },
                    {
                        'title': 'Real-time Analytics Platform',
                        'description': 'Build a real-time data processing and analytics platform',
                        'skills': ['kafka', 'spark', 'elasticsearch', 'kibana'],
                        'duration': '10-16 weeks',
                        'difficulty': 'advanced',
                        'category': 'big_data'
                    }
                ]
            }
        }
    
    def _load_learning_platforms(self) -> Dict[str, Dict[str, Any]]:
        """Load learning platforms and their course offerings"""
        
        return {
            'coursera': {
                'name': 'Coursera',
                'url': 'https://coursera.org',
                'specialties': ['data_science', 'machine_learning', 'business'],
                'certification': True,
                'cost': 'paid'
            },
            'udemy': {
                'name': 'Udemy',
                'url': 'https://udemy.com',
                'specialties': ['programming', 'web_development', 'design'],
                'certification': True,
                'cost': 'paid'
            },
            'codecademy': {
                'name': 'Codecademy',
                'url': 'https://codecademy.com',
                'specialties': ['programming_languages', 'web_technologies'],
                'certification': True,
                'cost': 'freemium'
            },
            'freecodecamp': {
                'name': 'freeCodeCamp',
                'url': 'https://freecodecamp.org',
                'specialties': ['web_development', 'programming_languages'],
                'certification': True,
                'cost': 'free'
            },
            'edx': {
                'name': 'edX',
                'url': 'https://edx.org',
                'specialties': ['computer_science', 'data_science', 'ai'],
                'certification': True,
                'cost': 'freemium'
            }
        }
    
    def _load_certification_providers(self) -> Dict[str, Dict[str, Any]]:
        """Load certification providers and their offerings"""
        
        return {
            'aws': {
                'name': 'Amazon Web Services',
                'certifications': [
                    {'name': 'AWS Cloud Practitioner', 'level': 'foundational', 'cost': 100},
                    {'name': 'AWS Solutions Architect Associate', 'level': 'associate', 'cost': 150},
                    {'name': 'AWS Solutions Architect Professional', 'level': 'professional', 'cost': 300}
                ],
                'renewal_period': 36
            },
            'google_cloud': {
                'name': 'Google Cloud Platform',
                'certifications': [
                    {'name': 'Google Cloud Digital Leader', 'level': 'foundational', 'cost': 99},
                    {'name': 'Professional Cloud Architect', 'level': 'professional', 'cost': 200}
                ],
                'renewal_period': 24
            },
            'microsoft': {
                'name': 'Microsoft',
                'certifications': [
                    {'name': 'Azure Fundamentals', 'level': 'foundational', 'cost': 99},
                    {'name': 'Azure Developer Associate', 'level': 'associate', 'cost': 165},
                    {'name': 'Azure Solutions Architect Expert', 'level': 'expert', 'cost': 165}
                ],
                'renewal_period': 12
            },
            'comptia': {
                'name': 'CompTIA',
                'certifications': [
                    {'name': 'CompTIA A+', 'level': 'entry', 'cost': 358},
                    {'name': 'CompTIA Security+', 'level': 'intermediate', 'cost': 370},
                    {'name': 'CompTIA CySA+', 'level': 'advanced', 'cost': 392}
                ],
                'renewal_period': 36
            }
        }
    
    async def get_recommendations(
        self, 
        user_skills: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get basic recommendations based on user skills"""
        
        recommendations = []
        
        # Analyze user skill levels
        skill_analysis = self._analyze_user_skills(user_skills)
        
        # Generate skill-based recommendations
        skill_recommendations = self._generate_skill_recommendations(skill_analysis)
        recommendations.extend(skill_recommendations)
        
        # Generate project recommendations
        project_recommendations = self._generate_project_recommendations(skill_analysis)
        recommendations.extend(project_recommendations)
        
        # Generate certification recommendations
        cert_recommendations = self._generate_certification_recommendations(skill_analysis)
        recommendations.extend(cert_recommendations)
        
        return recommendations[:10]  # Limit to top 10
    
    async def generate_comprehensive_recommendations(
        self,
        skills: List[Dict[str, Any]],
        gap_analysis: Optional[Dict[str, Any]] = None,
        career_goals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive personalized recommendations"""
        
        try:
            # Analyze current state
            skill_analysis = self._analyze_user_skills(skills)
            
            # Determine career path
            suggested_path = self._suggest_career_path(skill_analysis, career_goals)
            
            # Generate recommendations by category
            recommendations = {
                'immediate_actions': self._get_immediate_actions(skill_analysis, gap_analysis),
                'skill_development': self._get_skill_development_plan(skill_analysis, suggested_path),
                'project_suggestions': self._get_project_suggestions(skill_analysis),
                'certification_roadmap': self._get_certification_roadmap(skill_analysis, suggested_path),
                'learning_resources': self._get_learning_resources(skill_analysis),
                'career_roadmap': self._get_career_roadmap(suggested_path, skill_analysis),
                'networking_opportunities': self._get_networking_opportunities(suggested_path),
                'timeline': self._generate_development_timeline(skill_analysis, suggested_path)
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating comprehensive recommendations: {str(e)}")
            raise
    
    def _analyze_user_skills(self, skills: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user skills to determine current level and gaps"""
        
        analysis = {
            'skill_categories': {},
            'overall_level': 'beginner',
            'strongest_areas': [],
            'improvement_areas': [],
            'skill_count': len(skills),
            'experience_distribution': {'beginner': 0, 'intermediate': 0, 'advanced': 0}
        }
        
        # Group skills by category
        for skill in skills:
            category = skill.get('category', 'other')
            level = skill.get('experience_level', 'beginner')
            
            if category not in analysis['skill_categories']:
                analysis['skill_categories'][category] = []
            
            analysis['skill_categories'][category].append({
                'name': skill.get('normalized_name', skill.get('skill', '')),
                'level': level,
                'confidence': skill.get('confidence', 0.5),
                'importance': skill.get('importance_score', 0.5)
            })
            
            # Count experience levels
            analysis['experience_distribution'][level] += 1
        
        # Determine overall level
        total_skills = analysis['skill_count']
        if total_skills > 0:
            advanced_ratio = analysis['experience_distribution']['advanced'] / total_skills
            intermediate_ratio = analysis['experience_distribution']['intermediate'] / total_skills
            
            if advanced_ratio >= 0.3:
                analysis['overall_level'] = 'advanced'
            elif intermediate_ratio >= 0.4:
                analysis['overall_level'] = 'intermediate'
        
        # Identify strongest and weakest areas
        category_scores = {}
        for category, cat_skills in analysis['skill_categories'].items():
            if cat_skills:
                avg_importance = sum(skill['importance'] for skill in cat_skills) / len(cat_skills)
                category_scores[category] = avg_importance
        
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        analysis['strongest_areas'] = [cat for cat, score in sorted_categories[:3]]
        analysis['improvement_areas'] = [cat for cat, score in sorted_categories[-2:]]
        
        return analysis
    
    def _suggest_career_path(
        self, 
        skill_analysis: Dict[str, Any], 
        career_goals: Optional[Dict[str, Any]] = None
    ) -> str:
        """Suggest most suitable career path based on skills"""
        
        # If user has specified career goals, respect them
        if career_goals and career_goals.get('target_role'):
            return career_goals['target_role']
        
        # Analyze skill categories to suggest path
        categories = skill_analysis['skill_categories']
        
        # Simple heuristics for path suggestion
        if 'data_science' in categories and len(categories['data_science']) >= 3:
            return 'data_scientist'
        elif 'cloud_platforms' in categories and len(categories['cloud_platforms']) >= 2:
            return 'cloud_architect'
        elif 'programming_languages' in categories and 'web_technologies' in categories:
            return 'software_engineer'
        else:
            return 'software_engineer'  # Default path
    
    def _get_immediate_actions(
        self, 
        skill_analysis: Dict[str, Any], 
        gap_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get immediate actionable recommendations"""
        
        actions = []
        
        # Based on gap analysis
        if gap_analysis and gap_analysis.get('missing_skills'):
            critical_missing = gap_analysis['missing_skills'].get('critical', [])
            for skill in critical_missing[:3]:
                actions.append({
                    'type': 'skill_gap',
                    'title': f"Learn {skill.get('normalized_name', 'this skill')}",
                    'description': f"This skill is critical for your target role",
                    'priority': 'high',
                    'estimated_time': '2-4 weeks',
                    'category': skill.get('category', 'general')
                })
        
        # Based on current skill level
        if skill_analysis['overall_level'] == 'beginner':
            actions.append({
                'type': 'foundation_building',
                'title': 'Build Strong Fundamentals',
                'description': 'Focus on core concepts and basic projects',
                'priority': 'high',
                'estimated_time': '4-8 weeks',
                'category': 'foundation'
            })
        
        # Portfolio recommendations
        actions.append({
            'type': 'portfolio',
            'title': 'Update Your Portfolio',
            'description': 'Showcase your skills with recent projects',
            'priority': 'medium',
            'estimated_time': '1-2 weeks',
            'category': 'career_development'
        })
        
        return actions[:5]
    
    def _get_skill_development_plan(
        self, 
        skill_analysis: Dict[str, Any], 
        career_path: str
    ) -> List[Dict[str, Any]]:
        """Generate personalized skill development plan"""
        
        plan = []
        
        # Get skills progression for the career path
        path_data = self.recommendation_data['career_paths'].get(career_path, {})
        required_skill_categories = path_data.get('skills', [])
        
        user_level = skill_analysis['overall_level']
        
        for category in required_skill_categories:
            progression = self.recommendation_data['skills_progression'].get(category, {})
            
            # Determine next level skills to learn
            if user_level == 'beginner':
                next_skills = progression.get('intermediate', [])
                plan_level = 'intermediate'
            elif user_level == 'intermediate':
                next_skills = progression.get('advanced', [])
                plan_level = 'advanced'
            else:
                continue  # Already advanced
            
            # Check which skills user doesn't have yet
            user_skills_in_category = [
                skill['name'].lower() 
                for skill in skill_analysis['skill_categories'].get(category, [])
            ]
            
            missing_skills = [
                skill for skill in next_skills 
                if skill.lower() not in user_skills_in_category
            ]
            
            for skill in missing_skills[:3]:  # Limit to 3 per category
                plan.append({
                    'skill': skill,
                    'category': category,
                    'current_level': user_level,
                    'target_level': plan_level,
                    'priority': self._calculate_skill_priority(skill, category),
                    'estimated_time': self._estimate_learning_time(skill, plan_level),
                    'recommended_resources': self._get_skill_resources(skill)
                })
        
        # Sort by priority
        plan.sort(key=lambda x: x['priority'], reverse=True)
        
        return plan[:10]
    
    def _get_project_suggestions(
        self, 
        skill_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get personalized project suggestions"""
        
        user_level = skill_analysis['overall_level']
        user_categories = list(skill_analysis['skill_categories'].keys())
        
        # Get projects for user's level
        available_projects = self.recommendation_data['project_templates'].get(user_level, [])
        
        # Filter projects based on user's skill categories
        relevant_projects = []
        for project in available_projects:
            project_skills = project.get('skills', [])
            
            # Check if user has most of the required skills
            user_skill_names = [
                skill['name'].lower() 
                for category in user_categories 
                for skill in skill_analysis['skill_categories'][category]
            ]
            
            matching_skills = [
                skill for skill in project_skills 
                if skill.lower() in user_skill_names
            ]
            
            # Include project if user has at least 50% of required skills
            if len(matching_skills) >= len(project_skills) * 0.5:
                project_copy = project.copy()
                project_copy['skill_match_percentage'] = len(matching_skills) / len(project_skills) * 100
                project_copy['missing_skills'] = [
                    skill for skill in project_skills 
                    if skill.lower() not in user_skill_names
                ]
                relevant_projects.append(project_copy)
        
        # Sort by skill match percentage
        relevant_projects.sort(key=lambda x: x['skill_match_percentage'], reverse=True)
        
        return relevant_projects[:5]
    
    def _get_certification_roadmap(
        self, 
        skill_analysis: Dict[str, Any], 
        career_path: str
    ) -> List[Dict[str, Any]]:
        """Generate certification roadmap"""
        
        roadmap = []
        user_level = skill_analysis['overall_level']
        user_categories = skill_analysis['skill_categories']
        
        # Cloud certifications
        if 'cloud_platforms' in user_categories:
            cloud_skills = [skill['name'].lower() for skill in user_categories['cloud_platforms']]
            
            if any('aws' in skill for skill in cloud_skills):
                aws_certs = self._get_aws_certification_path(user_level)
                roadmap.extend(aws_certs)
            
            if any('azure' in skill for skill in cloud_skills):
                azure_certs = self._get_azure_certification_path(user_level)
                roadmap.extend(azure_certs)
        
        # Programming certifications
        if 'programming_languages' in user_categories:
            prog_certs = self._get_programming_certifications(user_level)
            roadmap.extend(prog_certs)
        
        return roadmap[:6]
    
    def _get_learning_resources(
        self, 
        skill_analysis: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get personalized learning resources"""
        
        resources = {
            'free_resources': [],
            'paid_courses': [],
            'books': [],
            'practice_platforms': []
        }
        
        user_categories = list(skill_analysis['skill_categories'].keys())
        
        # Free resources
        if 'programming_languages' in user_categories:
            resources['free_resources'].extend([
                {
                    'name': 'freeCodeCamp',
                    'url': 'https://freecodecamp.org',
                    'description': 'Free coding bootcamp with certificates',
                    'category': 'programming'
                },
                {
                    'name': 'MDN Web Docs',
                    'url': 'https://developer.mozilla.org',
                    'description': 'Comprehensive web development documentation',
                    'category': 'web_development'
                }
            ])
        
        # Practice platforms
        resources['practice_platforms'].extend([
            {
                'name': 'LeetCode',
                'url': 'https://leetcode.com',
                'description': 'Algorithm and data structure practice',
                'category': 'programming'
            },
            {
                'name': 'HackerRank',
                'url': 'https://hackerrank.com',
                'description': 'Coding challenges and competitions',
                'category': 'programming'
            }
        ])
        
        return resources
    
    def _get_career_roadmap(
        self, 
        career_path: str, 
        skill_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate career progression roadmap"""
        
        path_data = self.recommendation_data['career_paths'].get(career_path, {})
        
        if not path_data:
            return {}
        
        progression = path_data.get('progression', [])
        timeline = path_data.get('timeline_months', [])
        
        # Determine current position based on skills
        current_level = skill_analysis['overall_level']
        if current_level == 'beginner':
            current_position = 0
        elif current_level == 'intermediate':
            current_position = 1
        else:
            current_position = 2
        
        roadmap = {
            'career_path': career_path,
            'current_position': progression[current_position] if current_position < len(progression) else progression[-1],
            'next_steps': [],
            'long_term_goals': [],
            'estimated_timeline': {}
        }
        
        # Next 1-2 positions
        for i in range(current_position + 1, min(current_position + 3, len(progression))):
            roadmap['next_steps'].append({
                'position': progression[i],
                'timeline_months': timeline[i] - timeline[current_position] if i < len(timeline) else 24,
                'key_requirements': self._get_position_requirements(progression[i])
            })
        
        # Long-term goals
        for i in range(current_position + 3, len(progression)):
            roadmap['long_term_goals'].append({
                'position': progression[i],
                'timeline_months': timeline[i] - timeline[current_position] if i < len(timeline) else 48
            })
        
        return roadmap
    
    def _get_networking_opportunities(
        self, 
        career_path: str
    ) -> List[Dict[str, Any]]:
        """Get networking opportunities based on career path"""
        
        opportunities = [
            {
                'type': 'professional_communities',
                'title': 'Join Professional Communities',
                'description': 'Connect with professionals in your field',
                'platforms': ['LinkedIn Groups', 'Reddit Communities', 'Discord Servers'],
                'frequency': 'weekly'
            },
            {
                'type': 'conferences',
                'title': 'Attend Industry Conferences',
                'description': 'Learn from experts and network with peers',
                'examples': ['Local tech meetups', 'Online conferences', 'Industry summits'],
                'frequency': 'monthly'
            },
            {
                'type': 'mentorship',
                'title': 'Find a Mentor',
                'description': 'Get guidance from experienced professionals',
                'platforms': ['MentorCruise', 'ADPList', 'LinkedIn'],
                'frequency': 'ongoing'
            }
        ]
        
        return opportunities
    
    def _generate_development_timeline(
        self, 
        skill_analysis: Dict[str, Any], 
        career_path: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate 12-month development timeline"""
        
        timeline = {
            'month_1_3': [],
            'month_4_6': [],
            'month_7_9': [],
            'month_10_12': []
        }
        
        # Month 1-3: Foundation and immediate gaps
        timeline['month_1_3'].extend([
            {
                'type': 'skill_development',
                'title': 'Address Critical Skill Gaps',
                'description': 'Focus on most important missing skills',
                'priority': 'high'
            },
            {
                'type': 'project',
                'title': 'Complete Beginner Project',
                'description': 'Build confidence with achievable project',
                'priority': 'high'
            }
        ])
        
        # Month 4-6: Intermediate development
        timeline['month_4_6'].extend([
            {
                'type': 'skill_development',
                'title': 'Advance Core Skills',
                'description': 'Deepen knowledge in key areas',
                'priority': 'medium'
            },
            {
                'type': 'certification',
                'title': 'Pursue First Certification',
                'description': 'Validate skills with industry certification',
                'priority': 'medium'
            }
        ])
        
        # Month 7-9: Specialization
        timeline['month_7_9'].extend([
            {
                'type': 'specialization',
                'title': 'Choose Specialization',
                'description': 'Focus on specific domain expertise',
                'priority': 'medium'
            },
            {
                'type': 'project',
                'title': 'Advanced Project',
                'description': 'Demonstrate advanced capabilities',
                'priority': 'high'
            }
        ])
        
        # Month 10-12: Career advancement
        timeline['month_10_12'].extend([
            {
                'type': 'career',
                'title': 'Apply for Target Roles',
                'description': 'Leverage new skills for career advancement',
                'priority': 'high'
            },
            {
                'type': 'networking',
                'title': 'Expand Professional Network',
                'description': 'Build connections for future opportunities',
                'priority': 'medium'
            }
        ])
        
        return timeline
    
    # Helper methods
    
    def _generate_skill_recommendations(
        self, 
        skill_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate skill-based recommendations"""
        
        recommendations = []
        
        # Recommend skills to advance current level
        user_level = skill_analysis['overall_level']
        strongest_areas = skill_analysis['strongest_areas']
        
        for area in strongest_areas[:2]:  # Focus on top 2 areas
            progression = self.recommendation_data['skills_progression'].get(area, {})
            
            next_level = 'intermediate' if user_level == 'beginner' else 'advanced'
            next_skills = progression.get(next_level, [])
            
            if next_skills:
                recommendations.append({
                    'recommendation_type': 'skill_advancement',
                    'title': f'Advance Your {area.replace("_", " ").title()} Skills',
                    'description': f'Learn {next_skills[0]} to advance to {next_level} level',
                    'priority': 'high',
                    'estimated_time': '2-4 weeks',
                    'target_skills': next_skills[:3]
                })
        
        return recommendations
    
    def _generate_project_recommendations(
        self, 
        skill_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate project-based recommendations"""
        
        recommendations = []
        user_level = skill_analysis['overall_level']
        
        # Get appropriate projects for user level
        projects = self.recommendation_data['project_templates'].get(user_level, [])
        
        for project in projects[:2]:  # Recommend top 2 projects
            recommendations.append({
                'recommendation_type': 'project',
                'title': f'Build a {project["title"]}',
                'description': project['description'],
                'priority': 'medium',
                'estimated_time': project['duration'],
                'target_skills': project['skills']
            })
        
        return recommendations
    
    def _generate_certification_recommendations(
        self, 
        skill_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate certification recommendations"""
        
        recommendations = []
        user_categories = skill_analysis['skill_categories']
        
        # Cloud certifications
        if 'cloud_platforms' in user_categories:
            recommendations.append({
                'recommendation_type': 'certification',
                'title': 'Get AWS Cloud Practitioner Certification',
                'description': 'Validate your cloud knowledge with industry-recognized certification',
                'priority': 'medium',
                'estimated_time': '4-6 weeks',
                'cost': '$100'
            })
        
        return recommendations
    
    def _calculate_skill_priority(
        self, 
        skill: str, 
        category: str
    ) -> float:
        """Calculate priority score for a skill"""
        
        # Base priority
        priority = 0.5
        
        # High demand skills get higher priority
        high_demand_skills = [
            'python', 'javascript', 'react', 'aws', 'docker', 'kubernetes',
            'machine learning', 'data science', 'sql'
        ]
        
        if skill.lower() in high_demand_skills:
            priority += 0.3
        
        # Category-based priority
        high_priority_categories = ['programming_languages', 'cloud_platforms', 'data_science']
        if category in high_priority_categories:
            priority += 0.2
        
        return min(priority, 1.0)
    
    def _estimate_learning_time(
        self, 
        skill: str, 
        target_level: str
    ) -> str:
        """Estimate time required to learn a skill"""
        
        time_estimates = {
            'beginner': {
                'programming_language': '2-4 weeks',
                'tool': '1-2 weeks',
                'framework': '3-6 weeks',
                'concept': '1-3 weeks'
            },
            'intermediate': {
                'programming_language': '4-8 weeks',
                'tool': '2-4 weeks',
                'framework': '6-10 weeks',
                'concept': '3-6 weeks'
            },
            'advanced': {
                'programming_language': '8-16 weeks',
                'tool': '4-8 weeks',
                'framework': '10-20 weeks',
                'concept': '6-12 weeks'
            }
        }
        
        # Simple categorization (in production, use more sophisticated logic)
        if skill.lower() in ['python', 'java', 'javascript', 'go', 'rust']:
            skill_type = 'programming_language'
        elif skill.lower() in ['docker', 'kubernetes', 'git', 'jenkins']:
            skill_type = 'tool'
        elif skill.lower() in ['react', 'angular', 'vue', 'django', 'flask']:
            skill_type = 'framework'
        else:
            skill_type = 'concept'
        
        return time_estimates.get(target_level, {}).get(skill_type, '2-4 weeks')
    
    def _get_skill_resources(
        self, 
        skill: str
    ) -> List[Dict[str, str]]:
        """Get learning resources for a specific skill"""
        
        # Simplified resource mapping
        resources = [
            {
                'name': f'{skill.title()} Documentation',
                'type': 'documentation',
                'url': f'https://docs.{skill.lower().replace(" ", "")}.org'
            },
            {
                'name': f'Learn {skill.title()} - freeCodeCamp',
                'type': 'tutorial',
                'url': 'https://freecodecamp.org'
            },
            {
                'name': f'{skill.title()} Crash Course - YouTube',
                'type': 'video',
                'url': 'https://youtube.com'
            }
        ]
        
        return resources
    
    def _get_aws_certification_path(
        self, 
        user_level: str
    ) -> List[Dict[str, Any]]:
        """Get AWS certification recommendations"""
        
        aws_data = self.certification_providers['aws']
        certs = []
        
        if user_level == 'beginner':
            certs.append({
                'name': 'AWS Cloud Practitioner',
                'provider': 'AWS',
                'level': 'foundational',
                'cost': 100,
                'estimated_study_time': '4-6 weeks',
                'priority': 'high'
            })
        elif user_level == 'intermediate':
            certs.append({
                'name': 'AWS Solutions Architect Associate',
                'provider': 'AWS',
                'level': 'associate',
                'cost': 150,
                'estimated_study_time': '8-12 weeks',
                'priority': 'high'
            })
        
        return certs
    
    def _get_azure_certification_path(
        self, 
        user_level: str
    ) -> List[Dict[str, Any]]:
        """Get Azure certification recommendations"""
        
        certs = []
        
        if user_level == 'beginner':
            certs.append({
                'name': 'Azure Fundamentals',
                'provider': 'Microsoft',
                'level': 'foundational',
                'cost': 99,
                'estimated_study_time': '3-5 weeks',
                'priority': 'medium'
            })
        
        return certs
    
    def _get_programming_certifications(
        self, 
        user_level: str
    ) -> List[Dict[str, Any]]:
        """Get programming certification recommendations"""
        
        certs = []
        
        if user_level in ['intermediate', 'advanced']:
            certs.append({
                'name': 'Oracle Java Certification',
                'provider': 'Oracle',
                'level': 'professional',
                'cost': 245,
                'estimated_study_time': '6-10 weeks',
                'priority': 'medium'
            })
        
        return certs
    
    def _get_position_requirements(
        self, 
        position: str
    ) -> List[str]:
        """Get key requirements for a position"""
        
        requirements_map = {
            'senior engineer': [
                'Advanced technical skills',
                'Leadership experience',
                'System design knowledge',
                'Mentoring abilities'
            ],
            'tech lead': [
                'Technical expertise',
                'Team leadership',
                'Project management',
                'Architecture decisions'
            ],
            'senior data scientist': [
                'Advanced ML knowledge',
                'Business understanding',
                'Research skills',
                'Communication abilities'
            ]
        }
        
        return requirements_map.get(position.lower(), [
            'Relevant experience',
            'Technical skills',
            'Professional development'
        ])