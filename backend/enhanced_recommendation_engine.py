#!/usr/bin/env python3
"""
Enhanced Recommendation Engine with Proper Formatting
Fixes the "N/A" titles and 0.0% scores issue
"""

import asyncio
from typing import Dict, List, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)

class EnhancedRecommendationEngine:
    """Enhanced recommendation engine that generates properly formatted recommendations"""
    
    def __init__(self):
        self.skill_templates = {
            'python': {
                'beginner': ['Learn Python Basics', 'Master Python Syntax', 'Build First Python Project'],
                'intermediate': ['Advanced Python Features', 'Python Design Patterns', 'Python Performance Optimization'],
                'advanced': ['Python Architecture', 'Python Microservices', 'Python Machine Learning']
            },
            'javascript': {
                'beginner': ['JavaScript Fundamentals', 'DOM Manipulation', 'ES6+ Features'],
                'intermediate': ['Async JavaScript', 'JavaScript Frameworks', 'Node.js Development'],
                'advanced': ['JavaScript Architecture', 'Performance Optimization', 'Advanced React Patterns']
            },
            'react': {
                'beginner': ['React Basics', 'Components & Props', 'State Management'],
                'intermediate': ['React Hooks', 'React Router', 'Context API'],
                'advanced': ['Advanced React Patterns', 'React Performance', 'Custom Hooks']
            },
            'sql': {
                'beginner': ['SQL Basics', 'Database Design', 'Basic Queries'],
                'intermediate': ['Advanced SQL', 'Stored Procedures', 'Database Optimization'],
                'advanced': ['Database Architecture', 'Query Performance', 'Data Warehousing']
            },
            'machine learning': {
                'beginner': ['ML Fundamentals', 'Data Preprocessing', 'Basic Algorithms'],
                'intermediate': ['Advanced Algorithms', 'Feature Engineering', 'Model Evaluation'],
                'advanced': ['Deep Learning', 'MLOps', 'Production ML Systems']
            },
            'docker': {
                'beginner': ['Docker Basics', 'Container Fundamentals', 'Dockerfile Creation'],
                'intermediate': ['Docker Compose', 'Container Orchestration', 'Docker Networks'],
                'advanced': ['Docker Security', 'Multi-stage Builds', 'Docker in Production']
            },
            'aws': {
                'beginner': ['AWS Basics', 'EC2 & S3', 'AWS Console Navigation'],
                'intermediate': ['AWS Architecture', 'Auto Scaling', 'Load Balancers'],
                'advanced': ['AWS Security', 'Infrastructure as Code', 'AWS Well-Architected']
            }
        }
        
        self.career_paths = {
            'software_engineer': {
                'immediate': ['Update LinkedIn Profile', 'Build Portfolio Website', 'Practice Coding Challenges'],
                'short_term': ['Learn Advanced Framework', 'Contribute to Open Source', 'Get AWS Certification'],
                'long_term': ['Senior Developer Role', 'Tech Lead Position', 'Software Architect']
            },
            'data_scientist': {
                'immediate': ['Master Python/R', 'Learn SQL', 'Build Data Portfolio'],
                'short_term': ['Advanced ML Algorithms', 'Deep Learning', 'Cloud Platforms'],
                'long_term': ['Senior Data Scientist', 'ML Engineering', 'Data Science Manager']
            },
            'devops_engineer': {
                'immediate': ['Learn Docker', 'Master Git', 'Linux Administration'],
                'short_term': ['Kubernetes', 'CI/CD Pipelines', 'Infrastructure as Code'],
                'long_term': ['DevOps Lead', 'Site Reliability Engineer', 'Cloud Architect']
            }
        }
        
    async def generate_comprehensive_recommendations(
        self,
        skills: List[Dict[str, Any]],
        gap_analysis: Optional[Dict[str, Any]] = None,
        career_goals: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate enhanced recommendations with proper formatting"""
        
        try:
            logger.info("🎯 Generating enhanced recommendations...")
            
            # Analyze user skills
            user_skills = [skill['skill'].lower() for skill in skills]
            user_level = self._determine_user_level(skills)
            career_path = self._suggest_career_path(user_skills)
            
            # Generate recommendations by category
            recommendations = {
                'immediate_actions': self._generate_immediate_actions(user_skills, user_level),
                'skill_development': self._generate_skill_development(user_skills, user_level),
                'project_suggestions': self._generate_project_suggestions(user_skills, user_level),
                'certification_roadmap': self._generate_certifications(user_skills, user_level),
                'learning_resources': self._generate_learning_resources(user_skills),
                'career_roadmap': self._generate_career_roadmap(career_path, user_level),
                'networking_opportunities': self._generate_networking_opportunities(),
                'timeline': self._generate_timeline(user_level)
            }
            
            logger.info("✅ Enhanced recommendations generated successfully")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced recommendation generation: {e}")
            raise
    
    def _determine_user_level(self, skills: List[Dict[str, Any]]) -> str:
        """Determine user's overall skill level"""
        total_skills = len(skills)
        
        if total_skills <= 3:
            return 'beginner'
        elif total_skills <= 7:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _suggest_career_path(self, user_skills: List[str]) -> str:
        """Suggest career path based on skills"""
        
        if any(skill in ['machine learning', 'tensorflow', 'pytorch', 'data science'] for skill in user_skills):
            return 'data_scientist'
        elif any(skill in ['docker', 'kubernetes', 'aws', 'devops'] for skill in user_skills):
            return 'devops_engineer'
        else:
            return 'software_engineer'
    
    def _generate_immediate_actions(self, user_skills: List[str], user_level: str) -> List[Dict[str, Any]]:
        """Generate immediate action recommendations"""
        
        actions = [
            {
                'title': 'Update Your LinkedIn Profile',
                'description': 'Optimize your professional profile with latest skills and projects',
                'score': 0.85,
                'priority': 'high',
                'estimated_time': '2-3 hours',
                'category': 'profile_optimization'
            },
            {
                'title': 'Create GitHub Portfolio',
                'description': 'Showcase your coding projects and contributions',
                'score': 0.80,
                'priority': 'high',
                'estimated_time': '1-2 days',
                'category': 'portfolio'
            },
            {
                'title': 'Practice Daily Coding',
                'description': 'Maintain sharp coding skills with daily practice',
                'score': 0.75,
                'priority': 'medium',
                'estimated_time': '30 min/day',
                'category': 'skill_maintenance'
            }
        ]
        
        # Add skill-specific actions
        if 'python' in user_skills and user_level == 'beginner':
            actions.append({
                'title': 'Complete Python Basics Course',
                'description': 'Strengthen your Python foundation with structured learning',
                'score': 0.90,
                'priority': 'high',
                'estimated_time': '2-3 weeks',
                'category': 'skill_building'
            })
        
        return actions[:3]  # Return top 3
    
    def _generate_skill_development(self, user_skills: List[str], user_level: str) -> List[Dict[str, Any]]:
        """Generate skill development recommendations"""
        
        recommendations = []
        
        # Generate recommendations for each skill
        for skill in user_skills[:5]:  # Limit to top 5 skills
            skill_lower = skill.lower()
            
            if skill_lower in self.skill_templates:
                templates = self.skill_templates[skill_lower]
                level_templates = templates.get(user_level, templates.get('intermediate', []))
                
                if level_templates:
                    selected_template = random.choice(level_templates)
                    
                    # Calculate score based on skill popularity and user level
                    base_score = 0.6
                    if skill_lower in ['python', 'javascript', 'react']:
                        base_score += 0.2
                    if user_level == 'beginner':
                        base_score += 0.1
                    
                    recommendations.append({
                        'title': selected_template,
                        'description': f'Advance your {skill.title()} skills to the next level',
                        'score': min(base_score + random.uniform(0, 0.1), 0.95),
                        'skill': skill,
                        'current_level': user_level,
                        'estimated_time': self._estimate_learning_time(skill_lower, user_level),
                        'priority': self._calculate_priority(skill_lower),
                        'category': 'technical_skill'
                    })
        
        # Add complementary skills
        complementary_skills = self._get_complementary_skills(user_skills, user_level)
        recommendations.extend(complementary_skills)
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:6]
    
    def _get_complementary_skills(self, user_skills: List[str], user_level: str) -> List[Dict[str, Any]]:
        """Get complementary skills based on current skills"""
        
        recommendations = []
        user_skills_lower = [skill.lower() for skill in user_skills]
        
        # Programming language combinations
        if 'python' in user_skills_lower and 'sql' not in user_skills_lower:
            recommendations.append({
                'title': 'Learn SQL Database Management',
                'description': 'Essential for data manipulation and backend development',
                'score': 0.82,
                'skill': 'SQL',
                'estimated_time': '3-4 weeks',
                'priority': 'high',
                'category': 'complementary_skill'
            })
        
        if 'javascript' in user_skills_lower and 'react' not in user_skills_lower:
            recommendations.append({
                'title': 'Master React Framework',
                'description': 'Popular frontend framework for modern web development',
                'score': 0.78,
                'skill': 'React',
                'estimated_time': '4-6 weeks',
                'priority': 'high',
                'category': 'complementary_skill'
            })
        
        if any(skill in ['python', 'javascript'] for skill in user_skills_lower) and 'docker' not in user_skills_lower:
            recommendations.append({
                'title': 'Learn Docker Containerization',
                'description': 'Essential for modern application deployment',
                'score': 0.75,
                'skill': 'Docker',
                'estimated_time': '2-3 weeks',
                'priority': 'medium',
                'category': 'devops_skill'
            })
        
        return recommendations[:2]
    
    def _generate_project_suggestions(self, user_skills: List[str], user_level: str) -> List[Dict[str, Any]]:
        """Generate project suggestions based on skills"""
        
        projects = []
        user_skills_lower = [skill.lower() for skill in user_skills]
        
        # Python projects
        if 'python' in user_skills_lower:
            if user_level == 'beginner':
                projects.append({
                    'title': 'Personal Budget Tracker',
                    'description': 'Build a Python application to track personal expenses',
                    'score': 0.80,
                    'difficulty': 'beginner',
                    'estimated_time': '1-2 weeks',
                    'skills_used': ['Python', 'File I/O', 'Data Processing'],
                    'category': 'practical_application'
                })
            else:
                projects.append({
                    'title': 'Web Scraping & Analysis Tool',
                    'description': 'Create a tool to scrape and analyze web data',
                    'score': 0.85,
                    'difficulty': 'intermediate',
                    'estimated_time': '2-3 weeks',
                    'skills_used': ['Python', 'BeautifulSoup', 'Pandas', 'Data Analysis'],
                    'category': 'data_project'
                })
        
        # JavaScript/React projects
        if 'javascript' in user_skills_lower or 'react' in user_skills_lower:
            projects.append({
                'title': 'Task Management Web App',
                'description': 'Build a responsive task management application',
                'score': 0.82,
                'difficulty': 'intermediate',
                'estimated_time': '2-4 weeks',
                'skills_used': ['JavaScript', 'React', 'CSS', 'Local Storage'],
                'category': 'web_development'
            })
        
        # Machine Learning projects
        if 'machine learning' in user_skills_lower or 'tensorflow' in user_skills_lower:
            projects.append({
                'title': 'Predictive Analytics Dashboard',
                'description': 'Create an ML model with interactive dashboard',
                'score': 0.88,
                'difficulty': 'advanced',
                'estimated_time': '3-5 weeks',
                'skills_used': ['Python', 'TensorFlow', 'Pandas', 'Visualization'],
                'category': 'machine_learning'
            })
        
        return projects[:3]
    
    def _generate_certifications(self, user_skills: List[str], user_level: str) -> List[Dict[str, Any]]:
        """Generate certification recommendations"""
        
        certifications = []
        user_skills_lower = [skill.lower() for skill in user_skills]
        
        if 'aws' in user_skills_lower or 'cloud' in ' '.join(user_skills_lower):
            if user_level == 'beginner':
                certifications.append({
                    'title': 'AWS Cloud Practitioner',
                    'description': 'Foundational AWS certification for cloud basics',
                    'score': 0.85,
                    'provider': 'Amazon Web Services',
                    'cost': '$100',
                    'estimated_study_time': '4-6 weeks',
                    'difficulty': 'beginner',
                    'priority': 'high'
                })
            else:
                certifications.append({
                    'title': 'AWS Solutions Architect Associate',
                    'description': 'Professional-level AWS architecture certification',
                    'score': 0.90,
                    'provider': 'Amazon Web Services',
                    'cost': '$150',
                    'estimated_study_time': '8-12 weeks',
                    'difficulty': 'intermediate',
                    'priority': 'high'
                })
        
        if 'python' in user_skills_lower and user_level in ['intermediate', 'advanced']:
            certifications.append({
                'title': 'Python Institute PCAP',
                'description': 'Certified Associate in Python Programming',
                'score': 0.75,
                'provider': 'Python Institute',
                'cost': '$295',
                'estimated_study_time': '6-8 weeks',
                'difficulty': 'intermediate',
                'priority': 'medium'
            })
        
        return certifications[:3]
    
    def _generate_learning_resources(self, user_skills: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate learning resource recommendations"""
        
        return {
            'free_resources': [
                {
                    'title': 'freeCodeCamp',
                    'description': 'Comprehensive free coding bootcamp',
                    'url': 'https://freecodecamp.org',
                    'score': 0.90,
                    'category': 'comprehensive',
                    'time_commitment': 'Self-paced'
                },
                {
                    'title': 'MDN Web Docs',
                    'description': 'Authoritative web development documentation',
                    'url': 'https://developer.mozilla.org',
                    'score': 0.85,
                    'category': 'reference',
                    'time_commitment': 'As needed'
                }
            ],
            'paid_courses': [
                {
                    'title': 'Pluralsight Tech Skills',
                    'description': 'Professional technology learning platform',
                    'url': 'https://pluralsight.com',
                    'score': 0.88,
                    'cost': '$29/month',
                    'category': 'professional'
                },
                {
                    'title': 'Udemy Specialized Courses',
                    'description': 'Affordable specialized programming courses',
                    'url': 'https://udemy.com',
                    'score': 0.82,
                    'cost': '$10-200',
                    'category': 'specialized'
                }
            ],
            'practice_platforms': [
                {
                    'title': 'LeetCode',
                    'description': 'Algorithm and data structure practice',
                    'url': 'https://leetcode.com',
                    'score': 0.90,
                    'category': 'algorithms',
                    'focus': 'Problem solving'
                },
                {
                    'title': 'GitHub',
                    'description': 'Open source contribution and portfolio',
                    'url': 'https://github.com',
                    'score': 0.95,
                    'category': 'collaboration',
                    'focus': 'Real projects'
                }
            ]
        }
    
    def _generate_career_roadmap(self, career_path: str, user_level: str) -> Dict[str, Any]:
        """Generate career roadmap recommendations"""
        
        path_data = self.career_paths.get(career_path, self.career_paths['software_engineer'])
        
        roadmap = {
            'career_path': career_path.replace('_', ' ').title(),
            'current_level': user_level,
            'immediate_steps': [],
            'short_term_goals': [],
            'long_term_vision': []
        }
        
        # Add immediate steps with scores
        for step in path_data['immediate']:
            roadmap['immediate_steps'].append({
                'title': step,
                'description': f'Essential immediate action for {career_path.replace("_", " ")} growth',
                'score': random.uniform(0.7, 0.9),
                'timeline': '1-3 months',
                'priority': 'high'
            })
        
        # Add short-term goals
        for goal in path_data['short_term']:
            roadmap['short_term_goals'].append({
                'title': goal,
                'description': f'Medium-term objective for career advancement',
                'score': random.uniform(0.6, 0.8),
                'timeline': '3-12 months',
                'priority': 'medium'
            })
        
        # Add long-term vision
        for vision in path_data['long_term']:
            roadmap['long_term_vision'].append({
                'title': vision,
                'description': f'Long-term career milestone',
                'score': random.uniform(0.5, 0.7),
                'timeline': '1-3 years',
                'priority': 'low'
            })
        
        return roadmap
    
    def _generate_networking_opportunities(self) -> List[Dict[str, Any]]:
        """Generate networking opportunity recommendations"""
        
        return [
            {
                'title': 'Join Tech Communities',
                'description': 'Connect with professionals in Discord/Slack communities',
                'score': 0.80,
                'type': 'online_community',
                'frequency': 'Weekly',
                'platforms': ['Discord', 'Slack', 'Reddit'],
                'effort': 'Low'
            },
            {
                'title': 'Attend Virtual Meetups',
                'description': 'Participate in industry meetups and webinars',
                'score': 0.75,
                'type': 'events',
                'frequency': 'Monthly',
                'platforms': ['Meetup.com', 'Eventbrite', 'LinkedIn Events'],
                'effort': 'Medium'
            },
            {
                'title': 'Find a Mentor',
                'description': 'Connect with experienced professionals for guidance',
                'score': 0.85,
                'type': 'mentorship',
                'frequency': 'Ongoing',
                'platforms': ['ADPList', 'MentorCruise', 'LinkedIn'],
                'effort': 'High'
            }
        ]
    
    def _generate_timeline(self, user_level: str) -> Dict[str, List[Dict[str, Any]]]:
        """Generate 12-month development timeline"""
        
        return {
            'month_1_3': [
                {
                    'title': 'Foundation Building',
                    'description': 'Strengthen core skills and build portfolio',
                    'score': 0.90,
                    'type': 'skill_development',
                    'priority': 'high'
                },
                {
                    'title': 'Network Expansion',
                    'description': 'Join communities and attend virtual events',
                    'score': 0.75,
                    'type': 'networking',
                    'priority': 'medium'
                }
            ],
            'month_4_6': [
                {
                    'title': 'Specialization Focus',
                    'description': 'Deep dive into chosen specialization area',
                    'score': 0.85,
                    'type': 'specialization',
                    'priority': 'high'
                },
                {
                    'title': 'First Certification',
                    'description': 'Complete your first professional certification',
                    'score': 0.80,
                    'type': 'certification',
                    'priority': 'medium'
                }
            ],
            'month_7_9': [
                {
                    'title': 'Advanced Projects',
                    'description': 'Work on complex, portfolio-worthy projects',
                    'score': 0.88,
                    'type': 'project',
                    'priority': 'high'
                }
            ],
            'month_10_12': [
                {
                    'title': 'Career Advancement',
                    'description': 'Apply for senior roles or leadership positions',
                    'score': 0.70,
                    'type': 'career',
                    'priority': 'high'
                }
            ]
        }
    
    def _estimate_learning_time(self, skill: str, level: str) -> str:
        """Estimate learning time for a skill"""
        time_map = {
            'beginner': '2-4 weeks',
            'intermediate': '4-8 weeks',
            'advanced': '8-16 weeks'
        }
        return time_map.get(level, '4-6 weeks')
    
    def _calculate_priority(self, skill: str) -> str:
        """Calculate priority for a skill"""
        high_demand = ['python', 'javascript', 'react', 'aws', 'sql', 'docker']
        if skill in high_demand:
            return 'high'
        return 'medium'

# Test function
async def test_enhanced_engine():
    """Test the enhanced recommendation engine"""
    
    engine = EnhancedRecommendationEngine()
    
    test_skills = [
        {'skill': 'Python', 'normalized_name': 'python', 'experience_level': 'intermediate'},
        {'skill': 'JavaScript', 'normalized_name': 'javascript', 'experience_level': 'beginner'},
        {'skill': 'Machine Learning', 'normalized_name': 'machine learning', 'experience_level': 'intermediate'}
    ]
    
    recommendations = await engine.generate_comprehensive_recommendations(test_skills)
    
    print("🎯 Enhanced Recommendations Test:")
    for category, recs in recommendations.items():
        print(f"\n{category.upper()}:")
        if isinstance(recs, list):
            for rec in recs[:2]:
                if isinstance(rec, dict):
                    title = rec.get('title', 'N/A')
                    score = rec.get('score', 0)
                    print(f"  • {title} (Score: {score:.1%})")
        elif isinstance(recs, dict):
            print(f"  • Complex structure with {len(recs)} sections")

if __name__ == "__main__":
    asyncio.run(test_enhanced_engine())