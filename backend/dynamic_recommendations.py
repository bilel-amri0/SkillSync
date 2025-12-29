"""
 Advanced Dynamic Recommendation Engine for SkillSync
This module provides intelligent, CV-based personalized skill recommendations.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SkillRecommendation:
    """Data class for skill recommendations"""
    skill: str
    category: str
    priority: str
    reason: str
    learning_path: List[str]
    resources: List[str]

class DynamicRecommendationEngine:
    """
    Advanced recommendation engine that analyzes CV content and generates 
    personalized skill recommendations based on experience level, domain, and gaps.
    """
    
    def __init__(self):
        self.skill_domains = {
            "frontend": {
                "skills": ["React", "Vue.js", "Angular", "TypeScript", "HTML/CSS", "JavaScript", "Sass/SCSS", "Webpack", "Responsive Design"],
                "keywords": ["frontend", "ui", "ux", "interface", "react", "vue", "angular", "html", "css", "javascript"]
            },
            "backend": {
                "skills": ["Node.js", "Python/Django", "Java/Spring", "C#/.NET", "PHP/Laravel", "Ruby/Rails", "API Development", "Microservices"],
                "keywords": ["backend", "server", "api", "database", "node", "python", "java", "django", "spring", "laravel"]
            },
            "devops": {
                "skills": ["Docker", "Kubernetes", "AWS", "Azure", "CI/CD", "Jenkins", "Terraform", "Monitoring", "Linux Administration"],
                "keywords": ["devops", "docker", "kubernetes", "aws", "azure", "jenkins", "terraform", "deployment", "infrastructure"]
            },
            "data_science": {
                "skills": ["Python/Pandas", "Machine Learning", "TensorFlow", "PyTorch", "SQL", "R", "Tableau", "Data Visualization", "Statistics"],
                "keywords": ["data", "machine learning", "python", "pandas", "tensorflow", "pytorch", "analytics", "statistics", "sql"]
            },
            "mobile": {
                "skills": ["React Native", "Flutter", "Swift/iOS", "Kotlin/Android", "Mobile UI/UX", "App Store Optimization"],
                "keywords": ["mobile", "ios", "android", "react native", "flutter", "swift", "kotlin", "app development"]
            },
            "cybersecurity": {
                "skills": ["Ethical Hacking", "Network Security", "CISSP", "Vulnerability Assessment", "Incident Response", "Cryptography"],
                "keywords": ["security", "cybersecurity", "hacking", "network security", "vulnerability", "encryption", "firewall"]
            }
        }
        
        self.experience_levels = {
            "junior": {"years": (0, 2), "focus": "foundational skills"},
            "mid": {"years": (2, 5), "focus": "specialization and best practices"},
            "senior": {"years": (5, 100), "focus": "leadership and architecture"}
        }
    
    def analyze_cv_content(self, cv_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze CV content to extract key insights for recommendations
        """
        logger.info(" Analyzing CV content for personalized recommendations")
        
        try:
            # Extract text content from various sources
            text_content = ""
            if isinstance(cv_data, dict):
                if 'text' in cv_data:
                    text_content = cv_data['text']
                elif 'content' in cv_data:
                    text_content = cv_data['content']
                elif 'raw_text' in cv_data:
                    text_content = cv_data['raw_text']
            elif isinstance(cv_data, str):
                text_content = cv_data
            
            if not text_content:
                logger.warning(" No text content found in CV data")
                return self._get_default_analysis()
            
            text_lower = text_content.lower()
            
            # Determine experience level
            experience_level = self._determine_experience_level(text_lower)
            
            # Identify skill domains
            identified_domains = self._identify_skill_domains(text_lower)
            
            # Extract current skills
            current_skills = self._extract_current_skills(text_lower)
            
            # Identify role focus
            role_focus = self._identify_role_focus(text_lower)
            
            analysis = {
                "experience_level": experience_level,
                "primary_domains": identified_domains,
                "current_skills": current_skills,
                "role_focus": role_focus,
                "text_length": len(text_content),
                "has_technical_content": self._has_technical_content(text_lower)
            }
            
            logger.info(f" CV Analysis: {experience_level} level, domains: {identified_domains}, skills: {len(current_skills)}")
            return analysis
            
        except Exception as e:
            logger.error(f" Error analyzing CV content: {e}")
            logger.info(" Returning default analysis")
            return self._get_default_analysis()
    
    def _determine_experience_level(self, text: str) -> str:
        """Determine experience level based on CV content"""
        experience_indicators = {
            "senior": ["senior", "lead", "principal", "architect", "manager", "director", "team lead", "technical lead"],
            "mid": ["developer", "engineer", "analyst", "specialist", "consultant", "coordinator"],
            "junior": ["intern", "junior", "trainee", "assistant", "graduate", "entry"]
        }
        
        # Check for explicit experience mentions
        years_match = re.findall(r'(\d+)\s*(?:years?|ans?)', text)
        if years_match:
            max_years = max([int(year) for year in years_match])
            if max_years >= 5:
                return "senior"
            elif max_years >= 2:
                return "mid"
            else:
                return "junior"
        
        # Check for role indicators
        for level, indicators in experience_indicators.items():
            if any(indicator in text for indicator in indicators):
                return level
        
        return "mid"  # Default
    
    def _identify_skill_domains(self, text: str) -> List[str]:
        """Identify relevant skill domains from CV text"""
        identified = []
        
        for domain, info in self.skill_domains.items():
            keyword_matches = sum(1 for keyword in info["keywords"] if keyword in text)
            if keyword_matches >= 2:  # At least 2 keyword matches
                identified.append(domain)
        
        # If no domains identified, default to most common
        if not identified:
            identified = ["frontend", "backend"]
        
        return identified[:3]  # Max 3 domains
    
    def _extract_current_skills(self, text: str) -> List[str]:
        """Extract current skills mentioned in the CV"""
        skills = []
        
        for domain_info in self.skill_domains.values():
            for skill in domain_info["skills"]:
                if skill.lower() in text:
                    skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _identify_role_focus(self, text: str) -> str:
        """Identify the primary role focus"""
        role_indicators = {
            "fullstack": ["fullstack", "full-stack", "full stack"],
            "frontend": ["frontend", "front-end", "ui", "ux", "user interface"],
            "backend": ["backend", "back-end", "server", "api"],
            "devops": ["devops", "dev-ops", "infrastructure", "deployment"],
            "data": ["data scientist", "data analyst", "machine learning", "ai"],
            "mobile": ["mobile", "ios", "android", "app development"]
        }
        
        for role, indicators in role_indicators.items():
            if any(indicator in text for indicator in indicators):
                return role
        
        return "fullstack"  # Default
    
    def _has_technical_content(self, text: str) -> bool:
        """Check if CV has substantial technical content"""
        technical_keywords = ["programming", "development", "software", "coding", "technical", "engineering", "technology"]
        return any(keyword in text for keyword in technical_keywords)
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when CV content is not available"""
        return {
            "experience_level": "mid",
            "primary_domains": ["frontend", "backend"],
            "current_skills": [],
            "role_focus": "fullstack",
            "text_length": 0,
            "has_technical_content": True
        }
    
    def generate_skill_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific skill recommendations based on analysis"""
        logger.info(" Generating skill recommendations based on analysis")
        
        try:
            recommendations = []
            experience_level = analysis.get("experience_level", "mid")
            primary_domains = analysis.get("primary_domains", ["frontend", "backend"])
            current_skills = analysis.get("current_skills", [])
            
            # Generate recommendations for each domain
            for domain in primary_domains:
                if domain not in self.skill_domains:
                    logger.warning(f" Unknown domain: {domain}, skipping")
                    continue
                    
                domain_skills = self.skill_domains[domain]["skills"]
                
                # Filter out skills the person already has
                recommended_skills = [skill for skill in domain_skills if skill not in current_skills]
                
                # Select top skills based on experience level
                if experience_level == "junior":
                    selected_skills = recommended_skills[:2]  # 2 skills per domain
                    priority = "High"
                    learning_focus = "foundational understanding"
                elif experience_level == "mid":
                    selected_skills = recommended_skills[:3]  # 3 skills per domain
                    priority = "Medium"
                    learning_focus = "practical application"
                else:  # senior
                    selected_skills = recommended_skills[:2]  # 2 advanced skills
                    priority = "Low"
                    learning_focus = "mastery and leadership"
                
                # Create recommendations for selected skills
                for skill in selected_skills:
                    recommendation = {
                        "skill": skill,
                        "category": domain.title(),
                        "priority": priority,
                        "reason": f"Enhance your {domain} expertise with {learning_focus}",
                        "learning_path": self._get_learning_path(skill, experience_level),
                        "estimated_time": self._get_estimated_learning_time(skill, experience_level),
                        "difficulty": self._get_skill_difficulty(skill, experience_level)
                    }
                    recommendations.append(recommendation)
            
            # Add cross-domain recommendations
            if len(recommendations) < 5:
                cross_domain_recs = self._generate_cross_domain_recommendations(analysis)
                recommendations.extend(cross_domain_recs)
            
            # Limit to top 6 recommendations
            final_recommendations = recommendations[:6]
            logger.info(f" Successfully generated {len(final_recommendations)} skill recommendations")
            return final_recommendations
            
        except Exception as e:
            logger.error(f" Error generating skill recommendations: {e}")
            # Return basic fallback recommendations
            return [
                {
                    "skill": "JavaScript ES6+",
                    "category": "Frontend",
                    "priority": "Medium",
                    "reason": "Essential for modern web development",
                    "learning_path": ["Fundamentals", "Basic Implementation", "Best Practices"],
                    "estimated_time": "2-4 weeks",
                    "difficulty": "Intermediate"
                },
                {
                    "skill": "Git/Version Control",
                    "category": "General",
                    "priority": "High",
                    "reason": "Critical for all developers",
                    "learning_path": ["Basic Concepts", "Hands-on Practice", "Advanced Techniques"],
                    "estimated_time": "2-3 weeks",
                    "difficulty": "Intermediate"
                }
            ]
    
    def _get_learning_path(self, skill: str, experience_level: str) -> List[str]:
        """Get learning path for a specific skill"""
        paths = {
            "React": ["JavaScript Fundamentals", "React Basics", "Component Development", "State Management", "React Hooks"],
            "Node.js": ["JavaScript ES6+", "Node.js Basics", "Express.js", "Database Integration", "API Development"],
            "Docker": ["Containerization Concepts", "Docker Basics", "Dockerfile Creation", "Docker Compose", "Container Orchestration"],
            "Python/Django": ["Python Basics", "Django Framework", "Model-View-Template", "REST APIs", "Database ORM"],
            "AWS": ["Cloud Concepts", "AWS Core Services", "EC2 & S3", "IAM & Security", "Deployment Strategies"]
        }
        
        default_path = ["Fundamentals", "Basic Implementation", "Best Practices", "Advanced Concepts", "Real-world Projects"]
        return paths.get(skill, default_path)
    
    def _get_estimated_learning_time(self, skill: str, experience_level: str) -> str:
        """Estimate learning time based on skill complexity and experience level"""
        base_times = {
            "junior": "4-6 weeks",
            "mid": "2-4 weeks", 
            "senior": "1-2 weeks"
        }
        return base_times[experience_level]
    
    def _get_skill_difficulty(self, skill: str, experience_level: str) -> str:
        """Determine skill difficulty relative to experience level"""
        difficulty_map = {
            "junior": "Beginner",
            "mid": "Intermediate",
            "senior": "Advanced"
        }
        return difficulty_map[experience_level]
    
    def _generate_cross_domain_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cross-domain skill recommendations"""
        try:
            cross_domain_skills = {
                "Git/Version Control": "Essential for all developers",
                "Testing/TDD": "Critical for code quality",
                "Agile/Scrum": "Important for team collaboration",
                "System Design": "Crucial for scalable solutions"
            }
            
            recommendations = []
            for skill, reason in cross_domain_skills.items():
                recommendation = {
                    "skill": skill,
                    "category": "General",
                    "priority": "Medium",
                    "reason": reason,
                    "learning_path": ["Basic Concepts", "Hands-on Practice", "Advanced Techniques"],
                    "estimated_time": "2-3 weeks",
                    "difficulty": "Intermediate"
                }
                recommendations.append(recommendation)
            
            return recommendations[:2]  # Return max 2 cross-domain recommendations
            
        except Exception as e:
            logger.error(f" Error generating cross-domain recommendations: {e}")
            return []  # Return empty list if error occurs
    
    def generate_personalized_recommendations(self, cv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main method to generate personalized recommendations
        """
        logger.info(" Generating personalized recommendations based on CV content")
        
        # Analyze CV content
        analysis = self.analyze_cv_content(cv_data)
        
        # Generate skill recommendations
        recommendations = self.generate_skill_recommendations(analysis)
        
        logger.info(f" Generated {len(recommendations)} personalized skill recommendations")
        return recommendations

# Create global instance
dynamic_engine = DynamicRecommendationEngine()