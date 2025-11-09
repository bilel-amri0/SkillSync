"""
Simple CV Analyzer - Version sans dépendances ML lourdes
"""

import re
import json
from typing import Dict, List, Any
import PyPDF2
import io

class SimpleCVAnalyzer:
    """Analyseur CV simplifié sans ML"""
    
    def __init__(self):
        # Mots-clés techniques courants
        self.tech_skills = [
            # Langages de programmation
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'scala', 'kotlin', 'swift', 'dart', 'r', 'matlab', 'sql', 'html', 'css',
            
            # Frameworks et bibliothèques
            'react', 'vue', 'angular', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel',
            'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn', 'opencv',
            
            # Technologies et outils
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'jenkins', 'terraform',
            'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka',
            'linux', 'windows', 'macos', 'bash', 'powershell',
            
            # Méthodologies
            'agile', 'scrum', 'devops', 'ci/cd', 'microservices', 'api', 'rest', 'graphql'
        ]
        
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
            'creativity', 'adaptability', 'time management', 'project management', 'analytical',
            'collaboration', 'innovation', 'mentoring', 'presentation'
        ]

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extrait le texte d'un fichier PDF"""
        try:
            if hasattr(pdf_file, 'read'):
                content = pdf_file.read()
                if isinstance(content, str):
                    pdf_file = io.BytesIO(content.encode())
                else:
                    pdf_file = io.BytesIO(content)
            
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
            return text
        except Exception as e:
            print(f"Erreur lors de l'extraction PDF: {e}")
            return ""

    def extract_personal_info(self, text: str) -> Dict[str, str]:
        """Extrait les informations personnelles"""
        info = {}
        
        # Extraction du nom (souvent en début de CV)
        lines = text.split('\n')
        for line in lines[:5]:  # Cherche dans les 5 premières lignes
            line = line.strip()
            if len(line) > 2 and len(line) < 50 and not '@' in line:
                # Vérifie si ça ressemble à un nom
                if re.match(r'^[A-Za-z\s\-\'\.]+$', line):
                    info['name'] = line
                    break
        
        # Extraction email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            info['email'] = emails[0]
        
        # Extraction téléphone
        phone_pattern = r'(\+?[0-9\s\-\(\)]{8,15})'
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            cleaned_phone = re.sub(r'[^\d+]', '', phone)
            if len(cleaned_phone) >= 8:
                info['phone'] = phone.strip()
                break
        
        return info

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extrait les compétences techniques et soft skills"""
        text_lower = text.lower()
        
        found_tech_skills = []
        found_soft_skills = []
        
        # Recherche des compétences techniques
        for skill in self.tech_skills:
            if skill.lower() in text_lower:
                found_tech_skills.append(skill)
        
        # Recherche des soft skills
        for skill in self.soft_skills:
            if skill.lower() in text_lower:
                found_soft_skills.append(skill)
        
        return {
            'technical': found_tech_skills,
            'soft': found_soft_skills
        }

    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extrait l'expérience professionnelle"""
        experiences = []
        
        # Mots-clés pour identifier les sections d'expérience
        exp_keywords = ['experience', 'work', 'employment', 'career', 'professional', 'expérience']
        
        # Cherche la section expérience
        lines = text.split('\n')
        in_experience_section = False
        current_exp = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Détecte le début de la section expérience
            if any(keyword in line_lower for keyword in exp_keywords):
                in_experience_section = True
                continue
            
            # Détecte une nouvelle section (arrête l'analyse d'expérience)
            if in_experience_section and line.strip() and line.strip().upper() == line.strip():
                # Probable nouvelle section en majuscules
                section_keywords = ['education', 'skills', 'formation', 'compétences', 'projets', 'projects']
                if any(keyword in line_lower for keyword in section_keywords):
                    break
            
            if in_experience_section and line.strip():
                # Détection d'une date (années)
                date_pattern = r'(20\d{2}|19\d{2})'
                if re.search(date_pattern, line):
                    if current_exp:
                        experiences.append(current_exp)
                    current_exp = {
                        'title': 'Poste professionnel',
                        'company': 'Entreprise',
                        'date': line.strip(),
                        'description': ''
                    }
                elif current_exp and line.strip():
                    # Ajoute la description
                    if not current_exp['title'] or current_exp['title'] == 'Poste professionnel':
                        current_exp['title'] = line.strip()
                    else:
                        current_exp['description'] += line.strip() + ' '
        
        # Ajoute la dernière expérience
        if current_exp:
            experiences.append(current_exp)
        
        return experiences

    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extrait la formation"""
        education = []
        
        # Mots-clés pour identifier les sections d'éducation
        edu_keywords = ['education', 'formation', 'études', 'diplome', 'degree', 'university', 'école']
        
        lines = text.split('\n')
        in_education_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Détecte le début de la section éducation
            if any(keyword in line_lower for keyword in edu_keywords):
                in_education_section = True
                continue
            
            if in_education_section and line.strip():
                # Détection d'une date
                date_pattern = r'(20\d{2}|19\d{2})'
                if re.search(date_pattern, line):
                    education.append({
                        'degree': line.strip(),
                        'institution': 'Institution',
                        'date': re.search(date_pattern, line).group(),
                        'description': ''
                    })
                
                # Arrête à la prochaine section
                if line.strip().upper() == line.strip() and len(line.strip()) > 5:
                    break
        
        return education

    def analyze_cv(self, pdf_file) -> Dict[str, Any]:
        """Analyse complète du CV"""
        try:
            # Extraction du texte
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                return self._get_default_analysis()
            
            # Extraction des informations
            personal_info = self.extract_personal_info(text)
            skills = self.extract_skills(text)
            experience = self.extract_experience(text)
            education = self.extract_education(text)
            
            # Génération de recommandations basiques
            recommendations = self._generate_basic_recommendations(skills, experience)
            
            return {
                'personal_info': personal_info,
                'contact_info': {
                    'email': personal_info.get('email', ''),
                    'phone': personal_info.get('phone', '')
                },
                'skills': skills,
                'experience': experience,
                'education': education,
                'analysis': {
                    'total_skills': len(skills.get('technical', [])) + len(skills.get('soft', [])),
                    'years_experience': len(experience),
                    'education_level': len(education),
                    'recommendations': recommendations
                }
            }
        
        except Exception as e:
            print(f"Erreur lors de l'analyse: {e}")
            return self._get_default_analysis()

    def _generate_basic_recommendations(self, skills: Dict, experience: List) -> List[str]:
        """Génère des recommandations basiques"""
        recommendations = []
        
        tech_skills = skills.get('technical', [])
        exp_count = len(experience)
        
        if len(tech_skills) < 5:
            recommendations.append("Développer plus de compétences techniques")
        
        if exp_count < 2:
            recommendations.append("Acquérir plus d'expérience professionnelle")
        
        if 'python' in tech_skills and 'django' not in tech_skills:
            recommendations.append("Apprendre Django pour le développement web Python")
        
        if 'javascript' in tech_skills and 'react' not in tech_skills:
            recommendations.append("Apprendre React pour le développement frontend")
        
        recommendations.extend([
            "Mettre à jour régulièrement ses compétences",
            "Participer à des projets open source",
            "Développer son réseau professionnel",
            "Créer un portfolio en ligne"
        ])
        
        return recommendations[:4]  # Limite à 4 recommandations

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Retourne une analyse par défaut en cas d'erreur"""
        return {
            'personal_info': {'name': 'Nom non détecté'},
            'contact_info': {'email': '', 'phone': ''},
            'skills': {'technical': [], 'soft': []},
            'experience': [],
            'education': [],
            'analysis': {
                'total_skills': 0,
                'years_experience': 0,
                'education_level': 0,
                'recommendations': [
                    "Améliorer la structure du CV",
                    "Ajouter plus de détails sur l'expérience",
                    "Lister clairement les compétences",
                    "Inclure les informations de contact"
                ]
            }
        }

# Instance globale
cv_analyzer = SimpleCVAnalyzer()
