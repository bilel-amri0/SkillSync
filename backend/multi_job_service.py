"""
SkillSync - Multi Job Board Service
Intègre plusieurs APIs de job boards pour maximiser les résultats
"""

import asyncio
import aiohttp
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiJobService:
    def __init__(self):
        self.session = None
        
        # API Keys et configurations
        self.apis_config = {
            'adzuna': {
                'app_id': os.getenv('ADZUNA_APP_ID', ''),
                'app_key': os.getenv('ADZUNA_APP_KEY', ''),
                'base_url': 'https://api.adzuna.com/v1/api/jobs',
                'enabled': bool(os.getenv('ADZUNA_APP_ID'))
            },
            'rapidapi_jsearch': {
                'api_key': os.getenv('RAPIDAPI_KEY', ''),
                'base_url': 'https://jsearch.p.rapidapi.com',
                'enabled': bool(os.getenv('RAPIDAPI_KEY'))
            },
            'themuse': {
                'base_url': 'https://www.themuse.com/api/public',
                'enabled': True  # API publique gratuite
            },
            'poleemploi': {
                'client_id': os.getenv('POLE_EMPLOI_CLIENT_ID', ''),
                'client_secret': os.getenv('POLE_EMPLOI_CLIENT_SECRET', ''),
                'base_url': 'https://api.emploi-store.fr/partenaire',
                'enabled': bool(os.getenv('POLE_EMPLOI_CLIENT_ID'))
            },
            'tanitjob': {
                'base_url': 'https://www.tanitjobs.com',
                'enabled': True  # Scraping ou API si disponible
            }
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_all_sources(self, skills: List[str], location: str = "fr", max_results: int = 20) -> Dict[str, Any]:
        """
        Recherche sur toutes les sources disponibles en parallèle
        """
        logger.info(f"🔍 Searching jobs across all sources for skills: {skills}")
        
        # Créer les tâches pour chaque API
        tasks = []
        
        if self.apis_config['adzuna']['enabled']:
            tasks.append(self._search_adzuna(skills, location, max_results))
        
        if self.apis_config['rapidapi_jsearch']['enabled']:
            tasks.append(self._search_jsearch(skills, location, max_results))
        
        if self.apis_config['themuse']['enabled']:
            tasks.append(self._search_themuse(skills, max_results))
        
        if self.apis_config['poleemploi']['enabled']:
            tasks.append(self._search_poleemploi(skills, location, max_results))
        
        if self.apis_config['tanitjob']['enabled']:
            tasks.append(self._search_tanitjob(skills, max_results))
        
        # Exécuter toutes les recherches en parallèle
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"❌ Error in parallel search: {e}")
            results = []
        
        # Combiner et traiter les résultats
        combined_jobs = []
        sources_used = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"⚠️ API {i} failed: {result}")
                continue
            
            if result and 'jobs' in result:
                combined_jobs.extend(result['jobs'])
                sources_used.append(result.get('source', f'api_{i}'))
        
        # Dédupliquer et scorer
        unique_jobs = self._deduplicate_jobs(combined_jobs)
        scored_jobs = self._score_jobs(unique_jobs, skills)
        
        # Trier par score décroissant
        scored_jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return {
            'success': True,
            'total_jobs': len(scored_jobs),
            'jobs': scored_jobs[:max_results],
            'sources_used': sources_used,
            'search_parameters': {
                'skills_used': skills,
                'location': location,
                'max_results': max_results
            }
        }
    
    async def _search_adzuna(self, skills: List[str], location: str, max_results: int) -> Dict[str, Any]:
        """Recherche via l'API Adzuna"""
        try:
            config = self.apis_config['adzuna']
            query = " OR ".join(skills[:3])  # Limiter à 3 skills pour l'URL
            
            url = f"{config['base_url']}/{location}/search/1"
            params = {
                'app_id': config['app_id'],
                'app_key': config['app_key'],
                'what': query,
                'results_per_page': min(max_results, 20)
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    jobs = []
                    
                    for job in data.get('results', []):
                        jobs.append({
                            'job_id': f"adzuna_{job.get('id', '')}",
                            'title': job.get('title', ''),
                            'company': job.get('company', {}).get('display_name', ''),
                            'location': job.get('location', {}).get('display_name', ''),
                            'salary_min': job.get('salary_min'),
                            'salary_max': job.get('salary_max'),
                            'description': job.get('description', ''),
                            'url': job.get('redirect_url', ''),
                            'created_date': job.get('created', ''),
                            'source': 'Adzuna'
                        })
                    
                    return {'jobs': jobs, 'source': 'Adzuna'}
                
        except Exception as e:
            logger.warning(f"⚠️ Adzuna API failed: {e}")
        
        return {'jobs': [], 'source': 'Adzuna'}
    
    async def _search_jsearch(self, skills: List[str], location: str, max_results: int) -> Dict[str, Any]:
        """Recherche via RapidAPI JSearch (Indeed, LinkedIn, etc.)"""
        try:
            config = self.apis_config['rapidapi_jsearch']
            query = " ".join(skills[:3])
            
            url = f"{config['base_url']}/search"
            headers = {
                'X-RapidAPI-Key': config['api_key'],
                'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
            }
            params = {
                'query': query,
                'page': '1',
                'num_pages': '1',
                'country': 'FR' if location == 'fr' else 'US'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    jobs = []
                    
                    for job in data.get('data', [])[:max_results]:
                        jobs.append({
                            'job_id': f"jsearch_{job.get('job_id', '')}",
                            'title': job.get('job_title', ''),
                            'company': job.get('employer_name', ''),
                            'location': job.get('job_city', ''),
                            'salary_min': job.get('job_min_salary'),
                            'salary_max': job.get('job_max_salary'),
                            'description': job.get('job_description', ''),
                            'url': job.get('job_apply_link', ''),
                            'created_date': job.get('job_posted_at_datetime_utc', ''),
                            'source': 'JSearch (Indeed/LinkedIn)'
                        })
                    
                    return {'jobs': jobs, 'source': 'JSearch'}
                
        except Exception as e:
            logger.warning(f"⚠️ JSearch API failed: {e}")
        
        return {'jobs': [], 'source': 'JSearch'}
    
    async def _search_themuse(self, skills: List[str], max_results: int) -> Dict[str, Any]:
        """Recherche via The Muse API (gratuite)"""
        try:
            config = self.apis_config['themuse']
            
            url = f"{config['base_url']}/jobs"
            params = {
                'category': 'Engineering',  # Pour les jobs tech
                'page': 1,
                'descending': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    jobs = []
                    
                    for job in data.get('results', [])[:max_results]:
                        company = job.get('company', {})
                        jobs.append({
                            'job_id': f"muse_{job.get('id', '')}",
                            'title': job.get('name', ''),
                            'company': company.get('name', ''),
                            'location': ', '.join([loc.get('name', '') for loc in job.get('locations', [])]),
                            'salary_min': None,
                            'salary_max': None,
                            'description': job.get('contents', ''),
                            'url': job.get('refs', {}).get('landing_page', ''),
                            'created_date': job.get('publication_date', ''),
                            'source': 'The Muse'
                        })
                    
                    return {'jobs': jobs, 'source': 'The Muse'}
                
        except Exception as e:
            logger.warning(f"⚠️ The Muse API failed: {e}")
        
        return {'jobs': [], 'source': 'The Muse'}
    
    async def _search_poleemploi(self, skills: List[str], location: str, max_results: int) -> Dict[str, Any]:
        """Recherche via API Pôle Emploi"""
        try:
            # L'API Pôle Emploi nécessite une authentification OAuth2
            # Pour l'instant, retourner des jobs démo français
            return {
                'jobs': [
                    {
                        'job_id': 'pole_emploi_demo_1',
                        'title': 'Développeur Full Stack',
                        'company': 'Entreprise Française',
                        'location': 'Paris, France',
                        'salary_min': 35000,
                        'salary_max': 45000,
                        'description': 'Poste de développeur Full Stack dans une entreprise innovante.',
                        'url': 'https://pole-emploi.fr',
                        'created_date': datetime.now().isoformat(),
                        'source': 'Pôle Emploi'
                    }
                ],
                'source': 'Pôle Emploi'
            }
        except Exception as e:
            logger.warning(f"⚠️ Pôle Emploi API failed: {e}")
        
        return {'jobs': [], 'source': 'Pôle Emploi'}
    
    async def _search_tanitjob(self, skills: List[str], max_results: int) -> Dict[str, Any]:
        """Recherche via TanitJob (Tunisie)"""
        try:
            # Pour TanitJob, on peut implémenter du web scraping ou utiliser leur API si disponible
            # Pour l'instant, retourner des jobs démo tunisiens
            return {
                'jobs': [
                    {
                        'job_id': 'tanitjob_demo_1',
                        'title': 'Développeur Python',
                        'company': 'StartupTN',
                        'location': 'Tunis, Tunisie',
                        'salary_min': 1200,
                        'salary_max': 2000,
                        'description': 'Poste de développeur Python dans une startup tunisienne dynamique.',
                        'url': 'https://tanitjobs.com',
                        'created_date': datetime.now().isoformat(),
                        'source': 'TanitJob'
                    },
                    {
                        'job_id': 'tanitjob_demo_2',
                        'title': 'Ingénieur Logiciel',
                        'company': 'TechTunisia',
                        'location': 'Sfax, Tunisie',
                        'salary_min': 1000,
                        'salary_max': 1800,
                        'description': 'Ingénieur logiciel spécialisé en Java et React.',
                        'url': 'https://tanitjobs.com',
                        'created_date': datetime.now().isoformat(),
                        'source': 'TanitJob'
                    }
                ],
                'source': 'TanitJob'
            }
        except Exception as e:
            logger.warning(f"⚠️ TanitJob failed: {e}")
        
        return {'jobs': [], 'source': 'TanitJob'}
    
    def _deduplicate_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Supprime les doublons basés sur titre + entreprise"""
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            key = f"{job.get('title', '').lower()}_{job.get('company', '').lower()}"
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        return unique_jobs
    
    def _score_jobs(self, jobs: List[Dict], user_skills: List[str]) -> List[Dict]:
        """Score les jobs basé sur la correspondance des compétences"""
        user_skills_lower = [skill.lower() for skill in user_skills]
        
        for job in jobs:
            # Analyser le titre et la description pour les compétences
            text_to_analyze = f"{job.get('title', '')} {job.get('description', '')}".lower()
            
            matching_skills = []
            for skill in user_skills_lower:
                if skill in text_to_analyze:
                    matching_skills.append(skill)
            
            # Calculer le score de correspondance
            if user_skills_lower:
                match_score = (len(matching_skills) / len(user_skills_lower)) * 100
            else:
                match_score = 50  # Score par défaut
            
            # Bonus pour certaines sources
            if job.get('source') in ['Adzuna', 'JSearch']:
                match_score += 5
            
            job['match_score'] = round(min(match_score, 95), 1)  # Plafonner à 95%
            job['matching_skills'] = matching_skills
        
        return jobs

# Instance globale
multi_job_service = MultiJobService()

async def search_jobs_multi_source(skills: List[str], location: str = "fr", max_results: int = 20) -> Dict[str, Any]:
    """
    Fonction principale pour rechercher des emplois sur plusieurs sources
    """
    async with MultiJobService() as service:
        return await service.search_all_sources(skills, location, max_results)
