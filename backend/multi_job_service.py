"""
SkillSync - Multi Job Board Service
Intgre plusieurs APIs de job boards pour maximiser les rsultats
"""

import asyncio
import aiohttp
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from services.multi_job_api_service import MultiJobAPIService, JobResult

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


multi_api_service = MultiJobAPIService()
REMOTE_KEYWORDS = {"remote", "anywhere", "worldwide", "global", "telework", "work-from-home"}


def _select_job_query_skills(skills: List[str], min_terms: int = 5, max_terms: int = 7) -> List[str]:
    """Pick the richest mix of skills for live queries while avoiding duplicates."""
    seen = set()
    ordered: List[str] = []
    for raw in skills:
        if not isinstance(raw, str):
            continue
        normalized = raw.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)

    if not ordered:
        return []

    strong_terms = [skill for skill in ordered if len(skill) >= 3]
    prioritized = strong_terms[:max_terms]
    if len(prioritized) < min_terms:
        for skill in ordered:
            if skill in prioritized:
                continue
            prioritized.append(skill)
            if len(prioritized) >= max_terms:
                break

    return prioritized[:max_terms]


def _score_job_result(job: JobResult, user_skills: List[str]) -> Tuple[float, List[str]]:
    user_tokens = [skill.lower() for skill in user_skills if isinstance(skill, str)]
    title = job.title or ""
    description = job.description or ""
    text = f"{title} {description}".lower()
    matches = [skill for skill in user_tokens if skill and skill in text]
    score = (len(matches) / max(len(user_tokens), 1)) * 100 if user_tokens else 60.0
    return round(score, 1), [match.title() for match in matches]


def _location_tokens(location: str) -> List[str]:
    raw = (location or "").strip().lower()
    if not raw:
        return []
    return [token for token in re.split(r"[,/\-\s]+", raw) if token]


def _job_matches_location(job: JobResult, target_location: str) -> bool:
    if not target_location:
        return True
    target_tokens = _location_tokens(target_location)
    if not target_tokens:
        return True

    job_location = (job.location or "").lower()
    job_remote = bool(job.remote or ("remote" in job_location))

    # If user asked for remote roles explicitly
    if any(token in REMOTE_KEYWORDS for token in target_tokens):
        geo_tokens = [token for token in target_tokens if token not in REMOTE_KEYWORDS and token != "near"]
        if "near" in target_tokens:
            if not geo_tokens:
                return job_remote
            return job_remote or any(token in job_location for token in geo_tokens)
        if geo_tokens:
            return job_remote or any(token in job_location for token in geo_tokens)
        return job_remote

    # Non-remote location preference: require overlap in tokens
    if job_location:
        return any(token in job_location for token in target_tokens)

    # If no location provided but job is remote, treat as acceptable near match
    return job_remote


def _convert_job_results_to_payload(
    job_results: List[JobResult],
    user_skills: List[str],
    max_results: int,
    target_location: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    jobs: List[Dict[str, Any]] = []
    sources: List[str] = []
    for job in job_results:
        if len(jobs) >= max_results:
            break
        if not _job_matches_location(job, target_location):
            continue
        score, matches = _score_job_result(job, user_skills)
        description = (job.description or '')
        description_preview = description[:800] + '...' if len(description) > 800 else description
        job_dict = {
            'job_id': job.id,
            'title': job.title,
            'company': job.company,
            'location': job.location,
            'salary_min': None,
            'salary_max': None,
            'salary': job.salary,
            'description': description_preview,
            'url': job.url,
            'created_date': job.posted_date,
            'source': job.source,
            'match_score': score,
            'matching_skills': matches,
            'remote': job.remote
        }
        jobs.append(job_dict)
        if job.source and job.source not in sources:
            sources.append(job.source)
    return jobs, sources

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
        Recherche sur toutes les sources disponibles en parallle
        """
        logger.info(f" Searching jobs across all sources for skills: {skills}")
        
        # Crer les tches pour chaque API
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
        
        # Excuter toutes les recherches en parallle
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f" Error in parallel search: {e}")
            results = []
        
        # Combiner et traiter les rsultats
        combined_jobs = []
        sources_used = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f" API {i} failed: {result}")
                continue
            
            if result and 'jobs' in result:
                combined_jobs.extend(result['jobs'])
                sources_used.append(result.get('source', f'api_{i}'))
        
        # Ddupliquer et scorer
        unique_jobs = self._deduplicate_jobs(combined_jobs)
        scored_jobs = self._score_jobs(unique_jobs, skills)
        
        # Trier par score dcroissant
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
            query = " OR ".join(skills[:3])  # Limiter  3 skills pour l'URL
            
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
            logger.warning(f" Adzuna API failed: {e}")
        
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
            logger.warning(f" JSearch API failed: {e}")
        
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
            logger.warning(f" The Muse API failed: {e}")
        
        return {'jobs': [], 'source': 'The Muse'}
    
    async def _search_poleemploi(self, skills: List[str], location: str, max_results: int) -> Dict[str, Any]:
        """Recherche via API Ple Emploi"""
        try:
            logger.warning(" Ple Emploi integration not implemented yet - awaiting OAuth client + ingestion pipeline")
            # TODO: Implement OAuth2 client credential flow and persist live postings.
            return {'jobs': [], 'source': 'Ple Emploi'}
        except Exception as e:
            logger.warning(f" Ple Emploi API failed: {e}")
        
        return {'jobs': [], 'source': 'Ple Emploi'}
    
    async def _search_tanitjob(self, skills: List[str], max_results: int) -> Dict[str, Any]:
        """Recherche via TanitJob (Tunisie)"""
        try:
            logger.warning(" TanitJob integration not implemented yet - scraping/API connector pending")
            # TODO: Implement TanitJob connector via official API or compliant scraping pipeline.
            return {'jobs': [], 'source': 'TanitJob'}
        except Exception as e:
            logger.warning(f" TanitJob failed: {e}")
        
        return {'jobs': [], 'source': 'TanitJob'}
    
    def _deduplicate_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Supprime les doublons bass sur titre + entreprise"""
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            key = f"{job.get('title', '').lower()}_{job.get('company', '').lower()}"
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        return unique_jobs
    
    def _score_jobs(self, jobs: List[Dict], user_skills: List[str]) -> List[Dict]:
        """Score les jobs bas sur la correspondance des comptences"""
        user_skills_lower = [skill.lower() for skill in user_skills]
        
        for job in jobs:
            # Analyser le titre et la description pour les comptences
            text_to_analyze = f"{job.get('title', '')} {job.get('description', '')}".lower()
            
            matching_skills = []
            for skill in user_skills_lower:
                if skill in text_to_analyze:
                    matching_skills.append(skill)
            
            # Calculer le score de correspondance
            if user_skills_lower:
                match_score = (len(matching_skills) / len(user_skills_lower)) * 100
            else:
                match_score = 50  # Score par dfaut
            
            # Bonus pour certaines sources
            if job.get('source') in ['Adzuna', 'JSearch']:
                match_score += 5
            
            job['match_score'] = round(min(match_score, 95), 1)  # Plafonner  95%
            job['matching_skills'] = matching_skills
        
        return jobs

# Instance globale
multi_job_service = MultiJobService()

async def search_jobs_multi_source(skills: List[str], location: str = "fr", max_results: int = 20) -> Dict[str, Any]:
    """
    Fonction principale pour rechercher des emplois sur plusieurs sources
    """
    query_skills = _select_job_query_skills(skills)
    query = " ".join(query_skills) if query_skills else "developer"
    try:
        job_results = await multi_api_service.search_jobs(query=query, location=location, skills=skills)
        if job_results:
            jobs_payload, sources = _convert_job_results_to_payload(job_results, skills, max_results, location)
            if jobs_payload:
                return {
                    'success': True,
                    'total_jobs': len(job_results),
                    'jobs': jobs_payload,
                    'sources_used': sources,
                    'search_parameters': {
                        'skills_used': query_skills,
                        'location': location,
                        'max_results': max_results,
                        'query': query
                    }
                }
    except Exception as exc:
        logger.warning(f" Enhanced multi-API job search failed, falling back to legacy service: {exc}")

    async with MultiJobService() as service:
        return await service.search_all_sources(skills, location, max_results)
