"""Course catalog client for retrieving live learning resources."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class CourseCatalogClient:
    """Fetches learning resources from live course catalogs."""

    UDACITY_ENDPOINT = "https://catalog-api.udacity.com/v1/courses"

    def __init__(
        self,
        internal_catalog_url: Optional[str] = None,
        timeout_seconds: int = 15,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.internal_catalog_url = internal_catalog_url or os.getenv("COURSE_CATALOG_URL")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

        self._udacity_cache: List[Dict[str, Any]] = []
        self._udacity_cache_ts: float = 0.0
        self._udacity_cache_ttl = int(os.getenv("COURSE_CATALOG_TTL_SECONDS", "3600"))

    def fetch_resources_for_skills(
        self,
        skills: List[str],
        target_role: Optional[str] = None,
        overall_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return curated resources for the provided skills."""
        if not skills:
            return []

        per_skill_limit = max(1, overall_limit // len(skills))
        aggregated: List[Dict[str, Any]] = []

        for skill in skills:
            aggregated.extend(self._query_internal_catalog(skill, target_role, per_skill_limit))
            aggregated.extend(self._query_udacity_catalog(skill, target_role, per_skill_limit))

        unique_resources = self._deduplicate_resources(aggregated)
        ranked = sorted(unique_resources, key=lambda item: item.get("score", 0), reverse=True)
        return ranked[:overall_limit]

    # Internal catalog -------------------------------------------------
    def _query_internal_catalog(
        self,
        skill: str,
        target_role: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.internal_catalog_url:
            return []

        payload = {
            "skill": skill,
            "target_role": target_role,
            "limit": limit,
        }

        try:
            response = self.session.post(
                self.internal_catalog_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Course catalog endpoint failed (skill=%s): %s",
                skill,
                exc,
            )
            return []

        courses = data.get("courses") or data.get("results") or []
        resources: List[Dict[str, Any]] = []

        for course in courses:
            if len(resources) >= limit:
                break

            resource = self._normalize_resource(
                title=course.get("title") or course.get("name"),
                provider=course.get("provider") or course.get("platform") or data.get("provider"),
                url=course.get("url") or course.get("link"),
                duration=course.get("duration"),
                difficulty=course.get("level"),
                cost=course.get("cost") or course.get("price"),
                tags=course.get("skills") or course.get("tags"),
                skill=skill,
                target_role=target_role,
                source="internal_catalog",
            )
            if resource:
                resources.append(resource)

        return resources

    # Udacity ----------------------------------------------------------
    def _query_udacity_catalog(
        self,
        skill: str,
        target_role: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        catalog = self._get_udacity_catalog()
        if not catalog:
            return []

        matches: List[Dict[str, Any]] = []
        skill_lower = skill.lower()
        role_lower = (target_role or "").lower()

        for course in catalog:
            if len(matches) >= limit:
                break

            haystack = " ".join(
                [
                    str(course.get("title", "")),
                    str(course.get("short_summary", "")),
                    " ".join(course.get("skills", [])),
                    str(course.get("primary_subcategory", "")),
                ]
            ).lower()

            if skill_lower not in haystack:
                continue

            title = course.get("title") or ""
            url = course.get("homepage") or f"https://www.udacity.com/course/{course.get('slug', '')}"

            matches.append(
                self._normalize_resource(
                    title=title,
                    provider="Udacity",
                    url=url,
                    duration=self._format_duration(course),
                    difficulty=course.get("level"),
                    cost="paid",
                    tags=course.get("skills"),
                    skill=skill,
                    target_role=target_role,
                    source="udacity_catalog",
                    extra_score=1.0 if role_lower and role_lower in haystack else 0.0,
                )
            )

        return matches

    def _get_udacity_catalog(self) -> List[Dict[str, Any]]:
        now = time.time()
        if self._udacity_cache and (now - self._udacity_cache_ts) < self._udacity_cache_ttl:
            return self._udacity_cache

        try:
            response = self.session.get(self.UDACITY_ENDPOINT, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            self._udacity_cache = payload.get("courses", [])
            self._udacity_cache_ts = now
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to refresh Udacity catalog: %s", exc)
            self._udacity_cache = []

        return self._udacity_cache

    # Helpers ----------------------------------------------------------
    def _normalize_resource(
        self,
            title: Optional[str],
            provider: Optional[str],
            url: Optional[str],
            duration: Optional[str],
            difficulty: Optional[str],
            cost: Optional[str],
            tags: Optional[List[str]],
            skill: str,
            target_role: Optional[str],
            source: str,
            extra_score: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        if not title or not url:
            return None

        normalized = {
            "title": title.strip(),
            "provider": (provider or "Unknown").strip(),
            "url": url,
            "duration": duration or "self-paced",
            "difficulty": difficulty or "mixed",
            "cost": cost or "varies",
            "tags": tags or [],
            "skill_focus": skill,
            "target_role": target_role,
            "source": source,
        }

        normalized["score"] = self._score_resource(normalized, skill, target_role) + extra_score
        return normalized

    def _score_resource(
        self,
        resource: Dict[str, Any],
        skill: str,
        target_role: Optional[str],
    ) -> float:
        score = 0.0
        title = resource.get("title", "").lower()
        provider = resource.get("provider", "").lower()

        if skill.lower() in title:
            score += 1.5
        if target_role and target_role.lower() in title:
            score += 1.0
        if skill.lower() in " ".join(resource.get("tags", [])).lower():
            score += 0.5
        if provider in {"coursera", "udacity", "edx", "udemy"}:
            score += 0.3

        return score

    @staticmethod
    def _deduplicate_resources(resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for resource in resources:
            key = (
                resource.get("title", "").strip().lower(),
                resource.get("provider", "").strip().lower(),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            unique.append(resource)

        return unique

    @staticmethod
    def _format_duration(course: Dict[str, Any]) -> str:
        duration = course.get("expected_duration")
        unit = course.get("expected_duration_unit", "weeks")
        if not duration:
            return "self-paced"
        return f"{duration} {unit}"
