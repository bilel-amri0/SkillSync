"""Utility wrapper around the Gemini Generative Language API.

The implementation is intentionally lightweight. In production, you would
likely inject a long-lived HTTP client and handle retries, but this wrapper
keeps the SkillSync integration simple while still supporting real API calls
when a GEMINI_API_KEY is present.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Dict, List, Optional

import httpx

PROMPT_TEMPLATE = """You are an expert technical interviewer.
Generate {count} concise interview questions for the role "{job_title}".
Focus on the following skills: {skills}.
The candidate's CV summary is:
"""

ANALYSIS_PROMPT = """You are an AI assistant that scores interview transcripts.
Given the JSON transcript, return a JSON object with keys overall_score (0-100),
summary, strengths (list), weaknesses (list), and decision (Hire/Consider/No Hire).
Transcript JSON:
{transcript}
"""


class GeminiClient:
    """Minimal async client for Gemini with graceful fallback."""

    def __init__(self, api_key: Optional[str] = None, *, timeout: float = 25.0) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.timeout = timeout
        self.endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        )

    async def generate_questions(
        self,
        *,
        cv_text: str,
        job_description: str,
        job_title: str,
        difficulty: str,
        skills: List[str],
        count: int = 5,
    ) -> List[str]:
        payload_text = (
            PROMPT_TEMPLATE.format(count=count, job_title=job_title, skills=", ".join(skills or ["core engineering"]))
            + cv_text[:1500]
            + "\nJob description:\n"
            + (job_description or "Not provided")
            + f"\nDifficulty: {difficulty}."
        )

        if not self.api_key:
            return self._fallback_questions(job_title, skills, count)

        body = {"contents": [{"parts": [{"text": payload_text}]}]}
        params = {"key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, json=body, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return self._fallback_questions(job_title, skills, count)

        raw_text = self._extract_text(data)
        return self._split_questions(raw_text, fallback=self._fallback_questions(job_title, skills, count))

    async def analyze_transcript(self, *, job_title: str, transcript: List[Dict[str, str]]) -> Dict[str, object]:
        prompt = ANALYSIS_PROMPT.format(transcript=json.dumps(transcript, ensure_ascii=False, indent=2))

        if not self.api_key:
            return self._fallback_analysis(job_title, transcript)

        body = {"contents": [{"parts": [{"text": prompt}]}]}
        params = {"key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, json=body, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return self._fallback_analysis(job_title, transcript)

        raw_text = self._extract_text(data)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = self._fallback_analysis(job_title, transcript)

        parsed.setdefault("overall_score", random.randint(70, 90))
        parsed.setdefault("summary", f"Conversation assessed for {job_title}.")
        parsed.setdefault("strengths", ["Clear communication", "Relevant experience"])
        parsed.setdefault("weaknesses", ["Needs more detail on impact"])
        parsed.setdefault("decision", "Consider")
        return parsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_text(self, data: Dict[str, object]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        first = candidates[0]
        content = first.get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "")

    def _split_questions(self, blob: str, fallback: List[str]) -> List[str]:
        if not blob:
            return fallback
        lines = [line.strip("- ") for line in blob.splitlines() if line.strip()]
        questions = [line for line in lines if len(line.split()) > 3]
        return questions or fallback

    def _fallback_questions(self, job_title: str, skills: List[str], count: int) -> List[str]:
        base = skills or ["problem solving", "communication"]
        return [
            f"How have you applied {skill} in a recent {job_title} project?"
            for skill in (base * ((count // len(base)) + 1))[:count]
        ]

    def _fallback_analysis(self, job_title: str, transcript: List[Dict[str, str]]) -> Dict[str, object]:
        answered = sum(1 for item in transcript if item.get("answer"))
        return {
            "overall_score": 70 + answered * 3,
            "summary": f"Automated assessment for {job_title} interview.",
            "strengths": ["Thoughtful responses", "Baseline technical knowledge"],
            "weaknesses": ["Needs clearer impact metrics"],
            "decision": "Consider",
        }


# Convenience synchronous helpers for scripts/tests

def sync_generate_questions(**kwargs) -> List[str]:
    client = GeminiClient()
    return asyncio.get_event_loop().run_until_complete(client.generate_questions(**kwargs))


def sync_analyze_transcript(**kwargs) -> Dict[str, object]:
    client = GeminiClient()
    return asyncio.get_event_loop().run_until_complete(client.analyze_transcript(**kwargs))
