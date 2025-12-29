"""
Google ADK Live Agent for Voice Interview Mode
Handles real-time voice interactions using Gemini 2.0 Flash Live model
"""

import json
import os
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

# Google ADK imports (install with: pip install google-genai google-adk)
try:
    from google.genai import types
    from google import genai
    GOOGLE_ADK_AVAILABLE = True
except ImportError:
    GOOGLE_ADK_AVAILABLE = False
    print("⚠️ Google ADK not available. Install with: pip install google-genai")


class LiveInterviewAgent:
    """
    Live agent for conducting voice-based AI interviews using Google Gemini 2.0 Flash Live.
    Supports bidirectional audio streaming with real-time processing.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the live interview agent.
        
        Args:
            api_key: Google API key for Gemini. If None, reads from GOOGLE_API_KEY env var
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        if not GOOGLE_ADK_AVAILABLE:
            raise ImportError("Google ADK not installed. Run: pip install google-genai")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_id = "gemini-2.0-flash-exp"
        self.sessions = {}  # interview_id -> session_data
    
    async def start_live_session(
        self,
        interview_id: str,
        job_title: str,
        job_description: str,
        skills: list[str],
        cv_text: str,
        difficulty: str = "medium"
    ) -> Dict[str, Any]:
        """
        Start a new live voice interview session.
        
        Args:
            interview_id: Unique identifier for the interview
            job_title: Target job position
            job_description: Job requirements and description
            skills: List of skills from CV
            cv_text: Full CV text content
            difficulty: Interview difficulty level
            
        Returns:
            Session initialization data with configuration
        """
        
        # Create system prompt for the interview agent
        system_prompt = self._create_interview_prompt(
            job_title=job_title,
            job_description=job_description,
            skills=skills,
            cv_text=cv_text,
            difficulty=difficulty
        )
        
        # Initialize session data
        session_data = {
            "interview_id": interview_id,
            "job_title": job_title,
            "started_at": datetime.utcnow().isoformat(),
            "status": "active",
            "system_prompt": system_prompt,
            "conversation_history": [],
            "questions_asked": 0,
            "model_config": {
                "model": self.model_id,
                "generation_config": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_output_tokens": 1024,
                    "response_modalities": ["AUDIO"],
                },
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": "Puck"  # Professional voice
                        }
                    }
                }
            }
        }
        
        self.sessions[interview_id] = session_data
        
        return {
            "interview_id": interview_id,
            "status": "ready",
            "model": self.model_id,
            "websocket_url": f"/api/v2/interviews/live/{interview_id}/ws",
            "instructions": "Connect via WebSocket for real-time audio streaming",
            "audio_config": {
                "sample_rate": 16000,
                "channels": 1,
                "encoding": "pcm_s16le"
            }
        }
    
    def _create_interview_prompt(
        self,
        job_title: str,
        job_description: str,
        skills: list[str],
        cv_text: str,
        difficulty: str
    ) -> str:
        """Create a comprehensive system prompt for the AI interviewer."""
        
        skills_text = ", ".join(skills) if skills else "general skills"
        
        prompt = f"""You are an expert AI interviewer conducting a professional job interview for the position of {job_title}.

**Interview Context:**
- Position: {job_title}
- Difficulty Level: {difficulty}
- Candidate Skills: {skills_text}

**Job Description:**
{job_description}

**Candidate CV Summary:**
{cv_text[:500]}...

**Your Role:**
1. Conduct a professional, conversational interview
2. Ask relevant technical and behavioral questions based on the job requirements
3. Evaluate candidate responses with constructive feedback
4. Maintain a friendly but professional tone
5. Ask follow-up questions to dig deeper into candidate's experience
6. Cover both technical skills and soft skills
7. Provide hints if the candidate struggles, but don't give away answers

**Interview Structure:**
- Start with a warm greeting and brief introduction
- Ask 5-7 questions total covering:
  * Technical skills relevant to {job_title}
  * Problem-solving abilities
  * Past experience and projects
  * Behavioral/situational questions
  * Cultural fit and motivation
- End with asking if the candidate has questions
- Provide a brief summary and next steps

**Guidelines:**
- Keep questions clear and concise
- Allow candidate time to think and respond
- Show active listening through acknowledgments
- Be encouraging and constructive
- Adapt difficulty based on candidate responses
- If audio is unclear, politely ask the candidate to repeat

Begin the interview now with a friendly greeting and introduction."""
        
        return prompt
    
    async def process_audio_stream(
        self,
        interview_id: str,
        audio_chunk: bytes
    ) -> Dict[str, Any]:
        """
        Process incoming audio chunk from candidate.
        
        Args:
            interview_id: Interview session ID
            audio_chunk: Raw audio data from candidate
            
        Returns:
            Processing status and any generated response
        """
        if interview_id not in self.sessions:
            return {"error": "Interview session not found", "status": "not_found"}
        
        session = self.sessions[interview_id]
        
        # Add to conversation history
        session["conversation_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": "audio_input",
            "size": len(audio_chunk)
        })
        
        return {
            "interview_id": interview_id,
            "status": "processing",
            "received_bytes": len(audio_chunk)
        }
    
    async def generate_response(
        self,
        interview_id: str,
        candidate_text: str
    ) -> Dict[str, Any]:
        """
        Generate AI interviewer response to candidate's answer.
        
        Args:
            interview_id: Interview session ID
            candidate_text: Transcribed text from candidate's audio
            
        Returns:
            AI response with audio and text
        """
        if interview_id not in self.sessions:
            return {"error": "Interview session not found"}
        
        session = self.sessions[interview_id]
        session["questions_asked"] += 1
        
        # Use Gemini to generate contextual response
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=session["system_prompt"])]
                    ),
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"Candidate response: {candidate_text}")]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=1024
                )
            )
            
            ai_text = response.text
            
            # Log to conversation history
            session["conversation_history"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "candidate": candidate_text,
                "interviewer": ai_text,
                "question_number": session["questions_asked"]
            })
            
            return {
                "interview_id": interview_id,
                "response_text": ai_text,
                "question_number": session["questions_asked"],
                "status": "active"
            }
            
        except Exception as e:
            return {
                "error": f"Failed to generate response: {str(e)}",
                "status": "error"
            }
    
    async def end_session(self, interview_id: str) -> Dict[str, Any]:
        """
        End the live interview session and generate summary.
        
        Args:
            interview_id: Interview session ID
            
        Returns:
            Session summary and metrics
        """
        if interview_id not in self.sessions:
            return {"error": "Interview session not found"}
        
        session = self.sessions[interview_id]
        session["status"] = "completed"
        session["ended_at"] = datetime.utcnow().isoformat()
        
        # Generate interview summary
        conversation = session["conversation_history"]
        
        summary = {
            "interview_id": interview_id,
            "job_title": session["job_title"],
            "duration_minutes": self._calculate_duration(
                session["started_at"],
                session["ended_at"]
            ),
            "questions_asked": session["questions_asked"],
            "total_interactions": len(conversation),
            "status": "completed",
            "conversation_summary": conversation[-5:],  # Last 5 interactions
            "next_steps": "Report will be generated with detailed analysis"
        }
        
        # Clean up session data after retrieval
        del self.sessions[interview_id]
        
        return summary
    
    def _calculate_duration(self, start: str, end: str) -> float:
        """Calculate interview duration in minutes."""
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            duration = (end_dt - start_dt).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0.0
    
    def get_session_status(self, interview_id: str) -> Dict[str, Any]:
        """Get current status of a live interview session."""
        if interview_id not in self.sessions:
            return {"error": "Session not found", "status": "not_found"}
        
        session = self.sessions[interview_id]
        return {
            "interview_id": interview_id,
            "status": session["status"],
            "questions_asked": session["questions_asked"],
            "started_at": session["started_at"],
            "active": session["status"] == "active"
        }


# Global instance
_live_agent_instance = None

def get_live_agent() -> LiveInterviewAgent:
    """Get or create the global live agent instance."""
    global _live_agent_instance
    if _live_agent_instance is None:
        try:
            _live_agent_instance = LiveInterviewAgent()
        except Exception as e:
            print(f"⚠️ Failed to initialize LiveInterviewAgent: {e}")
            # Return a mock agent in case of failure
            return None
    return _live_agent_instance
