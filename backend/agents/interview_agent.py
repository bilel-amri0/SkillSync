"""
AI-powered interview agent for generating questions and analyzing responses
"""
import uuid
from typing import List, Dict, Any
from datetime import datetime
import random


class InterviewAgent:
    """
    Agent for managing AI-powered interview sessions
    
    This is a mock implementation for initial integration.
    In production, this would be connected to a generative AI service
    like Google AI, OpenAI, or similar.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def generate_questions(self, cv_text: str, job_description: str, num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate tailored interview questions based on CV and job description
        
        Args:
            cv_text: The candidate's CV content
            job_description: The target job description
            num_questions: Number of questions to generate
        
        Returns:
            List of question dictionaries
        """
        # Mock question generation
        # In production, this would use AI to analyze CV and job description
        # to generate highly relevant, personalized questions
        
        question_templates = [
            {
                "category": "technical",
                "templates": [
                    "Can you explain your experience with the technologies mentioned in your CV?",
                    "How would you approach solving a complex technical problem in this role?",
                    "Describe a challenging technical project you've worked on.",
                    "What technical skills from your background are most relevant to this position?",
                ]
            },
            {
                "category": "behavioral",
                "templates": [
                    "Tell me about a time when you had to work under pressure.",
                    "Describe a situation where you had to collaborate with a difficult team member.",
                    "How do you handle constructive criticism?",
                    "Give an example of when you showed leadership.",
                ]
            },
            {
                "category": "situational",
                "templates": [
                    "How would you prioritize multiple urgent tasks?",
                    "What would you do if you disagreed with your manager's approach?",
                    "How would you handle a project that's behind schedule?",
                    "Describe how you would approach learning a new technology for this role.",
                ]
            }
        ]
        
        questions = []
        categories = ["technical", "behavioral", "situational"]
        
        for i in range(num_questions):
            category = categories[i % len(categories)]
            category_data = next(q for q in question_templates if q["category"] == category)
            question_text = random.choice(category_data["templates"])
            
            questions.append({
                "question_id": i + 1,
                "question_text": question_text,
                "category": category
            })
        
        return questions
    
    def start_interview(self, cv_text: str, job_description: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Start a new interview session
        
        Args:
            cv_text: The candidate's CV content
            job_description: The target job description
            num_questions: Number of questions to generate
        
        Returns:
            Dictionary with interview_id and questions
        """
        interview_id = str(uuid.uuid4())
        questions = self.generate_questions(cv_text, job_description, num_questions)
        
        # Store session data
        self.sessions[interview_id] = {
            "cv_text": cv_text,
            "job_description": job_description,
            "questions": questions,
            "answers": {},
            "current_question_index": 0,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return {
            "interview_id": interview_id,
            "questions": questions
        }
    
    def submit_answer(self, interview_id: str, question_id: int, answer_text: str) -> Dict[str, Any]:
        """
        Submit an answer to a question
        
        Args:
            interview_id: The interview session ID
            question_id: The question ID
            answer_text: The candidate's answer
        
        Returns:
            Dictionary with submission status and next question if available
        """
        if interview_id not in self.sessions:
            raise ValueError(f"Interview session {interview_id} not found")
        
        session = self.sessions[interview_id]
        
        # Store the answer
        session["answers"][question_id] = answer_text
        session["current_question_index"] += 1
        
        # Check if there are more questions
        is_complete = session["current_question_index"] >= len(session["questions"])
        next_question = None
        
        if not is_complete:
            next_question = session["questions"][session["current_question_index"]]
        
        return {
            "is_complete": is_complete,
            "next_question": next_question
        }
    
    def analyze_interview(self, interview_id: str) -> Dict[str, Any]:
        """
        Analyze the complete interview and generate a report
        
        Args:
            interview_id: The interview session ID
        
        Returns:
            Dictionary with analysis results
        """
        if interview_id not in self.sessions:
            raise ValueError(f"Interview session {interview_id} not found")
        
        session = self.sessions[interview_id]
        
        # Mock analysis
        # In production, this would use AI to analyze answers and provide
        # detailed feedback on strengths, weaknesses, and overall performance
        
        num_answered = len(session["answers"])
        num_questions = len(session["questions"])
        
        # Calculate a mock score based on answer completeness
        score = (num_answered / num_questions) * 100 if num_questions > 0 else 0
        
        # Generate mock strengths and weaknesses
        strengths = [
            "Clear and structured responses",
            "Good understanding of technical concepts",
            "Demonstrates relevant experience",
        ]
        
        weaknesses = [
            "Could provide more specific examples",
            "Responses could be more concise",
            "Consider adding metrics to demonstrate impact",
        ]
        
        recommendations = [
            "Practice the STAR method (Situation, Task, Action, Result) for behavioral questions",
            "Prepare specific examples that demonstrate your key skills",
            "Research the company and role to tailor your responses",
        ]
        
        summary = (
            f"You completed {num_answered} out of {num_questions} questions. "
            f"Your responses show good potential, with clear communication and relevant experience. "
            f"Focus on providing more specific examples and quantifiable achievements to strengthen your answers."
        )
        
        return {
            "overall_score": score,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def get_report(self, interview_id: str) -> Dict[str, Any]:
        """
        Get the complete interview report
        
        Args:
            interview_id: The interview session ID
        
        Returns:
            Complete interview report with transcript and analysis
        """
        if interview_id not in self.sessions:
            raise ValueError(f"Interview session {interview_id} not found")
        
        session = self.sessions[interview_id]
        
        # Build transcript
        transcript = []
        for question in session["questions"]:
            question_id = question["question_id"]
            if question_id in session["answers"]:
                transcript.append({
                    "question_id": question_id,
                    "question_text": question["question_text"],
                    "answer_text": session["answers"][question_id],
                    "category": question["category"]
                })
        
        # Get analysis
        analysis = self.analyze_interview(interview_id)
        
        return {
            "interview_id": interview_id,
            "cv_text": session["cv_text"],
            "job_description": session["job_description"],
            "transcript": transcript,
            "analysis": analysis,
            "created_at": session["created_at"]
        }


# Singleton instance
_interview_agent = None


def get_interview_agent() -> InterviewAgent:
    """Get the singleton interview agent instance"""
    global _interview_agent
    if _interview_agent is None:
        _interview_agent = InterviewAgent()
    return _interview_agent
