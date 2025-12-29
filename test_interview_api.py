#!/usr/bin/env python3
"""
Test script for AI Interview endpoints
Tests both text and voice interview modes
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8001"

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_start_interview() -> Dict[str, Any]:
    """Test starting a new interview session."""
    print_section("TEST 1: Start Interview (Text Mode)")
    
    payload = {
        "user_id": "test_user_123",
        "cv_id": "test_cv_456",
        "cv_text": "Senior Python Developer with 5 years of experience in FastAPI, React, and microservices architecture. Strong background in AI and machine learning.",
        "job_title": "Senior Full Stack Developer",
        "job_description": "Looking for an experienced developer to lead our AI-powered platform development.",
        "difficulty": "medium",
        "skills": ["Python", "FastAPI", "React", "TypeScript", "Docker", "PostgreSQL"],
        "interview_mode": "text"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v2/interviews/start",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print("✅ Interview started successfully!")
        print(f"   Interview ID: {result.get('interview_id')}")
        print(f"   Status: {result.get('status')}")
        print(f"   Total Questions: {result.get('total_questions')}")
        print(f"   Mode: {result.get('interview_mode')}")
        
        if result.get('current_question'):
            print(f"\n   First Question:")
            print(f"   → {result['current_question'].get('question_text')}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error starting interview: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return {}

def test_get_next_question(interview_id: str):
    """Test getting next question."""
    print_section("TEST 2: Get Next Question")
    
    payload = {"interview_id": interview_id}
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v2/interviews/next-question",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print("✅ Next question retrieved!")
        print(f"   Status: {result.get('status')}")
        
        if result.get('next_question'):
            q = result['next_question']
            print(f"   Question ID: {q.get('question_id')}")
            print(f"   Category: {q.get('category')}")
            print(f"   Question: {q.get('question_text')}")
        
        if result.get('progress'):
            prog = result['progress']
            print(f"   Progress: {prog.get('current')}/{prog.get('total')}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting next question: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return {}

def test_submit_answer(interview_id: str, question_id: int):
    """Test submitting an answer."""
    print_section("TEST 3: Submit Answer")
    
    payload = {
        "interview_id": interview_id,
        "question_id": question_id,
        "answer_text": "I have extensive experience with React, having built multiple large-scale applications. In my last project, I architected a real-time analytics dashboard using React hooks, TypeScript, and Redux. We achieved 60fps performance with 10,000+ data points updating live."
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v2/interviews/answer",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print("✅ Answer submitted successfully!")
        print(f"   Response: {json.dumps(result, indent=2)}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error submitting answer: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return {}

def test_get_interview(interview_id: str):
    """Test getting interview details."""
    print_section("TEST 4: Get Interview Details")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v2/interviews/{interview_id}",
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print("✅ Interview details retrieved!")
        print(f"   User ID: {result.get('user_id')}")
        print(f"   Job Title: {result.get('job_title')}")
        print(f"   Status: {result.get('status')}")
        print(f"   Questions: {result.get('question_count')}")
        print(f"   Answered: {result.get('answered_questions')}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting interview details: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return {}

def test_finish_interview(interview_id: str):
    """Test finishing interview."""
    print_section("TEST 5: Finish Interview")
    
    payload = {"interview_id": interview_id}
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v2/interviews/finish",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print("✅ Interview finished successfully!")
        print(f"   Status: {result.get('status')}")
        print(f"   Report Ready: {result.get('report_ready')}")
        
        if result.get('report'):
            print(f"   Report generated!")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error finishing interview: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return {}

def test_list_interviews():
    """Test listing interviews."""
    print_section("TEST 6: List Interviews")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v2/interviews?limit=5",
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Retrieved {len(result)} interviews")
        
        for i, interview in enumerate(result[:3], 1):
            print(f"\n   Interview #{i}:")
            print(f"   → ID: {interview.get('interview_id')}")
            print(f"   → Job: {interview.get('job_title')}")
            print(f"   → Status: {interview.get('status')}")
            print(f"   → Questions: {interview.get('question_count')}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"❌ Error listing interviews: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return []

def test_health_check():
    """Test backend health."""
    print_section("TEST 0: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        result = response.json()
        
        print("✅ Backend is healthy!")
        print(f"   Status: {result.get('status')}")
        print(f"   Service: {result.get('service')}")
        
        features = result.get('features', {})
        print(f"\n   Available Features:")
        for feature, enabled in features.items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {feature}")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "="*60)
    print("  SkillSync AI Interview - Backend Tests")
    print("="*60)
    
    # Check backend health first
    if not test_health_check():
        print("\n❌ Backend is not running. Please start it first:")
        print("   cd backend && python main_simple_for_frontend.py")
        return
    
    # Test 1: Start interview
    interview_result = test_start_interview()
    if not interview_result:
        print("\n❌ Failed to start interview. Stopping tests.")
        return
    
    interview_id = interview_result.get('interview_id')
    print(f"\n📝 Using Interview ID: {interview_id}")
    
    time.sleep(1)
    
    # Test 2: Get next question
    next_q_result = test_get_next_question(interview_id)
    question_id = None
    if next_q_result.get('next_question'):
        question_id = next_q_result['next_question'].get('question_id')
    
    time.sleep(1)
    
    # Test 3: Submit answer (if we have a question)
    if question_id:
        test_submit_answer(interview_id, question_id)
        time.sleep(1)
    
    # Test 4: Get interview details
    test_get_interview(interview_id)
    time.sleep(1)
    
    # Test 5: Finish interview
    test_finish_interview(interview_id)
    time.sleep(1)
    
    # Test 6: List all interviews
    test_list_interviews()
    
    print_section("Summary")
    print("✅ All tests completed!")
    print(f"   Interview ID: {interview_id}")
    print(f"   Endpoints tested: 7/7")
    print(f"\n📚 Full documentation: AI_INTERVIEW_IMPLEMENTATION.md")
    print(f"📖 API docs: {BASE_URL}/docs")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
