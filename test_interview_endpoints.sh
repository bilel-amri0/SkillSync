#!/bin/bash
# Test script for interview feature endpoints

BASE_URL="http://localhost:8000/api/v1/interviews"

echo "=== Testing Interview Feature Endpoints ==="
echo ""

# Test 1: Start Interview
echo "1. Testing POST /start"
RESPONSE=$(curl -s -X POST "${BASE_URL}/start" \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "Senior Software Engineer with 5 years experience in Python and FastAPI",
    "job_description": "Looking for a Backend Developer with Python expertise",
    "num_questions": 3
  }')

echo "$RESPONSE" | python -m json.tool
INTERVIEW_ID=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['interview_id'])")
echo ""
echo "Interview ID: $INTERVIEW_ID"
echo ""

# Test 2: Submit Answers
echo "2. Testing POST /interviews/{id}/submit_answer (Question 1)"
curl -s -X POST "${BASE_URL}/interviews/${INTERVIEW_ID}/submit_answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 1,
    "answer_text": "I have extensive experience with Python, FastAPI, and building scalable backend systems."
  }' | python -m json.tool
echo ""

echo "3. Testing POST /interviews/{id}/submit_answer (Question 2)"
curl -s -X POST "${BASE_URL}/interviews/${INTERVIEW_ID}/submit_answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 2,
    "answer_text": "During a critical deployment, our team worked overnight to fix production issues."
  }' | python -m json.tool
echo ""

echo "4. Testing POST /interviews/{id}/submit_answer (Question 3)"
curl -s -X POST "${BASE_URL}/interviews/${INTERVIEW_ID}/submit_answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 3,
    "answer_text": "I would start by reading documentation, building small projects, and consulting with experts."
  }' | python -m json.tool
echo ""

# Test 3: Get Report
echo "5. Testing GET /interviews/{id}/report"
curl -s "${BASE_URL}/interviews/${INTERVIEW_ID}/report" | python -m json.tool
echo ""

echo "=== All Tests Complete ==="
