#!/bin/bash
# Test script for portfolio generation feature

BASE_URL="http://localhost:8000/api/v1/portfolio"

echo "=== Testing Portfolio Generation Feature ==="
echo ""

# Test 1: Get Templates
echo "1. Testing GET /templates"
curl -s "${BASE_URL}/templates" | python -m json.tool
echo ""

# Test 2: Generate Portfolio with Modern Template
echo "2. Testing POST /generate (Modern Template)"
RESPONSE=$(curl -s -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "cv_data": {
      "personal_info": {"name": "John Doe"},
      "job_titles": ["Senior Software Engineer"],
      "skills": ["Python", "FastAPI", "React", "TypeScript", "Docker", "AWS"],
      "experience_years": 5,
      "summary": "Experienced software engineer with expertise in full-stack development and cloud technologies."
    },
    "template": "modern"
  }')

echo "$RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(json.dumps({'portfolio_id': data['portfolio_id'], 'template': data['template'], 'status': data['status'], 'html_length': len(data['html_content'])}, indent=2))"
echo ""

# Test 3: Generate Portfolio with Tech Template
echo "3. Testing POST /generate (Tech Template)"
curl -s -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "cv_data": {
      "personal_info": {"name": "Jane Smith"},
      "job_titles": ["Data Scientist"],
      "skills": ["Python", "Machine Learning", "TensorFlow", "SQL", "Data Analysis"],
      "experience_years": 3,
      "summary": "Data scientist passionate about using ML to solve complex business problems."
    },
    "template": "tech"
  }' | python -c "import sys, json; data=json.load(sys.stdin); print(json.dumps({'portfolio_id': data['portfolio_id'], 'template': data['template'], 'status': data['status'], 'html_length': len(data['html_content'])}, indent=2))"
echo ""

# Test 4: Get Portfolio List
echo "4. Testing GET /list"
curl -s "${BASE_URL}/list" | python -m json.tool
echo ""

# Test 5: Export Portfolio
echo "5. Testing GET /export/{id}?format=html"
curl -s "${BASE_URL}/export/test-portfolio-id?format=html" | python -m json.tool
echo ""

echo "=== All Tests Complete ==="
