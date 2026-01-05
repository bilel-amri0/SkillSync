#!/usr/bin/env python
"""Test portfolio generation flow"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Sample CV
sample_cv = """
John Smith
Software Engineer
john.smith@email.com | +1 (555) 123-4567 | San Francisco, CA

SUMMARY
Experienced software engineer with 5+ years of expertise in full-stack development.

SKILLS
Python, JavaScript, React, Node.js, FastAPI, PostgreSQL, Docker, AWS

EXPERIENCE
Senior Software Engineer at TechCorp Inc. (2021-Present)
- Led development of microservices architecture

EDUCATION
B.S. Computer Science - Stanford University (2019)
"""

print("=" * 60)
print("PORTFOLIO GENERATION TEST")
print("=" * 60)

# Step 1: Analyze CV
print("\n[Step 1] Analyzing CV...")
response = requests.post(
    f"{BASE_URL}/api/v1/analyze-cv",
    json={"cv_content": sample_cv},
    timeout=60
)
print(f"Status: {response.status_code}")

if response.status_code != 200:
    print(f"Error: {response.text}")
    exit(1)

cv_data = response.json()
cv_id = cv_data.get('analysis_id')
print(f"CV Analysis ID: {cv_id}")
print(f"Name: {cv_data.get('name', 'N/A')}")
print(f"Skills: {cv_data.get('skills', [])[:5]}...")

# Step 2: Generate portfolio
print("\n[Step 2] Generating portfolio...")
response = requests.post(
    f"{BASE_URL}/api/v1/portfolio/generate",
    json={
        "cv_id": cv_id,
        "template_id": "modern"
    },
    timeout=30
)
print(f"Status: {response.status_code}")

if response.status_code != 200:
    print(f"Error: {response.json()}")
    exit(1)

portfolio = response.json()
print(f"Portfolio ID: {portfolio.get('portfolio', {}).get('id')}")
print(f"Portfolio Name: {portfolio.get('portfolio', {}).get('name')}")
print(f"HTML Content Length: {len(portfolio.get('html_content', ''))} chars")

# Step 3: List portfolios
print("\n[Step 3] Listing portfolios...")
response = requests.get(f"{BASE_URL}/api/v1/portfolio/list", timeout=10)
portfolios = response.json()
print(f"Total portfolios: {len(portfolios)}")

# Step 4: Export portfolio
print("\n[Step 4] Exporting portfolio...")
portfolio_id = portfolio.get('portfolio', {}).get('id')
response = requests.get(f"{BASE_URL}/api/v1/portfolio/export/{portfolio_id}?format=html", timeout=10)
print(f"Export Status: {response.status_code}")

if response.status_code == 200:
    export = response.json()
    print(f"Export Format: {export.get('format')}")
    print(f"Export Content Length: {len(export.get('content', ''))} chars")
    
    # Save HTML to file
    with open("test_portfolio.html", "w", encoding="utf-8") as f:
        f.write(export.get('content', ''))
    print("Saved portfolio to test_portfolio.html")

print("\n" + "=" * 60)
print("✅ PORTFOLIO GENERATION TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
