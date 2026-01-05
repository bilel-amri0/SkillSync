#!/usr/bin/env python3
"""Test name extraction in portfolio generation"""
import requests
import re

# Sample CV with clear name
cv = """Ahmed Ben Ali
ML Engineer
ahmed.benali@email.com

SKILLS
Python, TensorFlow, PyTorch, NLP, Computer Vision, MLOps

I explore cutting-edge fields like machine learning, deep learning, natural language processing (NLP), computer vision, MLOps, and multi-agent systems
"""

print("=" * 60)
print("TESTING NAME EXTRACTION IN PORTFOLIO")
print("=" * 60)

# Step 1: Analyze CV
print("\n[Step 1] Analyzing CV...")
r = requests.post('http://localhost:8000/api/v1/analyze-cv', json={'cv_content': cv}, timeout=120)

if r.status_code != 200:
    print(f"Error: {r.text}")
    exit(1)

data = r.json()
cv_id = data['analysis_id']
print(f"CV ID: {cv_id}")
print(f"Name from root: {data.get('name')}")
print(f"Name from personal_info: {data.get('personal_info', {}).get('name')}")

# Step 2: Generate portfolio
print("\n[Step 2] Generating portfolio...")
r2 = requests.post('http://localhost:8000/api/v1/portfolio/generate', json={'cv_id': cv_id, 'template_id': 'modern'}, timeout=30)

if r2.status_code != 200:
    print(f"Portfolio error: {r2.text}")
    exit(1)

portfolio = r2.json()
html = portfolio.get('html_content', '')

# Extract the h1 tag content
h1_match = re.search(r'<h1>(.*?)</h1>', html)
title_match = re.search(r'<title>(.*?)</title>', html)

print(f"\nResults:")
print(f"  Name in <h1>: {h1_match.group(1) if h1_match else 'NOT FOUND'}")
print(f"  <title>: {title_match.group(1) if title_match else 'NOT FOUND'}")

# Save HTML
with open('test_name_portfolio.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nSaved portfolio to test_name_portfolio.html")

print("\n" + "=" * 60)
print("✅ TEST COMPLETED")
print("=" * 60)
