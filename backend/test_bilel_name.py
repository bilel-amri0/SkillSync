#!/usr/bin/env python3
"""Test name extraction with Bilel's CV format"""
import requests
import re

# CV similar to the user's (Bilel Amri)
cv = """Bilel Amri bilela329@gmail.com ariana github.com/Bilel-Amri+21654505864 linkedin.com/in/amri-bilel-53092b283 Profile Student engineer at Tek-Up, fueled by a passion for artificial intelligence and its power to transform our world. I explore cutting-edge fields like machine learning, deep learning, natural language processing (NLP), computer vision, MLOps, and multi-agent systems. Fascinated by their ability to solve complex challenges, I'm committed to mastering these technologies through my studies and hands-on projects. I'm thrilled about AI's future Education 2022 – to Present ariana, TunisiaTek-up Technical Skills Programming Languages: Python, Sql, C/C++, Java, JavaScript Scientific Computing: MATLABTools & Platforms: Git, GitHub , Docker , Linux/Unix Web Development: HTML, CSS, Bootstrap , PHP Projects Tunisian Internship Chatbot with Ollama Integration Designed and implemented a chatbot application using FastAPI and React, integrated with Ollama to run local language models (e.g., Llama2:7B), enabling users to search for internships in Tunisia by processing PDF documents and providing structured responses about Ollama's functionality, such as model compatibility and usage instructions ChatMessagerie "Developed a real-time chat application utilizing JavaFX for an intuitive client interface, Spring Boot with WebSocket for seamless communication, and MongoDB for efficient message storage. The application supports general and private messaging, dynamic user list updates, and delivers reliable performance. This project demonstrates proficiency in modern technologies, effective problem-solving, and the ability to build scalable, user-friendly solutions for real-time communication." Clubs ESSAI ML ,IEEE Tekup, Securinets Tekup Languages Arabic English French Interests Sports Swimming Chess Camping Certification PCAP"""

print("=" * 60)
print("TESTING NAME EXTRACTION (Bilel Amri CV)")
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

# Extract the h1 tag content (the name)
h1_match = re.search(r'<h1>(.*?)</h1>', html)
title_match = re.search(r'<title>(.*?)</title>', html)

print(f"\n📋 RESULTS:")
print(f"  Name in <h1>: {h1_match.group(1) if h1_match else 'NOT FOUND'}")
print(f"  <title>: {title_match.group(1) if title_match else 'NOT FOUND'}")

# Check if name is correct
expected_name = "Bilel Amri"
actual_name = h1_match.group(1) if h1_match else ''

if expected_name.lower() in actual_name.lower():
    print(f"\n✅ SUCCESS! Name correctly extracted: {actual_name}")
else:
    print(f"\n❌ ISSUE: Expected '{expected_name}' but got '{actual_name}'")

# Save HTML
with open('test_bilel_portfolio.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\nSaved portfolio to test_bilel_portfolio.html")

print("\n" + "=" * 60)
