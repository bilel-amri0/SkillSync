#!/usr/bin/env python
"""Quick test for portfolio generation with Bilel's CV"""
import requests
import re
import sys

# Bilel's CV
cv_text = """Bilel Amri
+21654505864 bilela329@gmail.com
ariana
github.com/Bilel-Amri
Profile
linkedin.com/in/amri-bilel-53092b283
Student engineer at Tek-Up, fueled by a passion for artificial intelligence.
Education
2022 – to Present ariana, Tunisia
Tek-up
Technical Skills
Programming Languages: Python, Sql, C/C++, Java, JavaScript
Scientific Computing: MATLAB
Tools & Platforms: Git, GitHub, Docker, Linux/Unix
Web Development: HTML, CSS, Bootstrap, PHP
Projects
Tunisian Internship Chatbot with Ollama Integration
Designed and implemented a chatbot application using FastAPI and React
Certification
PCAP"""

def main():
    print("=" * 60)
    print("TESTING PORTFOLIO GENERATION FOR BILEL AMRI")
    print("=" * 60)
    
    # Step 1: Analyze CV
    print("\n[Step 1] Analyzing CV...")
    try:
        analysis_response = requests.post(
            "http://localhost:8000/api/v1/analyze-cv",
            json={"cv_content": cv_text},
            timeout=120
        )
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Make sure it's running on port 8000")
        sys.exit(1)
    
    if analysis_response.status_code != 200:
        print(f"ERROR: Analysis failed with status {analysis_response.status_code}")
        print(analysis_response.text[:500])
        sys.exit(1)
    
    data = analysis_response.json()
    analysis_id = data.get('analysis_id')
    name = data.get('name')
    skills = data.get('skills', [])
    
    print(f"  Name extracted: '{name}'")
    print(f"  Skills ({len(skills)}): {', '.join(skills[:10])}")
    print(f"  Analysis ID: {analysis_id}")
    
    # Step 2: Generate Portfolio
    print("\n[Step 2] Generating Portfolio...")
    portfolio_response = requests.post(
        "http://localhost:8000/api/v1/portfolio/generate",
        json={
            "cv_id": analysis_id,
            "template_id": "modern-professional"
        },
        timeout=30
    )
    
    if portfolio_response.status_code != 200:
        print(f"ERROR: Portfolio generation failed with status {portfolio_response.status_code}")
        print(portfolio_response.text[:500])
        sys.exit(1)
    
    portfolio_data = portfolio_response.json()
    html = portfolio_data.get('html_content', '')
    
    # Extract key elements
    title_match = re.search(r'<title>([^<]+)</title>', html)
    h1_match = re.search(r'<h1>([^<]+)</h1>', html)
    skill_tags = re.findall(r'<span class="skill-tag">([^<]+)</span>', html)
    
    print(f"  Portfolio Status: OK")
    print(f"  HTML Title: {title_match.group(1) if title_match else 'NOT FOUND'}")
    print(f"  H1 Name: {h1_match.group(1) if h1_match else 'NOT FOUND'}")
    print(f"  Skills in HTML ({len(skill_tags)}): {', '.join(skill_tags[:10])}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    errors = []
    
    # Check name
    if h1_match:
        h1_name = h1_match.group(1)
        if h1_name == "Bilel Amri":
            print("[PASS] Name correctly extracted: Bilel Amri")
        elif "bilel" in h1_name.lower():
            print(f"[WARN] Name partially correct: {h1_name}")
        else:
            print(f"[FAIL] Name incorrect: {h1_name}")
            errors.append("Name extraction failed")
    else:
        print("[FAIL] No H1 name found")
        errors.append("H1 missing")
    
    # Check skills - ensure no garbage
    garbage_skills = ['21654505864', 'chat', 'delivers', 'demonstrates', 'message', 'supports']
    found_garbage = [s for s in skill_tags if s.lower() in garbage_skills or s.isdigit()]
    
    if not found_garbage:
        print("[PASS] No garbage skills found")
    else:
        print(f"[FAIL] Garbage skills found: {found_garbage}")
        errors.append("Garbage skills present")
    
    # Check valid skills are present
    expected_skills = ['python', 'java', 'docker', 'git', 'html', 'css']
    found_expected = [s for s in skill_tags if s.lower() in expected_skills]
    
    if found_expected:
        print(f"[PASS] Valid skills found: {found_expected}")
    else:
        print("[WARN] Expected skills not found in portfolio")
    
    # Save portfolio
    output_file = 'test_bilel_final_portfolio.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n[INFO] Portfolio saved to: {output_file}")
    
    if errors:
        print(f"\n[RESULT] FAILED with {len(errors)} error(s)")
        sys.exit(1)
    else:
        print("\n[RESULT] ALL TESTS PASSED!")
        sys.exit(0)

if __name__ == "__main__":
    main()
