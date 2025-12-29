"""
Test script for improved CV parser
Validates all new features and improvements
"""

import sys
import json
from production_cv_parser_final import ProductionCVParser

# Test CV with all new features
test_cv = """
SARAH JOHNSON
Senior Full Stack Developer
sarah.johnson@email.com | +1-555-987-6543 | New York, NY

PROFESSIONAL SUMMARY
Senior software engineer with 8 years of experience in full-stack development.
Strong leadership, communication, and problem-solving skills.
Fluent in English, Spanish (Advanced - C1), and French (Intermediate - B2).

CERTIFICATIONS
- AWS Certified Solutions Architect (2022)
- Certified Scrum Master (CSM) - Scrum Alliance (2021)
- CompTIA Security+ (2020)
- Coursera Machine Learning Specialization (2023)

SKILLS
Technical: JavaScript, React, Node.js, Python, Django, PostgreSQL, MongoDB, 
Docker, Kubernetes, AWS, CI/CD, Git, TypeScript, HTML, CSS

Soft Skills: Leadership, Communication, Teamwork, Critical Thinking, 
Time Management, Adaptability, Problem Solving, Presentation

EXPERIENCE
2020 - Present: Senior Software Engineer at Tech Innovations Inc.
- Led team of 5 developers on microservices architecture migration
- Implemented CI/CD pipelines reducing deployment time by 60%
- Designed and built scalable React frontend serving 1M+ users
- Mentored junior developers and conducted code reviews
- Collaborated with product team to define technical roadmap

2016 - 2020: Software Developer at StartupXYZ
- Developed full-stack web applications using React and Django
- Built RESTful APIs and integrated third-party services
- Optimized database queries improving response time by 40%
- Participated in Agile sprint planning and retrospectives

EDUCATION
Master of Science in Computer Science
Stanford University, 2016
Graduated with Honors

Bachelor of Science in Software Engineering
MIT, 2014

LANGUAGES
- English (Native)
- Spanish (Fluent / C1)
- French (Intermediate / B2)
- German (Beginner / A2)
"""

def test_improvements():
    """Test all improvements"""
    print("=" * 70)
    print(" TESTING IMPROVED CV PARSER")
    print("=" * 70)
    
    # Initialize parser
    print("\n Loading models...")
    parser = ProductionCVParser()
    
    # Parse CV
    print("\n Parsing test CV...")
    result = parser.parse_cv(test_cv)
    
    # Convert to dict for display
    data = result.to_dict()
    
    # Test results
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    
    # Personal Info
    print(f"\n PERSONAL INFO:")
    print(f"   Name: {data['name']}")
    print(f"   Email: {data['email']}")
    print(f"   Phone: {data['phone']}")
    print(f"   Location: {data['location']}")
    
    # Professional
    print(f"\n PROFESSIONAL:")
    print(f"   Current Title: {data['current_title']}")
    print(f"   Seniority: {data['seniority_level']}")
    print(f"   Experience: {data['total_years_experience']} years")
    
    # Skills
    print(f"\n  TECHNICAL SKILLS ({len(data['skills'])}):")
    for cat, skills in data['skill_categories'].items():
        print(f"   {cat}: {', '.join(skills[:5])}")
    
    # NEW: Soft Skills
    print(f"\n SOFT SKILLS ({len(data['soft_skills'])}):")
    print(f"   {', '.join(data['soft_skills'])}")
    
    # NEW: Tech Stack Clusters
    print(f"\n TECH STACK CLUSTERS:")
    for cluster, skills in data['tech_stack_clusters'].items():
        print(f"   {cluster}: {', '.join(skills)}")
    
    # NEW: Certifications
    print(f"\n CERTIFICATIONS ({len(data['certifications'])}):")
    for cert in data['certifications']:
        print(f"    {cert['name']} ({cert['category']}) - {cert['issuer']}")
    
    # NEW: Languages
    print(f"\n LANGUAGES ({len(data['languages'])}):")
    for lang in data['languages']:
        print(f"    {lang['language']}: {lang['proficiency']}")
    
    # Education
    print(f"\n EDUCATION:")
    print(f"   Degree Level: {data['degree_level']}")
    print(f"   Graduation Year: {data['graduation_year']}")
    print(f"   Degrees: {len(data['degrees'])}")
    for deg in data['degrees']:
        print(f"    {deg}")
    
    # NEW: Responsibilities
    print(f"\n KEY RESPONSIBILITIES ({len(data['responsibilities'])}):")
    for resp in data['responsibilities'][:5]:
        print(f"    {resp}")
    
    # Metadata
    print(f"\n METADATA:")
    print(f"   Confidence: {data['confidence_score']:.2%}")
    print(f"   Processing Time: {data['processing_time_ms']}ms")
    
    # Validation
    print("\n" + "=" * 70)
    print(" VALIDATION")
    print("=" * 70)
    
    tests = {
        "Name extracted": data['name'] is not None,
        "Email extracted": data['email'] is not None,
        "Skills > 10": len(data['skills']) > 10,
        "Soft skills extracted": len(data['soft_skills']) > 0,
        "Certifications found": len(data['certifications']) >= 3,
        "Languages detected": len(data['languages']) >= 3,
        "Tech clusters created": len(data['tech_stack_clusters']) > 0,
        "Responsibilities extracted": len(data['responsibilities']) > 0,
        "Degree level detected": data['degree_level'] is not None,
        "Graduation year found": data['graduation_year'] is not None,
        "Experience years calculated": data['total_years_experience'] > 0,
        "Processing time < 300ms": data['processing_time_ms'] < 300
    }
    
    passed = sum(tests.values())
    total = len(tests)
    
    for test_name, result in tests.items():
        status = "" if result else ""
        print(f"{status} {test_name}")
    
    print(f"\n Score: {passed}/{total} tests passed ({passed/total:.0%})")
    
    # Save full result
    print("\n Saving full result to test_result.json...")
    with open('test_result.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(" TESTING COMPLETE")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    try:
        success = test_improvements()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
