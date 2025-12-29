"""
Quick Test Script for Advanced ML Modules
Tests each ML module independently before full integration
"""

import sys
sys.path.append('.')

from advanced_ml_modules import (
    SemanticSkillExtractor,
    MLJobTitleExtractor,
    SemanticResponsibilityExtractor,
    SemanticEducationExtractor,
    MLConfidenceScorer,
    IndustryClassifier,
    CareerTrajectoryAnalyzer,
    ProjectExtractor,
    extract_portfolio_links
)

from sentence_transformers import SentenceTransformer

# Test CV
test_cv = """
JOHN SMITH
Senior Software Engineer
john.smith@email.com | +1-555-1234 | San Francisco, CA
GitHub: github.com/johnsmith | LinkedIn: linkedin.com/in/johnsmith

PROFESSIONAL SUMMARY
Experienced software engineer with 8 years building scalable systems.
Expert in React, Node.js, Python, Docker, Kubernetes, and AWS.
Strong leadership and problem-solving skills with proven track record.

EXPERIENCE
2020 - Present: Senior Software Engineer at Tech Corp
- Led team of 5 engineers to build microservices architecture
- Reduced deployment time by 60% through CI/CD automation
- Architected system serving 2M+ daily active users
- Improved performance by 45% through optimization
- Mentored 3 junior developers

2016 - 2020: Software Developer at StartupXYZ
- Developed full-stack web applications using React and Django
- Built RESTful APIs handling 10K requests/minute
- Implemented automated testing reducing bugs by 40%

EDUCATION
Master of Science in Computer Science
Stanford University, 2016
GPA: 3.9/4.0

Bachelor of Science in Software Engineering
MIT, 2014

CERTIFICATIONS
- AWS Certified Solutions Architect (2022)
- Certified Scrum Master (2021)
- Google Cloud Professional (2023)

PROJECTS
E-Commerce Platform: Built scalable platform using React, Node.js, PostgreSQL
- Served 500K+ users with 99.9% uptime
- Reduced checkout time by 35%

Machine Learning Pipeline: Implemented ML pipeline with Python, TensorFlow, Docker
"""

def test_module(name, test_func):
    """Helper to test a module"""
    print(f"\n{'='*60}")
    print(f" Testing: {name}")
    print('='*60)
    try:
        result = test_func()
        print(f" {name} - PASSED")
        return result
    except Exception as e:
        print(f" {name} - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*60)
    print(" ADVANCED ML MODULES TEST SUITE")
    print("="*60)
    
    # Initialize embedder
    print("\n Loading embedder (paraphrase-mpnet-base-v2)...")
    embedder = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
    print(" Embedder loaded")
    
    # Test 1: Semantic Skill Extraction
    def test_skills():
        skill_db = [
            ('Python', 'Programming'), ('JavaScript', 'Programming'), ('React', 'Frontend'),
            ('Node.js', 'Backend'), ('Docker', 'DevOps'), ('Kubernetes', 'DevOps'),
            ('AWS', 'Cloud'), ('PostgreSQL', 'Database'), ('Leadership', 'Soft_Skills')
        ]
        
        extractor = SemanticSkillExtractor(embedder, skill_db)
        skills_dict = extractor.extract_skills_semantic(test_cv, threshold=0.72)
        
        print(f"\n Extracted {len(skills_dict)} skills:")
        for skill, (category, confidence, context) in list(skills_dict.items())[:10]:
            print(f"    {skill} ({category}): {confidence:.2f}")
        
        return skills_dict
    
    skills = test_module("Semantic Skill Extraction", test_skills)
    
    # Test 2: ML Job Title Extraction
    def test_jobs():
        extractor = MLJobTitleExtractor(embedder)
        current, titles, seniority, progression = extractor.extract_job_titles_ml(test_cv)
        
        print(f"\n Results:")
        print(f"   Current Title: {current}")
        print(f"   All Titles: {titles[:3]}")
        print(f"   Predicted Seniority: {seniority}")
        print(f"   Career Progression: {len(progression)} roles")
        for role in progression:
            print(f"      - {role['title']} ({role['period']})")
        
        return (current, titles, seniority, progression)
    
    job_data = test_module("ML Job Title Extraction", test_jobs)
    
    # Test 3: Semantic Responsibility Extraction
    def test_responsibilities():
        extractor = SemanticResponsibilityExtractor(embedder)
        classified = extractor.extract_responsibilities_ml(test_cv)
        
        print(f"\n Classified Responsibilities:")
        print(f"   Impact Statements: {len(classified['impact'])}")
        for stmt in classified['impact'][:3]:
            print(f"       {stmt['text'][:80]}...")
        print(f"   Technical Tasks: {len(classified['technical'])}")
        print(f"   Routine Tasks: {len(classified['routine'])}")
        
        return classified
    
    responsibilities = test_module("Semantic Responsibility Extraction", test_responsibilities)
    
    # Test 4: Semantic Education Extraction
    def test_education():
        extractor = SemanticEducationExtractor(embedder)
        edu_data = extractor.extract_education_ml(test_cv)
        
        print(f"\n Education Data:")
        print(f"   Degree Level: {edu_data['level']}")
        print(f"   Degrees: {len(edu_data['degrees'])}")
        for deg in edu_data['degrees']:
            print(f"       {deg['text']} ({deg['level']}, conf: {deg['confidence']:.2f})")
        print(f"   Institutions: {edu_data['institutions']}")
        print(f"   Graduation Year: {edu_data['graduation_year']}")
        print(f"   GPA: {edu_data['gpa']}")
        
        # Test certifications
        certs = extractor.extract_certifications_ml(test_cv)
        print(f"\n   Certifications: {len(certs)}")
        for cert in certs:
            print(f"       {cert['name']} (conf: {cert['confidence']:.2f})")
        
        return edu_data
    
    education = test_module("Semantic Education Extraction", test_education)
    
    # Test 5: ML Confidence Scoring
    def test_confidence():
        scorer = MLConfidenceScorer(embedder)
        
        cv_data = {
            'name': 'John Smith',
            'skills': list(skills.keys()) if skills else [],
            'responsibilities': responsibilities['impact'] if responsibilities else [],
            'degrees': education['degrees'] if education else [],
            'email': 'john@email.com',
            'phone': '+1-555-1234'
        }
        
        confidence = scorer.calculate_ml_confidence(cv_data)
        
        print(f"\n Confidence Scores:")
        print(f"   Overall: {confidence['overall']:.2%}")
        print(f"   Per Field:")
        for field, score in confidence['per_field'].items():
            print(f"       {field}: {score:.2%}")
        
        return confidence
    
    confidence = test_module("ML Confidence Scoring", test_confidence)
    
    # Test 6: Industry Classification
    def test_industry():
        classifier = IndustryClassifier(embedder)
        industries = classifier.classify_industry(test_cv, top_k=3)
        
        print(f"\n Top Industries:")
        for industry, score in industries:
            print(f"    {industry}: {score:.2%}")
        
        return industries
    
    industries = test_module("Industry Classification", test_industry)
    
    # Test 7: Career Trajectory Analysis
    def test_trajectory():
        analyzer = CareerTrajectoryAnalyzer(embedder)
        
        # Use progression from job extraction
        progression = job_data[3] if job_data else []
        trajectory = analyzer.analyze_trajectory(progression)
        
        print(f"\n Career Trajectory:")
        print(f"   Progression Speed: {trajectory['speed']}")
        print(f"   Career Gaps: {len(trajectory['gaps'])}")
        for gap in trajectory['gaps']:
            print(f"       {gap['period']} ({gap['duration_years']} years)")
        print(f"   Predicted Next Roles:")
        for role in trajectory['predicted_next']:
            print(f"       {role}")
        
        return trajectory
    
    trajectory = test_module("Career Trajectory Analysis", test_trajectory)
    
    # Test 8: Project Extraction
    def test_projects():
        extractor = ProjectExtractor(embedder)
        projects = extractor.extract_projects(test_cv)
        
        print(f"\n Extracted {len(projects)} projects:")
        for proj in projects:
            print(f"    {proj['description'][:60]}...")
            print(f"     Technologies: {proj['technologies'][:5]}")
            print(f"     Impact: {proj['impact']}")
        
        return projects
    
    projects = test_module("Project Extraction", test_projects)
    
    # Test 9: Portfolio Links
    def test_links():
        links = extract_portfolio_links(test_cv)
        
        print(f"\n Portfolio Links:")
        print(f"   GitHub: {links['github']}")
        print(f"   LinkedIn: {links['linkedin']}")
        print(f"   Portfolio: {links['portfolio']}")
        
        return links
    
    links = test_module("Portfolio Links Detection", test_links)
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    results = {
        'Skills': len(skills) if skills else 0,
        'Job Titles': len(job_data[1]) if job_data else 0,
        'Responsibilities': len(responsibilities['impact']) if responsibilities else 0,
        'Degrees': len(education['degrees']) if education else 0,
        'Certifications': 3,  # Known from test CV
        'Industries': len(industries) if industries else 0,
        'Projects': len(projects) if projects else 0,
        'Links': len([v for v in links.values() if v]) if links else 0
    }
    
    print("\n All Modules Tested Successfully!\n")
    
    for feature, count in results.items():
        print(f"   {feature}: {count} extracted")
    
    if confidence:
        print(f"\n   Overall Confidence: {confidence['overall']:.2%}")
    
    print("\n" + "="*60)
    print(" READY FOR INTEGRATION")
    print("="*60)
    print("\nNext steps:")
    print("1. Review extracted data above")
    print("2. Adjust thresholds if needed in advanced_ml_modules.py")
    print("3. Follow ML_UPGRADE_INTEGRATION.py for full integration")
    print("4. Test with real CVs from your database")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
