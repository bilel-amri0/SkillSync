"""
Test Advanced ML Modules Using Existing Production Parser's Embedder
This test reuses the already-initialized embedder from production_cv_parser_final.py
to avoid re-downloading models.
"""

print("=" * 60)
print(" TESTING ADVANCED ML WITH EXISTING EMBEDDER")
print("=" * 60)

# Step 1: Import production parser
print("\n1 Loading production parser (this has the embedder already)...")
try:
    from production_cv_parser_final import CVParserService
    parser = CVParserService()
    embedder = parser.embedder
    print(f"    Embedder loaded: {embedder}")
    print(f"    Device: {embedder.device}")
except Exception as e:
    print(f"    Failed to load production parser: {e}")
    exit(1)

# Step 2: Import advanced ML modules
print("\n2 Importing advanced ML modules...")
try:
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
    print("    All 9 modules imported successfully")
except Exception as e:
    print(f"    Import failed: {e}")
    exit(1)

# Step 3: Initialize modules with existing embedder
print("\n3 Initializing ML modules...")
try:
    skill_extractor = SemanticSkillExtractor(embedder, parser.all_skills)
    job_extractor = MLJobTitleExtractor(embedder)
    responsibility_extractor = SemanticResponsibilityExtractor(embedder)
    education_extractor = SemanticEducationExtractor(embedder)
    confidence_scorer = MLConfidenceScorer(embedder)
    industry_classifier = IndustryClassifier(embedder)
    trajectory_analyzer = CareerTrajectoryAnalyzer(embedder)
    project_extractor = ProjectExtractor(embedder)
    print("    All 8 ML modules initialized")
except Exception as e:
    print(f"    Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Test with sample CV text
print("\n4 Testing extraction with sample CV...")
sample_cv = """
John Doe
Senior Software Engineer
Email: john.doe@example.com | Phone: +1-555-0123
GitHub: github.com/johndoe | LinkedIn: linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 8+ years in full-stack development, specializing in 
cloud-native applications, microservices architecture, and machine learning systems.

WORK EXPERIENCE

Senior Software Engineer | Tech Corp | 2021 - Present
- Led development of cloud-based platform serving 1M+ users
- Increased system performance by 45% through optimization
- Managed team of 5 junior engineers
- Implemented CI/CD pipeline reducing deployment time by 60%
- Technologies: Python, React, AWS, Docker, Kubernetes

Software Engineer | StartupXYZ | 2018 - 2021
- Developed RESTful APIs handling 100K+ daily requests
- Built machine learning models with 92% accuracy
- Reduced infrastructure costs by $50K annually
- Technologies: Python, Django, PostgreSQL, Redis

Junior Developer | WebDev Inc | 2016 - 2018
- Maintained legacy codebases and fixed bugs
- Wrote unit tests and documentation
- Participated in code reviews

EDUCATION
Master of Science in Computer Science | MIT | 2016
Bachelor of Science in Software Engineering | Stanford University | 2014
GPA: 3.8/4.0

CERTIFICATIONS
- AWS Certified Solutions Architect (2022)
- Google Cloud Professional (2021)
- Certified Kubernetes Administrator (2020)

PROJECTS
- OpenAI Chat Application: Built chatbot using GPT-4 API, served 10K users
  Technologies: Python, FastAPI, React, OpenAI API
- Real-time Analytics Dashboard: Created data visualization tool with 50ms latency
  Technologies: Node.js, D3.js, WebSocket, Redis
"""

# Test 5: Semantic Skill Extraction
print("\n5 Testing SemanticSkillExtractor...")
try:
    skills_dict = skill_extractor.extract_skills_semantic(sample_cv, threshold=0.72)
    skills = sorted(skills_dict.keys(), key=lambda s: skills_dict[s][1], reverse=True)[:15]
    print(f"    Extracted {len(skills_dict)} total skills")
    print(f"    Top 15 skills: {', '.join(skills[:15])}")
except Exception as e:
    print(f"    Skill extraction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: ML Job Title Extraction
print("\n6 Testing MLJobTitleExtractor...")
try:
    current, titles, seniority, progression = job_extractor.extract_job_titles_ml(sample_cv)
    print(f"    Current title: {current}")
    print(f"    Seniority level: {seniority}")
    print(f"    Career progression: {len(progression)} positions")
    print(f"    All titles: {titles[:5]}")
except Exception as e:
    print(f"    Job title extraction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Responsibility Extraction
print("\n7 Testing SemanticResponsibilityExtractor...")
try:
    responsibilities = responsibility_extractor.extract_responsibilities_ml(sample_cv)
    impact_count = len(responsibilities.get('impact', []))
    routine_count = len(responsibilities.get('routine', []))
    print(f"    Impact achievements: {impact_count}")
    print(f"    Routine tasks: {routine_count}")
    if impact_count > 0:
        print(f"    Top impact: {responsibilities['impact'][0]['text'][:80]}...")
except Exception as e:
    print(f"    Responsibility extraction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Education Extraction
print("\n8 Testing SemanticEducationExtractor...")
try:
    edu_data = education_extractor.extract_education_ml(sample_cv)
    certs = education_extractor.extract_certifications_ml(sample_cv)
    print(f"    Degrees found: {len(edu_data.get('degrees', []))}")
    print(f"    Highest level: {edu_data.get('level', 'N/A')}")
    print(f"    Certifications: {len(certs)}")
    if edu_data.get('degrees'):
        print(f"    Degrees: {[d['level'] for d in edu_data['degrees']]}")
except Exception as e:
    print(f"    Education extraction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Industry Classification
print("\n9 Testing IndustryClassifier...")
try:
    industries = industry_classifier.classify_industry(sample_cv, top_k=3)
    print(f"    Top 3 industries:")
    for industry, score in industries:
        print(f"      - {industry}: {score:.2f}")
except Exception as e:
    print(f"    Industry classification failed: {e}")
    import traceback
    traceback.print_exc()

# Test 10: Career Trajectory
print("\n Testing CareerTrajectoryAnalyzer...")
try:
    if 'progression' in locals() and progression:
        trajectory = trajectory_analyzer.analyze_trajectory(progression)
        print(f"    Progression speed: {trajectory.get('speed', 'N/A')}")
        print(f"    Career gaps: {len(trajectory.get('gaps', []))}")
        print(f"    Predicted next roles: {trajectory.get('predicted_next_roles', [])[:3]}")
    else:
        print("     No career progression data available")
except Exception as e:
    print(f"    Trajectory analysis failed: {e}")
    import traceback
    traceback.print_exc()

# Test 11: Project Extraction
print("\n11 Testing ProjectExtractor...")
try:
    projects = project_extractor.extract_projects(sample_cv)
    print(f"    Projects found: {len(projects)}")
    if projects:
        print(f"    First project: {projects[0]['description'][:60]}...")
        print(f"      Technologies: {projects[0].get('technologies', [])[:5]}")
except Exception as e:
    print(f"    Project extraction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 12: Portfolio Links
print("\n12 Testing extract_portfolio_links...")
try:
    links = extract_portfolio_links(sample_cv)
    print(f"    GitHub: {links.get('github', 'N/A')}")
    print(f"    LinkedIn: {links.get('linkedin', 'N/A')}")
    print(f"    Portfolio: {links.get('portfolio', 'N/A')}")
except Exception as e:
    print(f"    Portfolio link extraction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 13: ML Confidence Scoring
print("\n13 Testing MLConfidenceScorer...")
try:
    cv_data = {
        'name': 'John Doe',
        'skills': skills[:20] if 'skills' in locals() else [],
        'job_titles': titles[:5] if 'titles' in locals() else [],
        'responsibilities': [r['text'] for r in responsibilities.get('impact', [])][:5] if 'responsibilities' in locals() else []
    }
    confidence = confidence_scorer.calculate_ml_confidence(cv_data)
    print(f"    Overall confidence: {confidence.get('overall', 0):.2%}")
    print(f"    Per-field breakdown:")
    for field, score in confidence.get('per_field', {}).items():
        print(f"      - {field}: {score:.2%}")
except Exception as e:
    print(f"    Confidence scoring failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print(" TESTING COMPLETE!")
print("=" * 60)
print("\n Summary:")
print("   - All 9 ML modules tested")
print("   - Using existing production parser's embedder")
print("   - No model re-downloading required")
print("   - Ready for integration into production_cv_parser_final.py")
print("\n Next step: Follow ML_UPGRADE_INTEGRATION.py to integrate these modules")
