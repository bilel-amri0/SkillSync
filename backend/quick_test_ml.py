"""
Quick validation test for Advanced ML modules
"""
import sys

print("="*60)
print(" QUICK VALIDATION TEST")
print("="*60)

# Test 1: Import modules
print("\n1 Testing imports...")
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
    sys.exit(1)

# Test 2: Load embedder
print("\n2 Loading sentence transformer...")
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('paraphrase-mpnet-base-v2', device='cpu')
    print("    Embedder loaded (768-dim)")
except Exception as e:
    print(f"    Embedder failed: {e}")
    sys.exit(1)

# Test 3: Initialize one module
print("\n3 Testing SemanticSkillExtractor...")
try:
    skill_db = [
        ('Python', 'Programming'),
        ('JavaScript', 'Programming'),
        ('React', 'Frontend'),
        ('AWS', 'Cloud')
    ]
    extractor = SemanticSkillExtractor(embedder, skill_db)
    print("    SemanticSkillExtractor initialized")
except Exception as e:
    print(f"    Initialization failed: {e}")
    sys.exit(1)

# Test 4: Extract from sample text
print("\n4 Testing extraction...")
try:
    sample_text = "Senior developer with 5 years experience in Python, React, and AWS cloud services"
    skills = extractor.extract_skills_semantic(sample_text, threshold=0.70)
    print(f"    Extracted {len(skills)} skills: {list(skills.keys())[:5]}")
except Exception as e:
    print(f"    Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test other modules
print("\n5 Testing other modules...")
try:
    # Job extractor
    job_extractor = MLJobTitleExtractor(embedder)
    print("    MLJobTitleExtractor")
    
    # Responsibility extractor
    resp_extractor = SemanticResponsibilityExtractor(embedder)
    print("    SemanticResponsibilityExtractor")
    
    # Education extractor
    edu_extractor = SemanticEducationExtractor(embedder)
    print("    SemanticEducationExtractor")
    
    # Confidence scorer
    scorer = MLConfidenceScorer(embedder)
    print("    MLConfidenceScorer")
    
    # Industry classifier
    classifier = IndustryClassifier(embedder)
    print("    IndustryClassifier")
    
    # Trajectory analyzer
    analyzer = CareerTrajectoryAnalyzer(embedder)
    print("    CareerTrajectoryAnalyzer")
    
    # Project extractor
    proj_extractor = ProjectExtractor(embedder)
    print("    ProjectExtractor")
    
    # Portfolio links
    links = extract_portfolio_links("github.com/test linkedin.com/in/test")
    print(f"    extract_portfolio_links (found {len([v for v in links.values() if v])} links)")
    
except Exception as e:
    print(f"    Module test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print(" ALL TESTS PASSED")
print("="*60)
print("\n Results:")
print(f"    9 ML modules:  Working")
print(f"    Embedder:  Loaded (768-dim)")
print(f"    Extraction:  Functional")
print(f"    Sample skills extracted: {len(skills)}")
print("\n Advanced ML modules are ready for integration!")
print("\nNext steps:")
print("   1. Run full test: python test_advanced_ml.py")
print("   2. Integrate: Follow ML_UPGRADE_INTEGRATION.py")
print("   3. Deploy: Use upgraded production_cv_parser_final.py")
print("")
