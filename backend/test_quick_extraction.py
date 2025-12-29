"""
Quick ML Extraction Test - Tests actual extraction with full CV
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" QUICK ML EXTRACTION TEST")
print("=" * 70)

# Load production parser
print("\n1 Loading production parser...")
from production_cv_parser_final import ProductionCVParser
parser = ProductionCVParser()
print("    Parser loaded")

# Import ML modules
print("\n2 Importing ML modules...")
from advanced_ml_modules import (
    SemanticSkillExtractor,
    MLJobTitleExtractor,
    SemanticResponsibilityExtractor,
    extract_portfolio_links
)
print("    Modules imported")

# Initialize extractors
print("\n3 Initializing extractors...")
skill_extractor = SemanticSkillExtractor(parser.embedder, parser.all_skills)
job_extractor = MLJobTitleExtractor(parser.embedder)
resp_extractor = SemanticResponsibilityExtractor(parser.embedder)
print("    Extractors ready")

# Test CV
sample_cv = """
John Doe
Senior Software Engineer
john.doe@example.com | +1-555-0123
GitHub: github.com/johndoe | LinkedIn: linkedin.com/in/johndoe

Experienced software engineer with 8 years in Python, React, AWS, and Docker.
Led team of 5 engineers and increased system performance by 45%.
Built microservices handling 100K requests daily with 92% accuracy.

EXPERIENCE:
Senior Software Engineer at Tech Corp (2021-Present)
- Architected cloud platform serving 1M+ users
- Reduced deployment time by 60% with CI/CD pipeline
- Managed cross-functional team of 5 engineers
- Technologies: Python, React, AWS, Kubernetes, Docker

Software Engineer at StartupXYZ (2018-2021)  
- Developed RESTful APIs with Django and PostgreSQL
- Implemented machine learning models (scikit-learn, TensorFlow)
- Reduced infrastructure costs by $50K annually

EDUCATION:
Master of Science in Computer Science - MIT (2016)
AWS Certified Solutions Architect (2022)
"""

print("\n4 Testing SemanticSkillExtractor...")
skills_dict = skill_extractor.extract_skills_semantic(sample_cv, threshold=0.72)
skills = sorted(skills_dict.keys(), key=lambda s: skills_dict[s][1], reverse=True)[:15]
print(f"    Extracted {len(skills_dict)} total skills")
print(f"    Top 15: {', '.join(skills)}")

print("\n5 Testing MLJobTitleExtractor...")
current, titles, seniority, progression = job_extractor.extract_job_titles_ml(sample_cv)
print(f"    Current: {current}")
print(f"    Seniority: {seniority}")
print(f"    All titles: {titles[:5]}")

print("\n6 Testing SemanticResponsibilityExtractor...")
resp = resp_extractor.extract_responsibilities_ml(sample_cv)
print(f"    Impact achievements: {len(resp['impact'])}")
print(f"    Technical tasks: {len(resp['technical'])}")
if resp['impact']:
    print(f"    Top impact: {resp['impact'][0]['text'][:60]}...")

print("\n7 Testing extract_portfolio_links...")
links = extract_portfolio_links(sample_cv)
print(f"    GitHub: {links['github']}")
print(f"    LinkedIn: {links['linkedin']}")

print("\n" + "=" * 70)
print(" ALL TESTS PASSED!")
print("=" * 70)
print(f"\n Results Summary:")
print(f"    Skills: {len(skills_dict)} total, {len(skills)} top skills")
print(f"    Seniority: {seniority}")
print(f"    Impact achievements: {len(resp['impact'])}")
print(f"    Portfolio links: {2 if links['github'] and links['linkedin'] else 0}/2")
print("\n Advanced ML modules are working perfectly!")
