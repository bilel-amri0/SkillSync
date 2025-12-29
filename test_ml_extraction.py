"""
Test ML CV Extraction with Real CV Data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from ai_cv_analyzer import AdvancedCVExtractor

# Sample CV texts
cv1 = """
BILEL AMRI
Senior Full Stack Developer

Email: bilel.amri@email.com
Phone: +33 6 12 34 56 78

EXPERIENCE:
2019-2023: Senior Full Stack Developer at TechCorp
- Developed enterprise web applications using React, Node.js, and PostgreSQL
- Led team of 5 developers
- Implemented CI/CD pipelines with Docker and Kubernetes

2016-2019: Software Engineer at StartupXYZ
- Built REST APIs with Python Django and Flask
- Managed AWS cloud infrastructure
- Worked with MongoDB and Redis

SKILLS:
Python, JavaScript, TypeScript, React, Angular, Node.js, Express, Django, Flask
PostgreSQL, MongoDB, Redis, MySQL
Docker, Kubernetes, AWS, Azure, Git, GitHub Actions
Machine Learning, TensorFlow, Scikit-learn

EDUCATION:
2012-2016: Bachelor in Computer Science, University of Tunis
"""

cv2 = """
SARAH MARTIN
Data Scientist & ML Engineer

Contact: sarah.martin@gmail.com | +1 555-123-4567

PROFESSIONAL EXPERIENCE:

Machine Learning Engineer | Google | 2021-Present
• Developed NLP models for search ranking using TensorFlow and PyTorch
• Implemented MLOps pipelines with Kubernetes and MLflow
• Worked with BigQuery, Spark, and Hadoop for data processing

Data Scientist | Facebook | 2018-2021
• Built recommendation systems using Python and Scikit-learn
• Analyzed user behavior with SQL and Pandas
• Deployed models to production using Docker and AWS

TECHNICAL SKILLS:
Languages: Python, R, SQL, Java, C++
ML/AI: TensorFlow, PyTorch, Keras, Scikit-learn, NLP, Computer Vision
Data: Pandas, NumPy, Matplotlib, Jupyter, Spark, Hadoop
Cloud: AWS, GCP, Azure
Tools: Docker, Git, Jenkins, MLflow

EDUCATION:
PhD in Machine Learning, Stanford University, 2018
MS in Computer Science, MIT, 2015
"""

def test_extraction(cv_text, label):
    print("\n" + "="*100)
    print(f"TESTING: {label}")
    print("="*100)
    
    extractor = AdvancedCVExtractor()
    result = extractor.parse_cv_advanced(cv_text)
    
    print("\n📊 EXTRACTION RESULTS:")
    print(f"   Name: {result.name} (confidence: {result.confidence_scores.get('name', 0):.2f})")
    print(f"   Title: {result.title} (confidence: {result.confidence_scores.get('title', 0):.2f})")
    print(f"   Email: {result.email}")
    print(f"   Phone: {result.phone}")
    print(f"   Skills ({len(result.skills)}): {', '.join(result.skills[:15])}")
    print(f"   Experience entries: {len(result.experience)}")
    print(f"   Education entries: {len(result.education)}")
    print(f"   Overall confidence: {result.confidence_scores.get('skills', 0):.2f}")

if __name__ == "__main__":
    print("\n🧪 TESTING ML CV EXTRACTION ENGINE")
    print("This will show detailed logs from the ML extraction process\n")
    
    test_extraction(cv1, "CV #1 - Bilel Amri (Full Stack Developer)")
    test_extraction(cv2, "CV #2 - Sarah Martin (Data Scientist)")
    
    print("\n" + "="*100)
    print("✅ TEST COMPLETE - Check logs above to verify:")
    print("   1. Different CVs produce DIFFERENT results")
    print("   2. Skills are extracted correctly (10-20 per CV)")
    print("   3. Names, titles, emails are found")
    print("   4. ML semantic matching is working")
    print("="*100)
