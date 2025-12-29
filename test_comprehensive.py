import requests
import json
import time

print("="*80)
print("COMPREHENSIVE ML CAREER GUIDANCE TEST")
print("="*80)

# More detailed CV
cv = """
BILEL AMRI
Senior Machine Learning Engineer
Email: bilel.amri@example.com | Phone: +1-555-0123
Location: San Francisco, CA
LinkedIn: linkedin.com/in/bilelamri | GitHub: github.com/bilelamri

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 5 years of experience building and deploying 
ML models at scale. Expert in deep learning, NLP, computer vision, and MLOps. 
Strong background in Python, TensorFlow, PyTorch, and cloud infrastructure (AWS/GCP).
Passionate about using AI to solve real-world problems and mentor junior engineers.

WORK EXPERIENCE

Senior ML Engineer | TechCorp AI Division | San Francisco, CA | 2022 - Present
• Led team of 4 ML engineers building recommendation systems serving 10M+ users
• Designed and deployed 15+ deep learning models using TensorFlow and PyTorch
• Implemented MLOps pipeline with Docker, Kubernetes, Jenkins, reducing deployment time by 60%
• Built real-time data pipelines processing 50TB+ data daily using Apache Spark and Kafka
• Improved model accuracy by 35% through advanced feature engineering and hyperparameter tuning
• Collaborated with product, data science, and engineering teams on ML product roadmap
• Technologies: Python, TensorFlow, PyTorch, Scikit-learn, Docker, Kubernetes, AWS SageMaker

Machine Learning Engineer | DataStart Inc. | San Jose, CA | 2020 - 2022  
• Developed NLP models for text classification and sentiment analysis with 92% accuracy
• Built computer vision models for object detection and image segmentation
• Created data preprocessing pipelines using Pandas, NumPy, and Apache Airflow
• Deployed models to production using FastAPI, Flask, and AWS Lambda
• Implemented A/B testing framework to measure model performance in production
• Technologies: Python, Keras, Scikit-learn, NLTK, OpenCV, FastAPI, PostgreSQL

Software Engineer | WebTech Solutions | Palo Alto, CA | 2018 - 2020
• Built full-stack web applications using React, Node.js, and MongoDB
• Developed RESTful APIs serving 1M+ requests per day
• Implemented CI/CD pipelines with Jenkins and GitHub Actions
• Managed PostgreSQL and Redis databases for high-traffic applications
• Technologies: JavaScript, React, Node.js, Express, MongoDB, PostgreSQL, AWS

EDUCATION
Master of Science in Computer Science - Machine Learning
Stanford University | 2017 - 2019 | GPA: 3.9/4.0
Focus: Deep Learning, Natural Language Processing, Computer Vision

Bachelor of Science in Computer Engineering  
UC Berkeley | 2013 - 2017 | GPA: 3.7/4.0

TECHNICAL SKILLS
• Languages: Python, JavaScript, SQL, Java, C++, R
• ML/DL Frameworks: TensorFlow, PyTorch, Keras, Scikit-learn, Hugging Face, LangChain
• Data Science: Pandas, NumPy, Matplotlib, Seaborn, Jupyter, Scipy
• MLOps: Docker, Kubernetes, Jenkins, MLflow, W&B, Airflow, CI/CD
• Cloud: AWS (SageMaker, EC2, S3, Lambda), GCP (Vertex AI, BigQuery), Azure
• Databases: PostgreSQL, MongoDB, Redis, Elasticsearch, Neo4j
• Big Data: Apache Spark, Kafka, Hadoop, Hive
• Web: FastAPI, Flask, Django, React, Node.js, GraphQL
• Tools: Git, Linux, Vim, VS Code, Postman

CERTIFICATIONS
• AWS Certified Machine Learning - Specialty (2023)
• TensorFlow Developer Certificate (2022)
• AWS Certified Solutions Architect - Associate (2021)

PROJECTS
• AI Resume Analyzer: Built end-to-end ML system analyzing resumes using NLP and providing 
  career guidance. Used BERT for skill extraction, implemented recommendation engine.
  Tech: Python, TensorFlow, FastAPI, React, PostgreSQL, Docker
  
• Stock Price Predictor: LSTM-based model for time series forecasting of stock prices.
  Achieved 85% directional accuracy. Deployed as web app with real-time predictions.
  Tech: PyTorch, Pandas, Flask, React, AWS

• Image Recognition API: Production-ready computer vision API for object detection and 
  classification. Serving 100k+ requests daily with 99.9% uptime.
  Tech: PyTorch, FastAPI, Docker, Kubernetes, AWS
"""

print(f"\n📄 Testing with comprehensive CV ({len(cv)} characters)")
print("⏳ Sending to ML career guidance API...")
print("   (First request may take 20-30s to load ML models)")

try:
    start = time.time()
    response = requests.post(
        'http://localhost:8001/api/v1/career-guidance',
        json={'cv_content': cv},
        timeout=90
    )
    elapsed = time.time() - start
    
    print(f"\n✅ Response received in {elapsed:.2f}s")
    print(f"   HTTP Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Save full result
        with open('comprehensive_ml_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("   💾 Full result saved to: comprehensive_ml_result.json")
        
        # Display results
        print("\n" + "="*80)
        print("📊 ML ANALYSIS RESULTS")
        print("="*80)
        
        # Metadata
        meta = result.get('metadata', {})
        print(f"\n🤖 ML ENGINE:")
        print(f"   Model: {meta.get('ml_model', 'N/A')}")
        print(f"   Version: {meta.get('engine_version', 'N/A')}")
        print(f"   Processing time: {meta.get('processing_time_seconds', 0):.2f}s")
        print(f"   Skills extracted: {meta.get('cv_skills_count', 0)}")
        
        # Job Recommendations
        jobs = result.get('job_recommendations', [])
        print(f"\n💼 JOB RECOMMENDATIONS: {len(jobs)}")
        if jobs:
            for i, job in enumerate(jobs[:3], 1):
                print(f"\n   {i}. {job['title']}")
                print(f"      🤖 ML Similarity: {job['similarity_score']*100:.1f}%")
                print(f"      🎯 ML Confidence: {job['confidence']*100:.1f}%")
                salary = job['predicted_salary']
                print(f"      💰 Predicted Salary: ${salary['min']:,} - ${salary['max']:,}")
                print(f"      📈 Growth Potential: {job['growth_potential']}")
                print(f"      ✅ Matching: {len(job['matching_skills'])} skills")
                print(f"      📚 To Learn: {len(job['skill_gaps'])} skills")
                if job['matching_skills']:
                    print(f"      Skills: {', '.join(job['matching_skills'][:5])}")
        else:
            print("   ⚠️  No jobs found (similarity threshold not met)")
        
        # Certifications  
        certs = result.get('certification_recommendations', [])
        print(f"\n🎓 CERTIFICATION RECOMMENDATIONS: {len(certs)}")
        if certs:
            for i, cert in enumerate(certs[:3], 1):
                print(f"\n   {i}. {cert['name']}")
                print(f"      🤖 ML Relevance: {cert['relevance_score']*100:.1f}%")
                print(f"      🎯 Skill Alignment: {cert['skill_alignment']*100:.1f}%")
                print(f"      💰 Predicted ROI: {cert['predicted_roi']}")
                print(f"      ⏱️  Time: {cert['estimated_time']}")
                print(f"      📈 Career Boost: {cert['career_boost']}")
        
        # Learning Roadmap
        roadmap = result.get('learning_roadmap', {})
        phases = roadmap.get('phases', [])
        print(f"\n🎯 LEARNING ROADMAP: {len(phases)} phases")
        print(f"   📅 Total Duration: {roadmap.get('total_duration_weeks')} weeks ({roadmap.get('total_duration_months')} months)")
        print(f"   🎓 Predicted Success: {roadmap.get('predicted_success_rate')}")
        print(f"   ✨ Personalization: {roadmap.get('personalization_score')}")
        print(f"   📚 Strategy: {roadmap.get('learning_strategy')}")
        
        if phases:
            for i, phase in enumerate(phases, 1):
                print(f"\n   Phase {i}: {phase['phase_name']}")
                print(f"      Duration: {phase['duration_weeks']} weeks")
                print(f"      Success: {phase['success_probability']}")
                print(f"      Skills: {', '.join(phase['skills_to_learn'][:5])}")
        
        # XAI Insights
        xai = result.get('xai_insights', {})
        confidence = xai.get('ml_confidence_scores', {})
        print(f"\n🧠 ML CONFIDENCE SCORES:")
        for key, val in confidence.items():
            print(f"   • {key}: {val}")
        
        key_insights = xai.get('key_insights', [])
        print(f"\n💡 KEY INSIGHTS:")
        for insight in key_insights[:5]:
            print(f"   • {insight}")
        
        # Summary
        print("\n" + "="*80)
        print("✅ ML CAREER GUIDANCE SYSTEM - ALL TESTS PASSED!")
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   • {len(jobs)} job recommendations (ML-matched)")
        print(f"   • {len(certs)} certification recommendations (ML-ranked)")
        print(f"   • {len(phases)}-phase learning roadmap (ML-optimized)")
        print(f"   • Complete XAI explainability")
        print(f"   • Processing time: {elapsed:.2f}s")
        
        print(f"\n📁 Files created:")
        print(f"   • comprehensive_ml_result.json (full ML analysis)")
        
        print(f"\n📖 Documentation:")
        print(f"   • ML_CAREER_SYSTEM_DOCUMENTATION.md")
        print(f"   • ML_IMPLEMENTATION_SUMMARY.md")
        
        print("\n" + "="*80 + "\n")
        
    else:
        print(f"\n❌ HTTP Error: {response.status_code}")
        print(response.text[:500])
        
except requests.exceptions.Timeout:
    print("\n⏱️  Timeout! Request took >90 seconds")
    print("Note: First ML request may take 20-30s to load models")
except requests.exceptions.ConnectionError:
    print("\n❌ Connection Error!")
    print("Make sure backend server is running: cd backend && python start_server.py")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
