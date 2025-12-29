"""
 Career Guidance Engine Test
Tests the complete flow: CV Upload  ML Analysis  Career Guidance  XAI Insights
"""
import requests
import json

BASE_URL = "http://localhost:8001"

# Sample CV for testing
SAMPLE_CV = """
Bilel Amri
bilela329@gmail.com
github.com/Bilel-Amri
linkedin.com/in/amri-bilel-53092b283

PROFILE
Student engineer at Tek-Up, fueled by a passion for artificial intelligence and its
applications in solving real-world problems.

SKILLS
Programming: Python, C, C++, JavaScript, HTML, CSS
AI/ML: Machine Learning, Deep Learning, NLP, Computer Vision, TensorFlow, PyTorch
Web Development: React, FastAPI, Django, Bootstrap
Tools: Docker, Git, GitHub, Linux, MATLAB
Soft Skills: Problem Solving, Teamwork, Communication

EDUCATION
Engineering Student | Tek-Up | 2022 - Present
Focus: Artificial Intelligence and Machine Learning

PROJECTS
- AI Chatbot: Built intelligent chatbot using NLP and machine learning
  Technologies: Python, TensorFlow, FastAPI, React
  
- Sports Analytics Platform: Developed analytics dashboard for sports data
  Technologies: Python, Pandas, React, FastAPI

LANGUAGES
- French: Professional working proficiency
- English: Professional working proficiency
- Arabic: Native proficiency

INTERNSHIPS
- Summer Internship | Tunisian Tech Company | 2024
  Focus: Machine Learning and web development
"""


def test_career_guidance():
    """Test complete career guidance flow"""
    print("=" * 80)
    print(" CAREER GUIDANCE ENGINE TEST")
    print("=" * 80)
    print()
    
    try:
        # Call career guidance endpoint
        print(" Sending CV for career guidance analysis...")
        response = requests.post(
            f"{BASE_URL}/api/v1/career-guidance",
            json={"cv_content": SAMPLE_CV},
            timeout=60
        )
        
        print(f" Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f" Error: {response.text}")
            return
        
        result = response.json()
        
        # Display results in beautiful format
        print("\n" + "=" * 80)
        print(" CV ANALYSIS SUMMARY")
        print("=" * 80)
        
        cv_analysis = result['cv_analysis']
        print(f"\n Seniority Level: {cv_analysis['seniority']}")
        print(f" Experience: {cv_analysis['experience_years']} years")
        print(f" Total Skills: {len(cv_analysis['skills'])}")
        print(f"   Top Skills: {', '.join(cv_analysis['skills'][:10])}")
        
        print(f"\n Top Industries:")
        for ind in cv_analysis['industries']:
            print(f"    {ind['name']}: {ind['confidence']*100:.1f}% confidence")
        
        if cv_analysis.get('portfolio_links'):
            print(f"\n Portfolio:")
            if cv_analysis['portfolio_links'].get('github'):
                print(f"    GitHub: {cv_analysis['portfolio_links']['github']}")
            if cv_analysis['portfolio_links'].get('linkedin'):
                print(f"    LinkedIn: {cv_analysis['portfolio_links']['linkedin']}")
        
        # Job Recommendations
        print("\n" + "=" * 80)
        print(" RECOMMENDED JOBS")
        print("=" * 80)
        
        for i, job in enumerate(result['recommended_jobs'], 1):
            print(f"\n{i}. {job['title']}")
            print(f"   Match Score: {job['match_score']}")
            print(f"   Salary Range: {job['salary_range']}")
            print(f"   Growth Potential: {job['growth_potential']}")
            print(f"   \n   Why this job?")
            for reason in job['reasons']:
                print(f"      {reason}")
            
            if job['missing_skills']:
                print(f"   \n    Skills to learn:")
                for skill in job['missing_skills'][:5]:
                    print(f"       {skill}")
        
        # Certification Recommendations
        print("\n" + "=" * 80)
        print(" RECOMMENDED CERTIFICATIONS")
        print("=" * 80)
        
        for i, cert in enumerate(result['recommended_certifications'], 1):
            print(f"\n{i}. {cert['name']}")
            print(f"   Provider: {cert['provider']}")
            print(f"   Priority: {cert['priority']}")
            print(f"   Duration: {cert['duration']}")
            print(f"   Cost: {cert['cost']}")
            print(f"   \n   Why this certification?")
            for reason in cert['reasons']:
                print(f"      {reason}")
            print(f"   \n    Impact: {cert['career_impact']}")
        
        # Learning Roadmap
        print("\n" + "=" * 80)
        print("  LEARNING ROADMAP")
        print("=" * 80)
        
        roadmap = result['learning_roadmap']
        print(f"\n Current Level: {roadmap['current_level']}")
        print(f" Target Level: {roadmap['target_level']}")
        print(f"  Timeline: {roadmap['timeline']}")
        
        print(f"\n Reasoning:")
        for reason in roadmap['reasoning']:
            print(f"   {reason}")
        
        print(f"\n Learning Phases:")
        for i, phase in enumerate(roadmap['phases'], 1):
            print(f"\n   Phase {i}: {phase['phase']}")
            print(f"   Duration: {phase['duration']}")
            print(f"   Priority: {phase['priority']}")
            print(f"   Skills to learn:")
            for skill in phase['skills']:
                print(f"       {skill}")
            print(f"   Reason: {phase['reason']}")
        
        # XAI Insights
        print("\n" + "=" * 80)
        print(" EXPLAINABLE AI (XAI) INSIGHTS")
        print("=" * 80)
        
        xai = result['xai_insights']
        
        print(f"\n Analysis Summary:")
        summary = xai['analysis_summary']
        print(f"    Total Skills: {summary['total_skills']}")
        print(f"    Top Industries: {', '.join(summary['top_industries'])}")
        print(f"    Skill Extraction: {summary['skill_extraction_method']}")
        
        print(f"\n Job Matching Logic:")
        job_logic = xai['job_matching_logic']
        print(f"    Algorithm: {job_logic['algorithm']}")
        print(f"    Threshold: {job_logic['threshold']}")
        print(f"    Top Match: {job_logic['top_match']['job']} ({job_logic['top_match']['score']})")
        
        print(f"\n Certification Logic:")
        cert_logic = xai['certification_logic']
        print(f"    Algorithm: {cert_logic['algorithm']}")
        print(f"    Threshold: {cert_logic['threshold']}")
        print(f"    Rationale: {cert_logic['rationale']}")
        
        print(f"\n  Roadmap Logic:")
        roadmap_logic = xai['roadmap_logic']
        print(f"    Phases: {roadmap_logic['phases']}")
        print(f"    Methodology: {roadmap_logic['methodology']}")
        print(f"    Personalization: {roadmap_logic['personalization']}")
        
        print(f"\n Confidence Scores:")
        conf = xai['confidence_scores']
        print(f"    Job Recommendations: {conf['job_recommendations']}")
        print(f"    Certification Fit: {conf['certification_fit']}")
        print(f"    Roadmap Accuracy: {conf['roadmap_accuracy']}")
        
        print(f"\n ML Features Used:")
        for feature in xai['ml_features_used']:
            print(f"    {feature}")
        
        # Save full JSON
        print("\n" + "=" * 80)
        print(" Saving full results to career_guidance_result.json")
        print("=" * 80)
        
        with open('career_guidance_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(" Saved!")
        
        print("\n" + "=" * 80)
        print(" CAREER GUIDANCE TEST COMPLETE")
        print("=" * 80)
        print("\n Summary:")
        print(f"    {len(result['recommended_jobs'])} job recommendations")
        print(f"    {len(result['recommended_certifications'])} certification recommendations")
        print(f"    {len(result['learning_roadmap']['phases'])} learning phases")
        print(f"    Complete XAI insights explaining all recommendations")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n Starting Career Guidance Engine Test\n")
    print("Prerequisites:")
    print("1. Backend server running: python -m uvicorn main_simple_for_frontend:app --reload --port 8001")
    print("2. ML models loaded (will take ~20s first time)")
    print()
    input("Press Enter to start test...")
    print()
    
    test_career_guidance()
