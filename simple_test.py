import requests
import json
import time

print("Testing ML Career Guidance...")
time.sleep(3)

cv = "Python Developer with TensorFlow, PyTorch, Docker, Kubernetes, 3 years experience"

try:
    print("Sending request...")
    r = requests.post(
        'http://localhost:8001/api/v1/career-guidance',
        json={'cv_content': cv},
        timeout=60
    )
    
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"✅ SUCCESS!")
        print(f"Jobs: {len(data.get('job_recommendations', []))}")
        print(f"Certs: {len(data.get('certification_recommendations', []))}")
        print(f"Phases: {len(data.get('learning_roadmap', {}).get('phases', []))}")
        
        # Show first job
        jobs = data.get('job_recommendations', [])
        if jobs:
            job = jobs[0]
            print(f"\nTop Job: {job['title']}")
            print(f"Similarity: {job['similarity_score']*100:.1f}%")
            print(f"Salary: ${job['predicted_salary']['min']:,} - ${job['predicted_salary']['max']:,}")
        
        with open('quick_result.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("\n💾 Saved to quick_result.json")
    else:
        print(f"❌ Error: {r.status_code}")
        print(r.text[:200])
        
except Exception as e:
    print(f"❌ Exception: {e}")
