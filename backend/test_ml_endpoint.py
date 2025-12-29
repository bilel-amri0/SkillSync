import requests  
import json  
  
cv_text = """John Doe >> test_ml_endpoint.py && echo Software Engineer >> test_ml_endpoint.py && echo john@example.com >> test_ml_endpoint.py && echo Python, React, AWS"""  
  
try:  
    response = requests.post('http://localhost:8001/api/v1/analyze-cv-advanced', json={'cv_content': cv_text})  
    print(f'Status: {response.status_code}')  
    if response.status_code == 200:  
        print(json.dumps(response.json(), indent=2))  
    else:  
        print(f'Error: {response.text}')  
except Exception as e:  
    print(f'Exception: {e}') 
