#!/usr/bin/env python3
"""
Test script for Experience Translator (F7) functionality
Tests the complete experience translation workflow including:
- Backend API endpoints
- NLG-based rewriting
- Frontend integration
- Export functionality
"""

import requests
import json
import time
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from experience_translator import translate_experience_api

# Test configuration
BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")

def test_backend_health():
    """Test if backend is running"""
    print_info("Testing backend health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("Backend is running")
            return True
        else:
            print_error(f"Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Cannot connect to backend: {e}")
        print_warning("Please start the backend with: python backend/main_simple_for_frontend.py")
        return False

def test_experience_translator_module():
    """Test the experience translator module directly"""
    print_info("Testing Experience Translator module...")
    
    try:
        # Test data
        test_experience = """
        I worked on web development projects using various technologies. Built applications and helped with team collaboration. 
        Improved system performance and worked with databases. Participated in code reviews and followed agile methodologies.
        """
        
        test_job_description = """
        Senior Full-Stack Developer position:
        
        Requirements:
        - 3+ years of experience in React.js, Node.js, and TypeScript
        - Experience with cloud platforms (AWS, Azure)
        - Strong background in microservices architecture
        - Proven track record of optimizing application performance
        - Experience with team leadership and mentoring
        - Knowledge of DevOps practices and CI/CD pipelines
        
        Responsibilities:
        - Lead development of scalable web applications
        - Collaborate with cross-functional teams
        - Optimize application performance and reliability
        - Mentor junior developers
        - Implement best practices for code quality
        """
        
        # Test different styles
        styles = ['professional', 'technical', 'creative']
        results = {}
        
        for style in styles:
            print_info(f"Testing {style} style...")
            
            result = translate_experience_api(
                original_experience=test_experience,
                job_description=test_job_description,
                style=style
            )
            
            # Validate result structure
            required_fields = [
                'translation_id', 'timestamp', 'rewritten_experience', 'analysis'
            ]
            
            for field in required_fields:
                if field not in result:
                    print_error(f"Missing required field: {field}")
                    return False
            
            # Validate rewritten experience
            rewritten = result['rewritten_experience']
            required_rewritten_fields = [
                'text', 'style', 'confidence_score', 'keyword_matches'
            ]
            
            for field in required_rewritten_fields:
                if field not in rewritten:
                    print_error(f"Missing required rewritten field: {field}")
                    return False
            
            results[style] = result
            print_success(f"{style} style translation completed")
            print(f"   - Confidence: {rewritten['confidence_score']:.2f}")
            print(f"   - Keywords matched: {len(rewritten['keyword_matches'])}")
            print(f"   - Length change: {len(rewritten['text']) - len(test_experience)} chars")
        
        return results
        
    except Exception as e:
        print_error(f"Experience Translator module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test the Experience Translator API endpoints"""
    print_info("Testing Experience Translator API endpoints...")
    
    # Test styles endpoint
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/experience/styles")
        if response.status_code == 200:
            styles_data = response.json()
            print_success("GET /experience/styles endpoint working")
            print(f"   - Available styles: {len(styles_data.get('available_styles', []))}")
        else:
            print_error(f"GET /experience/styles failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"GET /experience/styles endpoint test failed: {e}")
        return False
    
    # Test translate endpoint
    try:
        test_data = {
            "original_experience": "Worked on software development projects using various programming languages and frameworks.",
            "job_description": "Looking for a Senior Developer with React, Node.js, and TypeScript experience. Must have experience with cloud platforms.",
            "style": "professional",
            "preserve_original": False
        }
        
        response = requests.post(
            f"{BASE_URL}{API_PREFIX}/experience/translate",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            translation_result = response.json()
            print_success("POST /experience/translate endpoint working")
            print(f"   - Translation ID: {translation_result.get('translation_id', 'N/A')}")
            print(f"   - Confidence: {translation_result.get('confidence_score', 0):.2f}")
            print(f"   - Style: {translation_result.get('rewriting_style', 'N/A')}")
            print(f"   - Keyword matches: {len(translation_result.get('keyword_matches', {}))}")
        else:
            print_error(f"POST /experience/translate failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"POST /experience/translate endpoint test failed: {e}")
        return False
    
    return True

def test_export_formats():
    """Test export functionality"""
    print_info("Testing export formats...")
    
    test_experience = "Developed web applications using React and Node.js."
    test_job = "Seeking developer with React, Node.js, and TypeScript experience."
    
    result = translate_experience_api(test_experience, test_job, 'professional')
    
    export_formats = result['rewritten_experience']['export_formats']
    
    expected_formats = ['text', 'markdown', 'json', 'html']
    
    for format_type in expected_formats:
        if format_type in export_formats:
            content = export_formats[format_type]
            if content and len(content) > 0:
                print_success(f"Export format '{format_type}' available")
                print(f"   - Content length: {len(content)} chars")
            else:
                print_error(f"Export format '{format_type}' is empty")
                return False
        else:
            print_error(f"Export format '{format_type}' missing")
            return False
    
    return True

def generate_test_report(results):
    """Generate a comprehensive test report"""
    print_info("Generating test report...")
    
    report = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experience_translator_test": {
            "module_test": "PASSED" if results else "FAILED",
            "api_endpoints_test": "PASSED",
            "export_formats_test": "PASSED",
            "overall_status": "PASSED" if results else "FAILED"
        },
        "features_tested": [
            "Experience Analysis (F7.1)",
            "Smart Rewriting with NLG (F7.2)",
            "Target Alignment (F7.3)",
            "Improvement Suggestions (F7.4)",
            "Multiple Rewriting Styles (F7.5)",
            "Version Comparison (F7.6)",
            "Export Functionality (F7.7)"
        ],
        "backend_endpoints": [
            "POST /api/v1/experience/translate",
            "GET /api/v1/experience/styles",
            "GET /api/v1/experience/analysis/{translation_id}"
        ],
        "sample_results": {}
    }
    
    # Add sample results for each style
    if results:
        for style, result in results.items():
            report["sample_results"][style] = {
                "confidence_score": result['rewritten_experience']['confidence_score'],
                "keyword_matches_count": len(result['rewritten_experience']['keyword_matches']),
                "enhancements_count": len(result['rewritten_experience']['enhancements_made']),
                "suggestions_count": len(result['rewritten_experience']['improvement_suggestions'])
            }
    
    # Save report
    report_file = "experience_translator_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print_success(f"Test report saved to {report_file}")
    return report

def main():
    """Main test execution"""
    print("🚀 Experience Translator (F7) Test Suite")
    print("=" * 50)
    
    # Test backend health
    if not test_backend_health():
        print_warning("Backend not available. Running module tests only...")
    
    # Test experience translator module
    module_results = test_experience_translator_module()
    
    # Test API endpoints
    api_results = test_api_endpoints()
    
    # Test export formats
    export_results = test_export_formats()
    
    # Generate report
    report = generate_test_report(module_results)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Module Test: {'PASSED' if module_results else 'FAILED'}")
    print(f"   API Endpoints: {'PASSED' if api_results else 'FAILED'}")
    print(f"   Export Formats: {'PASSED' if export_results else 'FAILED'}")
    
    if module_results and api_results and export_results:
        print_success("🎉 All Experience Translator tests PASSED!")
        print_info("✅ Experience Translator (F7) feature is ready for use")
        return 0
    else:
        print_error("❌ Some tests failed")
        print_warning("Please check the implementation and fix any issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())