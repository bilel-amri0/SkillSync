#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORS Testing Script
Tests CORS configuration for all critical endpoints
"""

import requests
import json
from typing import Dict, Any
import time

# Configuration
BACKEND_URL = "http://localhost:8001"
FRONTEND_ORIGIN = "http://localhost:5175"

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_result(success: bool, message: str):
    """Print test result"""
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")

def test_endpoint(
    name: str,
    method: str,
    endpoint: str,
    data: Dict[str, Any] = None,
    files: Dict[str, Any] = None
) -> bool:
    """Test a single endpoint with CORS headers"""
    print_header(f"Test: {name}")
    print(f"Method: {method}")
    print(f"URL: {BACKEND_URL}{endpoint}")
    print(f"Origin: {FRONTEND_ORIGIN}")
    
    headers = {
        "Origin": FRONTEND_ORIGIN,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "content-type"
    }
    
    try:
        # First, test OPTIONS (preflight) for POST/PUT/DELETE
        if method in ["POST", "PUT", "DELETE", "PATCH"]:
            print("\n🔍 Testing preflight (OPTIONS)...")
            options_response = requests.options(
                f"{BACKEND_URL}{endpoint}",
                headers=headers,
                timeout=10
            )
            
            print(f"   Status: {options_response.status_code}")
            
            # Check CORS headers in OPTIONS response
            cors_headers = {
                "Access-Control-Allow-Origin": options_response.headers.get("Access-Control-Allow-Origin"),
                "Access-Control-Allow-Methods": options_response.headers.get("Access-Control-Allow-Methods"),
                "Access-Control-Allow-Headers": options_response.headers.get("Access-Control-Allow-Headers"),
            }
            
            print(f"   CORS Headers:")
            for key, value in cors_headers.items():
                print(f"      {key}: {value}")
            
            if options_response.status_code not in [200, 204]:
                print_result(False, f"Preflight failed with status {options_response.status_code}")
                return False
            
            if not cors_headers["Access-Control-Allow-Origin"]:
                print_result(False, "Missing Access-Control-Allow-Origin in preflight response")
                return False
            
            print_result(True, "Preflight passed")
        
        # Now test the actual request
        print(f"\n🔍 Testing actual {method} request...")
        
        request_headers = {"Origin": FRONTEND_ORIGIN}
        
        if method == "GET":
            response = requests.get(
                f"{BACKEND_URL}{endpoint}",
                headers=request_headers,
                timeout=30
            )
        elif method == "POST":
            if files:
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    headers=request_headers,
                    files=files,
                    timeout=30
                )
            else:
                request_headers["Content-Type"] = "application/json"
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    headers=request_headers,
                    json=data,
                    timeout=30
                )
        else:
            print_result(False, f"Method {method} not implemented in test script")
            return False
        
        print(f"   Status: {response.status_code}")
        
        # Check CORS headers in actual response
        cors_origin = response.headers.get("Access-Control-Allow-Origin")
        print(f"   CORS Origin: {cors_origin}")
        
        if response.status_code == 200:
            # Try to parse response
            try:
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
            except:
                print(f"   Response: {response.text[:200]}...")
            
            if cors_origin:
                print_result(True, f"{name} passed with CORS headers")
                return True
            else:
                print_result(False, f"{name} succeeded but missing CORS headers")
                return False
        else:
            print_result(False, f"{name} failed with status {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print_result(False, f"{name} timed out")
        return False
    except requests.exceptions.ConnectionError:
        print_result(False, f"{name} - Cannot connect to backend (is it running?)")
        return False
    except Exception as e:
        print_result(False, f"{name} - {str(e)}")
        return False

def main():
    """Run all CORS tests"""
    print("\n🚀 Starting CORS Tests for SkillSync Backend")
    print(f"Backend: {BACKEND_URL}")
    print(f"Frontend Origin: {FRONTEND_ORIGIN}")
    
    results = {}
    
    # Test 1: Health Check (Simple GET)
    results["health"] = test_endpoint(
        name="Health Check",
        method="GET",
        endpoint="/health"
    )
    time.sleep(1)
    
    # Test 2: Analytics Dashboard (GET)
    results["analytics"] = test_endpoint(
        name="Analytics Dashboard",
        method="GET",
        endpoint="/api/v1/analytics/dashboard"
    )
    time.sleep(1)
    
    # Test 3: Career Guidance (POST with preflight)
    results["career_guidance"] = test_endpoint(
        name="Career Guidance",
        method="POST",
        endpoint="/api/v1/career-guidance",
        data={
            "cv_content": "Senior Software Engineer with 5 years of experience in Python, JavaScript, React, and Node.js. Expert in building scalable web applications."
        }
    )
    time.sleep(1)
    
    # Test 4: CV Analysis (POST)
    results["cv_analysis"] = test_endpoint(
        name="CV Analysis",
        method="POST",
        endpoint="/api/v1/analyze-cv",
        data={
            "cv_text": "Python Developer with 3 years experience in Django and FastAPI"
        }
    )
    time.sleep(1)
    
    # Test 5: Extract Text (POST with file)
    results["extract_text"] = test_endpoint(
        name="Extract Text",
        method="POST",
        endpoint="/api/v1/extract-text",
        files={
            "file": ("test.txt", "Test CV content for extraction", "text/plain")
        }
    )
    
    # Summary
    print_header("Test Summary")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"\nSuccess Rate: {(passed/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, success in results.items():
        icon = "✅" if success else "❌"
        print(f"  {icon} {test_name.replace('_', ' ').title()}")
    
    if failed == 0:
        print("\n🎉 All CORS tests passed! Your application is ready to use.")
        print("\nNext steps:")
        print("1. Open http://localhost:5175 in your browser")
        print("2. Click '🤖 ML Career Guidance'")
        print("3. Upload a CV and test the complete flow")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("1. Make sure backend is running: python backend/main_simple_for_frontend.py")
        print("2. Check that port 5175 is in ALLOWED_ORIGINS")
        print("3. Restart backend after CORS changes")
        print("4. Hard refresh browser (Ctrl+Shift+R)")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
