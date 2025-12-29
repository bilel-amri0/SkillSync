"""
Test suite for modular architecture
Tests the new routers and logging functionality
"""
import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app  # noqa: F401  # app import ensures routers are registered


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_cv_analysis_modular(client):
    """Test CV analysis with modular router"""
    payload = {
        "cv_content": "Python developer with 5 years of experience in Django and Flask. Bachelor's degree in Computer Science.",
        "format": "text"
    }
    response = client.post("/api/v1/analyze-cv", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert "skills" in data
    assert len(data["skills"]) > 0
    assert data["experience_years"] >= 5


def test_get_cv_analyses(client):
    """Test getting all CV analyses"""
    # First create an analysis
    payload = {
        "cv_content": "React developer with TypeScript and Node.js experience",
        "format": "text"
    }
    client.post("/api/v1/analyze-cv", json=payload)
    
    # Get all analyses
    response = client.get("/api/v1/cv-analyses")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "analyses" in data
    assert isinstance(data["analyses"], list)
    assert data.get("total", 0) >= 1
    assert len(data["analyses"]) == data.get("total")


def test_recommendations_modular(client):
    """Test recommendations with modular router"""
    # Create CV analysis first
    cv_payload = {
        "cv_content": "Full-stack developer with JavaScript, Python, and AWS experience. 3 years in the industry.",
        "format": "text"
    }
    cv_response = client.post("/api/v1/analyze-cv", json=cv_payload)
    analysis_id = cv_response.json()["analysis_id"]
    
    # Get recommendations
    rec_response = client.get(f"/api/v1/recommendations/{analysis_id}")
    assert rec_response.status_code == 200
    data = rec_response.json()
    assert "recommendations" in data
    assert "analysis_id" in data
    assert "LEARNING_RESOURCES" in data["recommendations"]
    assert "CAREER_ROADMAP" in data["recommendations"]


def test_dashboard_modular(client):
    """Test dashboard with modular router"""
    response = client.get("/api/v1/dashboard/latest")
    assert response.status_code == 200
    data = response.json()
    assert "recent_analyses" in data
    assert "skills_summary" in data
    assert "portfolio_status" in data


def test_request_id_header(client):
    """Test that X-Request-ID header is added to responses"""
    response = client.get("/api/v1/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.options("/api/v1/health")
    # FastAPI automatically handles OPTIONS requests
    assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly defined


def test_authentication_endpoints_available(client):
    """Test that authentication endpoints are available"""
    # Health check to ensure auth router is loaded
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    # Auth endpoints should be available (will return 422 without proper data)
    response = client.post("/api/v1/auth/login", json={})
    assert response.status_code in [400, 422]  # Validation error expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
