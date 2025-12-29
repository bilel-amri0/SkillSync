"""Test authentication system using FastAPI TestClient."""


def test_auth_flow(client):
    """Test complete authentication flow"""
    register_data = {
        "email": "test@skillsync.com",
        "username": "testuser",
        "password": "SecurePassword123!",
        "full_name": "Test User"
    }

    response = client.post("/api/v1/auth/register", json=register_data)
    assert response.status_code == 201
    user = response.json()
    assert user["email"] == register_data["email"]

    login_data = {
        "username": register_data["username"],
        "password": register_data["password"]
    }
    response = client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200
    tokens = response.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    me_response = client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 200
    assert me_response.json()["email"] == register_data["email"]

    refresh_response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]}
    )
    assert refresh_response.status_code == 200
    refreshed = refresh_response.json()
    assert refreshed["access_token"] != tokens["access_token"]

    logout_response = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": tokens["refresh_token"]},
        headers=headers
    )
    assert logout_response.status_code == 200
