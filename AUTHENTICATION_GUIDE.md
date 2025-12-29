# 🔐 Authentication Guide

## Overview

SkillSync now includes a complete JWT-based authentication system with:
- ✅ User registration and login
- ✅ Access tokens (30 min expiry)
- ✅ Refresh tokens (7 days expiry)
- ✅ Password hashing with bcrypt
- ✅ Protected endpoints
- ✅ Token refresh mechanism

---

## Quick Start

### 1. Environment Setup

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Generate secure secret keys:
```bash
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('REFRESH_SECRET_KEY=' + secrets.token_urlsafe(32))"
```

Add these to your `.env` file.

### 2. Initialize Database

```bash
python -c "from database import init_db; init_db()"
```

### 3. Start Server

```bash
python main_simple_for_frontend.py
```

---

## API Endpoints

### Public Endpoints (No Authentication Required)

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "username",
  "password": "SecurePassword123!",
  "full_name": "John Doe"
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "username": "username",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2025-11-23T..."
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "username",
  "password": "SecurePassword123!"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200):**
```json
{
  "access_token": "new_access_token...",
  "refresh_token": "new_refresh_token...",
  "token_type": "bearer"
}
```

### Protected Endpoints (Authentication Required)

Add the access token to the `Authorization` header:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Get Current User
```http
GET /api/v1/auth/me
Authorization: Bearer <access_token>
```

**Response (200):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "username": "username",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2025-11-23T..."
}
```

#### Logout
```http
POST /api/v1/auth/logout
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200):**
```json
{
  "message": "Successfully logged out"
}
```

---

## Frontend Integration

### React/TypeScript Example

```typescript
// auth.service.ts
const API_URL = 'http://localhost:8000/api/v1/auth';

export const authService = {
  async register(email: string, username: string, password: string, fullName?: string) {
    const response = await fetch(`${API_URL}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, username, password, full_name: fullName })
    });
    return response.json();
  },

  async login(username: string, password: string) {
    const response = await fetch(`${API_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await response.json();
    
    // Store tokens
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    
    return data;
  },

  async getCurrentUser() {
    const token = localStorage.getItem('access_token');
    const response = await fetch(`${API_URL}/me`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    return response.json();
  },

  async refreshToken() {
    const refreshToken = localStorage.getItem('refresh_token');
    const response = await fetch(`${API_URL}/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken })
    });
    const data = await response.json();
    
    // Update tokens
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    
    return data;
  },

  async logout() {
    const token = localStorage.getItem('access_token');
    const refreshToken = localStorage.getItem('refresh_token');
    
    await fetch(`${API_URL}/logout`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ refresh_token: refreshToken })
    });
    
    // Clear tokens
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }
};
```

### Axios Interceptor (Automatic Token Refresh)

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1'
});

// Request interceptor - add token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handle token expiry
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If 401 and not already retried, refresh token
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        const response = await axios.post('/api/v1/auth/refresh', {
          refresh_token: refreshToken
        });

        const { access_token, refresh_token: newRefreshToken } = response.data;
        
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('refresh_token', newRefreshToken);

        // Retry original request with new token
        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return api(originalRequest);
      } catch (refreshError) {
        // Refresh failed - redirect to login
        localStorage.clear();
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

export default api;
```

---

## Testing

### Manual Testing

Run the test script:
```bash
# Make sure server is running first
python tests/test_auth.py
```

### Using cURL

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"testuser","password":"Test123!","full_name":"Test User"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"Test123!"}'

# Get user info (replace TOKEN with actual token from login)
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer TOKEN"
```

### API Documentation

Interactive API docs with authentication:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

Click "Authorize" button and enter your access token.

---

## Security Best Practices

### Production Configuration

1. **Strong Secret Keys**
   ```bash
   # Generate 32+ character random strings
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Environment Variables**
   - Never commit `.env` to git
   - Use different keys for dev/staging/prod
   - Rotate keys regularly

3. **HTTPS Only**
   - Always use HTTPS in production
   - Set secure cookie flags
   - Enable HSTS headers

4. **Token Expiry**
   - Access tokens: 15-30 minutes
   - Refresh tokens: 7-30 days
   - Implement token rotation

5. **Password Requirements**
   - Minimum 8 characters
   - Require uppercase, lowercase, numbers, symbols
   - Implement password strength meter

### Rate Limiting

Current limits:
- 100 requests/minute per IP (general)
- Consider stricter limits for auth endpoints:
  - Register: 5/hour
  - Login: 10/minute
  - Refresh: 20/minute

---

## Database Schema

### users
```sql
id               VARCHAR (UUID)    PRIMARY KEY
email            VARCHAR           UNIQUE, NOT NULL
username         VARCHAR           UNIQUE, NOT NULL
hashed_password  VARCHAR           NOT NULL
full_name        VARCHAR           NULL
is_active        BOOLEAN           DEFAULT TRUE
is_superuser     BOOLEAN           DEFAULT FALSE
created_at       TIMESTAMP         DEFAULT NOW()
updated_at       TIMESTAMP         ON UPDATE NOW()
```

### refresh_tokens
```sql
id          VARCHAR (UUID)    PRIMARY KEY
user_id     VARCHAR           FOREIGN KEY (users.id)
token       VARCHAR           UNIQUE, NOT NULL
expires_at  TIMESTAMP         NOT NULL
created_at  TIMESTAMP         DEFAULT NOW()
is_revoked  BOOLEAN           DEFAULT FALSE
```

---

## Troubleshooting

### "Could not validate credentials"
- Check token format: `Bearer <token>`
- Verify token not expired
- Ensure SECRET_KEY matches server

### "User already exists"
- Email or username taken
- Use unique credentials

### "Refresh token expired/revoked"
- Re-login to get new tokens
- Tokens auto-revoke on logout

### Import errors
```bash
pip install passlib[bcrypt] python-jose[cryptography] bcrypt
```

---

## Next Steps

- [ ] Add email verification
- [ ] Implement password reset
- [ ] Add OAuth2 (Google, GitHub)
- [ ] Implement role-based access control (RBAC)
- [ ] Add 2FA/MFA support
- [ ] Session management (track active sessions)

---

**Authentication is now production-ready!** 🔐
