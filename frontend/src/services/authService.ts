/**
 * Authentication Service - Handles all auth-related API calls
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8001';
const AUTH_API = `${API_BASE_URL}/api/v1/auth`;

// Token storage keys
const ACCESS_TOKEN_KEY = 'accessToken';
const REFRESH_TOKEN_KEY = 'refreshToken';
const USER_KEY = 'user';

// Types
export interface User {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface LoginCredentials {
  email?: string;
  username?: string;
  password: string;
}

export interface RegisterData {
  email: string;
  username: string;
  password: string;
  full_name?: string;
}

export interface AuthResponse {
  success: boolean;
  user?: User;
  tokens?: AuthTokens;
  error?: string;
}

// Token Management
export const getAccessToken = (): string | null => {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
};

export const getRefreshToken = (): string | null => {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
};

export const getStoredUser = (): User | null => {
  const userStr = localStorage.getItem(USER_KEY);
  if (userStr) {
    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }
  return null;
};

export const setTokens = (tokens: AuthTokens): void => {
  localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
  localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
};

export const setUser = (user: User): void => {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
};

export const clearAuthData = (): void => {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem('authToken'); // Legacy token
};

export const isAuthenticated = (): boolean => {
  const token = getAccessToken();
  return token !== null && token !== '';
};

// API Calls
export const authService = {
  /**
   * Register a new user
   */
  register: async (data: RegisterData): Promise<AuthResponse> => {
    try {
      console.log('Registering user with data:', { ...data, password: '***' });
      
      const response = await fetch(`${AUTH_API}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const responseData = await response.json();
      console.log('Registration response:', response.status, responseData);

      if (!response.ok) {
        // Handle validation errors (422)
        let errorMessage = 'Registration failed';
        if (responseData.detail) {
          if (Array.isArray(responseData.detail)) {
            // Pydantic validation errors
            errorMessage = responseData.detail.map((err: { msg?: string; loc?: string[] }) => 
              err.msg || JSON.stringify(err)
            ).join(', ');
          } else if (typeof responseData.detail === 'string') {
            errorMessage = responseData.detail;
          }
        }
        return {
          success: false,
          error: errorMessage,
        };
      }

      // Registration successful - now login automatically
      const loginResult = await authService.login({
        email: data.email,
        password: data.password,
      });

      return loginResult;
    } catch (error) {
      console.error('Registration error:', error);
      return {
        success: false,
        error: 'Network error. Please check your connection.',
      };
    }
  },

  /**
   * Login user with email or username
   */
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    try {
      const response = await fetch(`${AUTH_API}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const responseData = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: responseData.detail || 'Invalid credentials',
        };
      }

      // Store tokens
      const tokens: AuthTokens = responseData;
      setTokens(tokens);

      // Get user info
      const userResponse = await authService.getCurrentUser();
      if (userResponse.success && userResponse.user) {
        setUser(userResponse.user);
        // Also set legacy token for backward compatibility
        localStorage.setItem('authToken', tokens.access_token);
        return {
          success: true,
          user: userResponse.user,
          tokens,
        };
      }

      return {
        success: true,
        tokens,
      };
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        error: 'Network error. Please check your connection.',
      };
    }
  },

  /**
   * Get current authenticated user
   */
  getCurrentUser: async (): Promise<AuthResponse> => {
    try {
      const token = getAccessToken();
      if (!token) {
        return { success: false, error: 'No access token' };
      }

      const response = await fetch(`${AUTH_API}/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          // Token expired, try to refresh
          const refreshResult = await authService.refreshToken();
          if (refreshResult.success) {
            // Retry with new token
            return authService.getCurrentUser();
          }
          clearAuthData();
          return { success: false, error: 'Session expired. Please login again.' };
        }
        return { success: false, error: 'Failed to get user info' };
      }

      const user: User = await response.json();
      setUser(user);
      return { success: true, user };
    } catch (error) {
      console.error('Get user error:', error);
      return { success: false, error: 'Network error' };
    }
  },

  /**
   * Refresh access token using refresh token
   */
  refreshToken: async (): Promise<AuthResponse> => {
    try {
      const refreshToken = getRefreshToken();
      if (!refreshToken) {
        return { success: false, error: 'No refresh token' };
      }

      const response = await fetch(`${AUTH_API}/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!response.ok) {
        clearAuthData();
        return { success: false, error: 'Failed to refresh token' };
      }

      const tokens: AuthTokens = await response.json();
      setTokens(tokens);
      localStorage.setItem('authToken', tokens.access_token); // Legacy support
      return { success: true, tokens };
    } catch (error) {
      console.error('Refresh token error:', error);
      clearAuthData();
      return { success: false, error: 'Network error' };
    }
  },

  /**
   * Logout user and revoke refresh token
   */
  logout: async (): Promise<AuthResponse> => {
    try {
      const accessToken = getAccessToken();
      const refreshToken = getRefreshToken();

      if (accessToken && refreshToken) {
        await fetch(`${AUTH_API}/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
      }
    } catch (error) {
      console.error('Logout API error:', error);
      // Continue with local logout even if API fails
    }

    // Always clear local auth data
    clearAuthData();
    return { success: true };
  },

  /**
   * Check if user session is valid
   */
  validateSession: async (): Promise<boolean> => {
    const token = getAccessToken();
    if (!token) {
      return false;
    }

    const result = await authService.getCurrentUser();
    return result.success;
  },
};

export default authService;
