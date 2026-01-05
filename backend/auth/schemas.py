"""Authentication schemas for request/response validation"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import re


class UserCreate(BaseModel):
    """Schema for user registration"""
    email: str = Field(..., min_length=5, max_length=255)
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = None

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()


class UserLogin(BaseModel):
    """Schema for user login - accepts email or username"""
    email: Optional[str] = None
    username: Optional[str] = None
    password: str


class UserResponse(BaseModel):
    """Schema for user information response"""
    id: str
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """Schema for access token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for token payload data"""
    user_id: Optional[str] = None
    username: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request"""
    refresh_token: str
