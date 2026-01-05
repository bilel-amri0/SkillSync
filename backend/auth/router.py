"""Authentication router - endpoints for login, register, refresh"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from auth import schemas, utils, dependencies
from models import User, RefreshToken
from datetime import datetime

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()
    
    if existing_user:
        if existing_user.email == user_data.email:
            raise HTTPException(status_code=400, detail="Email already registered")
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user
    hashed_password = utils.get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        name=user_data.full_name,  # Backward compatibility
        is_active=True,
        is_superuser=False
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=schemas.Token)
async def login(credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    """Login and get access token - accepts email or username"""
    # Validate that at least one identifier is provided
    if not credentials.email and not credentials.username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide email or username"
        )
    
    # Find user by email or username
    if credentials.email:
        user = db.query(User).filter(User.email == credentials.email).first()
    else:
        user = db.query(User).filter(User.username == credentials.username).first()
    
    if not user or not utils.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Create tokens
    access_token = utils.create_access_token(data={"sub": user.id, "username": user.username})
    refresh_token, refresh_expires = utils.create_refresh_token(data={"sub": user.id})
    
    # Store refresh token in database
    db_refresh_token = RefreshToken(
        user_id=user.id,
        token=refresh_token,
        expires_at=refresh_expires,
        is_revoked=False
    )
    db.add(db_refresh_token)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=schemas.Token)
async def refresh_token(request: schemas.RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh access token using refresh token"""
    # Verify refresh token
    payload = utils.verify_token(request.refresh_token, token_type="refresh")
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Check if refresh token exists and is not revoked
    db_refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == request.refresh_token,
        RefreshToken.is_revoked == False
    ).first()
    
    if not db_refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found or revoked"
        )
    
    # Check if token expired
    if db_refresh_token.expires_at < datetime.utcnow():
        db_refresh_token.is_revoked = True
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired"
        )
    
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    # Create new tokens
    access_token = utils.create_access_token(data={"sub": user.id, "username": user.username})
    new_refresh_token, refresh_expires = utils.create_refresh_token(data={"sub": user.id})
    
    # Revoke old refresh token
    db_refresh_token.is_revoked = True
    
    # Store new refresh token
    new_db_refresh_token = RefreshToken(
        user_id=user.id,
        token=new_refresh_token,
        expires_at=refresh_expires,
        is_revoked=False
    )
    db.add(new_db_refresh_token)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.post("/logout")
async def logout(
    request: schemas.RefreshTokenRequest,
    current_user: User = Depends(dependencies.get_current_user),
    db: Session = Depends(get_db)
):
    """Logout and revoke refresh token"""
    # Revoke refresh token
    db_refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token == request.refresh_token,
        RefreshToken.user_id == current_user.id
    ).first()
    
    if db_refresh_token:
        db_refresh_token.is_revoked = True
        db.commit()
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=schemas.UserResponse)
async def get_me(current_user: User = Depends(dependencies.get_current_active_user)):
    """Get current user information"""
    return current_user
