"""
Security and input validation utilities for SkillSync backend
"""
import re
from typing import Optional
from fastapi import HTTPException, UploadFile
import magic  # pip install python-magic-bin (Windows)


# File validation constants
ALLOWED_CV_MIME_TYPES = {
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
    'application/msword',  # .doc
    'text/plain'
}

MAX_CV_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CV_TEXT_LENGTH = 50000  # characters


def sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent path traversal attacks
    """
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove non-alphanumeric except .-_
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename


def validate_cv_file(file: UploadFile) -> None:
    """
    Validate uploaded CV file for security
    Raises HTTPException if validation fails
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Check file size
    if file.size and file.size > MAX_CV_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_CV_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Sanitize filename
    safe_filename = sanitize_filename(file.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Validate file extension
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    file_ext = '.' + safe_filename.rsplit('.', 1)[-1].lower() if '.' in safe_filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )


def validate_cv_text(text: str) -> str:
    """
    Validate and sanitize CV text content
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="CV content cannot be empty")
    
    text = text.strip()
    
    if len(text) > MAX_CV_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"CV text too long. Maximum {MAX_CV_TEXT_LENGTH} characters"
        )
    
    # Basic XSS prevention (remove potential script tags)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text


def sanitize_sql_like_pattern(pattern: str) -> str:
    """
    Escape SQL LIKE pattern special characters
    """
    pattern = pattern.replace('\\', '\\\\')
    pattern = pattern.replace('%', '\\%')
    pattern = pattern.replace('_', '\\_')
    return pattern


def validate_analysis_id(analysis_id: str) -> str:
    """
    Validate analysis ID format (UUID)
    """
    # UUID v4 format
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    
    if not re.match(uuid_pattern, analysis_id, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Invalid analysis ID format")
    
    return analysis_id.lower()
