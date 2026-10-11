"""
Phase 2 — Authentication Router
=================================
Endpoints:
  POST /api/auth/register  — Create a new doctor/clinic account
  POST /api/auth/login     — Authenticate and receive JWT access token
  GET  /api/auth/me        — Return current authenticated user profile
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime
import uuid

from app.database import create_user, get_user_by_email
from app.services.auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter()


# ── Request / Response Models ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name:    str
    email:        str
    password:     str
    clinic_name:  Optional[str] = ""
    role:         Optional[str] = "Doctor"   # Doctor | Radiologist | Admin

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    full_name:    str
    role:         str


class UserProfile(BaseModel):
    id:           str
    full_name:    str
    email:        str
    clinic_name:  str
    role:         str
    created_at:   str


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """Create a new doctor/clinic account and return a JWT token."""
    # Check if email already registered
    existing = await get_user_by_email(req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists."
        )

    user_id = str(uuid.uuid4())
    user_data = {
        "id":           user_id,
        "full_name":    req.full_name,
        "email":        req.email.lower().strip(),
        "hashed_password": hash_password(req.password),
        "clinic_name":  req.clinic_name or "",
        "role":         req.role,
        "created_at":   datetime.now().isoformat()
    }
    await create_user(user_data)

    token = create_access_token({"sub": user_id, "email": req.email, "role": req.role})
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        full_name=req.full_name,
        role=req.role
    )


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate with email + password (OAuth2PasswordRequestForm uses 'username' field for email).
    Returns a JWT Bearer token.
    """
    user = await get_user_by_email(form_data.username.lower().strip())
    if not user or not verify_password(form_data.password, user.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({
        "sub":   user["id"],
        "email": user["email"],
        "role":  user.get("role", "Doctor")
    })
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        full_name=user["full_name"],
        role=user.get("role", "Doctor")
    )


@router.get("/me", response_model=UserProfile)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserProfile(
        id=current_user["id"],
        full_name=current_user["full_name"],
        email=current_user["email"],
        clinic_name=current_user.get("clinic_name", ""),
        role=current_user.get("role", "Doctor"),
        created_at=current_user.get("created_at", "")
    )
