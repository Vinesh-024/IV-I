"""
Phase 2 — JWT Authentication Service
======================================
Provides password hashing (bcrypt), JWT token creation/verification,
and a FastAPI dependency to get the current authenticated user.

Install dependencies:
    pip install python-jose[cryptography] passlib[bcrypt]
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("app.auth")

# ── Optional dependency guards ──────────────────────────────────────────────
JWT_AVAILABLE = False
try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    logger.warning("python-jose not installed — JWT auth disabled. pip install python-jose[cryptography]")

BCRYPT_AVAILABLE = False
try:
    from passlib.context import CryptContext
    _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    BCRYPT_AVAILABLE = True
except ImportError:
    logger.warning("passlib not installed — bcrypt hashing disabled. pip install passlib[bcrypt]")

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# ── JWT Configuration ───────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "iv-i-medical-ai-super-secret-key-change-in-production")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ── Password utilities ───────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    if BCRYPT_AVAILABLE:
        return _pwd_context.hash(plain_password)
    # Fallback: weak hash (dev/demo only — never production)
    import hashlib
    return "sha256:" + hashlib.sha256(plain_password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if BCRYPT_AVAILABLE and not hashed_password.startswith("sha256:"):
        return _pwd_context.verify(plain_password, hashed_password)
    # Fallback verification
    import hashlib
    return hashed_password == "sha256:" + hashlib.sha256(plain_password.encode()).hexdigest()


# ── Token utilities ──────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    if not JWT_AVAILABLE:
        # Return a simple base64-encoded payload as a pseudo-token for dev mode
        import base64, json
        payload = {**data, "exp": (datetime.utcnow() + timedelta(hours=8)).isoformat()}
        return "dev." + base64.b64encode(json.dumps(payload).encode()).decode()

    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    if not JWT_AVAILABLE:
        if token.startswith("dev."):
            import base64, json
            try:
                return json.loads(base64.b64decode(token[4:]).decode())
            except Exception:
                return None
        return None
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── FastAPI dependency ───────────────────────────────────────────────────────

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    """
    Validates the Bearer JWT token from the Authorization header.
    Returns the user dict from the DB or raises 401.
    Inject this as a FastAPI dependency on protected routes:
        @router.get("/protected")
        async def protected(user = Depends(get_current_user)):
            ...
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_exception

    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    # Import here to avoid circular imports
    from app.database import get_user_by_id
    user = await get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user


async def get_optional_user(token: Optional[str] = Depends(oauth2_scheme)):
    """
    Like get_current_user but does NOT raise on missing token.
    Returns None if unauthenticated. Use for semi-protected endpoints.
    """
    if not token:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None
