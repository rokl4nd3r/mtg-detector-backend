from __future__ import annotations

import secrets
from fastapi import Header, HTTPException, status

from .config import Settings


def require_bearer_token(
    settings: Settings,
    authorization: str | None = Header(default=None),
) -> None:
    if not settings.bearer_tokens:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is missing MTG_BEARER_TOKENS configuration",
        )

    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format (expected Bearer)",
        )

    provided = parts[1].strip()
    ok = any(secrets.compare_digest(provided, t) for t in settings.bearer_tokens)
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
