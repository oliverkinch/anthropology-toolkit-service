"""Session management: each browser session gets an isolated tmp directory."""

import shutil
import time
import uuid
from pathlib import Path

from fastapi import Cookie, Response

SESSION_ROOT = Path("tmp")
SESSION_TTL = 60 * 60 * 24  # 24 hours


def get_or_create_session(response: Response, session_id: str | None = Cookie(default=None)) -> str:
    if session_id is None or not (SESSION_ROOT / session_id).exists():
        session_id = str(uuid.uuid4())
        response.set_cookie("session_id", session_id, httponly=True, samesite="lax")
    session_dir = SESSION_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    # touch mtime so we can clean up stale sessions
    (session_dir / ".touch").write_text(str(time.time()))
    return session_id


def session_dir(session_id: str) -> Path:
    path = SESSION_ROOT / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_stale_sessions() -> None:
    if not SESSION_ROOT.exists():
        return
    now = time.time()
    for d in SESSION_ROOT.iterdir():
        touch = d / ".touch"
        if touch.exists():
            age = now - float(touch.read_text())
            if age > SESSION_TTL:
                shutil.rmtree(d, ignore_errors=True)
