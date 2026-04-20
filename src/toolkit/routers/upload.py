"""File upload endpoints."""

import shutil
from pathlib import Path

from fastapi import APIRouter, Cookie, HTTPException, Response, UploadFile
from pydantic import BaseModel

from toolkit.services.file_io import extract_text
from toolkit.session import get_or_create_session, session_dir

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Files with these suffixes are stored as originals; each gets an extracted .txt cache.
SOURCE_SUFFIXES = {".pdf", ".docx", ".doc"}
ALLOWED_SUFFIXES = SOURCE_SUFFIXES | {".txt"}


class UploadedFile(BaseModel):
    filename: str
    role: str  # transcript | guide | literature
    word_count: int


@router.post("/{role}", response_model=UploadedFile)
async def upload_file(
    role: str,
    file: UploadFile,
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> UploadedFile:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    role_dir = sdir / role
    role_dir.mkdir(exist_ok=True)
    dest = role_dir / (file.filename or "upload")
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text(dest)
    word_count = len(text.split())

    # Cache extracted text alongside the original
    (role_dir / (dest.stem + ".txt")).write_text(text, encoding="utf-8")

    return UploadedFile(filename=file.filename or "upload", role=role, word_count=word_count)


@router.get("", response_model=list[UploadedFile])
async def list_uploads(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> list[UploadedFile]:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)
    uploads = []
    for role in ("transcript", "guide", "literature"):
        role_dir = sdir / role
        if not role_dir.exists():
            continue
        for f in sorted(role_dir.iterdir()):
            suffix = f.suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                continue
            # Skip .txt files that are extraction caches for a non-txt source
            if suffix == ".txt" and any((role_dir / (f.stem + s)).exists() for s in SOURCE_SUFFIXES):
                continue
            txt_cache = role_dir / (f.stem + ".txt")
            wc = len(txt_cache.read_text(encoding="utf-8").split()) if txt_cache.exists() else 0
            uploads.append(UploadedFile(filename=f.name, role=role, word_count=wc))
    return uploads


@router.delete("/{role}/{filename}")
async def delete_upload(
    role: str,
    filename: str,
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> dict:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid) / role
    stem = Path(filename).stem
    for f in sdir.glob(f"{stem}*"):
        f.unlink(missing_ok=True)
    return {"ok": True}
