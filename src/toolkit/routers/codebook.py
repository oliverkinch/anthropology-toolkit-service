"""Codebook management endpoints."""

import json
import shutil

from fastapi import APIRouter, Cookie, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from toolkit.config import settings
from toolkit.routers._sse import parse_sse
from toolkit.services.codebook import (
    build_codebook,
    extract_codes_from_guide,
    load_codebook_csv,
    parse_custom_codes,
    save_codebook,
    save_codebook_atlas,
    save_codebook_markdown,
    save_codebook_nvivo,
)
from toolkit.session import get_or_create_session, session_dir

router = APIRouter(prefix="/api/codebook", tags=["codebook"])


class CustomCodesRequest(BaseModel):
    text: str  # one "LABEL: definition" per line


def _persist_all_formats(codebook: dict, sdir) -> None:
    save_codebook(codebook, sdir)
    save_codebook_markdown(codebook, sdir)
    save_codebook_atlas(codebook, sdir)
    save_codebook_nvivo(codebook, sdir)


@router.post("/build")
async def build_from_literature(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> StreamingResponse:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)

    lit_dir = sdir / "literature"
    if not lit_dir.exists() or not any(lit_dir.glob("*.txt")):
        raise HTTPException(400, "No literature uploaded")

    documents = {f.stem: f.read_text(encoding="utf-8") for f in sorted(lit_dir.glob("*.txt"))}

    async def stream():
        client = AsyncOpenAI(base_url=settings.inference_base_url, api_key=settings.inference_api_key)
        async for event in build_codebook(documents, client, settings.default_model):
            yield event
            if (payload := parse_sse(event)) and payload.get("type") == "done":
                codebook = payload.get("codebook", {})
                (sdir / "codebook.json").write_text(json.dumps(codebook))
                _persist_all_formats(codebook, sdir)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/from-guide")
async def build_from_guide(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> StreamingResponse:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)

    guide_dir = sdir / "guide"
    guide_texts = list(guide_dir.glob("*.txt")) if guide_dir.exists() else []
    if not guide_texts:
        raise HTTPException(400, "No interview guide uploaded")

    guide_text = "\n\n".join(f.read_text(encoding="utf-8") for f in guide_texts)

    async def stream():
        client = AsyncOpenAI(base_url=settings.inference_base_url, api_key=settings.inference_api_key)
        async for event in extract_codes_from_guide(guide_text, client, settings.default_model):
            yield event
            if (payload := parse_sse(event)) and payload.get("type") == "done":
                codebook = payload.get("codebook", {})
                (sdir / "codebook.json").write_text(json.dumps(codebook))
                _persist_all_formats(codebook, sdir)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/custom")
async def set_custom_codes(
    body: CustomCodesRequest,
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> dict:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)
    codebook = parse_custom_codes(body.text)
    (sdir / "codebook.json").write_text(json.dumps(codebook))
    _persist_all_formats(codebook, sdir)
    return {"ok": True, "code_count": len(codebook), "codes": list(codebook.keys())}


@router.post("/import")
async def import_codebook_csv(
    file: UploadFile,
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> dict:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)
    tmp = sdir / "_import_codebook.csv"
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    codebook = load_codebook_csv(tmp)
    tmp.unlink(missing_ok=True)
    (sdir / "codebook.json").write_text(json.dumps(codebook))
    _persist_all_formats(codebook, sdir)
    return {"ok": True, "code_count": len(codebook), "codes": list(codebook.keys())}


@router.get("")
async def get_codebook(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> dict:
    sid = get_or_create_session(response, session_id)
    cb_path = session_dir(sid) / "codebook.json"
    if not cb_path.exists():
        return {}
    return json.loads(cb_path.read_text())


@router.get("/download")
async def download_codebook_csv(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> FileResponse:
    sid = get_or_create_session(response, session_id)
    out = session_dir(sid) / "codebook.csv"
    if not out.exists():
        raise HTTPException(404, "No codebook available")
    return FileResponse(out, filename="codebook.csv", media_type="text/csv")


@router.get("/download/markdown")
async def download_codebook_markdown(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> FileResponse:
    sid = get_or_create_session(response, session_id)
    out = session_dir(sid) / "codebook.md"
    if not out.exists():
        raise HTTPException(404, "No codebook available")
    return FileResponse(out, filename="codebook.md", media_type="text/markdown")


@router.get("/download/atlas")
async def download_codebook_atlas(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> FileResponse:
    sid = get_or_create_session(response, session_id)
    out = session_dir(sid) / "codebook_atlas.json"
    if not out.exists():
        raise HTTPException(404, "No codebook available")
    return FileResponse(out, filename="codebook_atlas.json", media_type="application/json")


@router.get("/download/nvivo")
async def download_codebook_nvivo(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> FileResponse:
    sid = get_or_create_session(response, session_id)
    out = session_dir(sid) / "codebook_nvivo.csv"
    if not out.exists():
        raise HTTPException(404, "No codebook available")
    return FileResponse(out, filename="codebook_nvivo.csv", media_type="text/csv")
