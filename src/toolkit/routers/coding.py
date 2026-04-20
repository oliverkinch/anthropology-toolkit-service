"""Coding and thematic analysis endpoints."""

import json

from fastapi import APIRouter, Cookie, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from toolkit.config import settings
from toolkit.services.coding import (
    build_themes,
    merge_coding_results,
    run_deductive,
    run_inductive,
    save_coding_results,
    save_themes_docx,
)
from toolkit.session import get_or_create_session, session_dir

router = APIRouter(prefix="/api/code", tags=["coding"])


class CodingRequest(BaseModel):
    approach: str = "hybrid"  # "deductive" | "inductive" | "hybrid"


@router.post("")
async def run_coding(
    body: CodingRequest,
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> StreamingResponse:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)

    chunks_path = sdir / "chunks.json"
    if not chunks_path.exists():
        raise HTTPException(400, "No chunks available — run chunking first")
    chunks = json.loads(chunks_path.read_text())

    codebook_path = sdir / "codebook.json"
    codebook = json.loads(codebook_path.read_text()) if codebook_path.exists() else {}

    if body.approach != "inductive" and not codebook:
        raise HTTPException(400, "No codebook — set up deductive codes first or choose inductive approach")

    async def stream():
        client = AsyncOpenAI(base_url=settings.inference_base_url, api_key=settings.inference_api_key)
        model = settings.default_model

        deductive_results = []
        inductive_results = []
        inductive_codes = {}

        if body.approach in ("deductive", "hybrid"):
            async for event in run_deductive(chunks, codebook, client, model):
                yield event
                if event.startswith("data:"):
                    payload = json.loads(event[5:].strip())
                    if payload.get("type") == "deductive_done":
                        deductive_results = payload["results"]

        if body.approach in ("inductive", "hybrid"):
            existing = list(codebook.keys()) if codebook else []
            async for event in run_inductive(chunks, existing, client, model):
                yield event
                if event.startswith("data:"):
                    payload = json.loads(event[5:].strip())
                    if payload.get("type") == "inductive_done":
                        inductive_results = payload["results"]
                        inductive_codes = payload.get("discovered_codes", {})

        if not deductive_results:
            deductive_results = [
                {"chunk_id": c.get("chunk_id", i + 1), "deductive_codes": []} for i, c in enumerate(chunks)
            ]
        if not inductive_results:
            inductive_results = [
                {"chunk_id": c.get("chunk_id", i + 1), "inductive_codes": []} for i, c in enumerate(chunks)
            ]

        df = merge_coding_results(chunks, deductive_results, inductive_results)
        save_coding_results(df, sdir, codebook=codebook, inductive_codes=inductive_codes)
        df.to_json(sdir / "coded_data.json", orient="records")
        if inductive_codes:
            (sdir / "inductive_codes.json").write_text(json.dumps(inductive_codes))

        preview = df.head(10).to_dict("records")
        yield f"data: {json.dumps({'type': 'done', 'preview': preview, 'total_chunks': len(df)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/results")
async def get_results(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> list[dict]:
    sid = get_or_create_session(response, session_id)
    path = session_dir(sid) / "coded_data.json"
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_json(path)
    return df.head(50).to_dict("records")


@router.get("/download")
async def download_coding(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> FileResponse:
    sid = get_or_create_session(response, session_id)
    out = session_dir(sid) / "coded_data.xlsx"
    if not out.exists():
        raise HTTPException(404, "No coding results available")
    return FileResponse(
        out,
        filename="coded_data.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@router.post("/themes")
async def generate_themes(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> StreamingResponse:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)
    path = sdir / "coded_data.json"
    if not path.exists():
        raise HTTPException(400, "No coding results — run coding first")

    import pandas as pd

    df = pd.read_json(path)

    async def stream():
        client = AsyncOpenAI(base_url=settings.inference_base_url, api_key=settings.inference_api_key)
        themes_text = ""
        async for event in build_themes(df, client, settings.default_model):
            yield event
            if event.startswith("data:"):
                payload = json.loads(event[5:].strip())
                if payload.get("type") == "done":
                    themes_text = payload.get("themes", "")
                    (sdir / "themes.txt").write_text(themes_text)
                    save_themes_docx(themes_text, sdir)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/themes/download")
async def download_themes(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> FileResponse:
    sid = get_or_create_session(response, session_id)
    out = session_dir(sid) / "themes_report.docx"
    if not out.exists():
        raise HTTPException(404, "No thematic analysis available")
    return FileResponse(
        out,
        filename="themes_report.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get("/themes")
async def get_themes(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> dict:
    sid = get_or_create_session(response, session_id)
    path = session_dir(sid) / "themes.txt"
    if not path.exists():
        return {"themes": ""}
    return {"themes": path.read_text()}
