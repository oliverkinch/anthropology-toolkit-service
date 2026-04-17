"""Transcript chunking endpoints."""

import json

from fastapi import APIRouter, Cookie, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from toolkit.config import settings
from toolkit.services.chunker import (
    chunk_with_embeddings,
    chunk_with_llm,
    preprocess_text,
    save_chunks,
)
from toolkit.session import get_or_create_session, session_dir

router = APIRouter(prefix="/api/chunk", tags=["chunker"])


class ChunkRequest(BaseModel):
    method: str = "llm"  # "llm" | "embeddings"
    similarity_threshold: float = 0.5
    min_sentences: int = 1
    max_sentences: int = 8
    remove_timestamps: bool = True
    preserve_speakers: bool = True


@router.post("")
async def run_chunking(
    body: ChunkRequest,
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> StreamingResponse:
    sid = get_or_create_session(response, session_id)
    sdir = session_dir(sid)

    transcript_dir = sdir / "transcript"
    if not transcript_dir.exists() or not any(transcript_dir.iterdir()):
        raise HTTPException(400, "No transcripts uploaded")

    all_sentences: list[dict] = []
    for txt_file in sorted(transcript_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        sentences = preprocess_text(text, body.remove_timestamps, body.preserve_speakers)
        all_sentences.extend(sentences)

    async def stream():
        client = AsyncOpenAI(base_url=settings.inference_base_url, api_key=settings.inference_api_key)
        all_chunks: list[dict] = []

        if body.method == "embeddings":
            gen = chunk_with_embeddings(
                all_sentences,
                similarity_threshold=body.similarity_threshold,
                min_sentences=body.min_sentences,
                max_sentences=body.max_sentences,
            )
        else:
            gen = chunk_with_llm(all_sentences, client, settings.default_model, max_chunk_sentences=body.max_sentences)

        async for event in gen:
            yield event
            if event.startswith("data:"):
                payload = json.loads(event[5:].strip())
                if payload.get("type") == "done":
                    all_chunks = payload.get("chunks", [])
                    for i, c in enumerate(all_chunks):
                        c["chunk_id"] = i + 1
                    (sdir / "chunks.json").write_text(json.dumps(all_chunks))
                    save_chunks(all_chunks, sdir)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/results")
async def get_chunks(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> list[dict]:
    sid = get_or_create_session(response, session_id)
    chunks_path = session_dir(sid) / "chunks.json"
    if not chunks_path.exists():
        return []
    return json.loads(chunks_path.read_text())


@router.get("/download")
async def download_chunks(
    response: Response,
    session_id: str | None = Cookie(default=None),
) -> FileResponse:
    sid = get_or_create_session(response, session_id)
    out = session_dir(sid) / "chunks.xlsx"
    if not out.exists():
        raise HTTPException(404, "No chunks available — run chunking first")
    return FileResponse(
        out, filename="chunks.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
