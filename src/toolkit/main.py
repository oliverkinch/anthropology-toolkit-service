"""FastAPI application entry point."""

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from toolkit.config import settings
from toolkit.routers import chunker, codebook, coding, upload
from toolkit.session import cleanup_stale_sessions

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Anthropology Toolkit", version="0.1.0", docs_url="/docs")

app.include_router(upload.router)
app.include_router(chunker.router)
app.include_router(codebook.router)
app.include_router(coding.router)

# Serve frontend from static/
_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static, html=True), name="static")


@app.on_event("startup")
async def startup() -> None:
    Path("tmp").mkdir(exist_ok=True)
    cleanup_stale_sessions()


def run() -> None:
    uvicorn.run("toolkit.main:app", host=settings.host, port=settings.port, reload=True)
