"""Inference server configuration and connection testing."""

from fastapi import APIRouter, Cookie, Response
from pydantic import BaseModel

from toolkit.session import get_or_create_session, session_dir

router = APIRouter(prefix="/api/config", tags=["config"])


class ServerConfig(BaseModel):
    base_url: str
    api_key: str
    model: str


class TestResult(BaseModel):
    ok: bool
    message: str


@router.post("/test", response_model=TestResult)
async def test_connection(
    body: ServerConfig, response: Response, session_id: str | None = Cookie(default=None)
) -> TestResult:
    sid = get_or_create_session(response, session_id)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=body.base_url, api_key=body.api_key)
        await client.models.list()
        # Save config to session

        (session_dir(sid) / "server_config.json").write_text(body.model_dump_json())
        return TestResult(ok=True, message="Connection successful")
    except Exception as e:
        return TestResult(ok=False, message=str(e))


@router.post("/save")
async def save_config(body: ServerConfig, response: Response, session_id: str | None = Cookie(default=None)) -> dict:
    sid = get_or_create_session(response, session_id)
    (session_dir(sid) / "server_config.json").write_text(body.model_dump_json())
    return {"ok": True}
