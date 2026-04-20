"""Shared SSE utilities for routers."""

import json


def parse_sse(event: str) -> dict | None:
    """Parse an SSE data line into a dict, or return None if not a data event."""
    if event.startswith("data:"):
        return json.loads(event[5:].strip())
    return None
