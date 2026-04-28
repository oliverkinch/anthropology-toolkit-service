"""Tests that verify LLM calls are actually executed in parallel and that the
semaphore correctly caps the maximum number of simultaneous requests."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import numpy as np

from toolkit.services.codebook import (
    CodeEntry,
    _extract_codes_from_text,
    _refine_codebook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_MODEL = "test-model"
# chunk_size=500, overlap=50  →  stride = 450
_CHUNK_STRIDE = 450


def _fake_client() -> MagicMock:
    """Return a mock client — never actually called in these tests."""
    return MagicMock()


def _text_with_n_chunks(n: int) -> str:
    """Return a string that produces exactly n chunks.

    _chunk_text starts at 0, 450, 900, … (stride = chunk_size - overlap = 450).
    It creates a chunk for every start < len(words).
    So n chunks  →  (n-1)*450 < N  and  n*450 >= N  →  N = n*450.
    """
    return " ".join(["word"] * (n * _CHUNK_STRIDE))


def _make_entries(labels: list[str]) -> dict[str, CodeEntry]:
    return {label: CodeEntry(label=label, definition=f"Definition of {label}", frequency=1) for label in labels}


def _sim_matrix_for_pairs(labels: list[str], pairs: list[tuple[str, str, float]]) -> np.ndarray:
    """Build a square similarity matrix with given (a, b, score) pairs."""
    n = len(labels)
    mat = np.zeros((n, n))
    for a, b, s in pairs:
        i, j = labels.index(a), labels.index(b)
        mat[i][j] = mat[j][i] = s
    return mat


# ---------------------------------------------------------------------------
# _extract_codes_from_text — parallelism tests
# ---------------------------------------------------------------------------


async def test_chunk_extraction_runs_in_parallel() -> None:
    """With 0.1 s mock latency and 5 chunks, parallel execution should finish
    in roughly 0.1 s (one round), not 0.5 s (five sequential rounds)."""
    n_chunks = 5
    latency = 0.1

    async def slow_llm_json(client, model, prompt, max_tokens=2000):
        await asyncio.sleep(latency)
        return [{"label": "code_a", "definition": "some def"}]

    sem = asyncio.Semaphore(10)  # allow all chunks concurrently
    text = _text_with_n_chunks(n_chunks)

    with patch("toolkit.services.codebook._llm_json", side_effect=slow_llm_json):
        start = time.monotonic()
        await _extract_codes_from_text(text, "doc", _fake_client(), FAKE_MODEL, sem=sem)
        elapsed = time.monotonic() - start

    sequential_time = n_chunks * latency
    assert elapsed < sequential_time * 0.6, (
        f"Expected parallel execution (~{latency}s), but took {elapsed:.2f}s "
        f"(sequential would be {sequential_time:.2f}s)"
    )


async def test_semaphore_caps_concurrency() -> None:
    """Even with many chunks, no more than `cap` LLM calls should run at once."""
    n_chunks = 10
    cap = 3
    active: list[int] = []

    async def tracking_llm_json(client, model, prompt, max_tokens=2000):
        active.append(1)
        assert len(active) <= cap, f"Concurrency exceeded cap: {len(active)} > {cap}"
        await asyncio.sleep(0.05)
        active.pop()
        return []

    sem = asyncio.Semaphore(cap)
    text = _text_with_n_chunks(n_chunks)

    with patch("toolkit.services.codebook._llm_json", side_effect=tracking_llm_json):
        await _extract_codes_from_text(text, "doc", _fake_client(), FAKE_MODEL, sem=sem)


async def test_extraction_aggregates_results_correctly() -> None:
    """Codes returned across all chunks should be merged into one entries list."""
    call_count = 0

    async def llm_returns_unique_code(client, model, prompt, max_tokens=2000):
        nonlocal call_count
        call_count += 1
        idx = call_count
        return [{"label": f"code_{idx:02d}", "definition": "def"}]

    n_chunks = 3
    text = _text_with_n_chunks(n_chunks)

    with patch("toolkit.services.codebook._llm_json", side_effect=llm_returns_unique_code):
        entries = await _extract_codes_from_text(text, "doc", _fake_client(), FAKE_MODEL, sem=asyncio.Semaphore(10))

    assert call_count == n_chunks
    assert len(entries) == n_chunks


async def test_extraction_deduplicates_repeated_codes() -> None:
    """If the same label is returned by multiple chunks, frequency should accumulate."""

    async def llm_always_returns_same_code(client, model, prompt, max_tokens=2000):
        return [{"label": "shared_code", "definition": "shared def", "example": "ex"}]

    n_chunks = 4
    text = _text_with_n_chunks(n_chunks)

    with patch("toolkit.services.codebook._llm_json", side_effect=llm_always_returns_same_code):
        entries = await _extract_codes_from_text(text, "doc", _fake_client(), FAKE_MODEL, sem=asyncio.Semaphore(10))

    assert len(entries) == 1
    assert entries[0].label == "shared_code"
    assert entries[0].frequency == n_chunks


async def test_chunk_extraction_tolerates_llm_errors() -> None:
    """A failing LLM call on one chunk should not abort the whole extraction."""
    n_chunks = 3
    call_count = 0

    async def sometimes_fails(client, model, prompt, max_tokens=2000):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("LLM timeout")
        return [{"label": "ok_code", "definition": "def"}]

    text = _text_with_n_chunks(n_chunks)

    with patch("toolkit.services.codebook._llm_json", side_effect=sometimes_fails):
        entries = await _extract_codes_from_text(text, "doc", _fake_client(), FAKE_MODEL, sem=asyncio.Semaphore(10))

    assert call_count == n_chunks
    assert any(e.label == "ok_code" for e in entries)


# ---------------------------------------------------------------------------
# _refine_codebook — merge evaluation parallelism
# ---------------------------------------------------------------------------


async def _noop_progress(msg: str) -> None:
    pass


async def test_merge_evaluations_run_in_parallel() -> None:
    """With 6 pairs and 0.1 s latency each, parallel execution should be ~0.1 s."""
    n_pairs = 6
    latency = 0.1
    labels = [f"code_{i:02d}" for i in range(n_pairs + 1)]
    entries = _make_entries(labels)
    to_merge_pairs = [(labels[i], labels[i + 1], 0.9) for i in range(n_pairs)]

    async def slow_llm_json(client, model, prompt, max_tokens=2000):
        await asyncio.sleep(latency)
        return {"should_merge": False}

    mock_st = MagicMock()
    mock_st.encode.return_value = np.zeros((len(labels), 4))
    sim_matrix = _sim_matrix_for_pairs(labels, to_merge_pairs)

    with (
        patch("toolkit.services.codebook._llm_json", side_effect=slow_llm_json),
        patch("sentence_transformers.SentenceTransformer", return_value=mock_st),
        patch("sklearn.metrics.pairwise.cosine_similarity", return_value=sim_matrix),
    ):
        start = time.monotonic()
        await _refine_codebook(
            entries,
            _fake_client(),
            FAKE_MODEL,
            _noop_progress,
            sem=asyncio.Semaphore(20),
        )
        elapsed = time.monotonic() - start

    sequential_time = n_pairs * latency
    assert elapsed < sequential_time * 0.6, (
        f"Expected parallel merge evaluation (~{latency}s), but took {elapsed:.2f}s "
        f"(sequential would be {sequential_time:.2f}s)"
    )


async def test_merge_count_is_correct() -> None:
    """The progress message should report the correct number of merges applied."""
    labels = ["alpha", "beta", "gamma", "delta"]
    entries = _make_entries(labels)
    progress_messages: list[str] = []

    async def recording_progress(msg: str) -> None:
        progress_messages.append(msg)

    call_num = 0

    async def merge_first_pair_only(client, model, prompt, max_tokens=2000):
        nonlocal call_num
        call_num += 1
        return {
            "should_merge": call_num == 1,
            "merged_label": "alpha_beta",
            "merged_definition": "merged def",
        }

    # alpha↔beta and gamma↔delta both above threshold; only first merge accepted
    to_merge_pairs = [("alpha", "beta", 0.95), ("gamma", "delta", 0.95)]
    sim_matrix = _sim_matrix_for_pairs(labels, to_merge_pairs)
    mock_st = MagicMock()
    mock_st.encode.return_value = np.zeros((len(labels), 4))

    with (
        patch("toolkit.services.codebook._llm_json", side_effect=merge_first_pair_only),
        patch("sentence_transformers.SentenceTransformer", return_value=mock_st),
        patch("sklearn.metrics.pairwise.cosine_similarity", return_value=sim_matrix),
    ):
        await _refine_codebook(
            entries,
            _fake_client(),
            FAKE_MODEL,
            recording_progress,
            sem=asyncio.Semaphore(10),
        )

    final_msg = next(m for m in reversed(progress_messages) if "Refinement complete" in m)
    assert "1 merge(s)" in final_msg, f"Unexpected message: {final_msg}"
