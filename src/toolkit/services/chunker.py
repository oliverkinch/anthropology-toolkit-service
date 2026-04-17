"""Semantic transcript chunking — LLM-based or embedding-based."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

import nltk
import pandas as pd

logger = logging.getLogger(__name__)

# Download sentence tokenizer on first use
_nltk_ready = False


def _ensure_nltk() -> None:
    global _nltk_ready
    if not _nltk_ready:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        _nltk_ready = True


# ---------------------------------------------------------------------------
# Speaker label helpers (preserved from original notebook)
# ---------------------------------------------------------------------------

_SPEAKER_PATTERNS = [
    r"^\*\*(.+?):\*\*",
    r"^([A-Z][^:]{0,30}):\s",
    r"^\[(.+?)\]:",
    r"^(Q|A|I|R):\s",
    r"^(Interviewer|Respondent|Moderator|Participant):",
]
_SPEAKER_RE = re.compile("|".join(_SPEAKER_PATTERNS), re.IGNORECASE)


def _extract_speaker(line: str) -> str | None:
    m = _SPEAKER_RE.match(line.strip())
    if not m:
        return None
    for g in m.groups():
        if g:
            return g.strip()
    return None


def preprocess_text(
    text: str,
    remove_timestamps: bool = True,
    preserve_speakers: bool = True,
) -> list[dict]:
    """Return list of {text, speaker} sentence dicts."""
    _ensure_nltk()

    if remove_timestamps:
        text = re.sub(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?", "", text)

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    current_speaker: str | None = None
    sentences = []

    for para in paragraphs:
        speaker = _extract_speaker(para)
        if speaker:
            current_speaker = speaker
        sents = nltk.sent_tokenize(para)
        for s in sents:
            if s.strip():
                sentences.append({"text": s.strip(), "speaker": current_speaker})

    return sentences


# ---------------------------------------------------------------------------
# LLM-based chunking
# ---------------------------------------------------------------------------


async def chunk_with_llm(
    sentences: list[dict],
    client,
    model: str,
    max_chunk_sentences: int = 8,
) -> AsyncIterator[str]:
    """Yield SSE-style progress lines, then final JSON."""
    BATCH = 30
    chunks: list[dict] = []
    total_batches = max(1, len(sentences) // BATCH + (1 if len(sentences) % BATCH else 0))

    for batch_idx in range(0, len(sentences), BATCH):
        batch = sentences[batch_idx : batch_idx + BATCH]
        batch_text = "\n".join(s["text"] for s in batch)
        batch_num = batch_idx // BATCH + 1

        yield f"data: {json.dumps({'type': 'progress', 'message': f'Chunking batch {batch_num}/{total_batches}...'})}\n\n"

        prompt = (
            f"Please analyze this interview transcript and break it into semantically coherent chunks. Each chunk should:\n\n"
            f"1. Contain at most {max_chunk_sentences} sentences\n"
            f"2. Represent a distinct topic, question-answer pair, or conversation turn\n"
            f"3. Preserve speaker labels exactly as they appear (Q:, A:, Interviewer:, etc.)\n"
            f"4. Maintain natural conversation flow\n\n"
            f'Return ONLY the chunks, each separated by exactly "---CHUNK_BREAK---"\n\n'
            f"Text to chunk:\n{batch_text}"
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            parts = raw.split("---CHUNK_BREAK---")
        except Exception as e:
            logger.warning("LLM chunking batch %d failed: %s — falling back", batch_num, e)
            parts = [
                " ".join(s["text"] for s in batch[i : i + max_chunk_sentences])
                for i in range(0, len(batch), max_chunk_sentences)
            ]

        for part in parts:
            text = part.strip()
            if text:
                # infer speaker from first line
                first_speaker = batch[0]["speaker"] if batch else None
                chunks.append({"text": text, "speaker": first_speaker})

        await asyncio.sleep(0)  # yield control

    yield f"data: {json.dumps({'type': 'done', 'chunks': chunks})}\n\n"


# ---------------------------------------------------------------------------
# Embedding-based chunking
# ---------------------------------------------------------------------------


async def chunk_with_embeddings(
    sentences: list[dict],
    similarity_threshold: float = 0.5,
    min_sentences: int = 1,
    max_sentences: int = 8,
    model_name: str = "all-MiniLM-L6-v2",
) -> AsyncIterator[str]:
    """Yield SSE-style progress, then final JSON."""
    yield f"data: {json.dumps({'type': 'progress', 'message': 'Loading embedding model...'})}\n\n"
    await asyncio.sleep(0)

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    st_model = SentenceTransformer(model_name)
    texts = [s["text"] for s in sentences]

    yield f"data: {json.dumps({'type': 'progress', 'message': 'Computing embeddings...'})}\n\n"
    await asyncio.sleep(0)

    embeddings = st_model.encode(texts, show_progress_bar=False)

    chunks: list[dict] = []
    current: list[int] = []

    for i, _sent in enumerate(sentences):
        current.append(i)
        if len(current) < min_sentences:
            continue
        if len(current) >= max_sentences:
            chunks.append(_make_chunk(current, sentences))
            current = []
            continue
        if i + 1 < len(sentences):
            sim = float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
            if sim < similarity_threshold:
                chunks.append(_make_chunk(current, sentences))
                current = []

    if current:
        chunks.append(_make_chunk(current, sentences))

    yield f"data: {json.dumps({'type': 'done', 'chunks': chunks})}\n\n"


def _make_chunk(indices: list[int], sentences: list[dict]) -> dict:
    sents = [sentences[i] for i in indices]
    speaker = next((s["speaker"] for s in sents if s["speaker"]), None)
    return {"text": " ".join(s["text"] for s in sents), "speaker": speaker}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def chunks_to_df(chunks: list[dict]) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(chunks):
        rows.append(
            {
                "chunk_id": i + 1,
                "text": c["text"],
                "speaker": c.get("speaker", ""),
                "word_count": len(c["text"].split()),
                "sentence_count": len(nltk.sent_tokenize(c["text"])) if c["text"] else 0,
            }
        )
    return pd.DataFrame(rows)


def save_chunks(chunks: list[dict], session_path: Path) -> Path:
    df = chunks_to_df(chunks)
    out = session_path / "chunks.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Chunks", index=False)
        stats = pd.DataFrame(
            {
                "Metric": ["Total chunks", "Total words", "Avg words/chunk", "Min words", "Max words"],
                "Value": [
                    len(df),
                    df["word_count"].sum(),
                    round(df["word_count"].mean(), 1),
                    df["word_count"].min(),
                    df["word_count"].max(),
                ],
            }
        )
        stats.to_excel(writer, sheet_name="Statistics", index=False)
    return out
