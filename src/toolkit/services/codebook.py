"""Codebook building — from documents, interview guide, or custom codes."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from toolkit.config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500  # words
OVERLAP = 50
SIMILARITY_MERGE_THRESHOLD = 0.85
LLM_CONCURRENCY = settings.llm_concurrency  # max simultaneous LLM calls


@dataclass
class CodeEntry:
    label: str
    definition: str
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    frequency: int = 0
    source_documents: list[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Text chunking helpers
# ---------------------------------------------------------------------------


def _sanitize_label(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw.strip())[:25].lower()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def _buffered_progress() -> tuple[list[str], object]:
    """Return (event_buffer, async progress callback) for use with _refine_codebook."""
    events: list[str] = []

    async def _cb(msg: str) -> None:
        events.append(f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n")

    return events, _cb


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


async def _llm_json(client, model: str, prompt: str, max_tokens: int = 2000) -> list | dict:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    try:
        return json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


async def _extract_codes_from_text(
    text: str,
    doc_name: str,
    client,
    model: str,
    coding_strategy: str = "hybrid",
    sem: asyncio.Semaphore | None = None,
) -> list[CodeEntry]:
    chunks = _chunk_text(text)
    if sem is None:
        sem = asyncio.Semaphore(LLM_CONCURRENCY)

    async def _process_chunk(i: int, chunk: str) -> list[dict]:
        if coding_strategy == "inductive":
            prompt = (
                "Using inductive coding, extract potential codes from the provided text.\n"
                "Focus on:\n"
                "- Theoretical concepts and constructs\n"
                "- Methodological approaches\n"
                "- Key terms that appear multiple times\n"
                "- Conceptual frameworks\n\n"
                "For each code provide:\n"
                "- label: ≤25 characters, alphanumeric only, no spaces (use_underscores)\n"
                "- definition: One clear sentence defining the concept\n"
                "- example: Direct quote showing the concept\n"
                "- context: Why this is a meaningful code\n\n"
                f"Text: {chunk}\n\n"
                "Return ONLY valid JSON without markdown formatting:\n"
                '[{"label": "code_name", "definition": "...", "example": "...", "context": "..."}]'
            )
        elif coding_strategy == "deductive":
            prompt = (
                "Extract codes from the provided text based on established theoretical frameworks.\n"
                "Look specifically for:\n"
                "- Established theories\n"
                "- Standard methodological approaches\n"
                "- Common analytical frameworks\n"
                "- Disciplinary conventions\n\n"
                "Format requirements:\n"
                "- label: ≤25 characters, alphanumeric only\n"
                "- definition: One litmus sentence\n"
                "- example: Supporting quote\n\n"
                f"Text: {chunk}\n\n"
                "Return ONLY valid JSON:\n"
                '[{"label": "code_name", "definition": "...", "example": "..."}]'
            )
        else:  # hybrid
            prompt = (
                "Extract codes using a hybrid approach.\n\n"
                "First, identify standard theoretical/methodological codes:\n"
                "- Established frameworks and theories\n"
                "- Research design elements\n"
                "- Analytical approaches\n\n"
                "Then, identify emergent codes unique to this text:\n"
                "- Novel concepts introduced\n"
                "- Specific constructs defined\n"
                "- Unique methodological innovations\n\n"
                "For each code:\n"
                "- label: ≤25 characters, alphanumeric, no spaces\n"
                "- definition: One sentence 'litmus test' definition\n"
                "- code_type: 'deductive' or 'inductive'\n"
                "- example: Direct quote (50-150 words)\n"
                "- inclusion: When to use this code\n"
                "- exclusion: When NOT to use this code\n\n"
                f"Text: {chunk}\n\n"
                "Return ONLY valid JSON:\n"
                '[{"label": "code_name", "definition": "...", "code_type": "deductive", '
                '"example": "...", "inclusion": "...", "exclusion": "..."}]'
            )
        try:
            async with sem:
                items = await _llm_json(client, model, prompt)
            return items if isinstance(items, list) else []
        except Exception as e:
            logger.warning("Code extraction failed on chunk %d of %s: %s", i, doc_name, e)
            return []

    results = await asyncio.gather(*[_process_chunk(i, c) for i, c in enumerate(chunks)])

    entries: dict[str, CodeEntry] = {}
    for items in results:
        for item in items:
            if not isinstance(item, dict):
                continue
            label = _sanitize_label(str(item.get("label", "")))
            if not label:
                continue
            if label in entries:
                entries[label].frequency += 1
                ex = item.get("example", "")
                if ex and ex not in entries[label].examples:
                    entries[label].examples.append(ex)
            else:
                entries[label] = CodeEntry(
                    label=label,
                    definition=item.get("definition", ""),
                    inclusion_criteria=[item.get("inclusion", "")] if item.get("inclusion") else [],
                    exclusion_criteria=[item.get("exclusion", "")] if item.get("exclusion") else [],
                    examples=[item.get("example", "")] if item.get("example") else [],
                    frequency=1,
                    source_documents=[doc_name],
                )

    return list(entries.values())


# ---------------------------------------------------------------------------
# Refinement pipeline: deduplication + merge via embeddings + LLM
# ---------------------------------------------------------------------------


async def _refine_codebook(
    entries: dict[str, CodeEntry],
    client,
    model: str,
    progress_cb,  # async callable(msg: str)
    threshold: float = SIMILARITY_MERGE_THRESHOLD,
    sem: asyncio.Semaphore | None = None,
) -> dict[str, CodeEntry]:
    """Find near-duplicate codes, ask LLM whether to merge each pair."""
    if len(entries) < 2:
        return entries

    if sem is None:
        sem = asyncio.Semaphore(LLM_CONCURRENCY)

    await progress_cb("Computing code embeddings for deduplication...")

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    labels = list(entries.keys())
    texts = [f"{e.label} {e.definition}" for e in entries.values()]
    embeddings = st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    sims = cosine_similarity(embeddings)
    to_merge: list[tuple[str, str, float]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if sims[i][j] >= threshold:
                to_merge.append((labels[i], labels[j], float(sims[i][j])))

    if not to_merge:
        await progress_cb("No near-duplicate codes found.")
        return entries

    await progress_cb(f"Evaluating {len(to_merge)} potential merge(s)...")

    async def _eval_pair(label_a: str, label_b: str, sim: float) -> tuple[str, str, dict | None]:
        e_a = entries.get(label_a)
        e_b = entries.get(label_b)
        if e_a is None or e_b is None:
            return label_a, label_b, None
        ex_a = e_a.examples[0] if e_a.examples else ""
        ex_b = e_b.examples[0] if e_b.examples else ""
        prompt = (
            f"Evaluate whether these two qualitative codes should be merged.\n\n"
            f"Similarity score: {sim:.2f}\n\n"
            f"Code 1: {e_a.label}\n"
            f"Definition: {e_a.definition}\n"
            f"Example: {ex_a}\n\n"
            f"Code 2: {e_b.label}\n"
            f"Definition: {e_b.definition}\n"
            f"Example: {ex_b}\n\n"
            "Consider:\n"
            "- Are these conceptually distinct despite similar language?\n"
            "- Would merging lose important nuance?\n"
            "- Is one a subset of the other?\n\n"
            "Return ONLY valid JSON:\n"
            '{"should_merge": true, "rationale": "...", '
            '"merged_label": "suggested_label", "merged_definition": "comprehensive definition"}'
        )
        try:
            async with sem:
                result = await _llm_json(client, model, prompt, max_tokens=300)
            return label_a, label_b, result if isinstance(result, dict) else None
        except Exception as e:
            logger.warning("Merge evaluation failed for %s/%s: %s", label_a, label_b, e)
            return label_a, label_b, None

    decisions = await asyncio.gather(*[_eval_pair(a, b, s) for a, b, s in to_merge])

    # Apply merge decisions sequentially to correctly track merged_into
    merged_into: dict[str, str] = {}
    for label_a, label_b, result in decisions:
        if label_a in merged_into or label_b in merged_into:
            continue
        if result and result.get("should_merge"):
            e_a = entries.get(label_a)
            e_b = entries.get(label_b)
            if e_a is None or e_b is None:
                continue
            new_label = _sanitize_label(str(result.get("merged_label", label_a)))
            new_def = result.get("merged_definition", e_a.definition)
            merged = CodeEntry(
                label=new_label,
                definition=new_def,
                inclusion_criteria=e_a.inclusion_criteria + e_b.inclusion_criteria,
                exclusion_criteria=e_a.exclusion_criteria + e_b.exclusion_criteria,
                examples=(e_a.examples + e_b.examples)[:4],
                frequency=e_a.frequency + e_b.frequency,
                source_documents=list(set(e_a.source_documents + e_b.source_documents)),
            )
            entries.pop(label_a, None)
            entries.pop(label_b, None)
            entries[new_label] = merged
            merged_into[label_a] = new_label
            merged_into[label_b] = new_label
            logger.info("Merged %s + %s → %s", label_a, label_b, new_label)

    merges = len(set(merged_into.values()))
    await progress_cb(f"Refinement complete — {merges} merge(s) applied, {len(entries)} codes remaining.")
    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_codebook(
    documents: dict[str, str],
    client,
    model: str,
    coding_strategy: str = "hybrid",
    max_codes: int = 40,
    min_frequency: int = 2,
) -> AsyncIterator[str]:
    """Yield SSE events, finish with done+codebook."""
    logger.info("Codebook build started with llm_concurrency=%d", LLM_CONCURRENCY)
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    all_entries: dict[str, CodeEntry] = {}
    total = len(documents)

    async def _extract_doc(name: str, text: str) -> tuple[str, list[CodeEntry]]:
        entries = await _extract_codes_from_text(text, name, client, model, coding_strategy, sem)
        return name, entries

    tasks = [asyncio.create_task(_extract_doc(name, text)) for name, text in documents.items()]
    for idx, name in enumerate(documents.keys(), start=1):
        yield f"data: {json.dumps({'type': 'progress', 'message': f'Queued extraction for {name} ({idx}/{total})...'})}\n\n"

    for completed, done_task in enumerate(asyncio.as_completed(tasks), start=1):
        name, entries = await done_task
        yield f"data: {json.dumps({'type': 'progress', 'message': f'Completed extraction for {name} ({completed}/{total})...'})}\n\n"
        await asyncio.sleep(0)

        for e in entries:
            if e.label in all_entries:
                all_entries[e.label].frequency += e.frequency
                all_entries[e.label].examples.extend(e.examples)
            else:
                all_entries[e.label] = e

    # Prune rare codes
    pruned = {k: v for k, v in all_entries.items() if v.frequency >= min_frequency}

    # Refinement: deduplicate via embeddings + LLM merge
    events, _progress = _buffered_progress()
    refined = await _refine_codebook(pruned, client, model, _progress, sem=sem)
    for ev in events:
        yield ev

    # Limit to max_codes by frequency
    codebook = dict(sorted(refined.items(), key=lambda x: -x[1].frequency)[:max_codes])

    yield f"data: {json.dumps({'type': 'done', 'codebook': _codebook_to_dict(codebook)})}\n\n"


async def extract_codes_from_guide(
    guide_text: str,
    client,
    model: str,
) -> AsyncIterator[str]:
    """Extract deductive codes from an interview guide."""
    logger.info("Guide code extraction started with llm_concurrency=%d", LLM_CONCURRENCY)
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    yield f"data: {json.dumps({'type': 'progress', 'message': 'Extracting codes from interview guide...'})}\n\n"
    await asyncio.sleep(0)

    entries_list = await _extract_codes_from_text(guide_text, "interview_guide", client, model, "hybrid", sem)
    entries = {e.label: e for e in entries_list}

    events, _progress = _buffered_progress()
    refined = await _refine_codebook(entries, client, model, _progress, threshold=SIMILARITY_MERGE_THRESHOLD, sem=sem)
    for ev in events:
        yield ev

    yield f"data: {json.dumps({'type': 'done', 'codebook': _codebook_to_dict(refined)})}\n\n"


def parse_custom_codes(text: str) -> dict:
    """Parse textarea input — one 'LABEL: definition' per line."""
    codebook = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in (":", "—", "-"):
            if sep in line:
                label, _, definition = line.partition(sep)
                label = _sanitize_label(label)
                if label:
                    codebook[label] = {
                        "label": label,
                        "definition": definition.strip(),
                        "inclusion_criteria": [],
                        "exclusion_criteria": [],
                        "examples": [],
                        "frequency": 1,
                        "source_documents": ["custom"],
                    }
                break
    return codebook


def load_codebook_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    codebook = {}
    for _, row in df.iterrows():
        label = _sanitize_label(str(row.get("code_label", row.get("label", ""))))
        if not label:
            continue
        examples = [v for col in ("example_1", "example_2") if (v := str(row.get(col, "") or "")) not in ("", "nan")]
        codebook[label] = {
            "label": label,
            "definition": str(row.get("definition", "") or ""),
            "inclusion_criteria": [str(row["inclusion_criteria"])] if pd.notna(row.get("inclusion_criteria")) else [],
            "exclusion_criteria": [str(row["exclusion_criteria"])] if pd.notna(row.get("exclusion_criteria")) else [],
            "examples": examples,
            "frequency": int(row.get("frequency", 1) or 1),
            "source_documents": [],
        }
    return codebook


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def save_codebook(codebook: dict, session_path: Path) -> Path:
    """Save CSV (primary download format)."""
    rows = []
    for label, code in codebook.items():
        examples = code.get("examples", [])
        rows.append(
            {
                "code_label": label,
                "definition": code.get("definition", ""),
                "inclusion_criteria": "; ".join(code.get("inclusion_criteria", [])),
                "exclusion_criteria": "; ".join(code.get("exclusion_criteria", [])),
                "example_1": examples[0] if len(examples) > 0 else "",
                "example_2": examples[1] if len(examples) > 1 else "",
                "frequency": code.get("frequency", 0),
            }
        )
    df = pd.DataFrame(rows)
    out = session_path / "codebook.csv"
    df.to_csv(out, index=False)
    return out


def save_codebook_markdown(codebook: dict, session_path: Path) -> Path:
    lines = ["# Codebook\n"]
    for label, code in codebook.items():
        lines.append(f"## `{label}`\n")
        lines.append(f"**Definition:** {code.get('definition', '')}\n")
        inc = "; ".join(code.get("inclusion_criteria", []))
        exc = "; ".join(code.get("exclusion_criteria", []))
        if inc:
            lines.append(f"**Include when:** {inc}\n")
        if exc:
            lines.append(f"**Exclude when:** {exc}\n")
        for i, ex in enumerate(code.get("examples", [])[:2], 1):
            if ex:
                lines.append(f"**Example {i}:** _{ex}_\n")
        lines.append("")
    out = session_path / "codebook.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def save_codebook_atlas(codebook: dict, session_path: Path) -> Path:
    """ATLAS.ti compatible JSON format."""
    atlas = [
        {
            "name": label,
            "comment": code.get("definition", ""),
            "examples": [e for e in code.get("examples", []) if e],
        }
        for label, code in codebook.items()
    ]
    out = session_path / "codebook_atlas.json"
    out.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def save_codebook_nvivo(codebook: dict, session_path: Path) -> Path:
    """NVivo compatible CSV format."""
    rows = [{"Name": label, "Description": code.get("definition", ""), "Files": ""} for label, code in codebook.items()]
    df = pd.DataFrame(rows)
    out = session_path / "codebook_nvivo.csv"
    df.to_csv(out, index=False)
    return out


def _codebook_to_dict(codebook: dict[str, CodeEntry]) -> dict:
    return {
        k: {
            "label": v.label,
            "definition": v.definition,
            "inclusion_criteria": v.inclusion_criteria,
            "exclusion_criteria": v.exclusion_criteria,
            "examples": v.examples[:2],
            "frequency": v.frequency,
            "source_documents": v.source_documents,
        }
        for k, v in codebook.items()
    }
