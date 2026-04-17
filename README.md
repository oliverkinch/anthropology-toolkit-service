# Anthropology Toolkit Service

Web service exposing three AI-powered qualitative analysis tools as a browser application. Runs locally behind VPN, backed by the Alexandra Institute inference server.

## Tools

| Tool | What it does |
|---|---|
| **Semantic Chunker** | Splits interview transcripts into semantically coherent chunks, preserving speaker labels |
| **Codebook Builder** | Extracts a deductive codebook from source literature, an interview guide, or custom input |
| **Coding & Thematic Analysis** | Applies deductive and/or inductive coding to chunks, then synthesises 5–7 hierarchical themes |

## Requirements

- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- VPN access to the inference server

## Setup

```bash
cp .env.example .env
# Add your API key to .env
uv sync
```

**.env**
```
INFERENCE_BASE_URL=https://inference.projects.alexandrainst.dk/v1
INFERENCE_API_KEY=sk-...
DEFAULT_MODEL=qwen-235b
```

## Run

```bash
uv run serve
```

Opens at **http://localhost:8080**. Hot-reload is enabled by default.

## Workflow

The app is a four-step wizard:

### 1. Upload
Upload one or more interview transcripts (txt/docx/pdf). Optionally upload an interview guide (used as the basis for deductive codes) and source literature (used by the codebook builder).

### 2. Chunking
Splits all uploaded transcripts into semantically meaningful chunks.

- **LLM-based** (default) — asks the model to identify natural topic boundaries and conversation turns
- **Embedding-based** — uses `sentence-transformers` locally; no extra API calls

Download results as Excel.

### 3. Coding
Choose a codebook source, then run coding:

**Codebook sources**
| Option | When to use |
|---|---|
| Write custom codes | You already know your deductive codes — enter them as `LABEL: definition` |
| Use interview guide | Derive codes automatically from the uploaded interview guide |
| Build from literature | Run the full codebook builder pipeline on uploaded source literature |
| Import CSV | Re-use a codebook exported from a previous session |

**Approaches:** Deductive only · Inductive only · Hybrid (both)

Download coded data as Excel.

### 4. Analysis
Generates 5–7 hierarchical themes integrating both deductive and inductive codes. Download as Word (.docx).

## Test files

Sample files are provided in `test_files/`:

| File | Upload as |
|---|---|
| `interview_transcript.txt` | Interview Transcript |
| `interview_guide.txt` | Interview Guide |
| `source_literature.txt` | Source Literature |

## Project structure

```
src/toolkit/
├── main.py              # FastAPI app
├── config.py            # Settings from .env
├── session.py           # Per-session temp directories
├── routers/
│   ├── upload.py        # File upload endpoints
│   ├── chunker.py       # Chunking endpoints
│   ├── codebook.py      # Codebook endpoints
│   └── coding.py        # Coding & themes endpoints
├── services/
│   ├── file_io.py       # PDF / DOCX / TXT extraction
│   ├── chunker.py       # Chunking logic
│   ├── codebook.py      # Codebook building logic
│   └── coding.py        # Coding & thematic analysis logic
└── static/              # Frontend (Alpine.js + Tailwind CDN)
```

## Deployment

The service is hosted at **https://insights-tools.alexandrainst.dk/** on port 8080.

Build and start with Docker Compose:

```bash
cp .env.example .env
# Fill in INFERENCE_API_KEY in .env
docker compose up -d --build
```

The app reads all configuration from environment variables via `.env`. No code changes are needed to switch between local and deployed environments.
