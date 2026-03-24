# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies** (from repo root):
```bash
uv sync
```

**Run the server** (from repo root):
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is served at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

**Environment setup** — create `.env` in the repo root:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Architecture

This is a full-stack RAG chatbot. The **backend** is a Python FastAPI app (`backend/`); the **frontend** is plain HTML/JS/CSS (`frontend/`) served as static files by FastAPI itself.

### Backend component responsibilities

- **`rag_system.py`** — Central orchestrator. Wires together all components. Entry point for document ingestion (`add_course_document`, `add_course_folder`) and querying (`query`).
- **`app.py`** — FastAPI routes. Two endpoints: `POST /api/query` and `GET /api/courses`. On startup, auto-ingests `.txt` files from `../docs/` (skips already-ingested courses).
- **`document_processor.py`** — Parses course `.txt` files into `Course` + `CourseChunk` objects. Expects a specific format (Course Title / Course Link / Course Instructor on the first 3 lines, then `Lesson N: Title` markers). Chunks text at sentence boundaries with configurable size/overlap.
- **`vector_store.py`** — ChromaDB wrapper with two collections: `course_catalog` (one doc per course, used for fuzzy course-name resolution) and `course_content` (all text chunks, used for semantic search). Embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
- **`ai_generator.py`** — Anthropic Claude API wrapper. Makes up to **two API calls per query**: first to let Claude decide whether to search, second (if `stop_reason == "tool_use"`) to synthesize search results into a final answer.
- **`search_tools.py`** — Defines the `search_course_content` Anthropic tool and `ToolManager`. `CourseSearchTool` tracks `last_sources` after each search so they can be surfaced in the UI.
- **`session_manager.py`** — In-memory session store. History is formatted as a plain string and injected into the Claude system prompt. Sessions are lost on server restart.
- **`config.py`** — Single `Config` dataclass with all tuneable constants (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`, model names).

### Key design decisions

- **Tool-based retrieval**: Claude autonomously decides whether to call the search tool. There is no forced retrieval step — general knowledge questions bypass ChromaDB entirely.
- **ChromaDB is persistent on disk** at `./chroma_db` (relative to `backend/`). Re-running the server does not re-ingest already-loaded courses.
- **Course name resolution is fuzzy**: when Claude passes a `course_name` filter, `VectorStore._resolve_course_name` finds the best match via vector similarity against `course_catalog` before filtering `course_content`.
- **Session history is injected into the system prompt** (not as Anthropic multi-turn messages). Max 2 exchanges (4 messages) are kept per session.
- **All backend modules import from each other using plain module names** (not a package). The server must be run from the `backend/` directory for imports to resolve correctly — `run.sh` and `uvicorn` invocations handle this.

### Course document format

Files in `docs/` must follow:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>
Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>
Lesson 1: <lesson title>
...
```
