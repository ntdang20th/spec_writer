# Spec Writer — Complete Project Summary

## What it does
A codebase-aware AI tool that generates structured technical specifications for new features, epics, and tasks. Uses RAG + GraphRAG to understand your code patterns, dependencies, and architecture, then produces specs that follow your team's conventions.

## Architecture
```
                         localhost:8000
                              │
                         ┌────┴────┐
                         │ FastAPI  │ ← Web UI + REST API
                         │ main.py  │
                         └────┬────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              ┌─────┴──┐ ┌───┴───┐ ┌───┴────┐
              │ Vector  │ │ Graph │ │  Spec  │
              │  RAG    │ │  RAG  │ │ Writer │
              │embedder │ │graph_ │ │spec_   │
              │  .py    │ │builder│ │writer  │
              └────┬────┘ └───┬───┘ └───┬────┘
                   │          │         │
              ┌────┴──┐  ┌───┴───┐  ┌──┴───┐
              │ChromaDB│  │ Neo4j │  │Ollama│
              │:8100   │  │:7474  │  │:11434│
              └────────┘  └───────┘  └──────┘
```

## Tech Stack
| Component | Tool | Purpose |
|-----------|------|---------|
| Framework | LlamaIndex | RAG + GraphRAG orchestration |
| Vector store | ChromaDB (Docker) | Semantic code search |
| Graph store | Neo4j (Docker) | Entity-relationship knowledge graph |
| Local LLM | Ollama (host) | Text generation (llama3.2, qwen2.5-coder:7b) |
| Embeddings | nomic-embed-text via Ollama | Text-to-vector conversion |
| API | FastAPI | REST endpoints + Web UI |
| Code parsing | tree-sitter | AST-aware C# chunking |
| Language | Python 3.12 | All app code |

## Project Files
```
spec-writer/
├── docker-compose.yml          # All services
├── app/
│   ├── Dockerfile              # Python 3.12 + deps
│   ├── requirements.txt        # All Python packages
│   ├── config.py               # Model config (swappable)
│   ├── loader.py               # Load .cs/.json files with metadata
│   ├── chunker.py              # AST-aware C# chunking (tree-sitter)
│   ├── embedder.py             # Vector embeddings → ChromaDB
│   ├── graph_builder.py        # Knowledge graph → Neo4j
│   ├── spec_schema.py          # Pydantic spec output model
│   ├── spec_writer.py          # Spec generation engine
│   ├── query.py                # CLI query tool
│   ├── benchmark.py            # Compare models side by side
│   ├── incremental.py          # Incremental indexing (changed files only)
│   ├── scan.py                 # Codebase file inventory
│   ├── main.py                 # FastAPI REST API
│   └── static/
│       └── index.html          # Web UI
├── data/
│   ├── chroma/                 # Vector store data
│   └── graph/                  # Graph progress tracking
└── neo4j/
    └── data/                   # Neo4j database files
```

## Key URLs
- **Web UI**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474 (neo4j / specwriter123)

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /status | System health + stats |
| POST | /query | Ask questions about the codebase |
| POST | /spec | Generate a structured specification |
| POST | /index/vector | Full vector re-index |
| POST | /index/graph | Full graph re-index |
| POST | /index/incremental | Index only changed files |
| GET | /index/changes | Check what files changed |

## CLI Commands
```bash
# Start everything
docker compose up -d

# Generate a spec
docker compose exec app python spec_writer.py "your feature description"
docker compose exec app python spec_writer.py --export "feature"  # save as markdown

# Query the codebase
docker compose exec app python query.py --graph "how does X work?"
docker compose exec app python query.py --compare "question"  # RAG vs GraphRAG

# Benchmark models
docker compose exec app python benchmark.py --model qwen2.5-coder:7b "feature"
docker compose exec app python benchmark.py "feature"  # test all models

# Incremental indexing
docker compose exec app python incremental.py --status   # check changes
docker compose exec app python incremental.py            # index changes only

# Switch codebase
# 1. Edit docker-compose.yml volume mount
# 2. docker compose down
# 3. Clear data: rd /s /q data\chroma data\graph && mkdir data\chroma data\graph
# 4. docker compose up -d
# 5. docker compose exec app python graph_builder.py --reset
# 6. docker compose exec app python embedder.py
# 7. docker compose exec app python graph_builder.py

# Full graph build (all chunks, not just 100)
docker compose exec app python graph_builder.py --all
```

## Models
| Model | Used for | Pull command |
|-------|----------|-------------|
| llama3.2 | General queries, spec generation | ollama pull llama3.2 |
| nomic-embed-text | Embeddings (vector search) | ollama pull nomic-embed-text |
| qwen2.5-coder:7b | Graph extraction, better specs | ollama pull qwen2.5-coder:7b |

## Hardware Tested
- **PC**: RTX 3060 12GB, i5-12400F, 32GB RAM — runs everything well
- **Laptop**: MX350 2GB, i7-1165G7, 32GB RAM — too slow for 7B models, use llama3.2:1b or Claude API

## Known Issues & Tips
- **tree-sitter-language-pack** doesn't support C# — we build the parser manually from tree-sitter-c-sharp
- **WSL2 memory**: Cap it with `.wslconfig` (memory=8GB) or it eats all RAM
- **Graph build timeouts**: batch processing with resume — if it crashes, run again
- **uvloop conflict**: FastAPI needs `--loop asyncio` flag for nest_asyncio compatibility
- **First run after codebase switch**: must clear data/chroma and data/graph

## Future Improvements
- [ ] Claude API integration for higher quality specs
- [ ] Git hooks for auto-indexing on commit
- [ ] Board integration (Jira/Azure DevOps ticket → spec)
- [ ] Multi-repo support
- [ ] Agentic pipeline (analyze → research → draft → review)
- [ ] Full graph build with all chunks (currently limited to 100 for speed)