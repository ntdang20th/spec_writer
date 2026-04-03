# spec-writer/app/main.py
"""
Phase 6: FastAPI REST API for the spec writer engine.
Exposes spec generation, querying, and indexing as HTTP endpoints.
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
import config  # noqa: F401
from config import OLLAMA_URL

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spec Writer API",
    description="Codebase-aware specification generator using RAG + GraphRAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (Web UI)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request/Response models ───────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(description="Question about the codebase")
    mode: str = Field(default="graph", description="'vector' or 'graph'")
    top_k: int = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str

class SpecRequest(BaseModel):
    feature: str = Field(description="Feature description to generate spec for")
    use_graph: bool = Field(default=True, description="Use GraphRAG or vector-only")
    model: str = Field(default="", description="Override LLM model (empty = use default)")

class SpecResponse(BaseModel):
    spec: dict
    markdown: str
    model_used: str
    mode: str

class IndexRequest(BaseModel):
    full_build: bool = Field(default=False, description="True = all chunks, False = first 100")

class StatusResponse(BaseModel):
    status: str
    vector_count: int
    graph_entities: int
    graph_triplets: int
    llm_model: str
    embed_model: str


# ── Endpoints ─────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
def root():
    return FileResponse("static/index.html")


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Health check and system status."""
    import chromadb
    vector_count = 0
    graph_entities = 0
    graph_triplets = 0

    try:
        client = chromadb.HttpClient(host="chromadb", port=8000)
        col = client.get_or_create_collection("codebase")
        vector_count = col.count()
    except Exception:
        pass

    try:
        from graph_builder import _get_graph_store
        store = _get_graph_store()
        try:
            result = store.structured_query("MATCH (n) RETURN count(n) AS cnt")
            if result and isinstance(result, list) and result[0]:
                graph_entities = result[0].get("cnt", 0) if isinstance(result[0], dict) else 0
            result = store.structured_query("MATCH ()-[r]->() RETURN count(r) AS cnt")
            if result and isinstance(result, list) and result[0]:
                graph_triplets = result[0].get("cnt", 0) if isinstance(result[0], dict) else 0
        except Exception:
            # Fallback: fetch all triplets (slow on large graphs)
            triplets = store.get_triplets()
            entities = set()
            for s, r, o in triplets:
                entities.add(str(s))
                entities.add(str(o))
            graph_entities = len(entities)
            graph_triplets = len(triplets)
    except Exception:
        pass

    return StatusResponse(
        status="ok",
        vector_count=vector_count,
        graph_entities=graph_entities,
        graph_triplets=graph_triplets,
        llm_model=Settings.llm.model,
        embed_model=Settings.embed_model.model_name,
    )


@app.post("/query", response_model=QueryResponse)
def query_codebase(req: QueryRequest):
    """Ask a question about the codebase using RAG or GraphRAG."""
    try:
        if req.mode == "graph":
            from graph_builder import load_graph_index
            index = load_graph_index()
        else:
            from embedder import load_index
            index = load_index()

        engine = index.as_query_engine(
            similarity_top_k=req.top_k,
            llm=Settings.llm,
            include_text=True,
        )
        response = engine.query(req.question)

        sources = []
        for node in response.source_nodes:
            m = node.metadata if hasattr(node, "metadata") else {}
            sources.append({
                "file": m.get("file_name", "?"),
                "category": m.get("category", "?"),
                "score": round(node.score, 4) if hasattr(node, "score") and node.score else 0,
                "preview": node.text[:150] if hasattr(node, "text") else "",
            })

        return QueryResponse(
            answer=str(response.response),
            sources=sources,
            mode=req.mode,
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed")


@app.post("/spec", response_model=SpecResponse)
def generate_specification(req: SpecRequest):
    """Generate a structured specification for a feature."""
    from spec_writer import generate_spec, export_spec_markdown

    # Build a local LLM instance if overridden — never mutate global Settings.llm
    llm = None
    model_used = Settings.llm.model

    if req.model:
        llm = Ollama(
            model=req.model,
            base_url=OLLAMA_URL,
            request_timeout=600.0,
            temperature=0.1,
        )
        model_used = req.model

    try:
        spec = generate_spec(req.feature, use_graph=req.use_graph, llm=llm)
        md = export_spec_markdown(spec)

        return SpecResponse(
            spec=spec.model_dump(),
            markdown=md,
            model_used=model_used,
            mode="graph" if req.use_graph else "vector",
        )
    except Exception as e:
        logger.exception("Spec generation failed")
        raise HTTPException(status_code=500, detail="Spec generation failed")


@app.post("/index/vector")
def rebuild_vector_index():
    """Re-index the codebase into ChromaDB."""
    from loader import load_codebase
    from chunker import chunk_documents
    from embedder import build_index

    try:
        docs = load_codebase()
        nodes = chunk_documents(docs)
        build_index(nodes)
        return {"status": "ok", "documents": len(docs), "chunks": len(nodes)}
    except Exception as e:
        logger.exception("Vector index rebuild failed")
        raise HTTPException(status_code=500, detail="Vector index rebuild failed")


@app.post("/index/graph")
def rebuild_graph_index(req: IndexRequest):
    """Re-index the codebase into Neo4j knowledge graph."""
    from loader import load_codebase
    from chunker import chunk_documents
    from graph_builder import build_graph_index, reset_graph

    try:
        reset_graph()
        docs = load_codebase()
        nodes = chunk_documents(docs)
        if not req.full_build:
            nodes = nodes[:100]
        build_graph_index(nodes, batch_size=10)
        return {"status": "ok", "documents": len(docs), "chunks": len(nodes)}
    except Exception as e:
        logger.exception("Graph index rebuild failed")
        raise HTTPException(status_code=500, detail="Graph index rebuild failed")


@app.post("/index/incremental")
def incremental_update():
    """Index only files that changed since last run."""
    from incremental import detect_changes, incremental_index

    try:
        changes = detect_changes()
        summary = {
            "added": len(changes["added"]),
            "modified": len(changes["modified"]),
            "deleted": len(changes["deleted"]),
            "unchanged": changes["unchanged"],
        }

        if not changes["added"] and not changes["modified"] and not changes["deleted"]:
            return {"status": "up_to_date", **summary}

        incremental_index(changes=changes)
        return {"status": "ok", **summary}
    except Exception as e:
        logger.exception("Incremental indexing failed")
        raise HTTPException(status_code=500, detail="Incremental indexing failed")


@app.get("/index/changes")
def check_changes():
    """Check what files changed without indexing."""
    from incremental import detect_changes

    try:
        changes = detect_changes()
        return {
            "added": len(changes["added"]),
            "modified": len(changes["modified"]),
            "deleted": len(changes["deleted"]),
            "unchanged": changes["unchanged"],
            "added_files": changes["added"][:20],
            "modified_files": changes["modified"][:20],
            "deleted_files": changes["deleted"][:20],
        }
    except Exception as e:
        logger.exception("Change detection failed")
        raise HTTPException(status_code=500, detail="Change detection failed")