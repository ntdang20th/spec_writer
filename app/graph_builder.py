# spec-writer/app/graph_builder.py
"""
Phase 3: Build a knowledge graph from your codebase.
Uses Neo4j for persistent graph storage + batch processing with retry.
"""
import os
import json
import time
from pathlib import Path

from llama_index.core import (
    PropertyGraphIndex,
    Settings,
)
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from rich import print as rprint
import config  # noqa: F401
from config import OLLAMA_URL
GRAPH_LLM = Ollama(
    model=os.getenv("GRAPH_LLM_MODEL", "mistral:7b-instruct"),
    base_url=OLLAMA_URL,
    request_timeout=900.0,
    temperature=0.0,
)

# ── Neo4j connection ──────────────────────────────────────
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "specwriter123")

# ── Entity and relation types ─────────────────────────────
ENTITY_TYPES = [
    "Class", "Interface", "Method", "Service", "Repository",
    "Controller", "Handler", "Model", "DTO", "Table",
    "API_Endpoint", "Configuration", "Middleware", "Extension",
]

RELATION_TYPES = [
    "DEPENDS_ON", "IMPLEMENTS", "CALLS", "READS_FROM",
    "WRITES_TO", "RETURNS", "ACCEPTS", "INHERITS",
    "HAS_METHOD", "HAS_PROPERTY", "INJECTS", "CONFIGURES", "MAPS_TO",
]

PROGRESS_FILE = "/app/data/graph/progress.json"


def _get_graph_store() -> Neo4jPropertyGraphStore:
    """Create a Neo4j graph store connection."""
    return Neo4jPropertyGraphStore(
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )


def build_graph_index(nodes: list[TextNode], batch_size: int = 10) -> PropertyGraphIndex:
    """
    Build graph in batches with Neo4j persistence and retry logic.
    If it crashes, re-run and it picks up where it left off.
    """
    rprint(f"\n[bold]Building knowledge graph...[/bold]\n")
    rprint(f"  Entity types:   {len(ENTITY_TYPES)} defined")
    rprint(f"  Relation types: {len(RELATION_TYPES)} defined")
    rprint(f"  Chunks to process: {len(nodes)}")
    rprint(f"  Batch size: {batch_size}")
    rprint(f"  LLM: {GRAPH_LLM.model}")
    rprint(f"  Embed: {Settings.embed_model.model_name}")
    rprint(f"  Graph store: Neo4j at {NEO4J_URL}")

    # ── Check for previous progress ───────────────────────
    processed = _load_progress()
    all_processed = set(processed)

    # Build (original_index, node) pairs for remaining work
    remaining = [(i, n) for i, n in enumerate(nodes) if i not in all_processed]

    if processed:
        rprint(f"\n  [yellow]Resuming: {len(processed)} chunks already done[/yellow]")
        rprint(f"  Remaining: {len(remaining)} chunks\n")
    else:
        rprint()

    total_batches = (len(remaining) + batch_size - 1) // batch_size
    index = None

    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)

    schema_extractor = SchemaLLMPathExtractor(
        llm=GRAPH_LLM,
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        strict=False,
        num_workers=1,
        max_triplets_per_chunk=10,
    )

    implicit_extractor = ImplicitPathExtractor()

    graph_store = _get_graph_store()

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(remaining))
        batch_items = remaining[start:end]
        batch_nodes = [n for _, n in batch_items]
        batch_indices = [i for i, _ in batch_items]

        rprint(f"\n  [bold]Batch {batch_num + 1}/{total_batches}[/bold] ({len(batch_nodes)} chunks)")
        t0 = time.time()

        try:
            if batch_num == 0 and not processed:
                index = PropertyGraphIndex(
                    nodes=batch_nodes,
                    kg_extractors=[schema_extractor, implicit_extractor],
                    property_graph_store=graph_store,
                    embed_model=Settings.embed_model,
                    show_progress=True,
                )
            else:
                if index is None:
                    index = PropertyGraphIndex.from_existing(
                        property_graph_store=graph_store,
                        embed_model=Settings.embed_model,
                    )

                index.insert_nodes(batch_nodes)

            elapsed = time.time() - t0
            rprint(f"  [green]Done in {elapsed:.0f}s ({elapsed/len(batch_nodes):.1f}s/chunk)[/green]")

            for idx in batch_indices:
                all_processed.add(idx)
            _save_progress(all_processed)
            rprint(f"  [dim]Saved checkpoint[/dim]")

        except Exception as e:
            rprint(f"\n  [red]Batch {batch_num + 1} failed: {type(e).__name__}[/red]")
            rprint(f"  [red]{str(e)[:200]}[/red]")
            rprint(f"\n  [yellow]Progress saved. Run again to resume.[/yellow]")
            _save_progress(all_processed)
            break

    # If no index was created (all batches failed or resumed with no new work),
    # try to load the existing graph so callers get a usable index.
    if index is None:
        try:
            index = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                embed_model=Settings.embed_model,
            )
        except Exception:
            pass

    if index is not None:
        _print_graph_stats(graph_store)
    return index


def load_graph_index() -> PropertyGraphIndex:
    """Load graph index from Neo4j."""
    rprint(f"  Loading graph from Neo4j...")
    graph_store = _get_graph_store()
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=Settings.embed_model,
    )
    _print_graph_stats(graph_store)
    return index


def _load_progress() -> set:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def _save_progress(processed: set):
    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed), f)


def _print_graph_stats(graph_store: Neo4jPropertyGraphStore):
    """Print graph stats directly from Neo4j."""
    try:
        triplets = graph_store.get_triplets()
        entities = set()
        relations = set()
        for subj, rel, obj in triplets:
            entities.add(str(subj))
            entities.add(str(obj))
            relations.add(str(rel))

        rprint(f"\n  [bold]Graph stats:[/bold]")
        rprint(f"    Entities (nodes):       {len(entities)}")
        rprint(f"    Unique relationships:   {len(relations)}")
        rprint(f"    Total triplets:         {len(triplets)}")

        if triplets:
            rprint(f"\n  [bold]Sample triplets:[/bold]")
            for subj, rel, obj in triplets[:8]:
                rprint(f"    {subj} -> [{rel}] -> {obj}")
    except Exception as e:
        rprint(f"  [yellow]Could not read graph stats: {e}[/yellow]")


def reset_graph():
    """Clear all graph data from Neo4j and progress file."""
    rprint("  Clearing Neo4j graph data...")
    try:
        store = _get_graph_store()
        store.structured_query("MATCH (n) DETACH DELETE n")
        rprint("  [green]Neo4j cleared[/green]")
    except Exception as e:
        rprint(f"  [yellow]Could not clear Neo4j: {e}[/yellow]")

    if Path(PROGRESS_FILE).exists():
        os.remove(PROGRESS_FILE)
        rprint("  [green]Progress file cleared[/green]")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from loader import load_codebase
    from chunker import chunk_documents

    if "--reset" in sys.argv:
        reset_graph()
        sys.exit(0)

    rprint("\n[bold]Phase 3: Building Knowledge Graph[/bold]")
    rprint("=" * 45)

    full_build = "--all" in sys.argv

    rprint("\n[bold]Step 1: Loading documents...[/bold]\n")
    docs = load_codebase()
    rprint(f"  Loaded {len(docs)} documents")

    rprint("\n[bold]Step 2: Chunking...[/bold]\n")
    nodes = chunk_documents(docs)

    if not full_build:
        limit = 100
        rprint(f"\n  [yellow]Test mode: using first {limit} chunks[/yellow]")
        rprint(f"  Run with --all for full build\n")
        nodes = nodes[:limit]

    rprint(f"\n[bold]Step 3: Building graph + embeddings...[/bold]")
    index = build_graph_index(nodes, batch_size=10)

    rprint("\n[bold green]Graph build complete![/bold green]\n")