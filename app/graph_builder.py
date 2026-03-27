# spec-writer/app/graph_builder.py
"""
Phase 3: Build a knowledge graph from your codebase.
Uses batch processing with retry logic to handle LLM timeouts.
"""
import os
import json
import time
from pathlib import Path
from llama_index.core import (
    PropertyGraphIndex,
    Settings,
    StorageContext,
)
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama
from rich import print as rprint
import config  # noqa: F401

# ── Dedicated LLM for graph extraction ────────────────────
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GRAPH_LLM = Ollama(
    model=os.getenv("GRAPH_LLM_MODEL", "qwen2.5-coder:7b"),
    base_url=OLLAMA_URL,
    request_timeout=900.0,
    temperature=0.0,
)

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

GRAPH_PERSIST_DIR = "/app/data/graph"
PROGRESS_FILE = "/app/data/graph/progress.json"


def build_graph_index(nodes: list[TextNode], batch_size: int = 10) -> PropertyGraphIndex:
    """
    Build graph in batches with retry logic.
    If it crashes, re-run and it picks up where it left off.
    """
    rprint(f"\n[bold]Building knowledge graph...[/bold]\n")
    rprint(f"  Entity types:   {len(ENTITY_TYPES)} defined")
    rprint(f"  Relation types: {len(RELATION_TYPES)} defined")
    rprint(f"  Chunks to process: {len(nodes)}")
    rprint(f"  Batch size: {batch_size}")
    rprint(f"  LLM: {GRAPH_LLM.model}")
    rprint(f"  Embed: {Settings.embed_model.model_name}")

    # ── Check for previous progress ───────────────────────
    processed = _load_progress()
    if processed:
        rprint(f"\n  [yellow]Resuming: {len(processed)} chunks already done[/yellow]")
        nodes = [n for i, n in enumerate(nodes) if i not in processed]
        rprint(f"  Remaining: {len(nodes)} chunks\n")
    else:
        rprint()

    # ── Process in batches ────────────────────────────────
    all_processed = set(processed)
    total_batches = (len(nodes) + batch_size - 1) // batch_size

    # We'll build the index incrementally: start with implicit
    # extraction only (fast), then add LLM-extracted triplets
    # batch by batch.

    Path(GRAPH_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    schema_extractor = SchemaLLMPathExtractor(
        llm=GRAPH_LLM,
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        strict=False,
        num_workers=1,
        max_triplets_per_chunk=10,
    )

    implicit_extractor = ImplicitPathExtractor()

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(nodes))
        batch = nodes[start:end]

        rprint(f"\n  [bold]Batch {batch_num + 1}/{total_batches}[/bold] ({len(batch)} chunks)")
        t0 = time.time()

        try:
            if batch_num == 0 and not processed:
                # First batch: create the index
                index = PropertyGraphIndex(
                    nodes=batch,
                    kg_extractors=[schema_extractor, implicit_extractor],
                    embed_model=Settings.embed_model,
                    show_progress=True,
                )
            else:
                if batch_num == 0:
                    # Resuming: load existing index
                    storage_context = StorageContext.from_defaults(
                        persist_dir=GRAPH_PERSIST_DIR,
                    )
                    index = PropertyGraphIndex(
                        storage_context=storage_context,
                        embed_model=Settings.embed_model,
                    )

                # Insert new batch into existing index
                index.insert_nodes(
                    batch,
                    kg_extractors=[schema_extractor, implicit_extractor],
                    show_progress=True,
                )

            elapsed = time.time() - t0
            rprint(f"  [green]Done in {elapsed:.0f}s ({elapsed/len(batch):.1f}s/chunk)[/green]")

            # Save progress after each batch
            for i in range(start, end):
                all_processed.add(i + len(processed))
            _save_progress(all_processed)

            # Persist index after each batch
            index.storage_context.persist(persist_dir=GRAPH_PERSIST_DIR)
            rprint(f"  [dim]Saved checkpoint[/dim]")

        except Exception as e:
            rprint(f"\n  [red]Batch {batch_num + 1} failed: {type(e).__name__}[/red]")
            rprint(f"  [red]{str(e)[:200]}[/red]")
            rprint(f"\n  [yellow]Progress saved. Run again to resume.[/yellow]")

            # Save what we have so far
            _save_progress(all_processed)
            if 'index' in dir():
                try:
                    index.storage_context.persist(persist_dir=GRAPH_PERSIST_DIR)
                except Exception:
                    pass
            break

    rprint(f"\n  [green]Graph saved to {GRAPH_PERSIST_DIR}[/green]")
    _print_graph_stats(index)
    return index


def load_graph_index() -> PropertyGraphIndex:
    """Load a previously built graph index from disk."""
    if not Path(GRAPH_PERSIST_DIR).exists():
        raise ValueError(f"No graph at {GRAPH_PERSIST_DIR}. Run build_graph_index() first.")

    rprint(f"  Loading graph from {GRAPH_PERSIST_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=GRAPH_PERSIST_DIR)
    index = PropertyGraphIndex(
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    _print_graph_stats(index)
    return index


def _load_progress() -> set:
    """Load set of processed chunk indices."""
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def _save_progress(processed: set):
    """Save processed chunk indices."""
    Path(GRAPH_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(processed), f)


def _print_graph_stats(index: PropertyGraphIndex):
    """Print summary statistics about the knowledge graph."""
    try:
        graph_store = index.property_graph_store
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


def reset_progress():
    """Clear all graph data and progress to start fresh."""
    import shutil
    if Path(GRAPH_PERSIST_DIR).exists():
        shutil.rmtree(GRAPH_PERSIST_DIR)
    rprint("  [yellow]Graph data and progress cleared.[/yellow]")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from loader import load_codebase
    from chunker import chunk_documents

    if "--reset" in sys.argv:
        reset_progress()
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