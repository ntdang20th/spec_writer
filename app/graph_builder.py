# spec-writer/app/graph_builder.py
"""
Phase 3: Build a knowledge graph from your codebase using PropertyGraphIndex.
Extracts entities and relationships from code chunks using the LLM,
stores them in a graph, and enables hybrid vector+graph retrieval.
"""
import os
import json
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
from rich import print as rprint
import config  # noqa: F401


# ── 3.2: Entity types for .NET/C# codebases ──────────────
# These guide the LLM to extract the right kinds of nodes
ENTITY_TYPES = [
    "Class",
    "Interface",
    "Method",
    "Service",
    "Repository",
    "Controller",
    "Handler",
    "Model",
    "DTO",
    "Table",
    "API_Endpoint",
    "Configuration",
    "Middleware",
    "Extension",
]

# ── 3.3: Relationship types ──────────────────────────────
# These tell the LLM what connections to look for
RELATION_TYPES = [
    "DEPENDS_ON",
    "IMPLEMENTS",
    "CALLS",
    "READS_FROM",
    "WRITES_TO",
    "RETURNS",
    "ACCEPTS",
    "INHERITS",
    "HAS_METHOD",
    "HAS_PROPERTY",
    "INJECTS",
    "CONFIGURES",
    "MAPS_TO",
]

# ── Graph persistence path ────────────────────────────────
GRAPH_PERSIST_DIR = "/app/data/graph"


def build_graph_index(nodes: list[TextNode]) -> PropertyGraphIndex:
    """
    Build a PropertyGraphIndex from code chunks.

    The LLM reads each chunk and extracts (entity)-[relationship]-(entity)
    triplets. These get stored in a graph alongside the vector embeddings.

    Args:
        nodes: Chunked code from your codebase (output of chunk_documents)

    Returns:
        PropertyGraphIndex with both graph structure and vector search
    """
    rprint("\n[bold]Building knowledge graph...[/bold]\n")
    rprint(f"  Entity types:   {len(ENTITY_TYPES)} defined")
    rprint(f"  Relation types: {len(RELATION_TYPES)} defined")
    rprint(f"  Chunks to process: {len(nodes)}")
    rprint(f"  LLM: {Settings.llm.model}")
    rprint(f"  Embed: {Settings.embed_model.model_name}")
    rprint()

    # ── Schema-guided extraction ──────────────────────────
    # SchemaLLMPathExtractor uses the LLM to extract triplets
    # guided by our entity and relation types.
    schema_extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        strict=False,        # allow LLM to find types outside our list too
        num_workers=1,       # sequential — safer for single GPU
        max_triplets_per_chunk=10,  # reduce to speed up extraction
    )

    # ── Implicit extraction ───────────────────────────────
    # Also extracts relationships from node metadata (file_path, category)
    # This adds structural connections for free without LLM calls.
    implicit_extractor = ImplicitPathExtractor()

    rprint("[bold]Extracting entities and relationships...[/bold]")
    rprint("  This takes a while — the LLM reads every chunk.\n")

    # ── Build the index ───────────────────────────────────
    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[schema_extractor, implicit_extractor],
        embed_model=Settings.embed_model,
        show_progress=True,
    )

    # ── Persist ───────────────────────────────────────────
    Path(GRAPH_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=GRAPH_PERSIST_DIR)
    rprint(f"\n  [green]Graph saved to {GRAPH_PERSIST_DIR}[/green]")

    # ── Stats ─────────────────────────────────────────────
    _print_graph_stats(index)

    return index


def load_graph_index() -> PropertyGraphIndex:
    """
    Load a previously built graph index from disk.
    Use this after the first build_graph_index() call.
    """
    if not Path(GRAPH_PERSIST_DIR).exists():
        raise ValueError(
            f"No graph found at {GRAPH_PERSIST_DIR}. "
            "Run build_graph_index() first."
        )

    rprint(f"  Loading graph from {GRAPH_PERSIST_DIR}...")

    storage_context = StorageContext.from_defaults(
        persist_dir=GRAPH_PERSIST_DIR,
    )

    index = PropertyGraphIndex(
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )

    _print_graph_stats(index)
    return index


def _print_graph_stats(index: PropertyGraphIndex):
    """Print summary statistics about the knowledge graph."""
    try:
        graph_store = index.property_graph_store
        # Try to get triplet count
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

        # Show top entity types if available
        if triplets:
            rprint(f"\n  [bold]Sample triplets:[/bold]")
            for subj, rel, obj in triplets[:8]:
                rprint(f"    {subj} → [{rel}] → {obj}")
    except Exception as e:
        rprint(f"  [yellow]Could not read graph stats: {e}[/yellow]")


# ── Run directly to build the graph ───────────────────────
if __name__ == "__main__":
    import sys
    from loader import load_codebase
    from chunker import chunk_documents

    rprint("\n[bold]Phase 3: Building Knowledge Graph[/bold]")
    rprint("=" * 45)

    # Optional: limit chunks for testing
    # Usage: python graph_builder.py          → first 100 chunks (test)
    #        python graph_builder.py --all    → all chunks (full build)
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
    index = build_graph_index(nodes)

    rprint("\n[bold green]Graph build complete![/bold green]\n")