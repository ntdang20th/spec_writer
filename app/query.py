# spec-writer/app/query.py
"""
Query the codebase using RAG (vector-only) or GraphRAG (vector + graph).
Supports both modes so you can compare results side by side.
"""
import sys
from llama_index.core import Settings
from rich import print as rprint
from embedder import load_index
import config  # noqa: F401


def ask_vector(question: str, top_k: int = 5):
    """Phase 2 approach: vector-only RAG query."""
    rprint(f"\n[bold blue]── Vector RAG Query ──[/bold blue]")
    rprint(f"[bold]Question:[/bold] {question}\n")

    index = load_index()
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        llm=Settings.llm,
    )

    rprint("[dim]Retrieving via vector similarity...[/dim]\n")
    response = query_engine.query(question)

    rprint(f"[bold green]Answer:[/bold green]\n{response.response}\n")
    _print_sources(response.source_nodes)
    return response


def ask_graph(question: str, top_k: int = 5):
    """Phase 3 approach: GraphRAG hybrid query (vector + graph traversal)."""
    from graph_builder import load_graph_index

    rprint(f"\n[bold purple]── GraphRAG Hybrid Query ──[/bold purple]")
    rprint(f"[bold]Question:[/bold] {question}\n")

    index = load_graph_index()

    # The graph query engine combines:
    # 1. Vector similarity search (finds relevant chunks)
    # 2. Graph traversal (follows entity relationships)
    # 3. LLM synthesis (generates answer from combined context)
    query_engine = index.as_query_engine(
        llm=Settings.llm,
        include_text=True,     # include original chunk text in context
        similarity_top_k=top_k,
    )

    rprint("[dim]Retrieving via vector + graph traversal...[/dim]\n")
    response = query_engine.query(question)

    rprint(f"[bold green]Answer:[/bold green]\n{response.response}\n")
    _print_sources(response.source_nodes)
    return response


def ask_compare(question: str, top_k: int = 5):
    """Run both vector and graph queries to compare results."""
    rprint(f"\n{'='*60}")
    rprint(f"[bold]Comparing RAG vs GraphRAG[/bold]")
    rprint(f"{'='*60}")

    rprint("\n[dim]Running vector-only query...[/dim]")
    v_resp = ask_vector(question, top_k)

    rprint(f"\n{'─'*60}\n")

    rprint("[dim]Running graph+vector hybrid query...[/dim]")
    g_resp = ask_graph(question, top_k)

    rprint(f"\n{'='*60}")
    rprint(f"[bold]Summary[/bold]")
    rprint(f"{'='*60}")
    rprint(f"\n  Vector sources: {len(v_resp.source_nodes)} chunks")
    rprint(f"  Graph sources:  {len(g_resp.source_nodes)} chunks")
    rprint(f"\n  Compare the answers above — GraphRAG should include")
    rprint(f"  more context about related entities and dependencies.\n")


def _print_sources(source_nodes):
    """Print source chunks used to generate the answer."""
    rprint(f"[bold]Sources ({len(source_nodes)} chunks):[/bold]\n")
    for i, node in enumerate(source_nodes, 1):
        m = node.metadata if hasattr(node, "metadata") else {}
        score = node.score if hasattr(node, "score") else 0
        text = node.text if hasattr(node, "text") else str(node)
        preview = text[:100].replace("\n", " ").strip()
        cat = m.get("category", "?")
        fname = m.get("file_name", "?")
        rprint(f"  {i}. [{cat:12s}] {fname}")
        if score:
            rprint(f"     Score: {score:.4f} | \"{preview}...\"\n")
        else:
            rprint(f"     \"{preview}...\"\n")


# ── CLI interface ─────────────────────────────────────────
if __name__ == "__main__":
    # Usage:
    #   python query.py "your question"                  → vector only
    #   python query.py --graph "your question"          → graph only
    #   python query.py --compare "your question"        → both side by side

    args = sys.argv[1:]

    if not args:
        # Default test question
        ask_compare("What is this system ?")
    elif args[0] == "--graph":
        question = " ".join(args[1:]) or "What is this system ?"
        ask_graph(question)
    elif args[0] == "--compare":
        question = " ".join(args[1:]) or "What is this system ?"
        ask_compare(question)
    else:
        question = " ".join(args)
        ask_vector(question)