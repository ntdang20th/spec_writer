# spec-writer/app/chunker.py
"""
Smart chunking for C# codebases using tree-sitter AST parsing.
Splits by class/method boundaries instead of arbitrary character counts.
Falls back to token-based splitting for non-C# files.
"""
from llama_index.core import Document
from llama_index.core.node_parser import (
    CodeSplitter,
    SentenceSplitter,
)
from llama_index.core.schema import TextNode
from rich import print as rprint


def chunk_documents(docs: list[Document]) -> list[TextNode]:
    """
    Chunk documents using the best strategy per file type.
    - C# files: tree-sitter AST-aware splitting (class/method level)
    - JSON/MD/other: sentence/token-based splitting with overlap
    """
    cs_docs = [d for d in docs if d.metadata.get("extension") == ".cs"]
    other_docs = [d for d in docs if d.metadata.get("extension") != ".cs"]

    nodes = []

    # ── C# files: AST-aware chunking ──────────────────────
    if cs_docs:
        cs_splitter = CodeSplitter(
            language="c_sharp",
            chunk_lines=60,        # target ~60 lines per chunk
            chunk_lines_overlap=5, # small overlap for context
            max_chars=3000,        # hard cap per chunk
        )
        try:
            cs_nodes = cs_splitter.get_nodes_from_documents(cs_docs)
            nodes.extend(cs_nodes)
            rprint(f"  C# files:    {len(cs_docs):4d} docs → [green]{len(cs_nodes):5d} chunks[/green] (AST-aware)")
        except Exception as e:
            rprint(f"  [yellow]AST chunking failed, falling back to text splitting: {e}[/yellow]")
            fallback = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
            cs_nodes = fallback.get_nodes_from_documents(cs_docs)
            nodes.extend(cs_nodes)
            rprint(f"  C# files:    {len(cs_docs):4d} docs → [yellow]{len(cs_nodes):5d} chunks[/yellow] (fallback)")

    # ── Other files: token-based splitting ─────────────────
    if other_docs:
        text_splitter = SentenceSplitter(
            chunk_size=1024,     # ~1024 tokens per chunk
            chunk_overlap=128,   # overlap for context continuity
        )
        other_nodes = text_splitter.get_nodes_from_documents(other_docs)
        nodes.extend(other_nodes)
        rprint(f"  Other files: {len(other_docs):4d} docs → [green]{len(other_nodes):5d} chunks[/green] (text split)")

    rprint(f"  {'─'*45}")
    rprint(f"  Total:       {len(docs):4d} docs → [bold green]{len(nodes):5d} chunks[/bold green]")

    return nodes


# ── Run directly to test ───────────────────────────────────
if __name__ == "__main__":
    from loader import load_codebase

    rprint("\n[bold]Step 1: Loading documents...[/bold]\n")
    docs = load_codebase()
    rprint(f"\n  Loaded {len(docs)} documents\n")

    rprint("[bold]Step 2: Chunking...[/bold]\n")
    nodes = chunk_documents(docs)

    # Show chunk size distribution
    sizes = [len(n.text) for n in nodes]
    avg = sum(sizes) // len(sizes) if sizes else 0
    rprint(f"\n  Chunk sizes: min={min(sizes)}, avg={avg}, max={max(sizes)} chars")

    # Show sample chunks with metadata
    rprint(f"\n  [bold]Sample chunks:[/bold]\n")
    for n in nodes[:5]:
        m = n.metadata
        preview = n.text[:80].replace("\n", " ").strip()
        rprint(f"  [{m.get('category','?'):12s}] {m.get('file_name','?'):30s} → \"{preview}...\"")
    rprint()