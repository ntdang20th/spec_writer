# spec-writer/app/chunker.py
"""
Smart chunking for C# codebases.
Uses tree-sitter for AST-aware splitting by class/method boundaries.
Falls back to token-based splitting on parse errors.
"""
import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser
from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import TextNode
from rich import print as rprint


def _make_cs_parser() -> Parser:
    """Build a tree-sitter parser for C# from the installed grammar."""
    parser = Parser(Language(tscsharp.language()))
    return parser


def chunk_documents(docs: list[Document]) -> list[TextNode]:
    """
    Chunk C# documents using tree-sitter AST-aware splitting.
    Non-C# files are skipped (JSON/config can be added later).
    """
    cs_docs = [d for d in docs if d.metadata.get("extension") == ".cs"]
    skipped = len(docs) - len(cs_docs)

    if skipped:
        rprint(f"  Skipped {skipped} non-C# files")

    nodes = []

    if cs_docs:
        try:
            parser = _make_cs_parser()
            cs_splitter = CodeSplitter(
                language="c_sharp",
                parser=parser,
                chunk_lines=60,
                chunk_lines_overlap=5,
                max_chars=3000,
            )
            nodes = cs_splitter.get_nodes_from_documents(cs_docs)
            rprint(f"  C# files:  {len(cs_docs):4d} docs → [green]{len(nodes):5d} chunks[/green] (AST-aware)")
        except Exception as e:
            rprint(f"  [yellow]AST chunking failed, falling back to text splitting: {e}[/yellow]")
            fallback = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
            nodes = fallback.get_nodes_from_documents(cs_docs)
            rprint(f"  C# files:  {len(cs_docs):4d} docs → [yellow]{len(nodes):5d} chunks[/yellow] (fallback)")

    rprint(f"  {'─'*45}")
    rprint(f"  Total:     {len(cs_docs):4d} docs → [bold green]{len(nodes):5d} chunks[/bold green]")

    return nodes


# ── Run directly to test ───────────────────────────────────
if __name__ == "__main__":
    from loader import load_codebase

    rprint("\n[bold]Step 1: Loading documents...[/bold]\n")
    docs = load_codebase()
    rprint(f"\n  Loaded {len(docs)} documents\n")

    rprint("[bold]Step 2: Chunking...[/bold]\n")
    nodes = chunk_documents(docs)

    sizes = [len(n.text) for n in nodes]
    avg = sum(sizes) // len(sizes) if sizes else 0
    rprint(f"\n  Chunk sizes: min={min(sizes)}, avg={avg}, max={max(sizes)} chars")

    rprint(f"\n  [bold]Sample chunks:[/bold]\n")
    for n in nodes[:5]:
        m = n.metadata
        preview = n.text[:80].replace("\n", " ").strip()
        rprint(f"  [{m.get('category','?'):12s}] {m.get('file_name','?'):30s} → \"{preview}...\"")
    rprint()