# spec-writer/app/query.py
"""
Step 2.5: Query the vector store using RAG.
Retrieves relevant code chunks, sends them + your question to the LLM.
"""
from llama_index.core import Settings
from rich import print as rprint
from embedder import load_index
import config  # noqa: F401 — triggers Settings setup


def ask(question: str, top_k: int = 5):
    """Ask a question about the codebase using RAG."""
    rprint(f"\n[bold]Question:[/bold] {question}\n")

    # Load existing index from ChromaDB (no re-embedding)
    index = load_index()

    # Build query engine
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,    # retrieve top-k most relevant chunks
        llm=Settings.llm,
    )

    # Query
    rprint("[dim]Retrieving relevant code + generating answer...[/dim]\n")
    response = query_engine.query(question)

    # Show answer
    rprint(f"[bold green]Answer:[/bold green]\n{response.response}\n")

    # Show which chunks were used
    rprint(f"[bold]Sources ({len(response.source_nodes)} chunks):[/bold]\n")
    for i, node in enumerate(response.source_nodes, 1):
        m = node.metadata
        score = node.score or 0
        preview = node.text[:100].replace("\n", " ").strip()
        rprint(f"  {i}. [{m.get('category','?'):12s}] {m.get('file_name','?')}")
        rprint(f"     Score: {score:.4f} | \"{preview}...\"\n")


if __name__ == "__main__":
    ask("How does PurchaseInvoice work?")