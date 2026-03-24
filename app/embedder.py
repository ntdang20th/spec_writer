# spec-writer/app/embedder.py
"""
Step 2.3 + 2.4: Embed chunks and store in ChromaDB.
Uses nomic-embed-text via Ollama for embeddings.
Persists to ChromaDB so we don't re-embed every time.
"""
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich import print as rprint
import config  # noqa: F401 — triggers Settings.embed_model setup


def build_index(nodes: list[TextNode]) -> VectorStoreIndex:
    """
    Embed all chunks and store them in a persistent ChromaDB collection.
    Returns a VectorStoreIndex ready for querying.
    """
    # ── Connect to ChromaDB ───────────────────────────────
    rprint("[bold]Setting up ChromaDB...[/bold]")
    client = chromadb.HttpClient(host="chromadb", port=8000)
    collection = client.get_or_create_collection(
        name="codebase",
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    rprint(f"  Collection 'codebase': {collection.count()} existing vectors")

    # ── Set up vector store ───────────────────────────────
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Build index (embeds + stores) ─────────────────────
    embed_model = Settings.embed_model
    rprint(f"\n[bold]Embedding {len(nodes)} chunks...[/bold]")
    rprint(f"  Model: {embed_model.model_name}")
    rprint(f"  This will take a while on first run...\n")

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    rprint(f"\n  [green]Done![/green] Collection now has {collection.count()} vectors")
    return index


def load_index() -> VectorStoreIndex:
    """
    Load an existing index from ChromaDB (no re-embedding).
    Use this after the first build_index() call.
    """
    client = chromadb.HttpClient(host="chromadb", port=8000)
    collection = client.get_or_create_collection(
        name="codebase",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        raise ValueError("No vectors in ChromaDB. Run build_index() first.")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = Settings.embed_model

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    rprint(f"  Loaded index with {collection.count()} vectors from ChromaDB")
    return index


# ── Run directly to test ───────────────────────────────────
if __name__ == "__main__":
    from loader import load_codebase
    from chunker import chunk_documents

    rprint("\n[bold]Step 1: Loading documents...[/bold]\n")
    docs = load_codebase()
    rprint(f"  Loaded {len(docs)} documents")

    rprint("\n[bold]Step 2: Chunking...[/bold]\n")
    nodes = chunk_documents(docs)

    rprint(f"\n[bold]Step 3: Embedding + storing...[/bold]\n")
    index = build_index(nodes)

    rprint("\n[bold green]Pipeline complete![/bold green] Ready for queries.\n")