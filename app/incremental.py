# spec-writer/app/incremental.py
"""
Incremental indexing: only re-index files that changed since last run.
Tracks file hashes to detect changes. Works with both vector and graph indices.

Usage:
  python incremental.py              → index only changed files
  python incremental.py --full       → force full re-index
  python incremental.py --status     → show what changed without indexing
"""
import os
import sys
import json
import hashlib
import time
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from llama_index.core import Settings
from rich import print as rprint
from loader import load_codebase, CODEBASE
from chunker import chunk_documents
import config  # noqa: F401

HASH_FILE = "/app/data/file_hashes.json"


def _hash_file(filepath: str) -> str:
    """Fast MD5 hash of file content."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_hashes() -> dict:
    """Load previous file hashes."""
    if Path(HASH_FILE).exists():
        with open(HASH_FILE) as f:
            return json.load(f)
    return {}


def _save_hashes(hashes: dict):
    """Save current file hashes."""
    Path(HASH_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def detect_changes() -> dict:
    """
    Compare current files against stored hashes.
    Returns: {
        "added": [file_paths],
        "modified": [file_paths],
        "deleted": [file_paths],
        "unchanged": count
    }
    """
    old_hashes = _load_hashes()
    new_hashes = {}
    
    # Scan current codebase
    docs = load_codebase()
    for doc in docs:
        rel_path = doc.metadata["file_path"]
        full_path = os.path.join(CODEBASE, rel_path)
        if os.path.exists(full_path):
            new_hashes[rel_path] = _hash_file(full_path)

    added = [f for f in new_hashes if f not in old_hashes]
    deleted = [f for f in old_hashes if f not in new_hashes]
    modified = [
        f for f in new_hashes
        if f in old_hashes and new_hashes[f] != old_hashes[f]
    ]
    unchanged = len(new_hashes) - len(added) - len(modified)

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "unchanged": unchanged,
        "new_hashes": new_hashes,
    }


def incremental_index(update_vector: bool = True, update_graph: bool = True):
    """
    Index only files that changed since last run.
    - New/modified files: chunk, embed, and extract graph
    - Deleted files: remove from vector store (graph cleanup is manual)
    """
    rprint("\n[bold]Incremental indexing[/bold]\n")

    changes = detect_changes()
    added = changes["added"]
    modified = changes["modified"]
    deleted = changes["deleted"]

    rprint(f"  Added:     [green]{len(added)}[/green] files")
    rprint(f"  Modified:  [yellow]{len(modified)}[/yellow] files")
    rprint(f"  Deleted:   [red]{len(deleted)}[/red] files")
    rprint(f"  Unchanged: {changes['unchanged']} files\n")

    changed_paths = set(added + modified)

    if not changed_paths and not deleted:
        rprint("  [green]Everything up to date. Nothing to index.[/green]\n")
        _save_hashes(changes["new_hashes"])
        return

    # Load only changed files
    if changed_paths:
        rprint(f"  [bold]Loading {len(changed_paths)} changed files...[/bold]")
        all_docs = load_codebase()
        changed_docs = [d for d in all_docs if d.metadata["file_path"] in changed_paths]
        rprint(f"  Loaded {len(changed_docs)} documents")

        rprint(f"\n  [bold]Chunking...[/bold]")
        nodes = chunk_documents(changed_docs)
        rprint(f"  Created {len(nodes)} chunks")

        # Update vector index
        if update_vector and nodes:
            rprint(f"\n  [bold]Updating vector index...[/bold]")
            t0 = time.time()
            try:
                from embedder import build_index, load_index
                # For changed files, we add new vectors
                # ChromaDB handles deduplication by node ID
                import chromadb
                from llama_index.vector_stores.chroma import ChromaVectorStore
                from llama_index.core import StorageContext, VectorStoreIndex

                client = chromadb.HttpClient(host="chromadb", port=8000)
                collection = client.get_or_create_collection(
                    name="codebase",
                    metadata={"hnsw:space": "cosine"},
                )

                # Delete old vectors for modified files
                if modified:
                    rprint(f"    Removing old vectors for {len(modified)} modified files...")
                    for path in modified:
                        try:
                            collection.delete(where={"file_path": path})
                        except Exception:
                            pass

                # Add new vectors
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model,
                    show_progress=True,
                )
                elapsed = time.time() - t0
                rprint(f"    [green]Vector index updated in {elapsed:.0f}s[/green]")
                rprint(f"    Collection now has {collection.count()} vectors")
            except Exception as e:
                rprint(f"    [red]Vector update failed: {e}[/red]")

        # Update graph index
        if update_graph and nodes:
            rprint(f"\n  [bold]Updating knowledge graph...[/bold]")
            t0 = time.time()
            try:
                from graph_builder import load_graph_index
                index = load_graph_index()
                index.insert_nodes(nodes)
                elapsed = time.time() - t0
                rprint(f"    [green]Graph updated in {elapsed:.0f}s[/green]")
            except Exception as e:
                rprint(f"    [red]Graph update failed: {e}[/red]")

    # Handle deleted files
    if deleted and update_vector:
        rprint(f"\n  [bold]Removing {len(deleted)} deleted files from vector index...[/bold]")
        try:
            import chromadb
            client = chromadb.HttpClient(host="chromadb", port=8000)
            collection = client.get_or_create_collection("codebase")
            for path in deleted:
                try:
                    collection.delete(where={"file_path": path})
                except Exception:
                    pass
            rprint(f"    [green]Removed vectors for {len(deleted)} files[/green]")
        except Exception as e:
            rprint(f"    [red]Failed to remove deleted files: {e}[/red]")

    # Save new hashes
    _save_hashes(changes["new_hashes"])
    rprint(f"\n  [bold green]Incremental indexing complete![/bold green]\n")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    if "--status" in sys.argv:
        rprint("\n[bold]Change detection[/bold]\n")
        changes = detect_changes()
        rprint(f"  Added:     [green]{len(changes['added'])}[/green]")
        rprint(f"  Modified:  [yellow]{len(changes['modified'])}[/yellow]")
        rprint(f"  Deleted:   [red]{len(changes['deleted'])}[/red]")
        rprint(f"  Unchanged: {changes['unchanged']}")

        if changes["added"]:
            rprint(f"\n  [green]Added files:[/green]")
            for f in changes["added"][:20]:
                rprint(f"    + {f}")
            if len(changes["added"]) > 20:
                rprint(f"    ... and {len(changes['added']) - 20} more")

        if changes["modified"]:
            rprint(f"\n  [yellow]Modified files:[/yellow]")
            for f in changes["modified"][:20]:
                rprint(f"    ~ {f}")
            if len(changes["modified"]) > 20:
                rprint(f"    ... and {len(changes['modified']) - 20} more")

        if changes["deleted"]:
            rprint(f"\n  [red]Deleted files:[/red]")
            for f in changes["deleted"][:20]:
                rprint(f"    - {f}")
        rprint()

    elif "--full" in sys.argv:
        rprint("\n[bold]Force full re-index[/bold]\n")
        if Path(HASH_FILE).exists():
            os.remove(HASH_FILE)
        incremental_index()

    else:
        incremental_index()