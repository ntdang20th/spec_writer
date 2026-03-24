# spec-writer/app/loader.py
"""
Load Helpdesk-BED codebase into LlamaIndex Documents.
Each file becomes one Document with rich metadata.
"""
import os
from pathlib import Path
from llama_index.core import Document
from rich import print as rprint

CODEBASE = "/app/codebase"

# File types we care about
INDEX_EXTENSIONS = {".cs", ".json", ".md", ".yaml", ".yml"}

# Directories to skip
SKIP_DIRS = {"bin", "obj", "node_modules", ".git", ".vs", "packages",
             "TestResults", "Migrations"}

# Files to skip (too noisy, no spec value)
SKIP_FILES = {"AssemblyInfo.cs", "GlobalUsings.cs", ".editorconfig"}


def load_codebase(root: str = CODEBASE) -> list[Document]:
    """Walk the codebase and create a Document per file."""
    docs = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in INDEX_EXTENSIONS:
                continue
            if fname in SKIP_FILES:
                continue

            filepath = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(filepath, root)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                rprint(f"  [yellow]Skip[/yellow] {rel_path}: {e}")
                continue

            # Skip empty or tiny files
            if len(content.strip()) < 50:
                continue

            # Extract metadata from path
            # e.g. "src/Premex.FinanceBridge.Api/Controllers/InvoiceController.cs"
            parts = Path(rel_path).parts
            project = parts[0] if len(parts) > 1 else "root"

            # Detect file category from path/name patterns
            category = _detect_category(rel_path, content)

            doc = Document(
                text=content,
                metadata={
                    "file_path": rel_path,
                    "file_name": fname,
                    "extension": ext,
                    "project": project,
                    "category": category,
                    "language": "csharp" if ext == ".cs" else ext.lstrip("."),
                },
                excluded_llm_metadata_keys=["extension", "language"],
                excluded_embed_metadata_keys=["extension"],
            )
            docs.append(doc)

    return docs


def _detect_category(path: str, content: str) -> str:
    """Detect the role of a file based on path and content patterns."""
    p = path.lower()

    if "handler" in p or "consumer" in p:
        return "handler"
    if "controller" in p:
        return "controller"
    if "model" in p or "dto" in p or "request" in p or "response" in p:
        return "model"
    if "service" in p or "client" in p:
        return "service"
    if "repository" in p or "repo" in p:
        return "repository"
    if "test" in p or "spec" in p:
        return "test"
    if "migration" in p:
        return "migration"
    if "extension" in p or "helper" in p or "util" in p:
        return "utility"
    if "startup" in p or "program" in p or "config" in p:
        return "configuration"
    if "interface" in p or content.strip().startswith("public interface"):
        return "interface"
    if "appsettings" in p or path.endswith(".json"):
        return "config_file"

    return "other"


# ── Run directly to test ───────────────────────────────────
if __name__ == "__main__":
    rprint("\n[bold]Loading Helpdesk-BED codebase...[/bold]\n")
    docs = load_codebase()

    # Summary by category
    cats = {}
    for d in docs:
        cat = d.metadata["category"]
        cats[cat] = cats.get(cat, 0) + 1

    rprint(f"  Total documents: [green]{len(docs)}[/green]\n")
    rprint(f"  {'Category':20s}  Count")
    rprint(f"  {'-'*20:20s}  -----")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        rprint(f"  {cat:20s}  {count}")

    # Show a few examples
    rprint(f"\n  [bold]Sample documents:[/bold]")
    for d in docs[:5]:
        m = d.metadata
        rprint(f"  {m['category']:12s}  {m['file_path'][:60]}")
    rprint()