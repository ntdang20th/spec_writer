# spec-writer/app/config.py
"""
Centralised model configuration.
Change the provider/model here — everything else stays the same.
"""
import os
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ── Ollama connection ──────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── LLM (generation model) ────────────────────────────────
Settings.llm = Ollama(
    model=os.getenv("LLM_MODEL", "llama3.2"),
    base_url=OLLAMA_URL,
    request_timeout=120.0,
    temperature=0.1,
    context_window=32768,  # max for RTX 3060 12GB — tested at 99% GPU
)

# ── Embedding model ────────────────────────────────────────
Settings.embed_model = OllamaEmbedding(
    model_name=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    base_url=OLLAMA_URL,
)

# ── Query defaults ─────────────────────────────────────────
DEFAULT_TOP_K = 8  # max chunks that fit in 32K context

# ── Chunking defaults ─────────────────────────────────────
Settings.chunk_size = 1024
Settings.chunk_overlap = 128

# ── Verification ───────────────────────────────────────────
def verify_setup():
    """Quick check that Ollama is reachable and models are pulled."""
    from rich import print as rprint

    rprint("\n[bold]Spec Writer — Config Check[/bold]\n")
    rprint(f"  Ollama URL:      {OLLAMA_URL}")
    rprint(f"  LLM model:       {Settings.llm.model}")
    rprint(f"  Context window:  {Settings.llm.context_window}")
    rprint(f"  Embed model:     {Settings.embed_model.model_name}")
    rprint(f"  Chunk size:      {Settings.chunk_size}")
    rprint(f"  Chunk overlap:   {Settings.chunk_overlap}")
    rprint(f"  Default top_k:   {DEFAULT_TOP_K}")

    try:
        resp = Settings.llm.complete("Say 'hello' in one word.")
        rprint(f"  LLM test:        [green]OK[/green] → {resp.text.strip()[:50]}")
    except Exception as e:
        rprint(f"  LLM test:        [red]FAIL[/red] → {e}")

    try:
        vec = Settings.embed_model.get_text_embedding("test")
        rprint(f"  Embed test:      [green]OK[/green] → vector dim={len(vec)}")
    except Exception as e:
        rprint(f"  Embed test:      [red]FAIL[/red] → {e}")

    rprint()


if __name__ == "__main__":
    verify_setup()