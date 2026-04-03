# spec-writer/app/config.py
"""
Centralised model configuration.
Change the provider/model here — everything else stays the same.
"""
import os

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ── Ollama connection ──────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── LLM (generation model) ────────────────────────────────
# This is the model that writes your specs.
# Swap to any Ollama model: "llama3.2", "deepseek-coder-v2",
# "mistral", "codellama", "qwen2.5-coder"
#
# For Claude API (Phase 5), switch to:
#   from llama_index.llms.anthropic import Anthropic
#   Settings.llm = Anthropic(model="claude-sonnet-4-20250514")

Settings.llm = Ollama(
    model=os.getenv("LLM_MODEL", "llama3.2"),
    base_url=OLLAMA_URL,
    request_timeout=300.0,  # 5 min — graph extraction sends big prompts
    temperature=0.1,        # low temp for consistent specs
)

# ── Embedding model ────────────────────────────────────────
# This converts text to vectors for similarity search.
# nomic-embed-text is the best free local option.
# Alternative: "mxbai-embed-large" (better quality, slower)

Settings.embed_model = OllamaEmbedding(
    model_name=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    base_url=OLLAMA_URL,
)

# ── Chunking defaults ─────────────────────────────────────
Settings.chunk_size = 1024     # tokens per chunk
Settings.chunk_overlap = 128   # overlap between chunks

# ── Verification ───────────────────────────────────────────
def verify_setup():
    """Quick check that Ollama is reachable and models are pulled."""
    from rich import print as rprint

    rprint("\n[bold]Spec Writer — Config Check[/bold]\n")
    rprint(f"  Ollama URL:      {OLLAMA_URL}")
    rprint(f"  LLM model:       {Settings.llm.model}")
    rprint(f"  Embed model:     {Settings.embed_model.model_name}")
    rprint(f"  Chunk size:      {Settings.chunk_size}")
    rprint(f"  Chunk overlap:   {Settings.chunk_overlap}")

    # Test LLM
    try:
        resp = Settings.llm.complete("Say 'hello' in one word.")
        rprint(f"  LLM test:        [green]OK[/green] → {resp.text.strip()[:50]}")
    except Exception as e:
        rprint(f"  LLM test:        [red]FAIL[/red] → {e}")

    # Test embedding
    try:
        vec = Settings.embed_model.get_text_embedding("test")
        rprint(f"  Embed test:      [green]OK[/green] → vector dim={len(vec)}")
    except Exception as e:
        rprint(f"  Embed test:      [red]FAIL[/red] → {e}")

    rprint()


if __name__ == "__main__":
    verify_setup()