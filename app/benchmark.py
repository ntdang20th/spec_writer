# spec-writer/app/benchmark.py
"""
Phase 5: Benchmark spec quality across different LLM models.
Swaps models at runtime — no restart needed.
"""
import sys
import time
import os
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from rich import print as rprint
from spec_writer import generate_spec, print_spec, export_spec_markdown
import config  # noqa: F401

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Models to benchmark ───────────────────────────────────
# Pull these first:  ollama pull <model>
MODELS = {
    "llama3.2":           "Default 3B — fast, basic quality",
    "qwen2.5-coder:7b":  "7B code-focused — better at code patterns",
    "deepseek-coder-v2:lite": "16B lite — strong code understanding",
    "mistral:7b":         "7B general — good structured output",
    "llama3.1:8b":        "8B — better reasoning than 3.2",
}


def swap_model(model_name: str):
    """Hot-swap the generation LLM without restarting."""
    Settings.llm = Ollama(
        model=model_name,
        base_url=OLLAMA_URL,
        request_timeout=600.0,
        temperature=0.1,
    )
    rprint(f"  [green]Switched to: {model_name}[/green]")


def benchmark_single(model_name: str, feature: str):
    """Run spec generation with a specific model."""
    rprint(f"\n{'='*60}")
    rprint(f"[bold]Model: {model_name}[/bold]")
    if model_name in MODELS:
        rprint(f"[dim]{MODELS[model_name]}[/dim]")
    rprint(f"{'='*60}")

    swap_model(model_name)

    t0 = time.time()
    try:
        spec = generate_spec(feature)
        elapsed = time.time() - t0

        rprint(f"\n  [green]Generated in {elapsed:.0f}s[/green]")
        print_spec(spec)

        # Score the output
        score = 0
        if spec.entities_affected:
            score += len(spec.entities_affected)
        if spec.api_contracts:
            score += len(spec.api_contracts)
        if spec.implementation_steps:
            score += len(spec.implementation_steps)
        if spec.test_cases:
            score += len(spec.test_cases)
        if spec.data_model_changes:
            score += 1
        if spec.dependencies:
            score += 1

        return {
            "model": model_name,
            "time": elapsed,
            "entities": len(spec.entities_affected),
            "apis": len(spec.api_contracts),
            "steps": len(spec.implementation_steps),
            "tests": len(spec.test_cases),
            "richness_score": score,
            "success": True,
        }

    except Exception as e:
        elapsed = time.time() - t0
        rprint(f"\n  [red]Failed after {elapsed:.0f}s: {e}[/red]")
        return {
            "model": model_name,
            "time": elapsed,
            "richness_score": 0,
            "success": False,
        }


def benchmark_all(feature: str, models: list[str] = None):
    """Run spec generation across multiple models and compare."""
    if models is None:
        models = list(MODELS.keys())

    rprint(f"\n[bold]Benchmarking {len(models)} models[/bold]")
    rprint(f"Feature: {feature}\n")

    results = []
    for model in models:
        result = benchmark_single(model, feature)
        results.append(result)

    # ── Summary table ─────────────────────────────────────
    rprint(f"\n{'='*60}")
    rprint(f"[bold]Benchmark results[/bold]")
    rprint(f"{'='*60}\n")

    rprint(f"  {'Model':30s} {'Time':>6s} {'Entities':>8s} {'APIs':>5s} {'Steps':>6s} {'Tests':>6s} {'Score':>6s}")
    rprint(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")

    for r in sorted(results, key=lambda x: -x["richness_score"]):
        if r["success"]:
            rprint(
                f"  {r['model']:30s} {r['time']:5.0f}s "
                f"{r.get('entities', 0):8d} {r.get('apis', 0):5d} "
                f"{r.get('steps', 0):6d} {r.get('tests', 0):6d} "
                f"{r['richness_score']:6d}"
            )
        else:
            rprint(f"  {r['model']:30s} {r['time']:5.0f}s  [red]FAILED[/red]")

    # Restore default model
    swap_model(os.getenv("LLM_MODEL", "llama3.2"))
    rprint(f"\n  [dim]Restored default model[/dim]\n")


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        rprint("\n[bold]Usage:[/bold]")
        rprint("  python benchmark.py \"feature description\"               → benchmark all models")
        rprint("  python benchmark.py --model qwen2.5-coder:7b \"feature\"  → single model")
        rprint("  python benchmark.py --list                               → show available models")
        sys.exit(0)

    if args[0] == "--list":
        rprint("\n[bold]Available models:[/bold]\n")
        for name, desc in MODELS.items():
            rprint(f"  {name:30s} {desc}")
        rprint(f"\n  Pull with: ollama pull <model>\n")
        sys.exit(0)

    if args[0] == "--model":
        model = args[1]
        feature = " ".join(args[2:])
        benchmark_single(model, feature)
    else:
        feature = " ".join(args)
        benchmark_all(feature)