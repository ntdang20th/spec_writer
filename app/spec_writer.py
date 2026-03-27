# spec-writer/app/spec_writer.py
"""
Phase 4: Spec generation engine.
Uses GraphRAG to retrieve codebase context, then generates a structured
specification via the LLM constrained to the Pydantic schema.
"""
import sys
import json
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from rich import print as rprint
from rich.markdown import Markdown
from spec_schema import Specification
from graph_builder import load_graph_index
from embedder import load_index
import config  # noqa: F401


# ── Spec generation prompt ────────────────────────────────
SPEC_PROMPT = PromptTemplate(
    """You are a senior software architect generating a technical specification.

You have access to the following codebase context retrieved via RAG and knowledge graph:

--- CODEBASE CONTEXT ---
{context_str}
--- END CONTEXT ---

Based on the context above and your understanding of .NET/C# patterns,
generate a detailed specification for the following feature:

{query_str}

Requirements for the specification:
1. Identify ALL entities (classes, tables, services, DTOs) that need to be created or modified
2. Define API contracts with request/response shapes based on existing patterns in the codebase
3. Include message contracts if async communication is needed
4. List database schema changes
5. Provide ordered implementation steps that a developer can follow
6. Include test cases with Given/When/Then format
7. Follow the existing code patterns and conventions found in the context

Be specific — use actual class names, table names, and patterns from the codebase context.
"""
)


def generate_spec(feature_description: str, use_graph: bool = True) -> Specification:
    """
    Generate a specification for a feature using codebase context.

    Args:
        feature_description: What the feature should do
        use_graph: If True, use GraphRAG. If False, use vector-only RAG.

    Returns:
        A structured Specification object
    """
    rprint(f"\n[bold]Generating specification...[/bold]\n")
    rprint(f"  Feature: {feature_description}")
    rprint(f"  Mode: {'GraphRAG (vector + graph)' if use_graph else 'Vector RAG only'}")

    # ── Step 1: Retrieve context ──────────────────────────
    rprint(f"\n  [dim]Retrieving codebase context...[/dim]")

    if use_graph:
        index = load_graph_index()
    else:
        index = load_index()

    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(feature_description)

    context = "\n\n---\n\n".join([
        f"[{n.metadata.get('file_name', '?')} | {n.metadata.get('category', '?')}]\n{n.text}"
        for n in nodes
        if hasattr(n, 'text')
    ])

    rprint(f"  Retrieved {len(nodes)} context chunks")

    # ── Step 2: Generate structured spec ──────────────────
    rprint(f"  [dim]Generating specification via LLM...[/dim]\n")

    try:
        spec = Settings.llm.structured_predict(
            Specification,
            SPEC_PROMPT,
            context_str=context,
            query_str=feature_description,
        )
    except Exception as e:
        rprint(f"  [yellow]Structured output failed ({e}), falling back to text generation...[/yellow]")
        spec = _fallback_generate(context, feature_description)

    return spec


def _fallback_generate(context: str, feature_description: str) -> Specification:
    """Fallback: generate spec as text, then parse manually."""
    from llama_index.core.llms import ChatMessage

    prompt = SPEC_PROMPT.format(
        context_str=context,
        query_str=feature_description,
    )
    prompt += "\n\nRespond with a JSON object matching this structure: " + json.dumps(
        Specification.model_json_schema(), indent=2
    )

    response = Settings.llm.chat([ChatMessage(role="user", content=prompt)])
    text = response.message.content

    # Try to extract JSON from response
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        return Specification(**data)
    except Exception:
        # Last resort: return a minimal spec with the raw text
        return Specification(
            title=feature_description[:100],
            overview=text[:2000],
        )


def print_spec(spec: Specification):
    """Pretty-print a specification."""
    rprint(f"\n{'='*60}")
    rprint(f"[bold]SPECIFICATION: {spec.title}[/bold]")
    rprint(f"{'='*60}\n")

    rprint(f"[bold]Overview[/bold]\n{spec.overview}\n")

    if spec.entities_affected:
        rprint(f"[bold]Entities affected ({len(spec.entities_affected)})[/bold]")
        for e in spec.entities_affected:
            rprint(f"  [{e.action:9s}] {e.entity_type:12s} {e.name}")
            rprint(f"             {e.details}\n")

    if spec.api_contracts:
        rprint(f"[bold]API contracts ({len(spec.api_contracts)})[/bold]")
        for a in spec.api_contracts:
            rprint(f"  {a.method:6s} {a.path}")
            rprint(f"         {a.description}")
            if a.request_body:
                rprint(f"         Request:  {a.request_body}")
            if a.response_body:
                rprint(f"         Response: {a.response_body}")
            rprint()

    if spec.message_contracts:
        rprint(f"[bold]Message contracts ({len(spec.message_contracts)})[/bold]")
        for m in spec.message_contracts:
            rprint(f"  [{m.direction:7s}] {m.name}")
            rprint(f"            {m.description}")
            if m.fields:
                rprint(f"            Fields: {m.fields}")
            rprint()

    if spec.data_model_changes:
        rprint(f"[bold]Data model changes[/bold]\n{spec.data_model_changes}\n")

    if spec.dependencies:
        rprint(f"[bold]Dependencies[/bold]\n{spec.dependencies}\n")

    if spec.implementation_steps:
        rprint(f"[bold]Implementation steps ({len(spec.implementation_steps)})[/bold]")
        for i, step in enumerate(spec.implementation_steps, 1):
            rprint(f"  {i}. {step}")
        rprint()

    if spec.test_cases:
        rprint(f"[bold]Test cases ({len(spec.test_cases)})[/bold]")
        for t in spec.test_cases:
            rprint(f"  Scenario: {t.scenario}")
            rprint(f"    Given: {t.given}")
            rprint(f"    When:  {t.when}")
            rprint(f"    Then:  {t.then}\n")

    if spec.notes:
        rprint(f"[bold]Notes[/bold]\n{spec.notes}\n")


def export_spec_markdown(spec: Specification) -> str:
    """Export specification as markdown text."""
    md = f"# {spec.title}\n\n"
    md += f"## Overview\n{spec.overview}\n\n"

    if spec.entities_affected:
        md += "## Entities affected\n"
        md += "| Action | Type | Name | Details |\n|--------|------|------|--------|\n"
        for e in spec.entities_affected:
            md += f"| {e.action} | {e.entity_type} | {e.name} | {e.details} |\n"
        md += "\n"

    if spec.api_contracts:
        md += "## API contracts\n"
        for a in spec.api_contracts:
            md += f"### {a.method} {a.path}\n{a.description}\n"
            if a.request_body:
                md += f"- Request: `{a.request_body}`\n"
            if a.response_body:
                md += f"- Response: `{a.response_body}`\n"
            md += "\n"

    if spec.message_contracts:
        md += "## Message contracts\n"
        for m in spec.message_contracts:
            md += f"### {m.name} ({m.direction})\n{m.description}\n"
            if m.fields:
                md += f"- Fields: {m.fields}\n"
            md += "\n"

    if spec.data_model_changes:
        md += f"## Data model changes\n{spec.data_model_changes}\n\n"

    if spec.dependencies:
        md += f"## Dependencies\n{spec.dependencies}\n\n"

    if spec.implementation_steps:
        md += "## Implementation steps\n"
        for i, step in enumerate(spec.implementation_steps, 1):
            md += f"{i}. {step}\n"
        md += "\n"

    if spec.test_cases:
        md += "## Test cases\n"
        for t in spec.test_cases:
            md += f"### {t.scenario}\n"
            md += f"- **Given**: {t.given}\n"
            md += f"- **When**: {t.when}\n"
            md += f"- **Then**: {t.then}\n\n"

    if spec.notes:
        md += f"## Notes\n{spec.notes}\n"

    return md


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        feature = "Add a comment system to tickets where users can add comments and get notified"
    elif args[0] == "--vector":
        feature = " ".join(args[1:])
        spec = generate_spec(feature, use_graph=False)
        print_spec(spec)
        sys.exit(0)
    elif args[0] == "--export":
        feature = " ".join(args[1:])
        spec = generate_spec(feature)
        md = export_spec_markdown(spec)
        out_path = "/app/data/spec_output.md"
        with open(out_path, "w") as f:
            f.write(md)
        rprint(f"\n  [green]Spec exported to {out_path}[/green]\n")
        print_spec(spec)
        sys.exit(0)
    else:
        feature = " ".join(args)

    spec = generate_spec(feature)
    print_spec(spec)