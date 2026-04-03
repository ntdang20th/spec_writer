# RAG + GraphRAG Learning Path
## LlamaIndex → Codebase-Aware Spec Writer

---

## Phase 1: Foundations ✅
- [x] RAG & GraphRAG concepts
- [x] LlamaIndex core concepts (Documents, Nodes, Indices, Query Engines)
- [x] Chose LlamaIndex over LangChain
- [x] Environment setup (Docker + Ollama + ChromaDB)

## Phase 2: Basic RAG Pipeline ✅
- [x] Document loader (`loader.py`) — 942 docs from Helpdesk-BED
- [x] AST-aware C# chunking (`chunker.py`) — tree-sitter
- [x] Embeddings via Ollama nomic-embed-text (`embedder.py`)
- [x] VectorStoreIndex + ChromaDB (`indexer.py`)
- [x] Query engine working (`query_engine.py`, `query.py`)

## Phase 3: GraphRAG Layer ✅
- [x] **3.1** Understand PropertyGraphIndex — entities, relations, triplets
- [x] **3.2** Define entity types (Handler, Table, API, Message, Model, Service)
- [x] **3.3** Define relationship types (calls, reads_from, writes_to, depends_on)
- [x] **3.4** Extract graph using LLM (qwen2.5-coder:7b via Ollama)
- [x] **3.5** Store graph — Neo4j (474 entities, 1592 triplets)
- [x] **3.6** Hybrid query: vector retrieval + graph traversal
- [x] **3.7** Test: compare RAG-only vs GraphRAG results

## Phase 4: Spec Generation Engine ← YOU ARE HERE

## Phase 4: Spec Generation Engine
- [ ] Design Pydantic output schema for specs
- [ ] Build SpecWriter query engine with structured output
- [ ] Prompt engineering for team conventions
- [ ] SubQuestionQueryEngine for complex specs

## Phase 5: Model Flexibility
- [ ] Ollama model swapping (Llama 3, DeepSeek, Mistral)
- [ ] Claude API integration for high-quality output
- [ ] Benchmark spec quality across models

## Phase 6: Service Integration
- [ ] FastAPI wrapper
- [ ] Incremental indexing (git hooks)
- [ ] CLI tool / Board integration

## Phase 7: Advanced
- [ ] Agentic pipeline (analyze → research → draft → review)
- [ ] Multi-repo support
- [ ] LangGraph integration if needed

---
### Current Stack
| Component | Tool | Status |
|-----------|------|--------|
| Framework | LlamaIndex | ✅ |
| Vector store | ChromaDB (Docker) | ✅ |
| Graph store | SimpleGraphStore → Neo4j | 🔜 Phase 3 |
| Local LLM | Ollama llama3.2 | ✅ |
| Embeddings | Ollama nomic-embed-text | ✅ |
| Codebase | Helpdesk-BED (.NET/C#) | ✅ mounted |

**You are here → Phase 3.1**: PropertyGraphIndex concepts