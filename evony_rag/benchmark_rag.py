#!/usr/bin/env python3
"""
Comprehensive RAG Benchmark
===========================
Compares performance metrics across all RAG components.
"""
import sys
import time
import json
sys.path.insert(0, ".")

print("=" * 70)
print("RAG ULTIMATE COMPREHENSIVE BENCHMARK")
print("=" * 70)

results = {}

# 1. Knowledge Graph
print("\n[1] KNOWLEDGE GRAPH BENCHMARK")
print("-" * 50)
from evony_rag.knowledge_graph import get_knowledge_graph

start = time.time()
kg = get_knowledge_graph()
kg.load()
kg_load_time = time.time() - start

results["kg"] = {
    "entities": len(kg.entities),
    "relationships": len(kg.relationships),
    "load_time_ms": kg_load_time * 1000,
}

# KG query benchmark
queries = ["server", "attack", "command", "password", "config"]
kg_times = []
for q in queries:
    start = time.time()
    r = kg.enhanced_search(q, top_k=10)
    kg_times.append((time.time() - start) * 1000)

results["kg"]["avg_query_ms"] = sum(kg_times) / len(kg_times)
print(f"  Entities: {results['kg']['entities']:,}")
print(f"  Relationships: {results['kg']['relationships']:,}")
print(f"  Load time: {results['kg']['load_time_ms']:.0f}ms")
print(f"  Avg query: {results['kg']['avg_query_ms']:.2f}ms")

# 2. Hybrid Search
print("\n[2] HYBRID SEARCH BENCHMARK")
print("-" * 50)
from evony_rag.hybrid_search import get_hybrid_search

start = time.time()
hs = get_hybrid_search()
hs.load_index()
hs_load_time = time.time() - start

results["hybrid"] = {
    "chunks": len(hs.chunks),
    "embeddings_shape": str(hs.embeddings.shape) if hs.embeddings is not None else "None",
    "load_time_ms": hs_load_time * 1000,
}

# BM25 only benchmark
bm25_times = []
for q in queries:
    start = time.time()
    r = hs.bm25.search(q, top_k=20)
    bm25_times.append((time.time() - start) * 1000)

results["hybrid"]["bm25_avg_ms"] = sum(bm25_times) / len(bm25_times)
print(f"  Chunks: {results['hybrid']['chunks']:,}")
print(f"  Embeddings: {results['hybrid']['embeddings_shape']}")
print(f"  Load time: {results['hybrid']['load_time_ms']:.0f}ms")
print(f"  BM25 avg: {results['hybrid']['bm25_avg_ms']:.2f}ms")

# 3. Query Expansion
print("\n[3] QUERY EXPANSION BENCHMARK")
print("-" * 50)
from evony_rag.query_expansion import get_ultimate_expander

expander = get_ultimate_expander()

exp_times = []
kg_terms_count = 0
for q in ["troop attack", "server config", "npc farming"]:
    start = time.time()
    r = expander.expand(q)
    exp_times.append((time.time() - start) * 1000)
    kg_terms_count += len(r.get("kg_terms", []))

results["expansion"] = {
    "avg_time_ms": sum(exp_times) / len(exp_times),
    "avg_kg_terms": kg_terms_count / len(exp_times),
}
print(f"  Avg expansion time: {results['expansion']['avg_time_ms']:.2f}ms")
print(f"  Avg KG terms added: {results['expansion']['avg_kg_terms']:.1f}")

# 4. HyPE Index
print("\n[4] HYPE INDEX BENCHMARK")
print("-" * 50)
try:
    with open(r"G:\evony_rag_index\hype_chunks.json") as f:
        hype = json.load(f)
    total_q = sum(len(h.get("questions", [])) for h in hype)
    results["hype"] = {
        "chunks": len(hype),
        "questions": total_q,
        "avg_per_chunk": total_q / len(hype) if hype else 0,
    }
    print(f"  Chunks: {results['hype']['chunks']:,}")
    print(f"  Questions: {results['hype']['questions']:,}")
    print(f"  Avg per chunk: {results['hype']['avg_per_chunk']:.1f}")
except Exception as e:
    print(f"  Error: {e}")
    results["hype"] = {"error": str(e)}

# 5. Full Pipeline (without LLM)
print("\n[5] FULL PIPELINE BENCHMARK (no LLM)")
print("-" * 50)

def full_pipeline(query):
    """Run full pipeline without LLM generation."""
    # Expand
    expansion = expander.expand(query)
    expanded = expansion["expanded_query"]
    
    # Hybrid search
    search_results = hs.search(expanded, k_lexical=20, k_vector=20, final_k=5)
    
    # KG search
    kg_results = kg.enhanced_search(query, top_k=5)
    
    return len(search_results) + len(kg_results)

pipeline_queries = [
    "How do I send troops?",
    "What is the server address?",
    "Show NPC farming commands",
    "Find password encryption",
    "List available exploits",
]

pipeline_times = []
for q in pipeline_queries:
    start = time.time()
    count = full_pipeline(q)
    pipeline_times.append((time.time() - start) * 1000)

results["pipeline"] = {
    "avg_time_ms": sum(pipeline_times) / len(pipeline_times),
    "min_time_ms": min(pipeline_times),
    "max_time_ms": max(pipeline_times),
}
print(f"  Avg pipeline time: {results['pipeline']['avg_time_ms']:.0f}ms")
print(f"  Min: {results['pipeline']['min_time_ms']:.0f}ms, Max: {results['pipeline']['max_time_ms']:.0f}ms")

# Summary
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG ULTIMATE PERFORMANCE                         │
├─────────────────────────┬───────────────────────────────────────────┤
│ COMPONENT               │ METRICS                                   │
├─────────────────────────┼───────────────────────────────────────────┤""")
print(f"│ Knowledge Graph         │ {results['kg']['entities']:,} entities, {results['kg']['relationships']:,} rels    │")
print(f"│   Query speed           │ {results['kg']['avg_query_ms']:.2f}ms avg                               │")
print(f"│ Hybrid Search           │ {results['hybrid']['chunks']:,} chunks, {results['hybrid']['embeddings_shape']}  │")
print(f"│   BM25 speed            │ {results['hybrid']['bm25_avg_ms']:.2f}ms avg                              │")
print(f"│ Query Expansion         │ {results['expansion']['avg_kg_terms']:.1f} KG terms added                       │")
print(f"│ HyPE Index              │ {results.get('hype', {}).get('questions', 0):,} questions                        │")
print(f"│ Full Pipeline           │ {results['pipeline']['avg_time_ms']:.0f}ms avg (no LLM)                       │")
print("└─────────────────────────┴───────────────────────────────────────────┘")

# Save results
with open("evony_rag/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to evony_rag/benchmark_results.json")
