#!/usr/bin/env python3
"""Test hybrid search with embeddings."""
import sys
import time
sys.path.insert(0, ".")

from evony_rag.hybrid_search import get_hybrid_search

print("=" * 60)
print("HYBRID SEARCH TEST")
print("=" * 60)

hs = get_hybrid_search()
loaded = hs.load_index()

print(f"Loaded: {loaded}")
print(f"Chunks: {len(hs.chunks):,}")
print(f"Embeddings: {hs.embeddings.shape if hs.embeddings is not None else None}")

# Test queries
queries = [
    "troop attack command",
    "server configuration",
    "password encryption",
    "NPC farming script",
    "login authentication",
]

print("\nTesting queries:\n")
for query in queries:
    start = time.time()
    results = hs.search(query, k_lexical=20, k_vector=20, final_k=5)
    elapsed = (time.time() - start) * 1000
    
    print(f"'{query}' ({elapsed:.0f}ms): {len(results)} results")
    for r in results[:2]:
        print(f"  [{r.combined_score:.3f}] {r.file_path[-40:]} (lex:{r.lexical_score:.2f}, sem:{r.semantic_score:.2f})")
    print()

print("=" * 60)
print("HYBRID SEARCH TEST COMPLETE")
print("=" * 60)
