#!/usr/bin/env python3
"""Test integrated RAG + Enhanced KG."""
import sys
import time
sys.path.insert(0, ".")

print("=" * 60)
print("INTEGRATED RAG + ENHANCED KG TEST")
print("=" * 60)

# Test 1: Load KG directly
print("\n[1] Testing KG loading from G:\\evony_rag_index...")
from evony_rag.knowledge_graph import get_knowledge_graph

kg = get_knowledge_graph()
loaded = kg.load()
print(f"    Loaded: {loaded}")
print(f"    Entities: {len(kg.entities):,}")
print(f"    Relationships: {len(kg.relationships):,}")

# Test 2: Test enhanced search
print("\n[2] Testing enhanced_search...")
test_queries = [
    "server configuration",
    "attack commands",
    "password storage",
    "troop production",
    "encryption protocol",
]

for query in test_queries:
    start = time.time()
    results = kg.enhanced_search(query, top_k=5)
    elapsed = (time.time() - start) * 1000
    print(f"\n    '{query}' ({elapsed:.1f}ms): {len(results)} results")
    for r in results[:2]:
        if r.get("relation"):
            print(f"      {r['entity']} --[{r['relation']}]--> {r.get('related', '')}")
        else:
            print(f"      {r['entity']} ({r.get('entity_type', '')})")

# Test 3: Test RAG Ultimate with KG
print("\n[3] Testing RAG Ultimate pipeline...")
try:
    from evony_rag.rag_ultimate import RAGUltimate
    
    rag = RAGUltimate(use_lmstudio=False)
    rag._lazy_init()
    
    # Test KG search directly
    kg_results = rag._search_knowledge_graph("server password")
    print(f"    KG search: {len(kg_results)} results")
    for r in kg_results[:3]:
        print(f"      {r.get('content', '')[:60]}...")
    
    # Test combined search
    all_results = rag._search_all("server configuration", k=20)
    print(f"\n    Combined search: {len(all_results)} results")
    
    # Count by source
    sources = {}
    for r in all_results:
        src = r.get("source", "hybrid")
        sources[src] = sources.get(src, 0) + 1
    print(f"    Sources: {sources}")
    
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
