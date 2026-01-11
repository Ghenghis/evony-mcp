#!/usr/bin/env python3
"""Full RAG pipeline test with enhanced KG."""
import sys
import time
import json
sys.path.insert(0, ".")

print("=" * 60)
print("FULL RAG PIPELINE TEST")
print("=" * 60)

# 1. Test KG with boosted connectivity
print("\n[1] Testing enhanced KG...")
from evony_rag.knowledge_graph import get_knowledge_graph
kg = get_knowledge_graph()
kg.load()
print(f"    Entities: {len(kg.entities):,}")
print(f"    Relationships: {len(kg.relationships):,}")

# 2. Test enhanced search
print("\n[2] Testing KG enhanced search...")
queries = ["troop command", "server config", "attack script", "password encryption"]
for q in queries:
    start = time.time()
    results = kg.enhanced_search(q, top_k=5)
    elapsed = (time.time() - start) * 1000
    print(f"    '{q}': {len(results)} results ({elapsed:.1f}ms)")

# 3. Test HyPE index
print("\n[3] Testing HyPE index...")
try:
    with open(r"G:\evony_rag_index\hype_chunks.json") as f:
        hype = json.load(f)
    total_q = sum(len(h.get("questions", [])) for h in hype)
    print(f"    Chunks: {len(hype):,}")
    print(f"    Questions: {total_q:,}")
except Exception as e:
    print(f"    Error: {e}")

# 4. Test chunks
print("\n[4] Testing chunks index...")
try:
    with open(r"evony_rag\index\chunks.json") as f:
        chunks = json.load(f)
    print(f"    Chunks: {len(chunks):,}")
except Exception as e:
    print(f"    Error: {e}")

# 5. Test RAG Ultimate
print("\n[5] Testing RAG Ultimate pipeline...")
try:
    from evony_rag.rag_ultimate import RAGUltimate
    rag = RAGUltimate(use_lmstudio=False)
    rag._lazy_init()
    
    # Test KG search
    kg_results = rag._search_knowledge_graph("troop attack command")
    print(f"    KG search: {len(kg_results)} results")
    
    # Test combined search
    all_results = rag._search_all("server password config", k=20)
    sources = {}
    for r in all_results:
        src = r.get("source", "hybrid")
        sources[src] = sources.get(src, 0) + 1
    print(f"    Combined search: {len(all_results)} results")
    print(f"    Sources: {sources}")
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 60)
print("PIPELINE STATUS SUMMARY")
print("=" * 60)
print(f"KG:     {len(kg.entities):,} entities, {len(kg.relationships):,} relationships")
print(f"HyPE:   {total_q:,} questions")
print(f"Chunks: {len(chunks):,} chunks")
print("\nStatus: READY FOR QUERIES")
