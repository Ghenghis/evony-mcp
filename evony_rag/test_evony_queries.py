#!/usr/bin/env python3
"""Test with real Evony-related queries."""
import sys
import time
sys.path.insert(0, ".")

from evony_rag.knowledge_graph import get_knowledge_graph

print("=" * 60)
print("EVONY QUERY TEST")
print("=" * 60)

# Load KG
kg = get_knowledge_graph()
kg.load()
print(f"KG loaded: {len(kg.entities):,} entities, {len(kg.relationships):,} relationships\n")

# Real Evony queries
queries = [
    "How do I send troops to attack?",
    "What is the troop training command?",
    "Show the server connection protocol",
    "How does the bot automation work?",
    "What encryption is used?",
    "Find the NPC farming script",
    "Show command IDs for battles",
    "How to manage multiple cities?",
    "What are the resource gathering commands?",
    "Show the login authentication flow",
]

print("Testing queries:\n")
for i, query in enumerate(queries, 1):
    start = time.time()
    results = kg.enhanced_search(query, top_k=5)
    elapsed = (time.time() - start) * 1000
    
    print(f"[{i}] {query}")
    print(f"    Time: {elapsed:.1f}ms | Results: {len(results)}")
    
    if results:
        # Show top 2 results
        for r in results[:2]:
            if r.get("relation"):
                print(f"    → {r['entity']} --[{r['relation']}]--> {r.get('related', '')[:30]}")
            else:
                print(f"    → {r['entity']} ({r.get('entity_type', '')})")
    print()

print("=" * 60)
print("QUERY TEST COMPLETE")
print("=" * 60)
