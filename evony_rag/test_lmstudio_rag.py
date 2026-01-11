#!/usr/bin/env python3
"""
End-to-end RAG test with LM Studio.
Tests the complete pipeline: query -> expansion -> retrieval -> generation
"""
import sys
import time
import json
import requests
sys.path.insert(0, ".")

print("=" * 60)
print("END-TO-END RAG TEST WITH LM STUDIO")
print("=" * 60)

LMSTUDIO_URL = "http://localhost:1234/v1"

# Test LM Studio connection
print("\n[1] Testing LM Studio connection...")
try:
    resp = requests.get(f"{LMSTUDIO_URL}/models", timeout=5)
    models = resp.json()["data"]
    print(f"    Connected. {len(models)} models available.")
    # Find a chat model
    chat_models = [m["id"] for m in models if "embed" not in m["id"].lower()]
    model = chat_models[0] if chat_models else models[0]["id"]
    print(f"    Using model: {model}")
except Exception as e:
    print(f"    Error: {e}")
    sys.exit(1)

# Load components
print("\n[2] Loading RAG components...")
from evony_rag.knowledge_graph import get_knowledge_graph
from evony_rag.hybrid_search import get_hybrid_search
from evony_rag.query_expansion import get_ultimate_expander

kg = get_knowledge_graph()
kg.load()
print(f"    KG: {len(kg.entities):,} entities, {len(kg.relationships):,} relationships")

hs = get_hybrid_search()
hs.load_index()
print(f"    Hybrid Search: {len(hs.chunks):,} chunks, embeddings: {hs.embeddings.shape}")

expander = get_ultimate_expander()
print(f"    Query Expander: ready")

def rag_query(query: str) -> dict:
    """Execute full RAG pipeline."""
    start = time.time()
    
    # 1. Query expansion
    expansion = expander.expand(query)
    expanded_query = expansion["expanded_query"]
    
    # 2. Hybrid search
    search_results = hs.search(expanded_query, k_lexical=20, k_vector=20, final_k=5)
    
    # 3. KG search
    kg_results = kg.enhanced_search(query, top_k=5)
    
    # 4. Build context
    context_parts = []
    for r in search_results[:3]:
        context_parts.append(f"[{r.file_path}]\n{r.content[:500]}")
    for r in kg_results[:2]:
        if r.get("relation"):
            context_parts.append(f"[KG] {r['entity']} --[{r['relation']}]--> {r.get('related', '')}")
    
    context = "\n\n".join(context_parts)
    
    # 5. Generate answer with LM Studio
    prompt = f"""Based on the following context about Evony game internals, answer the question.

CONTEXT:
{context[:2000]}

QUESTION: {query}

Provide a concise, accurate answer based only on the context provided. If the context doesn't contain enough information, say so."""

    try:
        resp = requests.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300
            },
            timeout=60
        )
        answer = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    elapsed = time.time() - start
    
    return {
        "query": query,
        "expanded_query": expanded_query[:100],
        "kg_terms": expansion.get("kg_terms", []),
        "search_results": len(search_results),
        "kg_results": len(kg_results),
        "answer": answer,
        "time_ms": elapsed * 1000
    }

# Test queries
print("\n[3] Testing RAG queries...")
test_queries = [
    "What is the troop attack command?",
    "How does server authentication work?",
    "What commands are available for NPC farming?",
]

for query in test_queries:
    print(f"\n{'─' * 60}")
    print(f"Q: {query}")
    print(f"{'─' * 60}")
    
    result = rag_query(query)
    
    print(f"Time: {result['time_ms']:.0f}ms")
    print(f"Expanded: {result['expanded_query']}")
    print(f"KG terms: {result['kg_terms']}")
    print(f"Results: {result['search_results']} search + {result['kg_results']} KG")
    print(f"\nAnswer:\n{result['answer'][:400]}...")

print("\n" + "=" * 60)
print("END-TO-END RAG TEST COMPLETE")
print("=" * 60)
