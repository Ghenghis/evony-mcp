#!/usr/bin/env python3
"""Test RAG v3.0 features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_rag_v3():
    print("=" * 60)
    print("RAG v3.0 Test Suite")
    print("=" * 60)
    
    # Test 1: RAG Fusion
    print("\n[TEST 1] RAG Fusion Multi-Query")
    from evony_rag.rag_fusion import get_rag_fusion, QueryGenerator
    
    qg = QueryGenerator()
    queries = qg.generate_variations("What is command ID 45?")
    print(f"  Original: What is command ID 45?")
    print(f"  Generated {len(queries)} variations:")
    for q in queries:
        print(f"    - {q}")
    
    # Test 2: Cross-Encoder
    print("\n[TEST 2] Cross-Encoder Reranking")
    from evony_rag.cross_encoder import get_cross_encoder
    
    ce = get_cross_encoder("fast")
    docs = [
        "Command ID 45 is TROOP_PRODUCE for troop training",
        "The weather is nice today",
        "AMF3 protocol uses binary encoding",
    ]
    scores = ce.score_pairs("What is command ID 45?", docs)
    print(f"  Query: What is command ID 45?")
    print(f"  Scores:")
    for doc, score in zip(docs, scores):
        print(f"    {score:.3f}: {doc[:50]}...")
    
    # Test 3: Contextual Indexer
    print("\n[TEST 3] Contextual Chunking")
    from evony_rag.contextual_indexer import ContextGenerator
    
    cg = ContextGenerator()
    test_chunk = {
        "file_path": "source_code/TroopCommand.as",
        "category": "source_code",
        "content": "public function produceTroop(cityId:int, troopType:int):void",
        "start_line": 45,
        "end_line": 60,
        "symbols": ["produceTroop", "cityId", "troopType"],
    }
    context = cg.generate_context(test_chunk)
    print(f"  Original: {test_chunk['content'][:50]}...")
    print(f"  Context: {context}")
    
    # Test 4: Full RAG v3 Search
    print("\n[TEST 4] Full RAG v3.0 Search")
    try:
        from evony_rag.rag_v3 import get_rag_v3
        
        rag = get_rag_v3()
        results, meta = rag.search("LM Studio model switching", top_k=5)
        
        print(f"  Query type: {meta['query_type']}")
        print(f"  Confidence: {meta['confidence']:.0%}")
        print(f"  Queries used: {len(meta['queries_used'])}")
        print(f"  Results: {len(results)}")
        
        if results:
            print("\n  Top results:")
            for i, r in enumerate(results[:3], 1):
                score = r.get('final_score', r.get('combined_score', 0))
                print(f"    {i}. {r['file_path']}:{r['start_line']}-{r['end_line']} ({score:.2f})")
    except Exception as e:
        print(f"  [WARN] Search test skipped: {e}")
    
    # Test 5: Check indexed docs
    print("\n[TEST 5] Index Statistics")
    from evony_rag.config import INDEX_PATH
    import json
    
    chunks_file = INDEX_PATH / "chunks.json"
    if chunks_file.exists():
        with open(chunks_file) as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            chunks = data.get("chunks", [])
            metadata = data.get("metadata", {})
        else:
            chunks = data
            metadata = {}
        
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Version: {metadata.get('version', 'unknown')}")
        print(f"  Contextual: {metadata.get('contextual', False)}")
        
        # Count by category
        categories = {}
        for c in chunks:
            cat = c.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"  Categories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {cat}: {count}")
    else:
        print("  [WARN] No index found")
    
    print("\n" + "=" * 60)
    print("[OK] RAG v3.0 Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag_v3()
