#!/usr/bin/env python3
"""
Natural Question Benchmark - How Users Actually Ask
====================================================
Simple, direct questions like real users ask.
"""
import sys
sys.path.insert(0, ".")
import time

print("=" * 60)
print("NATURAL QUESTION BENCHMARK")
print("Simple questions like users actually ask")
print("=" * 60)

# Natural questions - how users ACTUALLY ask
TEST_QUERIES = [
    # Glitches/Exploits - simple
    {"q": "troop glitch how to use?", "cat": "exploits"},
    {"q": "food glitch script", "cat": "exploits"},
    {"q": "overflow exploit troops", "cat": "exploits"},
    
    # Scripts - simple
    {"q": "farmNPC script command", "cat": "scripts"},
    {"q": "setsilence command", "cat": "scripts"},
    {"q": "attack script example", "cat": "scripts"},
    
    # Source code - simple
    {"q": "StratagemCommands class", "cat": "source_code"},
    {"q": "SendTroops function", "cat": "source_code"},
    {"q": "ArmyCommands what does it do", "cat": "source_code"},
    
    # Protocol
    {"q": "AMF3 packet decode", "cat": "protocol"},
]

print(f"\nTesting {len(TEST_QUERIES)} natural questions...\n")

try:
    from evony_rag.precision_rag import get_precision_rag
    rag = get_precision_rag()
    
    results = []
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {test['q']}")
        
        start = time.time()
        r = rag.query(test["q"], use_cache=False)
        elapsed = (time.time() - start) * 1000
        
        has_answer = "don't have" not in r.answer.lower() and len(r.answer) > 50
        status = "✅" if has_answer else "❌"
        
        print(f"     {status} Conf: {r.confidence:.0%} | Grounded: {r.is_grounded} | {elapsed:.0f}ms")
        
        results.append({
            "query": test["q"],
            "category": test["cat"],
            "confidence": r.confidence,
            "grounded": r.is_grounded,
            "answered": has_answer,
            "time_ms": elapsed
        })
    
    # Stats
    total_answered = sum(1 for r in results if r["answered"])
    total_grounded = sum(1 for r in results if r["grounded"])
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    avg_time = sum(r["time_ms"] for r in results) / len(results)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Answered:       {total_answered}/{len(results)} ({total_answered/len(results)*100:.0f}%)")
    print(f"  Grounded:       {total_grounded}/{len(results)} ({total_grounded/len(results)*100:.0f}%)")
    print(f"  Avg Confidence: {avg_conf:.0%}")
    print(f"  Avg Time:       {avg_time:.0f}ms")
    
    overall = (total_answered/len(results) + avg_conf) / 2 * 100
    print(f"\n  OVERALL SCORE:  {overall:.0f}%")
    
    if overall >= 80:
        print("  STATUS: ✅ EXCELLENT")
    elif overall >= 60:
        print("  STATUS: 🟡 GOOD")
    else:
        print("  STATUS: ❌ NEEDS WORK")

except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
