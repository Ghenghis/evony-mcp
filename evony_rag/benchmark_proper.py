#!/usr/bin/env python3
"""
Proper Benchmark - Questions Model Was Trained On
==================================================
Tests reverse engineering, exploits, scripts, glitches, protocols.
NOT gameplay questions.
"""
import sys
sys.path.insert(0, ".")
import time

print("=" * 60)
print("PROPER BENCHMARK - Reverse Engineering Focus")
print("Model: F16 | Date: Jan 11, 2026")
print("=" * 60)

# Questions matching actual training data
TEST_QUERIES = [
    # Scripts & Automation
    {"q": "What does the farmNPC script command do?", "cat": "scripts"},
    {"q": "How do I write a script to attack NPCs?", "cat": "scripts"},
    {"q": "What script commands control troop movement?", "cat": "scripts"},
    
    # Source Code / Reverse Engineering
    {"q": "What does the SendTroopsCommand class do?", "cat": "source_code"},
    {"q": "How does the AMF3 protocol encode packets?", "cat": "protocol"},
    {"q": "What is command ID 45 in the protocol?", "cat": "protocol"},
    
    # Exploits & Glitches
    {"q": "How does the food glitch exploit work?", "cat": "exploits"},
    {"q": "What overflow exploits affect troop counts?", "cat": "exploits"},
    {"q": "How do I exploit integer overflow in Evony?", "cat": "exploits"},
    
    # CVEs
    {"q": "What CVEs affect Flash Player 10?", "cat": "cve"},
]

print(f"\nTesting {len(TEST_QUERIES)} queries...\n")

try:
    from evony_rag.precision_rag import get_precision_rag
    rag = get_precision_rag()
    
    results = []
    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {test['cat']}: {test['q'][:40]}...")
        
        start = time.time()
        r = rag.query(test["q"], use_cache=False)
        elapsed = (time.time() - start) * 1000
        
        # Check if real answer (not "I don't know")
        has_answer = "don't have" not in r.answer.lower() and len(r.answer) > 50
        
        status = "✅" if has_answer else "❌"
        print(f"     {status} Conf: {r.confidence:.0%} | {elapsed:.0f}ms")
        
        results.append({
            "query": test["q"],
            "category": test["cat"],
            "confidence": r.confidence,
            "grounded": r.is_grounded,
            "answered": has_answer,
            "time_ms": elapsed,
            "answer_preview": r.answer[:100]
        })
    
    # Calculate stats by category
    print("\n" + "=" * 60)
    print("RESULTS BY CATEGORY")
    print("=" * 60)
    
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    for cat, items in categories.items():
        answered = sum(1 for r in items if r["answered"])
        avg_conf = sum(r["confidence"] for r in items) / len(items)
        avg_time = sum(r["time_ms"] for r in items) / len(items)
        print(f"\n{cat.upper()}:")
        print(f"  Answered: {answered}/{len(items)} ({answered/len(items)*100:.0f}%)")
        print(f"  Avg Conf: {avg_conf:.0%}")
        print(f"  Avg Time: {avg_time:.0f}ms")
    
    # Overall stats
    total_answered = sum(1 for r in results if r["answered"])
    total_grounded = sum(1 for r in results if r["grounded"])
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_time = sum(r["time_ms"] for r in results) / len(results)
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"\n  Total Queries:    {len(results)}")
    print(f"  Answered:         {total_answered}/{len(results)} ({total_answered/len(results)*100:.0f}%)")
    print(f"  Grounded:         {total_grounded}/{len(results)} ({total_grounded/len(results)*100:.0f}%)")
    print(f"  Avg Confidence:   {avg_confidence:.0%}")
    print(f"  Avg Response:     {avg_time:.0f}ms")
    
    # Overall score
    answer_rate = total_answered / len(results)
    overall = (answer_rate + avg_confidence) / 2 * 100
    
    print(f"\n  OVERALL SCORE:    {overall:.0f}%")
    
    if overall >= 80:
        print("  STATUS: ✅ EXCELLENT")
    elif overall >= 60:
        print("  STATUS: 🟡 GOOD")
    else:
        print("  STATUS: ❌ NEEDS WORK")
    
    # Show sample answers
    print("\n" + "=" * 60)
    print("SAMPLE ANSWERS")
    print("=" * 60)
    for r in results[:3]:
        print(f"\nQ: {r['query']}")
        print(f"A: {r['answer_preview']}...")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
