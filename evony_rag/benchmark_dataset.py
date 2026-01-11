#!/usr/bin/env python3
"""
Dataset-Accurate Benchmark
==========================
Uses EXACT question formats from training data.
"""
import sys
sys.path.insert(0, ".")
import time

print("=" * 60)
print("DATASET-ACCURATE BENCHMARK")
print("Questions from actual training data")
print("=" * 60)

# EXACT questions from training dataset
TEST_QUERIES = [
    # Source Code - exact format from training
    {"q": "What does the StratagemCommands class/function do in Evony?", "cat": "source_code"},
    {"q": "What is the purpose of ArmyCommands.as in Evony?", "cat": "source_code"},
    {"q": "How does SendTroopsCommand work in the Evony client?", "cat": "source_code"},
    {"q": "What parameters does toDebugString accept?", "cat": "source_code"},
    {"q": "What does the MainFrame class/function do in Evony?", "cat": "source_code"},
    
    # Scripts - exact format  
    {"q": "What does this Evony script do?\n```\nsetsilence true\nlabel getMailboxContent\nset g $c.af.getMailCommands().receiveMailList(1,1,10000)$\n```", "cat": "scripts"},
    
    # Exploits - exact format
    {"q": "What is the purpose of Troop_Glitch.txt in Evony?", "cat": "exploits"},
    {"q": "What is the purpose of ultimate_overflow_chain.txt in Evony?", "cat": "exploits"},
    
    # More source code
    {"q": "What is the purpose of StoreListResponse_1.as in Evony?", "cat": "source_code"},
    {"q": "How does CastleBuffUpdate work in the Evony client?", "cat": "source_code"},
]

print(f"\nTesting {len(TEST_QUERIES)} queries from training data...\n")

try:
    from evony_rag.precision_rag import get_precision_rag
    rag = get_precision_rag()
    
    results = []
    for i, test in enumerate(TEST_QUERIES, 1):
        q_display = test["q"][:50] + "..." if len(test["q"]) > 50 else test["q"]
        print(f"[{i}/{len(TEST_QUERIES)}] {test['cat']}: {q_display}")
        
        start = time.time()
        r = rag.query(test["q"], use_cache=False)
        elapsed = (time.time() - start) * 1000
        
        # Check quality
        has_answer = "don't have" not in r.answer.lower() and len(r.answer) > 50
        has_code = "```" in r.answer or "function" in r.answer.lower() or "class" in r.answer.lower()
        
        status = "✅" if has_answer else "❌"
        print(f"     {status} Conf: {r.confidence:.0%} | Grounded: {r.is_grounded} | Code: {has_code} | {elapsed:.0f}ms")
        
        results.append({
            "query": test["q"][:60],
            "category": test["cat"],
            "confidence": r.confidence,
            "grounded": r.is_grounded,
            "answered": has_answer,
            "has_code": has_code,
            "time_ms": elapsed,
            "answer": r.answer[:200]
        })
    
    # Stats by category
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
        has_code = sum(1 for r in items if r["has_code"])
        print(f"\n{cat.upper()}:")
        print(f"  Answered:   {answered}/{len(items)} ({answered/len(items)*100:.0f}%)")
        print(f"  Avg Conf:   {avg_conf:.0%}")
        print(f"  Has Code:   {has_code}/{len(items)}")
    
    # Overall
    total_answered = sum(1 for r in results if r["answered"])
    total_grounded = sum(1 for r in results if r["grounded"])
    total_code = sum(1 for r in results if r["has_code"])
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    avg_time = sum(r["time_ms"] for r in results) / len(results)
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"\n  Total Queries:     {len(results)}")
    print(f"  Answered:          {total_answered}/{len(results)} ({total_answered/len(results)*100:.0f}%)")
    print(f"  Grounded:          {total_grounded}/{len(results)} ({total_grounded/len(results)*100:.0f}%)")
    print(f"  Contains Code:     {total_code}/{len(results)} ({total_code/len(results)*100:.0f}%)")
    print(f"  Avg Confidence:    {avg_conf:.0%}")
    print(f"  Avg Response Time: {avg_time:.0f}ms")
    
    # Score calculation
    answer_rate = total_answered / len(results)
    code_rate = total_code / len(results)
    overall = (answer_rate * 0.4 + avg_conf * 0.3 + code_rate * 0.3) * 100
    
    print(f"\n  OVERALL SCORE:     {overall:.0f}%")
    
    if overall >= 85:
        print("  STATUS: ✅ EXCELLENT")
    elif overall >= 70:
        print("  STATUS: 🟡 GOOD")
    else:
        print("  STATUS: ❌ NEEDS WORK")
    
    # Sample answers
    print("\n" + "=" * 60)
    print("SAMPLE ANSWERS")
    print("=" * 60)
    for r in results[:2]:
        print(f"\nQ: {r['query']}")
        print(f"A: {r['answer'][:150]}...")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
