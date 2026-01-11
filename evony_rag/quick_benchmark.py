#!/usr/bin/env python3
"""
Quick Lightweight Benchmark - Memory Safe
==========================================
Tests features without heavy memory usage.
"""
import sys
sys.path.insert(0, ".")
import time

print("=" * 60)
print("QUICK BENCHMARK - F16 Model")
print("=" * 60)

# Test 1: Features (no heavy loading)
print("\n[1] FEATURE CHECK")
print("-" * 40)

features = {}

# Question Formatter
try:
    from evony_rag.question_formatter import get_question_formatter
    f = get_question_formatter()
    r = f.format_question("how attack")
    features["Question Formatter"] = "✅ " + r.formatted[:40]
except Exception as e:
    features["Question Formatter"] = f"❌ {e}"

# Feedback Loop
try:
    from evony_rag.feedback_loop import get_feedback_collector
    fb = get_feedback_collector()
    s = fb.get_stats()
    features["Feedback Loop"] = f"✅ {s['total']} entries"
except Exception as e:
    features["Feedback Loop"] = f"❌ {e}"

# Reranker
try:
    from evony_rag.reranker import get_reranker
    rr = get_reranker()
    features["Reranker"] = "✅ Available"
except Exception as e:
    features["Reranker"] = f"❌ {e}"

# Cross-Encoder
try:
    from evony_rag.reranker import get_cross_encoder
    ce = get_cross_encoder()
    features["Cross-Encoder"] = "✅ Available"
except Exception as e:
    features["Cross-Encoder"] = f"❌ {e}"

# ChromaDB
try:
    from evony_rag.chroma_search import CHROMA_AVAILABLE
    features["ChromaDB"] = "✅ Installed" if CHROMA_AVAILABLE else "⚠️ Not installed"
except Exception as e:
    features["ChromaDB"] = f"❌ {e}"

for name, status in features.items():
    print(f"  {name}: {status}")

working = sum(1 for s in features.values() if "✅" in s)
print(f"\n  Features: {working}/{len(features)} working")

# Test 2: Quick RAG test (3 queries only)
print("\n[2] RAG QUERY TEST (3 queries)")
print("-" * 40)

try:
    from evony_rag.precision_rag import get_precision_rag
    rag = get_precision_rag()
    
    queries = [
        "What does farmNPC do?",
        "What is StratagemCommands?",
        "How do troops attack?",
    ]
    
    results = []
    for q in queries:
        print(f"\n  Q: {q}")
        start = time.time()
        r = rag.query(q, use_cache=False)
        elapsed = (time.time() - start) * 1000
        
        has_answer = "don't have" not in r.answer.lower() and len(r.answer) > 50
        status = "✅" if has_answer else "❌"
        
        print(f"  {status} Conf: {r.confidence:.0%} | Grounded: {r.is_grounded} | {elapsed:.0f}ms")
        print(f"     Answer: {r.answer[:80]}...")
        
        results.append({
            "conf": r.confidence,
            "grounded": r.is_grounded,
            "answered": has_answer,
            "time_ms": elapsed
        })
    
    # Summary
    avg_conf = sum(r["conf"] for r in results) / len(results)
    avg_time = sum(r["time_ms"] for r in results) / len(results)
    answered = sum(1 for r in results if r["answered"])
    grounded = sum(1 for r in results if r["grounded"])
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Features Working:  {working}/{len(features)} ({working/len(features)*100:.0f}%)")
    print(f"  Avg Confidence:    {avg_conf:.0%}")
    print(f"  Answered Queries:  {answered}/{len(queries)} ({answered/len(queries)*100:.0f}%)")
    print(f"  Grounded Answers:  {grounded}/{len(queries)} ({grounded/len(queries)*100:.0f}%)")
    print(f"  Avg Response Time: {avg_time:.0f}ms")
    
    overall = (working/len(features) + avg_conf + answered/len(queries)) / 3 * 100
    print(f"\n  OVERALL SCORE: {overall:.0f}%")
    
    if overall >= 80:
        print("  STATUS: ✅ EXCELLENT")
    elif overall >= 60:
        print("  STATUS: 🟡 GOOD - Minor improvements needed")
    else:
        print("  STATUS: ❌ NEEDS WORK")

except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
