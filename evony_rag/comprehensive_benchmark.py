#!/usr/bin/env python3
"""
Comprehensive Benchmark - Test All RAG Improvements
=====================================================
Tests all features from Claude's audit and measures:
1. Response time improvements
2. Answer quality (confidence, grounding)
3. Feature functionality
4. Areas needing more work

Compares: Basic RAG vs Enhanced RAG with all improvements
"""
import sys
sys.path.insert(0, ".")

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# Test queries covering different categories
TEST_QUERIES = [
    # Script-related (most important per user)
    {"q": "How do I use the farmNPC script command?", "category": "scripts", "expected_keywords": ["farm", "npc", "script"]},
    {"q": "What does the setsilence command do?", "category": "scripts", "expected_keywords": ["setsilence", "silence", "output"]},
    {"q": "Show me an attack script example", "category": "scripts", "expected_keywords": ["attack", "script", "loop"]},
    
    # Source code
    {"q": "What does the StratagemCommands class do in Evony?", "category": "source_code", "expected_keywords": ["stratagem", "command"]},
    {"q": "What parameters does ScriptManager accept?", "category": "source_code", "expected_keywords": ["script", "manager", "parameter"]},
    
    # Protocol
    {"q": "How does the troop attack command work?", "category": "protocol", "expected_keywords": ["troop", "attack", "command"]},
    {"q": "What is the army.newArmy protocol?", "category": "protocol", "expected_keywords": ["army", "protocol"]},
    
    # Exploits
    {"q": "What exploits are available for troops?", "category": "exploits", "expected_keywords": ["exploit", "troop", "glitch"]},
    {"q": "How does the food glitch work?", "category": "exploits", "expected_keywords": ["food", "glitch"]},
    
    # General
    {"q": "How do I send troops to attack an NPC?", "category": "general", "expected_keywords": ["troop", "attack", "npc"]},
]


@dataclass
class TestResult:
    """Result of a single test."""
    query: str
    category: str
    response_time_ms: float
    confidence: float
    is_grounded: bool
    citations_count: int
    answer_length: int
    keywords_found: int
    keywords_expected: int
    has_answer: bool  # Not "I don't know"
    error: str = ""


@dataclass
class FeatureTestResult:
    """Result of testing a specific feature."""
    feature_name: str
    working: bool
    response_time_ms: float
    error: str = ""
    details: str = ""


def test_basic_rag() -> List[TestResult]:
    """Test basic RAG without enhancements."""
    from evony_rag.hybrid_search import get_hybrid_search
    from evony_rag.knowledge_graph import get_knowledge_graph
    
    results = []
    
    # Initialize basic components
    hybrid = get_hybrid_search()
    hybrid.load_index()
    
    kg = get_knowledge_graph()
    kg.load()
    
    for test in TEST_QUERIES:
        start = time.time()
        try:
            # Basic search only
            search_results = hybrid.search(test["q"], final_k=5)
            kg_results = kg.enhanced_search(test["q"], limit=5) if hasattr(kg, 'enhanced_search') else []
            
            elapsed = (time.time() - start) * 1000
            
            # Count keywords in results
            all_content = " ".join([r.get("content", "")[:200] for r in search_results]).lower()
            keywords_found = sum(1 for kw in test["expected_keywords"] if kw.lower() in all_content)
            
            results.append(TestResult(
                query=test["q"],
                category=test["category"],
                response_time_ms=elapsed,
                confidence=0.5,  # No confidence in basic
                is_grounded=False,
                citations_count=len(search_results),
                answer_length=len(all_content),
                keywords_found=keywords_found,
                keywords_expected=len(test["expected_keywords"]),
                has_answer=len(search_results) > 0
            ))
        except Exception as e:
            results.append(TestResult(
                query=test["q"],
                category=test["category"],
                response_time_ms=0,
                confidence=0,
                is_grounded=False,
                citations_count=0,
                answer_length=0,
                keywords_found=0,
                keywords_expected=len(test["expected_keywords"]),
                has_answer=False,
                error=str(e)
            ))
    
    return results


def test_enhanced_rag() -> List[TestResult]:
    """Test enhanced RAG with all improvements."""
    from evony_rag.precision_rag import get_precision_rag
    
    results = []
    rag = get_precision_rag()
    
    for test in TEST_QUERIES:
        start = time.time()
        try:
            result = rag.query(test["q"], use_cache=False)
            elapsed = (time.time() - start) * 1000
            
            # Count keywords in answer
            answer_lower = result.answer.lower()
            keywords_found = sum(1 for kw in test["expected_keywords"] if kw.lower() in answer_lower)
            
            # Check if it's a real answer (not "I don't know")
            has_answer = "don't have enough information" not in result.answer.lower() and len(result.answer) > 50
            
            results.append(TestResult(
                query=test["q"],
                category=test["category"],
                response_time_ms=elapsed,
                confidence=result.confidence,
                is_grounded=result.is_grounded,
                citations_count=len(result.citations),
                answer_length=len(result.answer),
                keywords_found=keywords_found,
                keywords_expected=len(test["expected_keywords"]),
                has_answer=has_answer
            ))
        except Exception as e:
            results.append(TestResult(
                query=test["q"],
                category=test["category"],
                response_time_ms=0,
                confidence=0,
                is_grounded=False,
                citations_count=0,
                answer_length=0,
                keywords_found=0,
                keywords_expected=len(test["expected_keywords"]),
                has_answer=False,
                error=str(e)
            ))
    
    return results


def test_individual_features() -> List[FeatureTestResult]:
    """Test each individual feature."""
    results = []
    
    # 1. Question Formatter
    print("  Testing Question Formatter...")
    try:
        start = time.time()
        from evony_rag.question_formatter import get_question_formatter
        formatter = get_question_formatter()
        test_result = formatter.format_question("How do I attack?")
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Question Formatter",
            working=len(test_result.formatted) > 0 and test_result.formatted != "How do I attack?",
            response_time_ms=elapsed,
            details=f"Reformatted to: {test_result.formatted[:60]}..."
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Question Formatter",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 2. Cross-Encoder Reranker
    print("  Testing Cross-Encoder Reranker...")
    try:
        start = time.time()
        from evony_rag.reranker import get_cross_encoder, get_reranker
        ce = get_cross_encoder()
        reranker = get_reranker()
        
        # Test with dummy data
        test_docs = [
            {"content": "This is about farming NPCs in the game", "score": 0.5},
            {"content": "Attack commands for troops", "score": 0.6},
        ]
        reranked = reranker.rerank(test_docs, "farming")
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Cross-Encoder Reranker",
            working=len(reranked) > 0,
            response_time_ms=elapsed,
            details=f"Reranked {len(test_docs)} docs"
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Cross-Encoder Reranker",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 3. Feedback Loop
    print("  Testing Feedback Loop...")
    try:
        start = time.time()
        from evony_rag.feedback_loop import get_feedback_collector
        fb = get_feedback_collector()
        stats = fb.get_stats()
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Feedback Loop",
            working=True,
            response_time_ms=elapsed,
            details=f"Total feedback: {stats.get('total', 0)}, Accuracy: {stats.get('accuracy', 0):.0%}"
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Feedback Loop",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 4. Self-Consistency (Multi-Answer)
    print("  Testing Self-Consistency...")
    try:
        start = time.time()
        from evony_rag.precision_rag import get_precision_rag
        rag = get_precision_rag()
        has_method = hasattr(rag, 'query_with_consensus')
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Self-Consistency",
            working=has_method,
            response_time_ms=elapsed,
            details="query_with_consensus() method available" if has_method else "Method missing"
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Self-Consistency",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 5. Knowledge Graph
    print("  Testing Knowledge Graph...")
    try:
        start = time.time()
        from evony_rag.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        kg.load()
        entity_count = len(kg.entities)
        rel_count = len(kg.relationships)
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Knowledge Graph",
            working=entity_count > 0,
            response_time_ms=elapsed,
            details=f"{entity_count:,} entities, {rel_count:,} relationships"
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Knowledge Graph",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 6. Hybrid Search (BM25 + Semantic)
    print("  Testing Hybrid Search...")
    try:
        start = time.time()
        from evony_rag.hybrid_search import get_hybrid_search
        hybrid = get_hybrid_search()
        hybrid.load_index()
        test_results = hybrid.search("attack troops", final_k=3)
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Hybrid Search",
            working=len(test_results) > 0,
            response_time_ms=elapsed,
            details=f"Found {len(test_results)} results"
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Hybrid Search",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 7. Query Expansion
    print("  Testing Query Expansion...")
    try:
        start = time.time()
        from evony_rag.query_expansion import get_ultimate_expander
        expander = get_ultimate_expander()
        expanded = expander.expand("attack troops")
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Query Expansion",
            working=len(expanded.get("expanded_query", "")) > len("attack troops"),
            response_time_ms=elapsed,
            details=f"Expanded to: {expanded.get('expanded_query', '')[:60]}..."
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Query Expansion",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 8. Answer Cache
    print("  Testing Answer Cache...")
    try:
        start = time.time()
        from evony_rag.precision_rag import get_precision_rag
        rag = get_precision_rag()
        has_cache = rag._cache is not None
        cache_size = len(rag._cache.cache) if has_cache else 0
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="Answer Cache",
            working=has_cache,
            response_time_ms=elapsed,
            details=f"Cache size: {cache_size} entries"
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Answer Cache",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 9. ChromaDB (Optional)
    print("  Testing ChromaDB...")
    try:
        start = time.time()
        from evony_rag.chroma_search import CHROMA_AVAILABLE, get_chroma_search
        elapsed = (time.time() - start) * 1000
        
        results.append(FeatureTestResult(
            feature_name="ChromaDB (Optional)",
            working=CHROMA_AVAILABLE,
            response_time_ms=elapsed,
            details="Available" if CHROMA_AVAILABLE else "Not installed (optional)"
        ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="ChromaDB (Optional)",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    # 10. Embeddings
    print("  Testing Embeddings...")
    try:
        start = time.time()
        emb_path = Path(r"G:\evony_rag_index\embeddings.npy")
        if emb_path.exists():
            import numpy as np
            emb = np.load(emb_path)
            elapsed = (time.time() - start) * 1000
            results.append(FeatureTestResult(
                feature_name="Embeddings",
                working=True,
                response_time_ms=elapsed,
                details=f"Shape: {emb.shape}, Size: {emb_path.stat().st_size / 1024 / 1024:.1f}MB"
            ))
        else:
            results.append(FeatureTestResult(
                feature_name="Embeddings",
                working=False,
                response_time_ms=0,
                error="embeddings.npy not found"
            ))
    except Exception as e:
        results.append(FeatureTestResult(
            feature_name="Embeddings",
            working=False,
            response_time_ms=0,
            error=str(e)
        ))
    
    return results


def calculate_improvements(basic: List[TestResult], enhanced: List[TestResult]) -> Dict:
    """Calculate improvement percentages."""
    if not basic or not enhanced:
        return {}
    
    # Average metrics
    basic_time = sum(r.response_time_ms for r in basic) / len(basic)
    enhanced_time = sum(r.response_time_ms for r in enhanced) / len(enhanced)
    
    basic_confidence = sum(r.confidence for r in basic) / len(basic)
    enhanced_confidence = sum(r.confidence for r in enhanced) / len(enhanced)
    
    basic_grounded = sum(1 for r in basic if r.is_grounded) / len(basic)
    enhanced_grounded = sum(1 for r in enhanced if r.is_grounded) / len(enhanced)
    
    basic_citations = sum(r.citations_count for r in basic) / len(basic)
    enhanced_citations = sum(r.citations_count for r in enhanced) / len(enhanced)
    
    basic_keywords = sum(r.keywords_found for r in basic) / sum(r.keywords_expected for r in basic) if sum(r.keywords_expected for r in basic) > 0 else 0
    enhanced_keywords = sum(r.keywords_found for r in enhanced) / sum(r.keywords_expected for r in enhanced) if sum(r.keywords_expected for r in enhanced) > 0 else 0
    
    basic_answers = sum(1 for r in basic if r.has_answer) / len(basic)
    enhanced_answers = sum(1 for r in enhanced if r.has_answer) / len(enhanced)
    
    return {
        "response_time": {
            "basic_ms": basic_time,
            "enhanced_ms": enhanced_time,
            "change_pct": ((enhanced_time - basic_time) / basic_time * 100) if basic_time > 0 else 0,
            "note": "Slower but more accurate" if enhanced_time > basic_time else "Faster"
        },
        "confidence": {
            "basic": basic_confidence,
            "enhanced": enhanced_confidence,
            "improvement_pct": ((enhanced_confidence - basic_confidence) / max(basic_confidence, 0.01)) * 100
        },
        "grounding": {
            "basic_pct": basic_grounded * 100,
            "enhanced_pct": enhanced_grounded * 100,
            "improvement_pct": (enhanced_grounded - basic_grounded) * 100
        },
        "citations": {
            "basic_avg": basic_citations,
            "enhanced_avg": enhanced_citations,
            "improvement_pct": ((enhanced_citations - basic_citations) / max(basic_citations, 0.01)) * 100
        },
        "keyword_accuracy": {
            "basic_pct": basic_keywords * 100,
            "enhanced_pct": enhanced_keywords * 100,
            "improvement_pct": (enhanced_keywords - basic_keywords) * 100
        },
        "answer_rate": {
            "basic_pct": basic_answers * 100,
            "enhanced_pct": enhanced_answers * 100,
            "improvement_pct": (enhanced_answers - basic_answers) * 100
        }
    }


def identify_issues(basic: List[TestResult], enhanced: List[TestResult], features: List[FeatureTestResult]) -> List[Dict]:
    """Identify areas needing more work."""
    issues = []
    
    # Check features
    for f in features:
        if not f.working:
            issues.append({
                "area": f.feature_name,
                "severity": "HIGH" if "Optional" not in f.feature_name else "LOW",
                "issue": f.error or "Not working",
                "recommendation": f"Fix {f.feature_name} implementation"
            })
    
    # Check query results
    low_confidence = [r for r in enhanced if r.confidence < 0.5]
    if len(low_confidence) > len(enhanced) * 0.3:
        issues.append({
            "area": "Answer Confidence",
            "severity": "MEDIUM",
            "issue": f"{len(low_confidence)}/{len(enhanced)} queries have <50% confidence",
            "recommendation": "Improve retrieval or add more training data"
        })
    
    not_grounded = [r for r in enhanced if not r.is_grounded]
    if len(not_grounded) > len(enhanced) * 0.5:
        issues.append({
            "area": "Answer Grounding",
            "severity": "MEDIUM",
            "issue": f"{len(not_grounded)}/{len(enhanced)} answers not grounded in sources",
            "recommendation": "Improve citation extraction or retrieval"
        })
    
    no_answers = [r for r in enhanced if not r.has_answer]
    if len(no_answers) > len(enhanced) * 0.3:
        issues.append({
            "area": "Answer Coverage",
            "severity": "HIGH",
            "issue": f"{len(no_answers)}/{len(enhanced)} queries got 'I don't know' responses",
            "recommendation": "Add more training data or improve question formatting"
        })
    
    # Check by category
    categories = {}
    for r in enhanced:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    for cat, results in categories.items():
        avg_conf = sum(r.confidence for r in results) / len(results)
        if avg_conf < 0.5:
            issues.append({
                "area": f"Category: {cat}",
                "severity": "MEDIUM",
                "issue": f"Low confidence ({avg_conf:.0%}) for {cat} queries",
                "recommendation": f"Add more {cat} training data"
            })
    
    return issues


def run_benchmark():
    """Run the complete benchmark."""
    print("=" * 70)
    print("COMPREHENSIVE RAG BENCHMARK")
    print("Testing All Improvements from Claude's Audit")
    print("=" * 70)
    
    # 1. Test individual features
    print("\n[1/4] TESTING INDIVIDUAL FEATURES...")
    print("-" * 50)
    features = test_individual_features()
    
    working_count = sum(1 for f in features if f.working)
    print(f"\nFeature Results: {working_count}/{len(features)} working")
    
    for f in features:
        status = "✅" if f.working else "❌"
        print(f"  {status} {f.feature_name}: {f.details or f.error}")
    
    # 2. Test basic RAG (baseline)
    print("\n[2/4] TESTING BASIC RAG (BASELINE)...")
    print("-" * 50)
    basic_results = test_basic_rag()
    print(f"Completed {len(basic_results)} queries")
    
    # 3. Test enhanced RAG
    print("\n[3/4] TESTING ENHANCED RAG (WITH ALL IMPROVEMENTS)...")
    print("-" * 50)
    enhanced_results = test_enhanced_rag()
    print(f"Completed {len(enhanced_results)} queries")
    
    # 4. Calculate improvements
    print("\n[4/4] CALCULATING IMPROVEMENTS...")
    print("-" * 50)
    improvements = calculate_improvements(basic_results, enhanced_results)
    
    # 5. Identify issues
    issues = identify_issues(basic_results, enhanced_results, features)
    
    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    print("\n=== FEATURE STATUS ===")
    for f in features:
        status = "✅ WORKING" if f.working else "❌ BROKEN"
        print(f"  {f.feature_name}: {status} ({f.response_time_ms:.0f}ms)")
    
    print(f"\n  TOTAL: {working_count}/{len(features)} features working ({working_count/len(features)*100:.0f}%)")
    
    print("\n=== PERFORMANCE COMPARISON ===")
    if improvements:
        print(f"\n  Response Time:")
        print(f"    Basic:    {improvements['response_time']['basic_ms']:.0f}ms")
        print(f"    Enhanced: {improvements['response_time']['enhanced_ms']:.0f}ms")
        print(f"    Change:   {improvements['response_time']['change_pct']:+.1f}% ({improvements['response_time']['note']})")
        
        print(f"\n  Confidence Score:")
        print(f"    Basic:    {improvements['confidence']['basic']:.0%}")
        print(f"    Enhanced: {improvements['confidence']['enhanced']:.0%}")
        print(f"    Improvement: {improvements['confidence']['improvement_pct']:+.1f}%")
        
        print(f"\n  Answer Grounding:")
        print(f"    Basic:    {improvements['grounding']['basic_pct']:.0f}%")
        print(f"    Enhanced: {improvements['grounding']['enhanced_pct']:.0f}%")
        print(f"    Improvement: {improvements['grounding']['improvement_pct']:+.1f}%")
        
        print(f"\n  Citations per Query:")
        print(f"    Basic:    {improvements['citations']['basic_avg']:.1f}")
        print(f"    Enhanced: {improvements['citations']['enhanced_avg']:.1f}")
        print(f"    Improvement: {improvements['citations']['improvement_pct']:+.1f}%")
        
        print(f"\n  Keyword Accuracy:")
        print(f"    Basic:    {improvements['keyword_accuracy']['basic_pct']:.0f}%")
        print(f"    Enhanced: {improvements['keyword_accuracy']['enhanced_pct']:.0f}%")
        print(f"    Improvement: {improvements['keyword_accuracy']['improvement_pct']:+.1f}%")
        
        print(f"\n  Answer Rate (not 'I don't know'):")
        print(f"    Basic:    {improvements['answer_rate']['basic_pct']:.0f}%")
        print(f"    Enhanced: {improvements['answer_rate']['enhanced_pct']:.0f}%")
        print(f"    Improvement: {improvements['answer_rate']['improvement_pct']:+.1f}%")
    
    print("\n=== QUERY RESULTS BY CATEGORY ===")
    categories = {}
    for r in enhanced_results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    for cat, results in categories.items():
        avg_conf = sum(r.confidence for r in results) / len(results)
        avg_time = sum(r.response_time_ms for r in results) / len(results)
        answered = sum(1 for r in results if r.has_answer)
        print(f"  {cat}: {avg_conf:.0%} confidence, {answered}/{len(results)} answered, {avg_time:.0f}ms avg")
    
    print("\n=== ISSUES FOUND ===")
    if issues:
        for issue in issues:
            print(f"\n  [{issue['severity']}] {issue['area']}")
            print(f"    Issue: {issue['issue']}")
            print(f"    Fix: {issue['recommendation']}")
    else:
        print("  No major issues found!")
    
    # Overall score
    feature_score = working_count / len(features) * 100
    confidence_score = improvements.get('confidence', {}).get('enhanced', 0) * 100 if improvements else 0
    answer_score = improvements.get('answer_rate', {}).get('enhanced_pct', 0) if improvements else 0
    
    overall_score = (feature_score + confidence_score + answer_score) / 3
    
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    print(f"\n  Feature Score:    {feature_score:.0f}%")
    print(f"  Confidence Score: {confidence_score:.0f}%")
    print(f"  Answer Score:     {answer_score:.0f}%")
    print(f"\n  OVERALL SCORE:    {overall_score:.0f}%")
    
    if overall_score >= 80:
        print("\n  STATUS: ✅ EXCELLENT - System is performing well")
    elif overall_score >= 60:
        print("\n  STATUS: 🟡 GOOD - Minor improvements needed")
    else:
        print("\n  STATUS: ❌ NEEDS WORK - See issues above")
    
    # Save results
    results_file = Path(r"G:\evony_rag_index\benchmark_results.json")
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": [asdict(f) for f in features],
        "improvements": improvements,
        "issues": issues,
        "scores": {
            "feature_score": feature_score,
            "confidence_score": confidence_score,
            "answer_score": answer_score,
            "overall": overall_score
        }
    }
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n  Results saved to: {results_file}")
    print("=" * 70)
    
    return results_data


if __name__ == "__main__":
    run_benchmark()
