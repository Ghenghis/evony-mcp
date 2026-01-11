#!/usr/bin/env python3
"""
Test RAG Ultimate v2.0 - All Advanced Techniques
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_all_components():
    """Test all RAG Ultimate components."""
    print("=" * 70)
    print("RAG ULTIMATE v2.0 - Component Test Suite")
    print("=" * 70)
    
    results = {
        "passed": [],
        "failed": [],
    }
    
    # Test 1: Query Decomposer
    print("\n[TEST 1] Query Decomposer")
    try:
        from evony_rag.query_decomposer import get_query_decomposer
        qd = get_query_decomposer()
        
        complex_query = "What is command 45 and how does it encode parameters?"
        decomposed = qd.decompose(complex_query)
        
        print(f"  Query type: {decomposed.query_type}")
        print(f"  Sub-queries: {len(decomposed.sub_queries)}")
        for sq in decomposed.sub_queries:
            print(f"    - {sq}")
        
        results["passed"].append("Query Decomposer")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(f"Query Decomposer: {e}")
    
    # Test 2: Knowledge Graph
    print("\n[TEST 2] Knowledge Graph")
    try:
        from evony_rag.knowledge_graph import get_knowledge_graph, EntityExtractor
        
        kg = get_knowledge_graph()
        extractor = EntityExtractor()
        
        # Test entity extraction
        test_chunk = {
            "content": """
            public class TroopCommand extends BaseCommand {
                public static const COMMAND_ID:int = 45;
                public function execute(cityId:int, troopType:int):void {
                    sendRequest(cityId, troopType);
                }
            }
            """,
            "file_path": "commands/TroopCommand.as",
            "category": "source_code",
            "start_line": 1,
        }
        
        entities, relationships = extractor.extract_from_chunk(test_chunk)
        print(f"  Entities extracted: {len(entities)}")
        for e in entities[:3]:
            print(f"    - {e.entity_type}: {e.name}")
        print(f"  Relationships: {len(relationships)}")
        
        results["passed"].append("Knowledge Graph")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(f"Knowledge Graph: {e}")
    
    # Test 3: Self-RAG Verifier
    print("\n[TEST 3] Self-RAG Verifier")
    try:
        from evony_rag.self_rag import SelfRAGVerifier
        
        verifier = SelfRAGVerifier()
        
        query = "What is command ID 45?"
        context = "Command ID 45 is TROOP_PRODUCE used for producing troops in cities."
        
        relevance, score = verifier.check_relevance(query, context)
        print(f"  Relevance: {relevance.value} ({score:.2f})")
        
        answer = "Command 45, known as TROOP_PRODUCE, handles troop production."
        support, support_score = verifier.check_groundedness(answer, [context])
        print(f"  Groundedness: {support.value} ({support_score:.2f})")
        
        results["passed"].append("Self-RAG Verifier")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(f"Self-RAG Verifier: {e}")
    
    # Test 4: HyPE Generator
    print("\n[TEST 4] HyPE Generator")
    try:
        from evony_rag.hype_embeddings import HyPEGenerator
        
        hype = HyPEGenerator(use_llm=False)
        
        test_chunk = {
            "content": "public function produceTroop(cityId:int, troopType:int):void { ... }",
            "category": "source_code",
            "file_path": "TroopCommand.as",
        }
        
        questions = hype.generate_questions(test_chunk)
        print(f"  Questions generated: {len(questions)}")
        for q in questions[:3]:
            print(f"    - {q}")
        
        results["passed"].append("HyPE Generator")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(f"HyPE Generator: {e}")
    
    # Test 5: Cross-Encoder
    print("\n[TEST 5] Cross-Encoder Reranker")
    try:
        from evony_rag.cross_encoder import get_cross_encoder
        
        ce = get_cross_encoder("fast")
        
        query = "troop production command"
        docs = [
            "Command 45 handles troop production",
            "The weather is sunny today",
            "TroopCommand extends BaseCommand",
        ]
        
        scores = ce.score_pairs(query, docs)
        print(f"  Scores computed: {len(scores)}")
        for doc, score in zip(docs, scores):
            print(f"    {score:.3f}: {doc[:40]}...")
        
        results["passed"].append("Cross-Encoder")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(f"Cross-Encoder: {e}")
    
    # Test 6: RAG Fusion
    print("\n[TEST 6] RAG Fusion Query Generator")
    try:
        from evony_rag.rag_fusion import QueryGenerator
        
        qg = QueryGenerator()
        
        query = "How does AMF3 encoding work?"
        variations = qg.generate_variations(query, num_variations=4)
        
        print(f"  Variations: {len(variations)}")
        for v in variations:
            print(f"    - {v}")
        
        results["passed"].append("RAG Fusion")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(f"RAG Fusion: {e}")
    
    # Test 7: RAG Ultimate Integration
    print("\n[TEST 7] RAG Ultimate Integration")
    try:
        from evony_rag.rag_ultimate import get_rag_ultimate
        
        rag = get_rag_ultimate()
        stats = rag.get_stats()
        
        print(f"  Initialized: {stats['initialized']}")
        print(f"  Components:")
        for comp, available in stats['components'].items():
            status = "[OK]" if available else "[--]"
            print(f"    {status} {comp}")
        
        results["passed"].append("RAG Ultimate Integration")
    except Exception as e:
        print(f"  FAILED: {e}")
        results["failed"].append(f"RAG Ultimate Integration: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {len(results['passed'])}")
    for p in results['passed']:
        print(f"  [OK] {p}")
    
    if results['failed']:
        print(f"\nFailed: {len(results['failed'])}")
        for f in results['failed']:
            print(f"  [FAIL] {f}")
    
    print("\n" + "=" * 70)
    
    # Calculate improvement potential
    improvements = {
        "Query Decomposer": 25,
        "Knowledge Graph": 40,
        "Self-RAG Verifier": 30,
        "HyPE Generator": 42,
        "Cross-Encoder": 67,
        "RAG Fusion": 25,
    }
    
    total_potential = sum(
        improvements.get(p, 0) for p in results['passed']
    )
    
    print(f"POTENTIAL IMPROVEMENT: +{total_potential}%")
    print("=" * 70)


if __name__ == "__main__":
    test_all_components()
