#!/usr/bin/env python3
"""Confidence Scorer for RAG Responses"""

from typing import List, Dict, Tuple

class ConfidenceScorer:
    """Scores confidence of RAG responses"""
    
    def score_results(self, results: List[Dict], query: str) -> Tuple[float, str]:
        """
        Score search results confidence.
        Returns (score 0-1, explanation)
        """
        if not results:
            return (0.0, "No results found")
        
        score = 0.0
        reasons = []
        
        # Factor 1: Number of results (max 0.2)
        result_score = min(0.2, len(results) * 0.04)
        score += result_score
        reasons.append(f"{len(results)} results")
        
        # Factor 2: Relevance scores (max 0.3)
        if results:
            avg_relevance = sum(r.get("score", 0) for r in results) / len(results)
            relevance_score = min(0.3, avg_relevance * 0.3)
            score += relevance_score
            reasons.append(f"avg relevance {avg_relevance:.2f}")
        
        # Factor 3: Query term matches (max 0.3)
        query_terms = set(query.lower().split())
        matches = 0
        for result in results[:5]:
            content = result.get("snippet", result.get("content", "")).lower()
            matches += sum(1 for term in query_terms if term in content)
        
        match_score = min(0.3, matches * 0.05)
        score += match_score
        reasons.append(f"{matches} term matches")
        
        # Factor 4: Source diversity (max 0.2)
        files = set(r.get("file", "") for r in results[:10])
        diversity_score = min(0.2, len(files) * 0.04)
        score += diversity_score
        reasons.append(f"{len(files)} unique files")
        
        explanation = ", ".join(reasons)
        return (min(1.0, score), explanation)
    
    def should_answer(self, score: float) -> Tuple[bool, str]:
        """Determine if confidence is high enough to answer"""
        if score >= 0.7:
            return (True, "High confidence - proceed with answer")
        elif score >= 0.4:
            return (True, "Medium confidence - answer with caveats")
        else:
            return (False, "Low confidence - request more evidence")
    
    def format_confidence_indicator(self, score: float) -> str:
        """Format confidence as visual indicator"""
        if score >= 0.8:
            return "🟢 High Confidence"
        elif score >= 0.6:
            return "🟡 Medium Confidence"
        elif score >= 0.4:
            return "🟠 Low Confidence"
        else:
            return "🔴 Insufficient Evidence"


# Singleton
_scorer = None

def get_scorer() -> ConfidenceScorer:
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer
