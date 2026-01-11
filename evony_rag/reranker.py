#!/usr/bin/env python3
"""Result Reranker for improved RAG precision"""

from typing import List, Dict, Tuple
import re
import requests

LMSTUDIO_URL = "http://localhost:1234/v1"


class CrossEncoderReranker:
    """
    LM Studio-based cross-encoder reranking.
    Uses the LLM to score query-document relevance.
    More accurate but slower than heuristic reranking.
    """
    
    def __init__(self, model: str = "evony-7b-3800"):
        self.model = model
        self.enabled = True
    
    def score_relevance(self, query: str, document: str) -> float:
        """Score relevance of document to query using LLM."""
        prompt = f"""Rate how relevant this document is to the query on a scale of 0-10.
Only output a single number.

Query: {query}

Document:
{document[:500]}

Relevance score (0-10):"""
        
        try:
            resp = requests.post(
                f"{LMSTUDIO_URL}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 5
                },
                timeout=10
            )
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            # Extract number
            nums = re.findall(r'(\d+(?:\.\d+)?)', answer)
            if nums:
                return min(float(nums[0]) / 10.0, 1.0)
        except:
            pass
        return 0.5  # Default if scoring fails
    
    def rerank(self, results: List[Dict], query: str, top_k: int = 5) -> List[Dict]:
        """Rerank top results using cross-encoder scoring."""
        if not self.enabled or len(results) == 0:
            return results
        
        # Only rerank top candidates (expensive operation)
        candidates = results[:min(len(results), top_k * 2)]
        
        scored = []
        for r in candidates:
            content = r.get("content", r.get("snippet", ""))
            score = self.score_relevance(query, content)
            r["cross_encoder_score"] = score
            scored.append((score, r))
        
        # Sort by cross-encoder score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Return reranked + remaining
        reranked = [r for _, r in scored]
        remaining = results[len(candidates):]
        
        return reranked + remaining


class ResultReranker:
    """Reranks search results for better relevance"""
    
    def __init__(self):
        self.boost_patterns = {
            # Boost exact matches
            "exact_match": 2.0,
            # Boost code definitions
            "definition": 1.5,
            # Boost public APIs
            "public_api": 1.3,
            # Boost documentation
            "documentation": 1.2,
        }
    
    def rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Rerank results based on query relevance"""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        scored_results = []
        for result in results:
            score = result.get("score", 0.5)
            content = result.get("snippet", result.get("content", "")).lower()
            file_path = result.get("file", "").lower()
            
            # Boost exact term matches
            for term in query_terms:
                if len(term) > 3 and term in content:
                    score *= 1.2
                if term in file_path:
                    score *= 1.1
            
            # Boost definitions
            if re.search(r"(public|private|protected)\s+(static\s+)?(const|var|function)", content):
                score *= self.boost_patterns["definition"]
            
            # Boost public APIs
            if "public static" in content or "public function" in content:
                score *= self.boost_patterns["public_api"]
            
            # Boost .as files for ActionScript queries
            if ".as" in file_path and any(kw in query_lower for kw in ["command", "bean", "constant"]):
                score *= 1.3
            
            # Penalize very short snippets
            if len(content) < 50:
                score *= 0.7
            
            result["reranked_score"] = score
            scored_results.append(result)
        
        # Sort by reranked score
        scored_results.sort(key=lambda x: x.get("reranked_score", 0), reverse=True)
        
        return scored_results
    
    def deduplicate(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate or near-duplicate results"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content = result.get("snippet", result.get("content", ""))[:100]
            content_hash = hash(content.strip())
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results


# Singletons
_reranker = None
_cross_encoder = None

def get_reranker() -> ResultReranker:
    global _reranker
    if _reranker is None:
        _reranker = ResultReranker()
    return _reranker

def get_cross_encoder() -> CrossEncoderReranker:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker()
    return _cross_encoder
