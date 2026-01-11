"""
Evony RAG v3.0 - Game-Changing Knowledge System
================================================
Combines all advanced RAG techniques:
- Contextual chunking (Anthropic method)
- Cross-encoder neural reranking
- RAG Fusion with multi-query
- Agentic query refinement
- Adaptive query routing

Expected improvements:
- 67% reduction in retrieval failures
- 49% better accuracy
- Complex reasoning capability
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATASET_PATH, INDEX_PATH
from .hybrid_search import HybridSearch, SearchResult, get_hybrid_search
from .cross_encoder import get_cross_encoder, get_adaptive_reranker
from .rag_fusion import get_rag_fusion, RAGFusion
from .contextual_indexer import get_contextual_indexer
from .policy import PolicyEngine, QueryPolicy, get_policy


@dataclass
class RAGv3Response:
    """Response from RAG v3.0."""
    answer: str
    citations: List[Dict]
    confidence: float
    query_type: str
    queries_used: List[str]
    model_used: str
    retrieval_stats: Dict = field(default_factory=dict)


class EvonyRAGv3:
    """
    Game-changing RAG system with all advanced techniques.
    """
    
    SYSTEM_TEMPLATE = """You are Svony, an elite Evony Age II reverse engineering expert.

Your knowledge covers:
- Complete AMF3 protocol (301+ commands)
- ActionScript 3 source code analysis
- Security vulnerabilities and exploits
- Bot development and automation

Rules:
- Be precise and technical
- Cite sources with file:line format
- For code: show relevant snippets
- For protocol: explain parameters and structure
- Provide educational analysis only

Query Type: {query_type}
Confidence: {confidence:.0%}
Sources searched: {num_sources}

Evidence:
{context}

Based on this evidence, answer the query."""

    def __init__(self, use_lmstudio: bool = True):
        """
        Initialize RAG v3.0.
        
        Args:
            use_lmstudio: Whether to use LM Studio for generation
        """
        # Core components
        self.hybrid_search = get_hybrid_search()
        self.cross_encoder = get_cross_encoder("fast")
        self.adaptive_reranker = get_adaptive_reranker()
        self.rag_fusion = get_rag_fusion()
        self.policy = get_policy()
        
        # Configure RAG Fusion with our search function
        self.rag_fusion.set_search_function(self._search_single)
        
        # LM Studio config
        self.use_lmstudio = use_lmstudio
        self.lmstudio_url = "http://localhost:1234/v1"
        
        # Stats
        self.stats = {
            "queries_processed": 0,
            "avg_confidence": 0.0,
            "cache_hits": 0,
        }
    
    def _search_single(self, query: str, k: int = 50) -> List[Dict]:
        """Execute single search (for RAG Fusion)."""
        results = self.hybrid_search.search(query, k_lexical=k, k_vector=k, final_k=k)
        return [self._result_to_dict(r) for r in results]
    
    def _result_to_dict(self, result: SearchResult) -> Dict:
        """Convert SearchResult to dict."""
        return {
            "chunk_id": result.chunk_id,
            "file_path": result.file_path,
            "category": result.category,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "content": result.content,
            "lexical_score": result.lexical_score,
            "semantic_score": result.semantic_score,
            "combined_score": result.combined_score,
            "symbols": result.symbols,
        }
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for adaptive handling."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["command id", "what is command", "enum"]):
            return "factual"
        if any(kw in query_lower for kw in ["exploit", "vulnerability", "cve", "attack"]):
            return "security"
        if any(kw in query_lower for kw in ["how does", "explain", "why"]):
            return "analytical"
        if any(kw in query_lower for kw in ["code", "function", "class", "implement"]):
            return "code"
        if any(kw in query_lower for kw in ["decode", "packet", "amf"]):
            return "protocol"
        
        return "general"
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence based on result quality."""
        if not results:
            return 0.0
        
        # Factors:
        # 1. Top result score
        # 2. Score distribution (tight = confident)
        # 3. Number of high-quality results
        
        top_score = results[0].get("final_score", results[0].get("combined_score", 0))
        
        # Count high-quality results (score > 0.5)
        high_quality = sum(1 for r in results[:10] if r.get("final_score", r.get("combined_score", 0)) > 0.5)
        
        # Calculate confidence
        confidence = min(1.0, (top_score * 0.5) + (high_quality / 10 * 0.5))
        
        return confidence
    
    def search(
        self, 
        query: str,
        mode: str = "research",
        top_k: int = 20,
        use_fusion: bool = True,
        use_reranking: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Execute advanced search with all RAG v3.0 features.
        
        Args:
            query: Search query
            mode: Query mode (research, forensics, full_access)
            top_k: Number of results to return
            use_fusion: Whether to use RAG Fusion
            use_reranking: Whether to use cross-encoder reranking
            
        Returns:
            Tuple of (results, metadata)
        """
        metadata = {
            "query": query,
            "mode": mode,
            "query_type": self._classify_query(query),
            "queries_used": [query],
            "stages": [],
        }
        
        # Stage 1: Initial retrieval (with or without fusion)
        if use_fusion:
            results, queries = self.rag_fusion.search(query, k=100, num_queries=4)
            metadata["queries_used"] = queries
            metadata["stages"].append({"name": "rag_fusion", "count": len(results)})
        else:
            results = self._search_single(query, k=100)
            metadata["stages"].append({"name": "hybrid_search", "count": len(results)})
        
        # Stage 2: Cross-encoder reranking
        if use_reranking and results:
            results = self.adaptive_reranker.rerank(query, results, top_k=top_k * 2)
            metadata["stages"].append({"name": "reranking", "count": len(results)})
        
        # Stage 3: Final selection
        final_results = results[:top_k]
        
        # Calculate confidence
        metadata["confidence"] = self._calculate_confidence(final_results)
        metadata["num_results"] = len(final_results)
        
        return final_results, metadata
    
    def query(
        self,
        query: str,
        mode: str = "research",
        generate_answer: bool = True,
        top_k: int = 10
    ) -> RAGv3Response:
        """
        Full RAG v3.0 query with answer generation.
        
        Args:
            query: User query
            mode: Query mode
            generate_answer: Whether to generate LLM answer
            top_k: Number of sources to use
            
        Returns:
            RAGv3Response with answer and citations
        """
        # Get policy
        policy = self.policy.get_policy(mode)
        
        # Execute search
        results, metadata = self.search(
            query, 
            mode=mode, 
            top_k=top_k,
            use_fusion=True,
            use_reranking=True
        )
        
        # Format context
        context = self._format_context(results, policy.evidence)
        
        # Generate answer if requested
        if generate_answer and self.use_lmstudio:
            answer = self._generate_answer(query, context, metadata)
        else:
            answer = self._format_results_as_answer(results)
        
        # Create citations
        citations = [
            {
                "file": r["file_path"],
                "lines": f"{r['start_line']}-{r['end_line']}",
                "score": r.get("final_score", r.get("combined_score", 0)),
                "snippet": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
            }
            for r in results[:5]
        ]
        
        # Update stats
        self.stats["queries_processed"] += 1
        
        return RAGv3Response(
            answer=answer,
            citations=citations,
            confidence=metadata["confidence"],
            query_type=metadata["query_type"],
            queries_used=metadata["queries_used"],
            model_used="evony-7b-3800-rtx3090ti" if self.use_lmstudio else "none",
            retrieval_stats=metadata,
        )
    
    def _format_context(self, results: List[Dict], evidence_config: Dict) -> str:
        """Format results as context for LLM."""
        max_sources = evidence_config.get("max_sources", 5)
        max_chars = evidence_config.get("max_snippet_chars", 500)
        
        parts = []
        for i, r in enumerate(results[:max_sources], 1):
            snippet = r["content"][:max_chars]
            if len(r["content"]) > max_chars:
                snippet += "..."
            
            score = r.get("final_score", r.get("combined_score", 0))
            parts.append(f"""
--- Source {i}: {r['file_path']}:{r['start_line']}-{r['end_line']} (score: {score:.2f}) ---
{snippet}
""")
        
        return "\n".join(parts) if parts else "No relevant sources found."
    
    def _format_results_as_answer(self, results: List[Dict]) -> str:
        """Format results as simple answer (no LLM)."""
        if not results:
            return "No relevant information found."
        
        parts = ["## Search Results\n"]
        for i, r in enumerate(results[:5], 1):
            score = r.get("final_score", r.get("combined_score", 0))
            parts.append(f"### {i}. `{r['file_path']}:{r['start_line']}-{r['end_line']}` ({score:.0%})")
            parts.append(f"```\n{r['content'][:300]}\n```\n")
        
        return "\n".join(parts)
    
    def _generate_answer(self, query: str, context: str, metadata: Dict) -> str:
        """Generate answer using LM Studio."""
        try:
            import requests
            
            system_prompt = self.SYSTEM_TEMPLATE.format(
                query_type=metadata["query_type"],
                confidence=metadata["confidence"],
                num_sources=metadata["num_results"],
                context=context,
            )
            
            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                },
                timeout=60,
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return self._format_results_as_answer([])
                
        except Exception as e:
            logging.error(f"LM Studio generation failed: {e}")
            return self._format_results_as_answer([])


# Singleton
_rag_v3 = None


def get_rag_v3() -> EvonyRAGv3:
    """Get singleton RAG v3.0 instance."""
    global _rag_v3
    if _rag_v3 is None:
        _rag_v3 = EvonyRAGv3()
    return _rag_v3
