"""
Cross-Encoder Neural Reranking for RAG v2.0
============================================
Uses transformer cross-encoder for precise query-document relevance scoring.
Impact: -67% retrieval failures (Anthropic research)
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Suppress library output for MCP compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


@dataclass
class RankedResult:
    """Result with cross-encoder score."""
    chunk_id: str
    file_path: str
    category: str
    start_line: int
    end_line: int
    content: str
    original_score: float
    cross_encoder_score: float
    final_score: float


class CrossEncoderReranker:
    """
    Neural cross-encoder for precise reranking.
    
    Cross-encoders evaluate query-document pairs jointly, enabling:
    - Fine-grained intent matching
    - Semantic alignment detection
    - Answer completeness scoring
    """
    
    # Model options (from fastest to most accurate)
    MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",      # ~50ms/100 pairs
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2", # ~100ms/100 pairs
        "accurate": "BAAI/bge-reranker-base",                # ~150ms/100 pairs
        "best": "BAAI/bge-reranker-large",                   # ~300ms/100 pairs
    }
    
    def __init__(self, model_name: str = "fast", device: str = "cpu"):
        """
        Initialize cross-encoder.
        
        Args:
            model_name: One of "fast", "balanced", "accurate", "best"
            device: "cpu" or "cuda"
        """
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = device
        self.model = None
        self._loaded = False
        
    def _lazy_load(self):
        """Lazy load model on first use."""
        if self._loaded:
            return
            
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            self._loaded = True
        except ImportError:
            logging.warning("sentence-transformers not available, using fallback scoring")
            self._loaded = True
        except Exception as e:
            logging.error(f"Failed to load cross-encoder: {e}")
            self._loaded = True
    
    def score_pairs(self, query: str, documents: List[str]) -> List[float]:
        """
        Score query-document pairs.
        
        Args:
            query: The search query
            documents: List of document texts
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        self._lazy_load()
        
        if self.model is None:
            # Fallback: simple keyword overlap scoring
            return self._fallback_score(query, documents)
        
        # Create pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    def _fallback_score(self, query: str, documents: List[str]) -> List[float]:
        """Fallback scoring using keyword overlap."""
        query_terms = set(query.lower().split())
        scores = []
        
        for doc in documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)
            scores.append(score)
            
        return scores
    
    def rerank(
        self, 
        query: str, 
        results: List[Dict],
        top_k: int = 20,
        content_key: str = "content"
    ) -> List[Dict]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search result dicts
            top_k: Number of results to return
            content_key: Key for document content in result dict
            
        Returns:
            Reranked and filtered results
        """
        if not results:
            return []
        
        # Extract document contents
        documents = [r.get(content_key, r.get("snippet", "")) for r in results]
        
        # Score with cross-encoder
        scores = self.score_pairs(query, documents)
        
        # Add scores to results
        scored_results = []
        for result, score in zip(results, scores):
            result = result.copy()
            result["cross_encoder_score"] = float(score)
            # Combine with original score (weighted average)
            original = result.get("combined_score", result.get("score", 0.5))
            result["final_score"] = 0.3 * original + 0.7 * float(score)
            scored_results.append(result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return scored_results[:top_k]


class AdaptiveReranker:
    """
    Adaptive reranking that adjusts strategy based on query type.
    """
    
    def __init__(self):
        self.cross_encoder = CrossEncoderReranker("fast")
        
    def classify_query(self, query: str) -> str:
        """Classify query type for adaptive strategy."""
        query_lower = query.lower()
        
        # Factual queries (direct lookups)
        if any(kw in query_lower for kw in ["what is", "command id", "enum", "constant"]):
            return "factual"
        
        # Analytical queries (need synthesis)
        if any(kw in query_lower for kw in ["explain", "how does", "why", "analyze"]):
            return "analytical"
        
        # Code queries
        if any(kw in query_lower for kw in ["code", "function", "class", "implement"]):
            return "code"
        
        # Exploit/security queries
        if any(kw in query_lower for kw in ["exploit", "vulnerability", "cve", "attack"]):
            return "security"
        
        return "general"
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 20) -> List[Dict]:
        """
        Adaptively rerank based on query type.
        """
        query_type = self.classify_query(query)
        
        # Apply type-specific boosting
        for result in results:
            boost = 1.0
            content = result.get("content", "").lower()
            file_path = result.get("file_path", "").lower()
            
            if query_type == "factual":
                # Boost enum/constant files
                if "constant" in file_path or "enum" in file_path:
                    boost *= 1.3
                    
            elif query_type == "code":
                # Boost source code
                if ".as" in file_path or "source_code" in file_path:
                    boost *= 1.3
                # Boost definitions
                if "public function" in content or "public class" in content:
                    boost *= 1.2
                    
            elif query_type == "security":
                # Boost exploit/CVE content
                if "exploit" in file_path or "cve" in file_path:
                    boost *= 1.4
                    
            elif query_type == "analytical":
                # Boost documentation
                if ".md" in file_path or "doc" in file_path:
                    boost *= 1.2
            
            result["type_boost"] = boost
        
        # Apply cross-encoder reranking
        reranked = self.cross_encoder.rerank(query, results, top_k * 2)
        
        # Apply type boost to final score
        for result in reranked:
            result["final_score"] *= result.get("type_boost", 1.0)
        
        # Re-sort and return top_k
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        
        return reranked[:top_k]


# Singleton instances
_cross_encoder = None
_adaptive_reranker = None


def get_cross_encoder(model: str = "fast") -> CrossEncoderReranker:
    """Get singleton cross-encoder instance."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker(model)
    return _cross_encoder


def get_adaptive_reranker() -> AdaptiveReranker:
    """Get singleton adaptive reranker instance."""
    global _adaptive_reranker
    if _adaptive_reranker is None:
        _adaptive_reranker = AdaptiveReranker()
    return _adaptive_reranker
