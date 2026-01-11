"""
Evony RAG - Hybrid Search (Lexical + Semantic Fusion)
======================================================
BM25 lexical search + embedding semantic search with rank fusion.
This is the #1 upgrade for "smart" code RAG.
"""

import re
import os
import sys
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from .config import INDEX_PATH, DATASET_PATH


def _suppress_library_output():
    """Suppress stdout/stderr from libraries (for MCP compatibility)."""
    import warnings
    # Suppress all warnings (numpy, etc.)
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    # Suppress TensorFlow output
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL only
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    # Suppress transformers/sentence_transformers logging
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


@dataclass
class SearchResult:
    """A search result with combined score."""
    chunk_id: str
    file_path: str
    category: str
    start_line: int
    end_line: int
    content: str
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0
    symbols: List[str] = field(default_factory=list)


class BM25Index:
    """BM25 lexical search index for exact matching."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Keep code symbols intact
        text = text.lower()
        # Split on whitespace and punctuation but keep dots in identifiers
        tokens = re.findall(r'\b[\w.]+\b', text)
        return tokens
    
    def build(self, documents: List[Dict]):
        """Build BM25 index from documents."""
        self.documents = documents
        self.doc_lengths = []
        self.doc_freqs = defaultdict(int)
        self.inverted_index = defaultdict(list)
        
        # First pass: collect document frequencies
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc.get('content', ''))
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
            
            # Update inverted index and document frequencies
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_idx, freq))
                self.doc_freqs[term] += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
    
    def _score_bm25(self, doc_idx: int, term_freq: int, term: str) -> float:
        """Calculate BM25 score for a term in a document."""
        N = len(self.documents)
        df = self.doc_freqs.get(term, 0)
        
        if df == 0:
            return 0.0
        
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        doc_len = self.doc_lengths[doc_idx]
        tf_component = (term_freq * (self.k1 + 1)) / (
            term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
        )
        
        return idf * tf_component
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search for documents matching query."""
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Score each document
        scores = defaultdict(float)
        
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_idx, term_freq in self.inverted_index[token]:
                    scores[doc_idx] += self._score_bm25(doc_idx, term_freq, token)
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_results[:top_k]
    
    def save(self, path: Path):
        """Save BM25 index."""
        data = {
            'k1': self.k1,
            'b': self.b,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'doc_freqs': dict(self.doc_freqs),
            'inverted_index': {k: v for k, v in self.inverted_index.items()},
        }
        with open(path / 'bm25_index.json', 'w') as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> bool:
        """Load BM25 index."""
        try:
            with open(path / 'bm25_index.json', 'r') as f:
                data = json.load(f)
            self.k1 = data['k1']
            self.b = data['b']
            self.doc_lengths = data['doc_lengths']
            self.avg_doc_length = data['avg_doc_length']
            self.doc_freqs = defaultdict(int, data['doc_freqs'])
            self.inverted_index = defaultdict(list)
            for k, v in data['inverted_index'].items():
                self.inverted_index[k] = [tuple(x) for x in v]
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            import logging
            logging.getLogger(__name__).debug(f"BM25 index load failed: {e}")
            return False


class SymbolIndex:
    """Index for code symbols (classes, functions, variables)."""
    
    # Patterns for extracting symbols
    PATTERNS = {
        'as_class': r'class\s+(\w+)',
        'as_function': r'function\s+(\w+)',
        'as_var': r'(?:var|const)\s+(\w+)',
        'py_class': r'class\s+(\w+)',
        'py_function': r'def\s+(\w+)',
        'py_var': r'^(\w+)\s*=',
        'command': r'["\'](\w+\.\w+)["\']',
        'constant': r'([A-Z_]{3,})\s*[=:]',
    }
    
    def __init__(self):
        self.symbols: Dict[str, List[Dict]] = defaultdict(list)
        
    def extract_symbols(self, content: str, file_path: str, 
                       category: str, start_line: int) -> List[str]:
        """Extract code symbols from content."""
        found = []
        
        for pattern_name, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, content, re.MULTILINE):
                symbol = match.group(1)
                if len(symbol) >= 2:
                    found.append(symbol)
                    
                    # Calculate approximate line number
                    line_offset = content[:match.start()].count('\n')
                    
                    self.symbols[symbol.lower()].append({
                        'symbol': symbol,
                        'file': file_path,
                        'category': category,
                        'line': start_line + line_offset,
                        'context': content[max(0, match.start()-50):match.end()+100],
                    })
        
        return found
    
    def find_symbol(self, name: str) -> List[Dict]:
        """Find all occurrences of a symbol."""
        name_lower = name.lower()
        
        # Exact match
        if name_lower in self.symbols:
            return self.symbols[name_lower]
        
        # Partial match
        results = []
        for sym, occurrences in self.symbols.items():
            if name_lower in sym or sym in name_lower:
                results.extend(occurrences)
        
        return results[:20]
    
    def save(self, path: Path):
        """Save symbol index."""
        with open(path / 'symbol_index.json', 'w') as f:
            json.dump(dict(self.symbols), f)
    
    def load(self, path: Path) -> bool:
        """Load symbol index."""
        try:
            with open(path / 'symbol_index.json', 'r') as f:
                data = json.load(f)
            self.symbols = defaultdict(list, data)
            return True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            import logging
            logging.getLogger(__name__).debug(f"Symbol index load failed: {e}")
            return False


class HybridSearch:
    """Hybrid search combining BM25 and embeddings with rank fusion."""
    
    def __init__(self):
        self.bm25 = BM25Index()
        self.symbols = SymbolIndex()
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.embedding_model = None
        
    def load_index(self, index_path: Path = INDEX_PATH) -> bool:
        """Load all indexes."""
        try:
            # Load chunks
            with open(index_path / 'chunks.json', 'r') as f:
                self.chunks = json.load(f)
            
            # Load embeddings from LARGE_INDEX_PATH (configurable via EVONY_INDEX_PATH env var)
            from .config import LARGE_INDEX_PATH
            embeddings_path = LARGE_INDEX_PATH / 'embeddings.npy'
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
            else:
                # Fallback to index_path
                self.embeddings = np.load(index_path / 'embeddings.npy')
            
            # Load or build BM25
            if not self.bm25.load(index_path):
                self.bm25.build(self.chunks)
                self.bm25.save(index_path)
            
            # Load or build symbol index
            if not self.symbols.load(index_path):
                for chunk in self.chunks:
                    self.symbols.extract_symbols(
                        chunk['content'],
                        chunk['file_path'],
                        chunk['category'],
                        chunk['start_line']
                    )
                self.symbols.save(index_path)
            
            # Use LM Studio for embeddings (avoids torch DLL issues)
            self.embedding_model = "lmstudio"  # Flag for _semantic_search
            
            return True
        except Exception as e:
            # Log error to file (NOT stdout/stderr - would corrupt MCP)
            import traceback
            log_file = INDEX_PATH.parent / "logs" / "index_error.log"
            log_file.parent.mkdir(exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"\n=== load_index error ===\n{traceback.format_exc()}\n")
            return False
    
    def _semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Semantic search using embeddings."""
        if self.embedding_model is None or self.embeddings is None:
            # Fallback: return empty results if model/embeddings not loaded
            return []
        
        try:
            # Use LM Studio API for query embedding
            if self.embedding_model == "lmstudio":
                import requests
                resp = requests.post(
                    "http://localhost:1234/v1/embeddings",
                    json={"model": "text-embedding-nomic-embed-text-v1.5", "input": [query]},
                    timeout=30
                )
                query_embedding = np.array(resp.json()["data"][0]["embedding"])
            else:
                query_embedding = self.embedding_model.encode([query])[0]
            
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
            )
        except Exception:
            return []
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(idx, float(similarities[idx])) for idx in top_indices]
    
    def _reciprocal_rank_fusion(self, 
                                lexical_results: List[Tuple[int, float]],
                                semantic_results: List[Tuple[int, float]],
                                k: int = 60) -> List[Tuple[int, float, float, float]]:
        """Combine results using Reciprocal Rank Fusion."""
        scores = defaultdict(lambda: {'lexical': 0.0, 'semantic': 0.0, 'rrf': 0.0})
        
        # Score lexical results
        for rank, (doc_idx, score) in enumerate(lexical_results):
            scores[doc_idx]['lexical'] = score
            scores[doc_idx]['rrf'] += 1.0 / (k + rank + 1)
        
        # Score semantic results
        for rank, (doc_idx, score) in enumerate(semantic_results):
            scores[doc_idx]['semantic'] = score
            scores[doc_idx]['rrf'] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score
        sorted_results = sorted(
            [(idx, s['rrf'], s['lexical'], s['semantic']) 
             for idx, s in scores.items()],
            key=lambda x: -x[1]
        )
        
        return sorted_results
    
    def search(self, query: str, 
               k_lexical: int = 20,
               k_vector: int = 20,
               final_k: int = 8,
               categories: List[str] = None,
               min_score: float = 0.001) -> List[SearchResult]:
        """Hybrid search with rank fusion."""
        
        # Get lexical results
        lexical_results = self.bm25.search(query, top_k=k_lexical)
        
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k=k_vector)
        
        # Fuse results
        fused = self._reciprocal_rank_fusion(lexical_results, semantic_results)
        
        # Build results
        results = []
        for doc_idx, rrf_score, lex_score, sem_score in fused[:final_k * 2]:
            # Guard against invalid index
            if doc_idx < 0 or doc_idx >= len(self.chunks):
                continue
            chunk = self.chunks[doc_idx]
            
            # Filter by category
            if categories and chunk['category'] not in categories:
                continue
            
            # Filter by minimum score
            if rrf_score < min_score:
                continue
            
            results.append(SearchResult(
                chunk_id=chunk['id'],
                file_path=chunk['file_path'],
                category=chunk['category'],
                start_line=chunk['start_line'],
                end_line=chunk['end_line'],
                content=chunk['content'],
                lexical_score=lex_score,
                semantic_score=sem_score,
                combined_score=rrf_score,
            ))
            
            if len(results) >= final_k:
                break
        
        return results
    
    def find_symbol(self, name: str) -> List[Dict]:
        """Find symbol definitions/usages."""
        return self.symbols.find_symbol(name)


# Singleton with thread safety
import threading
_hybrid_search = None
_hybrid_search_lock = threading.Lock()

def get_hybrid_search() -> HybridSearch:
    """Get singleton hybrid search instance (thread-safe)."""
    global _hybrid_search
    if _hybrid_search is None:
        with _hybrid_search_lock:
            # Double-check after acquiring lock
            if _hybrid_search is None:
                # Suppress library output before any loading (for MCP compatibility)
                _suppress_library_output()
                hs = HybridSearch()
                hs.load_index()
                _hybrid_search = hs
    return _hybrid_search

def reload_hybrid_search() -> HybridSearch:
    """Force reload the hybrid search index (use after updating chunks)."""
    global _hybrid_search
    with _hybrid_search_lock:
        _suppress_library_output()
        hs = HybridSearch()
        hs.load_index()
        _hybrid_search = hs
    return _hybrid_search
