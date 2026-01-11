"""
Evony RAG v2 - Enhanced Engine
===============================
Hybrid search + policy controls + symbol resolution.
"""

import json
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATASET_PATH
from .hybrid_search import HybridSearch, SearchResult, get_hybrid_search
from .policy import PolicyEngine, QueryPolicy, get_policy


@dataclass
class Citation:
    """Citation with full metadata."""
    file_path: str
    category: str
    start_line: int
    end_line: int
    lexical_score: float
    semantic_score: float
    combined_score: float
    snippet: str
    
    def format_brief(self) -> str:
        return f"`{self.file_path}:{self.start_line}-{self.end_line}` ({self.combined_score:.0%})"
    
    def format_full(self) -> str:
        return f"""📄 **{self.file_path}**:{self.start_line}-{self.end_line}
Category: {self.category} | Score: {self.combined_score:.0%} (lex:{self.lexical_score:.2f}, sem:{self.semantic_score:.2f})
```
{self.snippet}
```"""


@dataclass
class RAGResponse:
    """Response from RAG v2."""
    answer: str
    citations: List[Citation]
    policy: QueryPolicy
    model_used: str
    symbols_found: List[Dict] = field(default_factory=list)


class EvonyRAGv2:
    """Enhanced RAG with hybrid search and policy controls."""
    
    SYSTEM_TEMPLATE = """You are an Evony reverse engineering expert. Answer based on the provided context.

Rules:
- Be precise and technical
- Cite sources with file:line format
- For code: show relevant snippets
- For protocol: explain parameters and structure
- Do NOT provide operational cheating instructions
- Educational analysis is OK

Mode: {mode}
Categories searched: {categories}

Context:
{context}
"""
    
    def __init__(self):
        self.search = get_hybrid_search()
        self.policy = get_policy()
        self.lmstudio_url = "http://localhost:1234/v1"
        
    def _format_context(self, results: List[SearchResult], 
                        evidence_config: Dict) -> str:
        """Format search results as context."""
        max_sources = evidence_config.get('max_sources', 5)
        show_snippets = evidence_config.get('show_snippets', True)
        max_chars = evidence_config.get('max_snippet_chars', 500)
        
        parts = []
        for i, r in enumerate(results[:max_sources], 1):
            if show_snippets:
                snippet = r.content[:max_chars]
                if len(r.content) > max_chars:
                    snippet += "..."
                parts.append(f"""
--- Source {i}: {r.file_path}:{r.start_line}-{r.end_line} (score: {r.combined_score:.2f}) ---
{snippet}
""")
            else:
                parts.append(f"- {r.file_path}:{r.start_line}-{r.end_line}")
        
        return "\n".join(parts) if parts else "No relevant context found."
    
    def _create_citations(self, results: List[SearchResult],
                          evidence_config: Dict) -> List[Citation]:
        """Create citations from results."""
        max_sources = evidence_config.get('max_sources', 5)
        max_chars = evidence_config.get('max_snippet_chars', 500)
        
        citations = []
        for r in results[:max_sources]:
            snippet = r.content[:max_chars]
            if len(r.content) > max_chars:
                snippet += "..."
            
            citations.append(Citation(
                file_path=r.file_path,
                category=r.category,
                start_line=r.start_line,
                end_line=r.end_line,
                lexical_score=r.lexical_score,
                semantic_score=r.semantic_score,
                combined_score=r.combined_score,
                snippet=snippet,
            ))
        
        return citations
    
    def _call_lmstudio(self, prompt: str, system: str) -> Optional[str]:
        """Call LM Studio for generation."""
        try:
            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except:
            return None
    
    def _generate_standalone(self, query: str, 
                             citations: List[Citation]) -> str:
        """Generate answer without LLM."""
        parts = [f"**Query:** {query}\n"]
        
        if citations:
            parts.append("**Relevant Sources:**\n")
            for cit in citations:
                parts.append(cit.format_full())
                parts.append("")
        else:
            parts.append("No relevant sources found.")
        
        return "\n".join(parts)
    
    def query(self, query: str,
              mode: str = None,
              include: List[str] = None,
              exclude: List[str] = None,
              evidence_level: str = None,
              final_k: int = None,
              use_llm: bool = True) -> RAGResponse:
        """Query with policy controls."""
        
        # Evaluate policy
        policy = self.policy.evaluate(
            query=query,
            mode=mode,
            include=include,
            exclude=exclude,
            evidence_level=evidence_level,
            final_k=final_k,
        )
        
        # Check if blocked
        if policy.is_blocked:
            return RAGResponse(
                answer=f"⚠️ {policy.block_reason}",
                citations=[],
                policy=policy,
                model_used="none",
            )
        
        # Get retrieval config
        retrieval = self.policy.get_retrieval_config()
        evidence_config = self.policy.get_evidence_config(policy.evidence_level)
        
        # Hybrid search
        results = self.search.search(
            query=query,
            k_lexical=retrieval.get('k_lexical', 20),
            k_vector=retrieval.get('k_vector', 20),
            final_k=policy.final_k,
            categories=list(policy.include_categories) if policy.include_categories else None,
            min_score=policy.min_score,
        )
        
        # Create citations
        citations = self._create_citations(results, evidence_config)
        
        # Check for symbols
        symbols_found = []
        # Extract potential symbol names from query
        import re
        potential_symbols = re.findall(r'\b([A-Z][A-Za-z0-9_]+|[a-z]+\.[a-z]+)\b', query)
        for sym in potential_symbols[:3]:
            found = self.search.find_symbol(sym)
            if found:
                symbols_found.extend(found[:3])
        
        # Generate answer
        model_used = "standalone"
        
        if use_llm:
            context = self._format_context(results, evidence_config)
            system = self.SYSTEM_TEMPLATE.format(
                mode=policy.mode,
                categories=", ".join(policy.include_categories),
                context=context,
            )
            
            llm_answer = self._call_lmstudio(query, system)
            if llm_answer:
                answer = llm_answer
                model_used = "lmstudio"
            else:
                answer = self._generate_standalone(query, citations)
        else:
            answer = self._generate_standalone(query, citations)
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            policy=policy,
            model_used=model_used,
            symbols_found=symbols_found,
        )
    
    def search_only(self, query: str,
                    include: List[str] = None,
                    exclude: List[str] = None,
                    k: int = 10) -> List[SearchResult]:
        """Search without generation (retrieval-only mode)."""
        policy = self.policy.evaluate(query, include=include, exclude=exclude)

        retrieval = self.policy.get_retrieval_config()

        return self.search.search(
            query=query,
            k_lexical=retrieval.get('k_lexical', 20),
            k_vector=retrieval.get('k_vector', 20),
            final_k=k,
            categories=list(policy.include_categories) if policy.include_categories else None,
            min_score=policy.min_score,
        )
    
    def find_symbol(self, name: str) -> List[Dict]:
        """Find symbol definitions."""
        return self.search.find_symbol(name)
    
    def trace(self, topic: str, depth: int = 3) -> List[Dict]:
        """Multi-hop trace: follow connections between concepts using hybrid search."""
        import re
        visited = set()
        trace_results = []
        current_topics = [topic]
        
        for hop in range(depth):
            next_topics = []
            
            for t in current_topics:
                if t in visited:
                    continue
                visited.add(t)
                
                # Use hybrid search which properly maps results to chunks
                search_results = self.search.search(t, final_k=5)
                
                for result in search_results:
                    # Handle SearchResult objects (dataclass) from hybrid search
                    if hasattr(result, 'file_path'):
                        # It's a SearchResult dataclass
                        trace_results.append({
                            'hop': hop + 1,
                            'topic': t,
                            'file': result.file_path,
                            'lines': f"{result.start_line}-{result.end_line}",
                            'score': round(result.combined_score, 3),
                            'snippet': result.content[:200] if result.content else '',
                        })
                        content = result.content or ''
                    else:
                        # It's a dict
                        chunk = result
                        score = chunk.get('score', 0.0)
                        trace_results.append({
                            'hop': hop + 1,
                            'topic': t,
                            'file': chunk.get('file_path', chunk.get('id', 'unknown')),
                            'lines': f"{chunk.get('start_line', 1)}-{chunk.get('end_line', 1)}",
                            'score': round(score, 3) if isinstance(score, float) else 0,
                            'snippet': chunk.get('content', '')[:200],
                        })
                        content = chunk.get('content', '')
                    
                    # Extract next topics: function calls, class names, commands
                    # Find function calls like doSomething(), class names, AMF commands
                    patterns = [
                        r'\.([a-z][A-Za-z0-9_]+)\s*\(',  # method calls
                        r'\b([A-Z][A-Za-z0-9_]+)\b',      # class names
                        r'["\'](\w+\.\w+)["\']',          # AMF commands like "troop.produceTroop"
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        next_topics.extend(matches[:3])
            
            # Filter and limit next topics
            current_topics = list(set(next_topics) - visited)[:5]
            if not current_topics:
                break
        
        return trace_results[:50]  # Limit total results
    
    def get_file(self, path: str, 
                 start_line: int = None, 
                 end_line: int = None) -> Optional[str]:
        """Get file content with optional line range."""
        full_path = DATASET_PATH / path
        
        if not full_path.exists():
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            if start_line and end_line:
                lines = lines[start_line-1:end_line]
            elif start_line:
                lines = lines[start_line-1:]
            
            return ''.join(lines)
        except:
            return None
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'chunks': len(self.search.chunks),
            'symbols': len(self.search.symbols.symbols),
            'mode': self.policy.current_mode,
            'modes_available': self.policy.get_modes(),
        }


# Singleton with thread safety
import threading
_rag_v2 = None
_rag_v2_lock = threading.Lock()

def get_rag_v2() -> EvonyRAGv2:
    """Get singleton RAG v2 instance (thread-safe)."""
    global _rag_v2
    if _rag_v2 is None:
        with _rag_v2_lock:
            if _rag_v2 is None:
                _rag_v2 = EvonyRAGv2()
    return _rag_v2
