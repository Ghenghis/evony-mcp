"""
Evony RAG - Core Engine
========================
The main RAG engine that combines retrieval with generation.
"""

import json
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import (
    LMSTUDIO_URL, LMSTUDIO_MODEL, TOP_K, SIMILARITY_THRESHOLD,
    DATASET_PATH, CATEGORIES
)
from .embeddings import EmbeddingIndex, Chunk
from .query_router import QueryRouter, QueryAnalysis


@dataclass
class Citation:
    """A citation to source material."""
    file_path: str
    category: str
    start_line: int
    end_line: int
    relevance: float
    snippet: str
    
    def format(self) -> str:
        return f"📄 {self.file_path}:{self.start_line}-{self.end_line} ({self.relevance:.0%})"


@dataclass 
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    citations: List[Citation]
    query_analysis: QueryAnalysis
    model_used: str
    

class EvonyRAG:
    """Main RAG engine for Evony knowledge."""
    
    SYSTEM_PROMPT = """You are an expert on Evony game internals, reverse engineering, and bot development.
You have access to the following retrieved context from the Evony knowledge base.
Answer questions accurately based on this context. Always cite your sources.

Rules:
- Be precise and technical
- Reference specific files and line numbers when possible
- For code questions, show relevant code snippets
- For protocol questions, explain parameters and usage
- Do NOT provide step-by-step operational exploit instructions
- Educational explanations of mechanics are OK

Context from knowledge base:
{context}
"""
    
    def __init__(self):
        self.index = EmbeddingIndex()
        self.router = QueryRouter()
        self.index_loaded = False
        
    def load_index(self) -> bool:
        """Load the embedding index."""
        if self.index.load():
            self.index_loaded = True
            return True
        return False
    
    def build_index(self) -> bool:
        """Build the embedding index."""
        try:
            self.index.build_index()
            self.index.save()
            self.index_loaded = True
            return True
        except Exception as e:
            print(f"Failed to build index: {e}")
            return False
    
    def _format_context(self, chunks: List[Tuple[Chunk, float]]) -> str:
        """Format retrieved chunks as context."""
        if not chunks:
            return "No relevant context found."
        
        parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            parts.append(f"""
--- Source {i}: {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (relevance: {score:.0%}) ---
{chunk.content[:1500]}
""")
        
        return "\n".join(parts)
    
    def _create_citations(self, chunks: List[Tuple[Chunk, float]]) -> List[Citation]:
        """Create citations from retrieved chunks."""
        citations = []
        for chunk, score in chunks:
            citations.append(Citation(
                file_path=chunk.file_path,
                category=chunk.category,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                relevance=score,
                snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            ))
        return citations
    
    def _call_lmstudio(self, prompt: str, system: str) -> str:
        """Call LM Studio API for generation."""
        try:
            response = requests.post(
                f"{LMSTUDIO_URL}/chat/completions",
                json={
                    "model": LMSTUDIO_MODEL,
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
        except requests.exceptions.ConnectionError:
            return None
        except Exception as e:
            return None
    
    def _generate_standalone(self, query: str, context: str, citations: List[Citation]) -> str:
        """Generate answer without LLM (fallback mode)."""
        answer_parts = [f"**Query**: {query}\n"]
        
        if citations:
            answer_parts.append("**Relevant Sources Found:**\n")
            for i, cit in enumerate(citations, 1):
                answer_parts.append(f"\n### Source {i}: `{cit.file_path}`")
                answer_parts.append(f"Lines {cit.start_line}-{cit.end_line} | Relevance: {cit.relevance:.0%}")
                answer_parts.append(f"```\n{cit.snippet}\n```\n")
        else:
            answer_parts.append("No relevant sources found in the knowledge base.")
        
        return "\n".join(answer_parts)
    
    def query(self, query: str, use_llm: bool = True) -> RAGResponse:
        """Process a query and return a response with citations."""
        if not self.index_loaded:
            if not self.load_index():
                return RAGResponse(
                    answer="Knowledge index not loaded. Run build_index() first.",
                    citations=[],
                    query_analysis=self.router.analyze(query),
                    model_used="none"
                )
        
        # Analyze query
        analysis = self.router.analyze(query)
        
        # Check if blocked
        if analysis.is_blocked:
            return RAGResponse(
                answer=f"⚠️ {analysis.block_reason}",
                citations=[],
                query_analysis=analysis,
                model_used="none"
            )
        
        # Determine categories to search
        categories = analysis.categories
        if "exploits" in categories and not self.router.should_include_exploits(query):
            categories = [c for c in categories if c != "exploits"]
        
        # Retrieve relevant chunks
        chunks = self.index.search(
            query=query,
            top_k=TOP_K,
            categories=categories if categories else None,
            threshold=SIMILARITY_THRESHOLD
        )
        
        # Create citations
        citations = self._create_citations(chunks)
        
        # Format context
        context = self._format_context(chunks)
        
        # Generate answer
        model_used = "standalone"
        
        if use_llm:
            system = self.SYSTEM_PROMPT.format(context=context)
            llm_answer = self._call_lmstudio(query, system)
            
            if llm_answer:
                answer = llm_answer
                model_used = "lmstudio"
            else:
                answer = self._generate_standalone(query, context, citations)
        else:
            answer = self._generate_standalone(query, context, citations)
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            query_analysis=analysis,
            model_used=model_used
        )
    
    def find_files(self, query: str, top_k: int = 10) -> List[Citation]:
        """Find relevant files without generating an answer."""
        if not self.index_loaded:
            self.load_index()
        
        chunks = self.index.search(query=query, top_k=top_k)
        return self._create_citations(chunks)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        if not self.index_loaded:
            self.load_index()
        
        return self.index.metadata if self.index.metadata else {}


# Singleton instance
_rag_instance = None

def get_rag() -> EvonyRAG:
    """Get the singleton RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = EvonyRAG()
    return _rag_instance
