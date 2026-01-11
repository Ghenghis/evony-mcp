"""
Evony Knowledge RAG System
===========================
MCP-enabled Retrieval-Augmented Generation for Evony expertise.

Features:
- Local embeddings index over curated dataset
- Query router with intent classification  
- Citations with exact file/line references
- MCP server for Windsurf IDE integration
- LM Studio API compatibility
- Safety filters for operational exploit blocking
"""

__version__ = "2.0.0"
__author__ = "Evony RE Team"

# Public API exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core classes (lazy imports to avoid circular dependencies)
    "get_rag_v2",
    "get_hybrid_search",
    "get_policy",
]


def get_rag_v2():
    """Get the RAG v2 engine singleton."""
    from .rag_v2 import get_rag_v2 as _get_rag_v2
    return _get_rag_v2()


def get_hybrid_search():
    """Get the hybrid search singleton."""
    from .hybrid_search import get_hybrid_search as _get_hybrid_search
    return _get_hybrid_search()


def get_policy():
    """Get the policy engine singleton."""
    from .policy import get_policy as _get_policy
    return _get_policy()
