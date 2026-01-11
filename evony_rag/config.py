"""
Evony RAG Configuration
========================
Central configuration for the RAG system with environment-based path resolution.
"""

import os
from pathlib import Path

# ============================================================================
# BASE PATH RESOLUTION
# ============================================================================

def _get_base_path() -> Path:
    """
    Resolve the base path for the Evony project.
    
    Priority:
    1. EVONY_BASE_PATH environment variable
    2. Parent of the evony_rag package directory
    """
    env_path = os.environ.get("EVONY_BASE_PATH")
    if env_path:
        return Path(env_path)
    
    # Default: parent of evony_rag package
    return Path(__file__).parent.parent


def _get_index_path() -> Path:
    """
    Resolve the index path for large indices.
    
    Priority:
    1. EVONY_INDEX_PATH environment variable
    2. BASE_PATH / evony_rag / index
    """
    env_path = os.environ.get("EVONY_INDEX_PATH")
    if env_path:
        return Path(env_path)
    
    return RAG_PATH / "index"


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths - resolved at import time
BASE_PATH = _get_base_path()
RAG_PATH = BASE_PATH / "evony_rag"
DATASET_PATH = BASE_PATH / "Evony_Training_Dataset"
INDEX_PATH = RAG_PATH / "index"

# Large index path - for embeddings and knowledge graph
LARGE_INDEX_PATH = _get_index_path()
KG_INDEX_PATH = LARGE_INDEX_PATH
HYPE_INDEX_PATH = LARGE_INDEX_PATH

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Embedding model (runs locally, no API needed)
EMBEDDING_MODEL = os.environ.get(
    "EVONY_EMBEDDING_MODEL", 
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.environ.get("EVONY_EMBEDDING_DIM", "384"))

# ============================================================================
# CHUNK SETTINGS
# ============================================================================

CHUNK_SIZE = int(os.environ.get("EVONY_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("EVONY_CHUNK_OVERLAP", "50"))
MAX_CHUNKS_PER_FILE = int(os.environ.get("EVONY_MAX_CHUNKS_PER_FILE", "100"))

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

TOP_K = int(os.environ.get("EVONY_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.environ.get("EVONY_SIMILARITY_THRESHOLD", "0.3"))

# ============================================================================
# LM STUDIO SETTINGS
# ============================================================================

LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "local-model")

# ============================================================================
# MCP SERVER SETTINGS
# ============================================================================

MCP_HOST = os.environ.get("EVONY_MCP_HOST", "localhost")
MCP_PORT = int(os.environ.get("EVONY_MCP_PORT", "8765"))

# ============================================================================
# CATEGORIES WITH SAFETY LEVELS
# ============================================================================

CATEGORIES = {
    "source_code": {"safe": True, "description": "AS3/Python source files"},
    "documentation": {"safe": True, "description": "Protocol docs, guides"},
    "protocol": {"safe": True, "description": "Protocol specifications"},
    "keys": {"safe": True, "description": "Encryption keys (educational)"},
    "scripts": {"safe": True, "description": "Bot automation scripts"},
    "exploits": {"safe": False, "description": "Glitch documentation"},
    "game_data": {"safe": True, "description": "Game mechanics data"},
    "tools": {"safe": True, "description": "Analysis utilities"},
}

# ============================================================================
# BLOCKED QUERY PATTERNS (OPERATIONAL EXPLOIT REQUESTS)
# ============================================================================

BLOCKED_PATTERNS = [
    r"how (do i|can i|to) (use|exploit|abuse) .* (glitch|bug|exploit)",
    r"give me .* (exploit|hack|cheat)",
    r"step.?by.?step .* (glitch|exploit)",
    r"working (exploit|hack|cheat)",
]

# ============================================================================
# QUERY INTENT CATEGORIES
# ============================================================================

QUERY_INTENTS = {
    "code_explain": "Explain code/function behavior",
    "protocol_info": "Protocol command information",
    "find_files": "Find relevant files/snippets",
    "howto": "How to implement something",
    "exploit_info": "Educational exploit information",
    "general": "General Evony knowledge",
}

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Paths
    "BASE_PATH",
    "RAG_PATH",
    "DATASET_PATH",
    "INDEX_PATH",
    "LARGE_INDEX_PATH",
    "KG_INDEX_PATH",
    "HYPE_INDEX_PATH",
    # Model
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    # Chunk
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "MAX_CHUNKS_PER_FILE",
    # Retrieval
    "TOP_K",
    "SIMILARITY_THRESHOLD",
    # LM Studio
    "LMSTUDIO_URL",
    "LMSTUDIO_MODEL",
    # MCP
    "MCP_HOST",
    "MCP_PORT",
    # Categories
    "CATEGORIES",
    "BLOCKED_PATTERNS",
    "QUERY_INTENTS",
]
