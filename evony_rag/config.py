"""
Evony RAG Configuration
========================
Central configuration for the RAG system.
"""

from pathlib import Path

# Base paths
BASE_PATH = Path("c:/Users/Admin/Downloads/Evony_Decrypted")
RAG_PATH = BASE_PATH / "evony_rag"
DATASET_PATH = BASE_PATH / "Evony_Training_Dataset"
INDEX_PATH = RAG_PATH / "index"

# Index paths - use G:\ for large indices (Downloads folder has file system issues)
LARGE_INDEX_PATH = Path("G:/evony_rag_index")
KG_INDEX_PATH = LARGE_INDEX_PATH  # 101,853 entities  
HYPE_INDEX_PATH = LARGE_INDEX_PATH  # 169,620 questions

# Embedding model (runs locally, no API needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Chunk settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_CHUNKS_PER_FILE = 100

# Retrieval settings
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# LM Studio settings
LMSTUDIO_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "local-model"

# MCP Server settings
MCP_HOST = "localhost"
MCP_PORT = 8765

# Categories with safety levels
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

# Blocked query patterns (operational exploit requests)
BLOCKED_PATTERNS = [
    r"how (do i|can i|to) (use|exploit|abuse) .* (glitch|bug|exploit)",
    r"give me .* (exploit|hack|cheat)",
    r"step.?by.?step .* (glitch|exploit)",
    r"working (exploit|hack|cheat)",
]

# Query intent categories
QUERY_INTENTS = {
    "code_explain": "Explain code/function behavior",
    "protocol_info": "Protocol command information", 
    "find_files": "Find relevant files/snippets",
    "howto": "How to implement something",
    "exploit_info": "Educational exploit information",
    "general": "General Evony knowledge",
}
