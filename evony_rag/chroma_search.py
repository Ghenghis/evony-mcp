#!/usr/bin/env python3
"""
ChromaDB Integration for Evony RAG
===================================
OPTIONAL addition - does NOT replace existing system.
Can be used alongside current numpy embeddings.

Benefits:
- Faster ANN search for large datasets
- Built-in persistence
- Metadata filtering
- Scales better than numpy

Self-hosting: ChromaDB runs locally, no external services needed.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Check if chromadb is available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed. Run: pip install chromadb")


CHROMA_PATH = Path(r"G:\evony_rag_index\chroma_db")
CHUNKS_PATH = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\index\chunks.json")


@dataclass
class ChromaResult:
    """Search result from ChromaDB."""
    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    distance: float  # Lower is better
    metadata: Dict


class ChromaSearch:
    """
    ChromaDB-based semantic search.
    
    This is an OPTIONAL addition to the existing RAG system.
    It can be used instead of or alongside the numpy-based search.
    """
    
    def __init__(self, persist_directory: str = None):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed")
        
        self.persist_dir = persist_directory or str(CHROMA_PATH)
        self.client = None
        self.collection = None
        self._initialized = False
    
    def init(self):
        """Initialize ChromaDB client and collection."""
        if self._initialized:
            return
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="evony_chunks",
            metadata={"description": "Evony source code and scripts"}
        )
        
        self._initialized = True
        print(f"ChromaDB initialized: {self.collection.count()} documents")
    
    def index_chunks(self, chunks: List[Dict] = None, embeddings_file: str = None):
        """
        Index chunks into ChromaDB.
        
        Can use:
        1. Pre-computed embeddings from numpy file
        2. Let ChromaDB compute embeddings (slower but simpler)
        """
        self.init()
        
        # Load chunks if not provided
        if chunks is None:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        
        print(f"Indexing {len(chunks):,} chunks into ChromaDB...")
        
        # Check if we have pre-computed embeddings
        embeddings = None
        if embeddings_file:
            import numpy as np
            embeddings = np.load(embeddings_file)
            print(f"Using pre-computed embeddings: {embeddings.shape}")
        
        # Batch index
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [f"chunk_{i+j}" for j in range(len(batch))]
            documents = [c.get("content", "")[:8000] for c in batch]  # Limit size
            metadatas = [
                {
                    "file_path": c.get("file_path", ""),
                    "start_line": c.get("start_line", 0),
                    "end_line": c.get("end_line", 0),
                    "category": c.get("category", ""),
                }
                for c in batch
            ]
            
            if embeddings is not None:
                batch_embeddings = embeddings[i:i + batch_size].tolist()
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=batch_embeddings
                )
            else:
                # Let ChromaDB compute embeddings
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            if (i + batch_size) % 2000 == 0:
                print(f"  Indexed {i + batch_size:,} chunks...")
        
        print(f"Indexing complete: {self.collection.count():,} documents in ChromaDB")
    
    def search(self, query: str, top_k: int = 10, 
               filter_category: str = None,
               query_embedding: List[float] = None) -> List[ChromaResult]:
        """
        Search ChromaDB for similar chunks.
        
        Args:
            query: Search query text
            top_k: Number of results
            filter_category: Optional category filter
            query_embedding: Pre-computed query embedding (optional)
        """
        self.init()
        
        where_filter = None
        if filter_category:
            where_filter = {"category": filter_category}
        
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        
        # Convert to ChromaResult objects
        output = []
        for i in range(len(results["ids"][0])):
            output.append(ChromaResult(
                chunk_id=results["ids"][0][i],
                content=results["documents"][0][i],
                file_path=results["metadatas"][0][i].get("file_path", ""),
                start_line=results["metadatas"][0][i].get("start_line", 0),
                end_line=results["metadatas"][0][i].get("end_line", 0),
                distance=results["distances"][0][i],
                metadata=results["metadatas"][0][i]
            ))
        
        return output
    
    def get_stats(self) -> Dict:
        """Get ChromaDB collection statistics."""
        self.init()
        return {
            "count": self.collection.count(),
            "name": self.collection.name,
            "persist_dir": self.persist_dir,
        }


# Singleton
_chroma_search = None

def get_chroma_search() -> Optional[ChromaSearch]:
    """Get ChromaDB search instance (if available)."""
    global _chroma_search
    if not CHROMA_AVAILABLE:
        return None
    if _chroma_search is None:
        _chroma_search = ChromaSearch()
    return _chroma_search


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    print("=" * 60)
    print("CHROMADB INTEGRATION TEST")
    print("=" * 60)
    
    if not CHROMA_AVAILABLE:
        print("ChromaDB not installed. Run: pip install chromadb")
        sys.exit(1)
    
    chroma = get_chroma_search()
    chroma.init()
    
    # Check if we need to index
    if chroma.collection.count() == 0:
        print("\nNo documents in ChromaDB. Indexing...")
        # Use pre-computed embeddings
        embeddings_file = r"G:\evony_rag_index\embeddings.npy"
        if Path(embeddings_file).exists():
            chroma.index_chunks(embeddings_file=embeddings_file)
        else:
            print("No pre-computed embeddings found. ChromaDB will compute them.")
            chroma.index_chunks()
    
    # Test search
    print("\nTesting search...")
    queries = ["StratagemCommands", "farmNPC script", "troop attack"]
    
    for q in queries:
        results = chroma.search(q, top_k=3)
        print(f"\nQuery: {q}")
        for r in results:
            print(f"  [{r.distance:.3f}] {r.file_path} - {r.content[:50]}...")
    
    print("\n" + "=" * 60)
    print(f"ChromaDB Stats: {chroma.get_stats()}")
    print("=" * 60)
