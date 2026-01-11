"""
Evony RAG - Embedding Index Builder
====================================
Creates and manages the vector index over the curated dataset.
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    DATASET_PATH, INDEX_PATH, EMBEDDING_MODEL, EMBEDDING_DIM,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_FILE, CATEGORIES
)


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    id: str
    file_path: str
    category: str
    start_line: int
    end_line: int
    content: str
    embedding: np.ndarray = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "category": self.category,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
        }


class EmbeddingIndex:
    """Vector index for semantic search over Evony knowledge base."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = None
        self.metadata: Dict = {}
        
    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def _read_file(self, path: Path) -> Tuple[str, List[str]]:
        """Read file and return content with lines."""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                lines = content.split('\n')
            return content, lines
        except Exception as e:
            return "", []
    
    def _chunk_content(self, content: str, lines: List[str], 
                       file_path: str, category: str) -> List[Chunk]:
        """Split content into overlapping chunks with line tracking."""
        chunks = []
        
        if len(content) < CHUNK_SIZE:
            # Small file - single chunk
            chunks.append(Chunk(
                id=f"{file_path}:1-{len(lines)}",
                file_path=file_path,
                category=category,
                start_line=1,
                end_line=len(lines),
                content=content
            ))
            return chunks
        
        # Calculate lines per chunk (approximate)
        avg_line_len = len(content) / max(len(lines), 1)
        lines_per_chunk = max(10, int(CHUNK_SIZE / max(avg_line_len, 1)))
        overlap_lines = max(2, int(CHUNK_OVERLAP / max(avg_line_len, 1)))
        
        start = 0
        chunk_num = 0
        
        while start < len(lines) and chunk_num < MAX_CHUNKS_PER_FILE:
            end = min(start + lines_per_chunk, len(lines))
            chunk_content = '\n'.join(lines[start:end])
            
            if chunk_content.strip():
                chunks.append(Chunk(
                    id=f"{file_path}:{start+1}-{end}",
                    file_path=file_path,
                    category=category,
                    start_line=start + 1,
                    end_line=end,
                    content=chunk_content
                ))
                chunk_num += 1
            
            start = end - overlap_lines
            if start >= len(lines) - overlap_lines:
                break
        
        return chunks
    
    def build_index(self, dataset_path: Path = DATASET_PATH) -> int:
        """Build the embedding index from the dataset."""
        print("\n" + "="*60)
        print("BUILDING EVONY KNOWLEDGE INDEX")
        print("="*60)
        
        self.load_model()
        self.chunks = []
        
        # Process each category
        for category_dir in dataset_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            if category not in CATEGORIES:
                continue
            
            print(f"\nIndexing {category}...")
            files = list(category_dir.glob('*'))
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                content, lines = self._read_file(file_path)
                if not content:
                    continue
                
                # Get relative path for citation
                rel_path = str(file_path.relative_to(dataset_path))
                
                # Create chunks
                file_chunks = self._chunk_content(content, lines, rel_path, category)
                self.chunks.extend(file_chunks)
            
            print(f"  {len([c for c in self.chunks if c.category == category])} chunks")
        
        print(f"\nTotal chunks: {len(self.chunks)}")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        texts = [c.content for c in self.chunks]
        
        # Batch embedding for efficiency
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 500 == 0:
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        
        self.embeddings = np.array(all_embeddings)
        
        # Store metadata
        self.metadata = {
            "model": self.model_name,
            "num_chunks": len(self.chunks),
            "embedding_dim": EMBEDDING_DIM,
            "categories": {cat: len([c for c in self.chunks if c.category == cat]) 
                         for cat in CATEGORIES},
        }
        
        print(f"\nIndex built: {len(self.chunks)} chunks, {self.embeddings.shape}")
        return len(self.chunks)
    
    def save(self, index_path: Path = INDEX_PATH):
        """Save index to disk."""
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(index_path / "embeddings.npy", self.embeddings)
        
        # Save chunks metadata (without embeddings)
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(index_path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)
        
        # Save index metadata
        with open(index_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Index saved to: {index_path}")
    
    def load(self, index_path: Path = INDEX_PATH) -> bool:
        """Load index from disk."""
        try:
            self.load_model()
            
            # Load embeddings
            self.embeddings = np.load(index_path / "embeddings.npy")
            
            # Load chunks
            with open(index_path / "chunks.json", 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            self.chunks = [Chunk(**c) for c in chunks_data]
            
            # Load metadata
            with open(index_path / "metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print(f"Index loaded: {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, 
               categories: List[str] = None,
               threshold: float = 0.3) -> List[Tuple[Chunk, float]]:
        """Search for relevant chunks."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Embed query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Filter by category if specified
        if categories:
            mask = np.array([c.category in categories for c in self.chunks])
            similarities = np.where(mask, similarities, -1)
        
        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append((self.chunks[idx], float(similarities[idx])))
        
        return results


def build_index():
    """Build and save the embedding index."""
    index = EmbeddingIndex()
    index.build_index()
    index.save()
    return index


if __name__ == "__main__":
    build_index()
