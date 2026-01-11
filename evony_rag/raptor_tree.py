"""
RAPTOR - Recursive Abstractive Processing for Tree-Organized Retrieval
=======================================================================
GAME-CHANGER: +20% on complex multi-hop questions

RAPTOR creates a hierarchical tree of summaries:
1. Cluster similar chunks together
2. Summarize each cluster
3. Recursively cluster and summarize summaries
4. Query at multiple levels of abstraction

This enables answering questions that require understanding
relationships across many documents.

Reference: https://arxiv.org/abs/2401.18059
"""

import json
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

import numpy as np


@dataclass
class RAPTORNode:
    """A node in the RAPTOR tree."""
    node_id: str
    level: int  # 0 = leaf (original chunks), 1+ = summaries
    content: str
    summary: str  # Empty for leaves
    children: List[str] = field(default_factory=list)  # Child node IDs
    parent: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)


class RAPTORClusterer:
    """
    Clusters chunks/nodes for RAPTOR tree building.
    Uses simple k-means style clustering based on embeddings.
    """
    
    def __init__(self, cluster_size: int = 10):
        """
        Initialize clusterer.
        
        Args:
            cluster_size: Target number of items per cluster
        """
        self.cluster_size = cluster_size
    
    def cluster(self, embeddings: np.ndarray, k: int = None) -> List[List[int]]:
        """
        Cluster embeddings into groups.
        
        Args:
            embeddings: (N, D) array of embeddings
            k: Number of clusters (auto-computed if None)
            
        Returns:
            List of lists, each containing indices of items in that cluster
        """
        n = len(embeddings)
        
        if k is None:
            k = max(1, n // self.cluster_size)
        
        if k >= n:
            # Each item is its own cluster
            return [[i] for i in range(n)]
        
        # Simple k-means clustering
        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = embeddings[indices].copy()
        
        for _ in range(10):  # Max iterations
            # Assign to nearest centroid
            distances = np.zeros((n, k))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(embeddings - centroid, axis=1)
            
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centroids[i] = embeddings[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Build cluster lists
        clusters = defaultdict(list)
        for idx, cluster_id in enumerate(assignments):
            clusters[cluster_id].append(idx)
        
        return list(clusters.values())


class RAPTORSummarizer:
    """
    Summarizes clusters of chunks for RAPTOR.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.lmstudio_url = "http://localhost:1234/v1"
    
    def summarize(self, texts: List[str], level: int = 1) -> str:
        """
        Summarize a cluster of texts.
        
        Args:
            texts: List of text chunks to summarize
            level: Tree level (higher = more abstract)
            
        Returns:
            Summary text
        """
        if not texts:
            return ""
        
        combined = "\n\n---\n\n".join(texts[:10])  # Limit to 10 texts
        
        if self.use_llm:
            return self._llm_summarize(combined, level)
        else:
            return self._extractive_summarize(texts)
    
    def _llm_summarize(self, text: str, level: int) -> str:
        """Summarize using LLM."""
        try:
            import requests
            
            abstraction = "detailed" if level == 1 else "high-level" if level == 2 else "abstract"
            
            prompt = f"""Create a {abstraction} summary of the following content.
Focus on key technical concepts, commands, functions, and relationships.
Be concise but preserve important details.

Content:
{text[:3000]}

Summary:"""

            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
                
        except Exception:
            pass
        
        return self._extractive_summarize([text])
    
    def _extractive_summarize(self, texts: List[str]) -> str:
        """Simple extractive summary (first sentences)."""
        summary_parts = []
        for text in texts[:5]:
            sentences = text.split('.')[:2]
            summary_parts.append('. '.join(sentences))
        return ' '.join(summary_parts)[:500]


class RAPTORTree:
    """
    RAPTOR Tree for hierarchical retrieval.
    
    Structure:
    - Level 0: Original chunks (leaves)
    - Level 1: Summaries of chunk clusters
    - Level 2: Summaries of summary clusters
    - Level N: Root summary
    """
    
    def __init__(self, max_levels: int = 3, cluster_size: int = 10):
        """
        Initialize RAPTOR tree.
        
        Args:
            max_levels: Maximum tree depth
            cluster_size: Target items per cluster
        """
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.nodes: Dict[str, RAPTORNode] = {}
        self.levels: Dict[int, List[str]] = defaultdict(list)
        self.clusterer = RAPTORClusterer(cluster_size)
        self.summarizer = RAPTORSummarizer(use_llm=True)
        self.embedding_model = None
    
    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is not None:
            return
        
        try:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
    
    def build_from_chunks(self, chunks: List[Dict], progress_callback=None):
        """
        Build RAPTOR tree from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content', 'chunk_id', etc.
            progress_callback: Optional callback(message, current, total)
        """
        self._load_embedding_model()
        
        if progress_callback:
            progress_callback("Creating leaf nodes...", 0, len(chunks))
        
        # Level 0: Create leaf nodes
        for i, chunk in enumerate(chunks):
            node_id = f"L0_{chunk.get('chunk_id', i)}"
            content = chunk.get('content', '')
            
            node = RAPTORNode(
                node_id=node_id,
                level=0,
                content=content,
                summary="",
                metadata={
                    "file_path": chunk.get("file_path", ""),
                    "category": chunk.get("category", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                }
            )
            
            self.nodes[node_id] = node
            self.levels[0].append(node_id)
        
        # Compute embeddings for leaves
        if self.embedding_model:
            if progress_callback:
                progress_callback("Computing embeddings...", 0, 1)
            
            contents = [self.nodes[nid].content for nid in self.levels[0]]
            embeddings = self.embedding_model.encode(contents, show_progress_bar=False)
            
            for nid, emb in zip(self.levels[0], embeddings):
                self.nodes[nid].embedding = emb.tolist()
        
        # Build higher levels
        current_level = 0
        while current_level < self.max_levels - 1 and len(self.levels[current_level]) > 1:
            if progress_callback:
                progress_callback(f"Building level {current_level + 1}...", current_level, self.max_levels)
            
            self._build_level(current_level + 1)
            current_level += 1
    
    def _build_level(self, level: int):
        """Build a summary level from the level below."""
        parent_level = level - 1
        parent_nodes = [self.nodes[nid] for nid in self.levels[parent_level]]
        
        if not parent_nodes:
            return
        
        # Get embeddings
        embeddings = np.array([
            n.embedding if n.embedding else np.zeros(384)
            for n in parent_nodes
        ])
        
        # Cluster
        clusters = self.clusterer.cluster(embeddings)
        
        # Create summary nodes for each cluster
        for cluster_idx, cluster_indices in enumerate(clusters):
            if not cluster_indices:
                continue
            
            # Get texts to summarize
            texts = []
            child_ids = []
            for idx in cluster_indices:
                node = parent_nodes[idx]
                texts.append(node.summary if node.summary else node.content)
                child_ids.append(node.node_id)
            
            # Generate summary
            summary = self.summarizer.summarize(texts, level)
            
            # Create summary node
            node_id = f"L{level}_{cluster_idx}"
            
            # Compute embedding for summary
            embedding = None
            if self.embedding_model:
                embedding = self.embedding_model.encode([summary])[0].tolist()
            
            node = RAPTORNode(
                node_id=node_id,
                level=level,
                content="\n".join(texts[:3]),  # Keep some original content
                summary=summary,
                children=child_ids,
                embedding=embedding,
            )
            
            # Update children's parent
            for child_id in child_ids:
                self.nodes[child_id].parent = node_id
            
            self.nodes[node_id] = node
            self.levels[level].append(node_id)
    
    def search(self, query: str, top_k: int = 10, levels: List[int] = None) -> List[Tuple[RAPTORNode, float]]:
        """
        Search RAPTOR tree at multiple levels.
        
        Args:
            query: Search query
            top_k: Number of results
            levels: Which levels to search (None = all)
            
        Returns:
            List of (node, score) tuples
        """
        if not self.embedding_model:
            self._load_embedding_model()
        
        if not self.embedding_model:
            return []
        
        # Embed query
        query_emb = self.embedding_model.encode([query])[0]
        
        # Search specified levels
        if levels is None:
            levels = list(self.levels.keys())
        
        results = []
        for level in levels:
            for node_id in self.levels.get(level, []):
                node = self.nodes[node_id]
                if node.embedding:
                    score = np.dot(query_emb, node.embedding) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-8
                    )
                    results.append((node, float(score)))
        
        # Sort by score
        results.sort(key=lambda x: -x[1])
        
        return results[:top_k]
    
    def get_with_context(self, node: RAPTORNode) -> Dict:
        """
        Get node with its hierarchical context.
        
        Returns the node plus its parent summaries for full context.
        """
        context = {
            "content": node.content,
            "summary": node.summary,
            "level": node.level,
            "metadata": node.metadata,
            "parent_summaries": [],
        }
        
        # Walk up the tree
        current = node
        while current.parent:
            parent = self.nodes.get(current.parent)
            if parent and parent.summary:
                context["parent_summaries"].append(parent.summary)
            current = parent if parent else None
            if current is None:
                break
        
        return context
    
    def save(self, path: Path):
        """Save RAPTOR tree."""
        data = {
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "level": n.level,
                    "content": n.content[:1000],  # Truncate for storage
                    "summary": n.summary,
                    "children": n.children,
                    "parent": n.parent,
                    "metadata": n.metadata,
                }
                for nid, n in self.nodes.items()
            },
            "levels": dict(self.levels),
        }
        
        with open(path / "raptor_tree.json", "w") as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> bool:
        """Load RAPTOR tree."""
        try:
            with open(path / "raptor_tree.json") as f:
                data = json.load(f)
            
            for nid, n_data in data["nodes"].items():
                node = RAPTORNode(
                    node_id=n_data["node_id"],
                    level=n_data["level"],
                    content=n_data["content"],
                    summary=n_data["summary"],
                    children=n_data["children"],
                    parent=n_data["parent"],
                    metadata=n_data.get("metadata", {}),
                )
                self.nodes[nid] = node
            
            self.levels = defaultdict(list, {int(k): v for k, v in data["levels"].items()})
            return True
        except:
            return False
    
    def get_stats(self) -> Dict:
        """Get tree statistics."""
        return {
            "total_nodes": len(self.nodes),
            "levels": {level: len(nodes) for level, nodes in self.levels.items()},
            "max_depth": max(self.levels.keys()) if self.levels else 0,
        }


# Singleton
_raptor_tree = None


def get_raptor_tree() -> RAPTORTree:
    """Get singleton RAPTOR tree."""
    global _raptor_tree
    if _raptor_tree is None:
        _raptor_tree = RAPTORTree()
    return _raptor_tree
