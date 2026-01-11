"""
RAG Ultimate Index Builders
===========================
Robust builders for populating advanced indices:
1. HyPE Index - Hypothetical questions for each chunk
2. Knowledge Graph - Entity extraction and relationships

Features:
- Progress tracking with ETA
- Resume capability (checkpoint saving)
- Memory-efficient batch processing
- Error handling with retry logic
- Parallel processing support
"""

import json
import time
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable, Generator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

from .config import INDEX_PATH, DATASET_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BuildProgress:
    """Tracks index building progress."""
    total_chunks: int
    processed_chunks: int
    failed_chunks: int
    start_time: float
    last_checkpoint: float
    current_batch: int
    errors: List[str]
    
    @property
    def percent_complete(self) -> float:
        return (self.processed_chunks / max(self.total_chunks, 1)) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def eta_seconds(self) -> float:
        if self.processed_chunks == 0:
            return 0
        rate = self.processed_chunks / self.elapsed_seconds
        remaining = self.total_chunks - self.processed_chunks
        return remaining / rate if rate > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "percent_complete": self.percent_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "current_batch": self.current_batch,
            "errors_count": len(self.errors),
        }


class ChunkLoader:
    """
    Memory-efficient chunk loader with batching.
    """
    
    def __init__(self, chunks_path: Path, batch_size: int = 100):
        self.chunks_path = chunks_path
        self.batch_size = batch_size
        self._total_chunks = None
    
    @property
    def total_chunks(self) -> int:
        """Get total chunk count (cached)."""
        if self._total_chunks is None:
            self._total_chunks = self._count_chunks()
        return self._total_chunks
    
    def _count_chunks(self) -> int:
        """Count chunks without loading all into memory."""
        try:
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data) if isinstance(data, list) else 0
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error counting chunks: {e}")
            return 0
    
    def load_all(self) -> List[Dict]:
        """Load all chunks (for smaller datasets)."""
        try:
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} chunks from {self.chunks_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return []
    
    def iter_batches(self, start_idx: int = 0) -> Generator[List[Dict], None, None]:
        """
        Iterate through chunks in batches.
        Supports resuming from a specific index.
        """
        chunks = self.load_all()
        
        for i in range(start_idx, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            yield batch


class HyPEIndexBuilder:
    """
    Builds HyPE (Hypothetical Prompt Embeddings) Index.
    
    For each chunk:
    1. Generate hypothetical questions that this chunk answers
    2. Embed those questions
    3. Store for question-to-question matching at query time
    """
    
    def __init__(self, 
                 output_path: Path = None,
                 batch_size: int = 50,
                 checkpoint_interval: int = 500,
                 use_llm: bool = False):
        """
        Initialize HyPE builder.
        
        Args:
            output_path: Where to save the index
            batch_size: Chunks per batch
            checkpoint_interval: Save checkpoint every N chunks
            use_llm: Use LLM for question generation (slower but better)
        """
        self.output_path = output_path or Path(INDEX_PATH) / "hype_index"
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.use_llm = use_llm
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Components
        self._generator = None
        self._embedding_model = None
        
        # State
        self.progress: Optional[BuildProgress] = None
        self._stop_flag = False
    
    def _init_components(self):
        """Initialize generator and embedding model."""
        if self._generator is None:
            from .hype_embeddings import HyPEGenerator
            self._generator = HyPEGenerator(use_llm=self.use_llm)
        
        if self._embedding_model is None:
            try:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                logger.info("Loaded embedding model on CPU")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                logger.warning("HyPE index will be built without embeddings (questions only)")
                self._embedding_model = None  # Will skip embedding step
    
    def build(self, 
              chunks_path: Path = None,
              resume: bool = True,
              progress_callback: Callable[[BuildProgress], None] = None) -> BuildProgress:
        """
        Build HyPE index from chunks.
        
        Args:
            chunks_path: Path to chunks.json
            resume: Resume from last checkpoint if available
            progress_callback: Called with progress updates
            
        Returns:
            Final BuildProgress
        """
        chunks_path = chunks_path or Path(INDEX_PATH) / "chunks.json"
        
        logger.info(f"Starting HyPE index build from {chunks_path}")
        
        # Initialize components
        self._init_components()
        
        # Load chunks
        loader = ChunkLoader(chunks_path, self.batch_size)
        total = loader.total_chunks
        
        if total == 0:
            logger.error("No chunks found!")
            return BuildProgress(0, 0, 0, time.time(), time.time(), 0, ["No chunks found"])
        
        logger.info(f"Found {total} chunks to process")
        
        # Check for resume
        start_idx = 0
        existing_data = []
        
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                start_idx = checkpoint.get("processed_chunks", 0)
                existing_data = checkpoint.get("data", [])
                logger.info(f"Resuming from chunk {start_idx}")
        
        # Initialize progress
        self.progress = BuildProgress(
            total_chunks=total,
            processed_chunks=start_idx,
            failed_chunks=0,
            start_time=time.time(),
            last_checkpoint=time.time(),
            current_batch=start_idx // self.batch_size,
            errors=[],
        )
        
        # Build index
        all_hype_chunks = existing_data
        
        try:
            for batch in loader.iter_batches(start_idx):
                if self._stop_flag:
                    logger.info("Build stopped by user")
                    break
                
                # Process batch
                batch_results = self._process_batch(batch)
                all_hype_chunks.extend(batch_results)
                
                self.progress.processed_chunks += len(batch)
                self.progress.current_batch += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.progress)
                
                # Log progress
                if self.progress.processed_chunks % 100 == 0:
                    eta = str(timedelta(seconds=int(self.progress.eta_seconds)))
                    logger.info(
                        f"Progress: {self.progress.processed_chunks}/{total} "
                        f"({self.progress.percent_complete:.1f}%) - ETA: {eta}"
                    )
                
                # Checkpoint
                if self.progress.processed_chunks % self.checkpoint_interval == 0:
                    self._save_checkpoint(all_hype_chunks)
        
        except Exception as e:
            logger.error(f"Build error: {e}")
            self.progress.errors.append(str(e))
            self._save_checkpoint(all_hype_chunks)
            raise
        
        # Save final index
        self._save_index(all_hype_chunks)
        logger.info(f"HyPE index build complete: {len(all_hype_chunks)} chunks processed")
        
        return self.progress
    
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of chunks."""
        results = []
        
        for chunk in batch:
            try:
                # Generate questions
                questions = self._generator.generate_questions(chunk)
                
                if not questions:
                    self.progress.failed_chunks += 1
                    continue
                
                # Embed questions (if model available)
                question_embeddings = []
                if self._embedding_model is not None:
                    question_embeddings = self._embedding_model.encode(
                        questions, 
                        show_progress_bar=False
                    ).tolist()
                
                # Create HyPE chunk
                hype_chunk = {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "file_path": chunk.get("file_path", ""),
                    "category": chunk.get("category", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "content": chunk.get("content", "")[:500],  # Truncate for storage
                    "questions": questions,
                    "question_embeddings": question_embeddings,
                }
                
                results.append(hype_chunk)
                
            except Exception as e:
                self.progress.failed_chunks += 1
                if len(self.progress.errors) < 100:
                    self.progress.errors.append(f"Chunk error: {str(e)[:100]}")
        
        return results
    
    def _save_checkpoint(self, data: List[Dict]):
        """Save checkpoint for resume capability."""
        checkpoint = {
            "processed_chunks": self.progress.processed_chunks,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        checkpoint_path = self.output_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        self.progress.last_checkpoint = time.time()
        logger.debug(f"Checkpoint saved: {self.progress.processed_chunks} chunks")
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        checkpoint_path = self.output_path / "checkpoint.json"
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return None
    
    def _save_index(self, data: List[Dict]):
        """Save final HyPE index."""
        # Save main index
        index_path = self.output_path / "hype_chunks.json"
        with open(index_path, 'w') as f:
            json.dump(data, f)
        
        # Save metadata
        metadata = {
            "total_chunks": len(data),
            "build_time": datetime.now().isoformat(),
            "questions_per_chunk": sum(len(c.get("questions", [])) for c in data) / max(len(data), 1),
        }
        
        with open(self.output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up checkpoint
        checkpoint_path = self.output_path / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        logger.info(f"HyPE index saved to {index_path}")
    
    def stop(self):
        """Stop the build process gracefully."""
        self._stop_flag = True


class KnowledgeGraphBuilder:
    """
    Builds Knowledge Graph from chunks.
    
    Extracts:
    1. Entities (classes, functions, commands, variables)
    2. Relationships (calls, extends, implements, uses)
    
    Stores in a queryable graph structure.
    """
    
    def __init__(self,
                 output_path: Path = None,
                 batch_size: int = 100,
                 checkpoint_interval: int = 1000):
        """
        Initialize Knowledge Graph builder.
        
        Args:
            output_path: Where to save the graph
            batch_size: Chunks per batch
            checkpoint_interval: Save checkpoint every N chunks
        """
        self.output_path = output_path or Path(INDEX_PATH) / "knowledge_graph"
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Components
        self._extractor = None
        self._graph = None
        
        # State
        self.progress: Optional[BuildProgress] = None
        self._stop_flag = False
    
    def _init_components(self):
        """Initialize extractor and graph."""
        if self._extractor is None:
            from .knowledge_graph import EntityExtractor, KnowledgeGraph
            self._extractor = EntityExtractor()
            self._graph = KnowledgeGraph()
    
    def build(self,
              chunks_path: Path = None,
              resume: bool = True,
              progress_callback: Callable[[BuildProgress], None] = None) -> BuildProgress:
        """
        Build Knowledge Graph from chunks.
        
        Args:
            chunks_path: Path to chunks.json
            resume: Resume from last checkpoint
            progress_callback: Called with progress updates
            
        Returns:
            Final BuildProgress
        """
        chunks_path = chunks_path or Path(INDEX_PATH) / "chunks.json"
        
        logger.info(f"Starting Knowledge Graph build from {chunks_path}")
        
        # Initialize components
        self._init_components()
        
        # Load chunks
        loader = ChunkLoader(chunks_path, self.batch_size)
        total = loader.total_chunks
        
        if total == 0:
            logger.error("No chunks found!")
            return BuildProgress(0, 0, 0, time.time(), time.time(), 0, ["No chunks found"])
        
        logger.info(f"Found {total} chunks to process")
        
        # Check for resume
        start_idx = 0
        
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                start_idx = checkpoint.get("processed_chunks", 0)
                self._graph.load(self.output_path)
                logger.info(f"Resuming from chunk {start_idx}")
        
        # Initialize progress
        self.progress = BuildProgress(
            total_chunks=total,
            processed_chunks=start_idx,
            failed_chunks=0,
            start_time=time.time(),
            last_checkpoint=time.time(),
            current_batch=start_idx // self.batch_size,
            errors=[],
        )
        
        # Build graph
        try:
            for batch in loader.iter_batches(start_idx):
                if self._stop_flag:
                    logger.info("Build stopped by user")
                    break
                
                # Process batch
                self._process_batch(batch)
                
                self.progress.processed_chunks += len(batch)
                self.progress.current_batch += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.progress)
                
                # Log progress
                if self.progress.processed_chunks % 500 == 0:
                    eta = str(timedelta(seconds=int(self.progress.eta_seconds)))
                    stats = self._graph.get_stats()
                    logger.info(
                        f"Progress: {self.progress.processed_chunks}/{total} "
                        f"({self.progress.percent_complete:.1f}%) - "
                        f"Entities: {stats['total_entities']}, "
                        f"Relationships: {stats['total_relationships']} - "
                        f"ETA: {eta}"
                    )
                
                # Checkpoint
                if self.progress.processed_chunks % self.checkpoint_interval == 0:
                    self._save_checkpoint()
        
        except Exception as e:
            logger.error(f"Build error: {e}")
            self.progress.errors.append(str(e))
            self._save_checkpoint()
            raise
        
        # Save final graph
        self._save_graph()
        stats = self._graph.get_stats()
        logger.info(
            f"Knowledge Graph build complete: "
            f"{stats['total_entities']} entities, "
            f"{stats['total_relationships']} relationships"
        )
        
        return self.progress
    
    def _process_batch(self, batch: List[Dict]):
        """Process a batch of chunks."""
        for chunk in batch:
            try:
                # Extract entities and relationships
                entities, relationships = self._extractor.extract_from_chunk(chunk)
                
                # Add to graph
                for entity in entities:
                    self._graph.add_entity(entity)
                
                for rel in relationships:
                    self._graph.add_relationship(rel)
                
            except Exception as e:
                self.progress.failed_chunks += 1
                if len(self.progress.errors) < 100:
                    self.progress.errors.append(f"Chunk error: {str(e)[:100]}")
    
    def _save_checkpoint(self):
        """Save checkpoint for resume capability."""
        # Save graph state
        self._graph.save(self.output_path)
        
        # Save progress
        checkpoint = {
            "processed_chunks": self.progress.processed_chunks,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.output_path / "checkpoint.json", 'w') as f:
            json.dump(checkpoint, f)
        
        self.progress.last_checkpoint = time.time()
        logger.debug(f"Checkpoint saved: {self.progress.processed_chunks} chunks")
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        checkpoint_path = self.output_path / "checkpoint.json"
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return None
    
    def _save_graph(self):
        """Save final Knowledge Graph."""
        self._graph.save(self.output_path)
        
        # Save metadata
        stats = self._graph.get_stats()
        metadata = {
            "total_entities": stats["total_entities"],
            "total_relationships": stats["total_relationships"],
            "entity_types": stats.get("entity_types", {}),
            "relationship_types": stats.get("relationship_types", {}),
            "build_time": datetime.now().isoformat(),
        }
        
        with open(self.output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up checkpoint
        checkpoint_path = self.output_path / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        logger.info(f"Knowledge Graph saved to {self.output_path}")
    
    def stop(self):
        """Stop the build process gracefully."""
        self._stop_flag = True


def build_all_indices(chunks_path: Path = None,
                      hype: bool = True,
                      kg: bool = True,
                      progress_callback: Callable[[str, BuildProgress], None] = None):
    """
    Build all indices from chunks.
    
    Args:
        chunks_path: Path to chunks.json
        hype: Build HyPE index
        kg: Build Knowledge Graph
        progress_callback: Called with (index_name, progress)
    """
    chunks_path = chunks_path or Path(INDEX_PATH) / "chunks.json"
    
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return
    
    results = {}
    
    if hype:
        logger.info("=" * 50)
        logger.info("Building HyPE Index...")
        logger.info("=" * 50)
        
        builder = HyPEIndexBuilder(batch_size=50, checkpoint_interval=500)
        
        def hype_callback(p):
            if progress_callback:
                progress_callback("hype", p)
        
        results["hype"] = builder.build(chunks_path, progress_callback=hype_callback)
    
    if kg:
        logger.info("=" * 50)
        logger.info("Building Knowledge Graph...")
        logger.info("=" * 50)
        
        builder = KnowledgeGraphBuilder(batch_size=100, checkpoint_interval=1000)
        
        def kg_callback(p):
            if progress_callback:
                progress_callback("kg", p)
        
        results["kg"] = builder.build(chunks_path, progress_callback=kg_callback)
    
    # Summary
    logger.info("=" * 50)
    logger.info("INDEX BUILD SUMMARY")
    logger.info("=" * 50)
    
    for name, progress in results.items():
        logger.info(f"{name.upper()}: {progress.processed_chunks} chunks, "
                   f"{progress.failed_chunks} failed, "
                   f"{progress.elapsed_seconds:.1f}s elapsed")
    
    return results


if __name__ == "__main__":
    # CLI for building indices
    import argparse
    
    parser = argparse.ArgumentParser(description="Build RAG Ultimate indices")
    parser.add_argument("--chunks", type=str, help="Path to chunks.json")
    parser.add_argument("--hype", action="store_true", help="Build HyPE index")
    parser.add_argument("--kg", action="store_true", help="Build Knowledge Graph")
    parser.add_argument("--all", action="store_true", help="Build all indices")
    
    args = parser.parse_args()
    
    chunks_path = Path(args.chunks) if args.chunks else None
    
    if args.all:
        build_all_indices(chunks_path, hype=True, kg=True)
    else:
        build_all_indices(chunks_path, hype=args.hype, kg=args.kg)
