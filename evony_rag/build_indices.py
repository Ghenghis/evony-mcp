#!/usr/bin/env python3
"""
RAG Ultimate Index Builder CLI
==============================
Build HyPE and Knowledge Graph indices from chunks.

Usage:
    python build_indices.py --all              # Build all indices
    python build_indices.py --hype             # Build HyPE index only
    python build_indices.py --kg               # Build Knowledge Graph only
    python build_indices.py --status           # Check index status
    
Options:
    --chunks PATH    Path to chunks.json (default: index/chunks.json)
    --no-resume      Don't resume from checkpoint
    --batch SIZE     Batch size (default: 50)
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evony_rag.config import INDEX_PATH


def print_progress(index_name: str, progress):
    """Print progress bar."""
    pct = progress.percent_complete
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "=" * filled + "-" * (bar_len - filled)
    
    eta_min = progress.eta_seconds / 60
    
    print(f"\r[{index_name.upper()}] [{bar}] {pct:.1f}% "
          f"({progress.processed_chunks}/{progress.total_chunks}) "
          f"ETA: {eta_min:.1f}min", end="", flush=True)


def check_status():
    """Check status of all indices."""
    print("=" * 60)
    print("RAG ULTIMATE INDEX STATUS")
    print("=" * 60)
    
    index_path = Path(INDEX_PATH)
    
    # Check chunks
    chunks_path = index_path / "chunks.json"
    if chunks_path.exists():
        import json
        try:
            with open(chunks_path, encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"\n[CHUNKS] {len(chunks):,} chunks available")
        except json.JSONDecodeError as e:
            print(f"\n[CHUNKS] ERROR: Corrupted ({e})")
    else:
        print(f"\n[CHUNKS] NOT FOUND at {chunks_path}")
    
    # Check HyPE index
    hype_path = index_path / "hype_index"
    if (hype_path / "hype_chunks.json").exists():
        try:
            with open(hype_path / "metadata.json") as f:
                meta = json.load(f)
            print(f"\n[HYPE INDEX] READY")
            print(f"    Chunks: {meta.get('total_chunks', 0):,}")
            print(f"    Questions/chunk: {meta.get('questions_per_chunk', 0):.1f}")
            print(f"    Built: {meta.get('build_time', 'unknown')}")
        except:
            print(f"\n[HYPE INDEX] Incomplete")
    elif (hype_path / "checkpoint.json").exists():
        try:
            with open(hype_path / "checkpoint.json") as f:
                ckpt = json.load(f)
            print(f"\n[HYPE INDEX] IN PROGRESS")
            print(f"    Checkpoint: {ckpt.get('processed_chunks', 0):,} chunks")
            print(f"    Saved: {ckpt.get('timestamp', 'unknown')}")
        except:
            print(f"\n[HYPE INDEX] Checkpoint exists but unreadable")
    else:
        print(f"\n[HYPE INDEX] NOT BUILT")
    
    # Check Knowledge Graph
    kg_path = index_path / "knowledge_graph"
    if (kg_path / "knowledge_graph.json").exists():
        try:
            with open(kg_path / "metadata.json") as f:
                meta = json.load(f)
            print(f"\n[KNOWLEDGE GRAPH] READY")
            print(f"    Entities: {meta.get('total_entities', 0):,}")
            print(f"    Relationships: {meta.get('total_relationships', 0):,}")
            print(f"    Built: {meta.get('build_time', 'unknown')}")
        except:
            print(f"\n[KNOWLEDGE GRAPH] Incomplete")
    elif (kg_path / "checkpoint.json").exists():
        try:
            with open(kg_path / "checkpoint.json") as f:
                ckpt = json.load(f)
            print(f"\n[KNOWLEDGE GRAPH] IN PROGRESS")
            print(f"    Checkpoint: {ckpt.get('processed_chunks', 0):,} chunks")
        except:
            print(f"\n[KNOWLEDGE GRAPH] Checkpoint exists")
    else:
        print(f"\n[KNOWLEDGE GRAPH] NOT BUILT")
    
    print("\n" + "=" * 60)


def build_hype(chunks_path: Path, resume: bool, batch_size: int):
    """Build HyPE index."""
    from evony_rag.index_builders import HyPEIndexBuilder
    
    print("\n" + "=" * 60)
    print("BUILDING HYPE INDEX")
    print("=" * 60)
    print(f"Source: {chunks_path}")
    print(f"Resume: {resume}")
    print(f"Batch size: {batch_size}")
    print("=" * 60 + "\n")
    
    builder = HyPEIndexBuilder(
        batch_size=batch_size,
        checkpoint_interval=500,
        use_llm=False  # Use template-based for speed
    )
    
    def callback(p):
        print_progress("hype", p)
    
    try:
        progress = builder.build(chunks_path, resume=resume, progress_callback=callback)
        print("\n")
        print(f"\nHyPE Index Build Complete!")
        print(f"  Processed: {progress.processed_chunks:,}")
        print(f"  Failed: {progress.failed_chunks:,}")
        print(f"  Time: {progress.elapsed_seconds:.1f}s")
        return True
    except KeyboardInterrupt:
        print("\n\nBuild interrupted. Progress saved to checkpoint.")
        return False
    except Exception as e:
        print(f"\n\nBuild error: {e}")
        return False


def build_kg(chunks_path: Path, resume: bool, batch_size: int):
    """Build Knowledge Graph."""
    from evony_rag.index_builders import KnowledgeGraphBuilder
    
    print("\n" + "=" * 60)
    print("BUILDING KNOWLEDGE GRAPH")
    print("=" * 60)
    print(f"Source: {chunks_path}")
    print(f"Resume: {resume}")
    print(f"Batch size: {batch_size}")
    print("=" * 60 + "\n")
    
    builder = KnowledgeGraphBuilder(
        batch_size=batch_size,
        checkpoint_interval=1000
    )
    
    def callback(p):
        print_progress("kg", p)
    
    try:
        progress = builder.build(chunks_path, resume=resume, progress_callback=callback)
        print("\n")
        print(f"\nKnowledge Graph Build Complete!")
        print(f"  Processed: {progress.processed_chunks:,}")
        print(f"  Failed: {progress.failed_chunks:,}")
        print(f"  Time: {progress.elapsed_seconds:.1f}s")
        return True
    except KeyboardInterrupt:
        print("\n\nBuild interrupted. Progress saved to checkpoint.")
        return False
    except Exception as e:
        print(f"\n\nBuild error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RAG Ultimate Index Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--all", action="store_true", help="Build all indices")
    parser.add_argument("--hype", action="store_true", help="Build HyPE index")
    parser.add_argument("--kg", action="store_true", help="Build Knowledge Graph")
    parser.add_argument("--status", action="store_true", help="Check index status")
    parser.add_argument("--chunks", type=str, help="Path to chunks.json")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--batch", type=int, default=50, help="Batch size")
    
    args = parser.parse_args()
    
    if args.status:
        check_status()
        return
    
    if not (args.all or args.hype or args.kg):
        parser.print_help()
        print("\nUse --status to check current index status")
        return
    
    # Determine chunks path
    chunks_path = Path(args.chunks) if args.chunks else Path(INDEX_PATH) / "chunks.json"
    
    if not chunks_path.exists():
        print(f"ERROR: Chunks file not found: {chunks_path}")
        print("\nPlease ensure chunks.json exists in the index directory.")
        return
    
    resume = not args.no_resume
    batch_size = args.batch
    
    # Build requested indices
    if args.all or args.hype:
        build_hype(chunks_path, resume, batch_size)
    
    if args.all or args.kg:
        build_kg(chunks_path, resume, batch_size)
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    check_status()


if __name__ == "__main__":
    main()
