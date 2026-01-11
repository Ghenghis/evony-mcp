#!/usr/bin/env python3
"""
Index New Documents for RAG v2.0
================================
Adds new documentation to the RAG knowledge base with contextual chunking.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evony_rag.config import INDEX_PATH, DATASET_PATH
from evony_rag.contextual_indexer import get_contextual_indexer


class NewDocumentIndexer:
    """
    Indexes new documentation files into the RAG.
    """
    
    # Documents to index (relative to Evony_Decrypted root)
    NEW_DOCS = [
        # LM Studio integration
        ("Docs/LMSTUDIO_CLAUDE_WINDSURF_ACCESS.md", "documentation"),
        ("Docs/LMSTUDIO_F16_CONFIG.md", "documentation"),
        ("lmstudio_presets/README.md", "documentation"),
        ("lmstudio_presets/evony-master-expert.preset.json", "config"),
        ("lmstudio_presets/evony-exploit-hunter.preset.json", "config"),
        ("lmstudio_presets/evony-protocol-decoder.preset.json", "config"),
        ("lmstudio_presets/evony-code-auditor.preset.json", "config"),
        ("lmstudio_presets/evony-quick-reference.preset.json", "config"),
        ("lmstudio_presets/evony-forensic-analyst.preset.json", "config"),
        ("lmstudio_presets/evony-creative-writer.preset.json", "config"),
        
        # Control scripts
        ("evony_rte/lmstudio_control.py", "source_code"),
        
        # Coordination docs
        ("Windsurf.md", "documentation"),
        ("CLAUDE_DESKTOP_ASSISTANCE.md", "documentation"),
        
        # RAG documentation
        ("evony_rag/RAG_V2_ARCHITECTURE.md", "documentation"),
        
        # Training documentation
        ("Docs/LLM_TRAINING_COMPLETE.md", "documentation"),
        ("Docs/EPIC_TRAINING_PLAN.md", "documentation"),
        
        # Protocol documentation
        ("Docs/COMPLETE_PROTOCOL_REFERENCE.md", "protocol"),
        ("Docs/COMMAND_REFERENCE.md", "protocol"),
        ("Docs/NETWORK_PROTOCOL.md", "protocol"),
        
        # Security documentation
        ("Docs/HACKING_TOOLKIT.md", "exploit"),
        ("Docs/HIDDEN_COMMANDS.md", "exploit"),
        ("Docs/SECURITY_VALIDATION_RULES.md", "exploit"),
    ]
    
    def __init__(self, root_path: Optional[str] = None):
        """Initialize indexer with project root path."""
        if root_path:
            self.root = Path(root_path)
        else:
            # Auto-detect from this file's location
            self.root = Path(__file__).parent.parent
        
        self.indexer = get_contextual_indexer()
        self.index_path = INDEX_PATH
        
    def read_document(self, rel_path: str) -> Optional[str]:
        """Read document content."""
        full_path = self.root / rel_path
        
        if not full_path.exists():
            print(f"⚠️  Not found: {rel_path}")
            return None
        
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error reading {rel_path}: {e}")
            return None
    
    def load_existing_index(self) -> Dict:
        """Load existing chunk index."""
        chunks_file = self.index_path / "chunks.json"
        
        if not chunks_file.exists():
            return {"chunks": [], "metadata": {}}
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            return {"chunks": data, "metadata": {}}
        return data
    
    def save_index(self, data: Dict):
        """Save updated index."""
        chunks_file = self.index_path / "chunks.json"
        
        # Ensure directory exists
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def get_content_hash(self, content: str) -> str:
        """Get hash of content for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def index_all_new_docs(self, force: bool = False) -> Dict:
        """
        Index all new documents.
        
        Args:
            force: Re-index even if already indexed
            
        Returns:
            Summary of indexing results
        """
        results = {
            "indexed": [],
            "skipped": [],
            "errors": [],
            "total_chunks": 0,
        }
        
        # Load existing index
        existing = self.load_existing_index()
        existing_chunks = existing.get("chunks", [])
        existing_paths = {c.get("file_path") for c in existing_chunks}
        
        new_chunks = []
        
        for rel_path, category in self.NEW_DOCS:
            # Check if already indexed
            if rel_path in existing_paths and not force:
                results["skipped"].append(rel_path)
                continue
            
            # Read document
            content = self.read_document(rel_path)
            if content is None:
                results["errors"].append(rel_path)
                continue
            
            # Create document dict
            doc = {
                "file_path": rel_path,
                "content": content,
                "category": category,
            }
            
            # Index with contextual chunking
            chunks = self.indexer.index_new_documents([doc])
            
            new_chunks.extend(chunks)
            results["indexed"].append(rel_path)
            results["total_chunks"] += len(chunks)
            
            print(f"✅ Indexed: {rel_path} ({len(chunks)} chunks)")
        
        # Merge with existing chunks (remove old versions of indexed files)
        merged_chunks = [
            c for c in existing_chunks 
            if c.get("file_path") not in {p for p, _ in self.NEW_DOCS if p in results["indexed"]}
        ]
        merged_chunks.extend(new_chunks)
        
        # Update index
        updated_data = {
            "chunks": merged_chunks,
            "metadata": {
                "total_chunks": len(merged_chunks),
                "last_updated": datetime.now().isoformat(),
                "version": "2.0",
                "contextual": True,
            }
        }
        
        self.save_index(updated_data)
        
        return results
    
    def index_custom_docs(self, doc_paths: List[tuple]) -> Dict:
        """
        Index custom document paths.
        
        Args:
            doc_paths: List of (rel_path, category) tuples
            
        Returns:
            Indexing results
        """
        results = {"indexed": [], "errors": [], "total_chunks": 0}
        
        existing = self.load_existing_index()
        existing_chunks = existing.get("chunks", [])
        
        new_chunks = []
        
        for rel_path, category in doc_paths:
            content = self.read_document(rel_path)
            if content is None:
                results["errors"].append(rel_path)
                continue
            
            doc = {
                "file_path": rel_path,
                "content": content,
                "category": category,
            }
            
            chunks = self.indexer.index_new_documents([doc])
            new_chunks.extend(chunks)
            results["indexed"].append(rel_path)
            results["total_chunks"] += len(chunks)
        
        # Merge
        indexed_paths = set(results["indexed"])
        merged_chunks = [c for c in existing_chunks if c.get("file_path") not in indexed_paths]
        merged_chunks.extend(new_chunks)
        
        updated_data = {
            "chunks": merged_chunks,
            "metadata": {
                "total_chunks": len(merged_chunks),
                "last_updated": datetime.now().isoformat(),
                "version": "2.0",
            }
        }
        
        self.save_index(updated_data)
        
        return results


def main():
    """Run indexer from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index new documents for RAG v2.0")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-index all")
    parser.add_argument("--root", "-r", type=str, help="Project root path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAG v2.0 Document Indexer")
    print("=" * 60)
    
    indexer = NewDocumentIndexer(args.root)
    results = indexer.index_all_new_docs(force=args.force)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"✅ Indexed: {len(results['indexed'])} documents")
    print(f"⏭️  Skipped: {len(results['skipped'])} (already indexed)")
    print(f"❌ Errors:  {len(results['errors'])}")
    print(f"📦 Total chunks: {results['total_chunks']}")
    
    if results["errors"]:
        print("\nErrors:")
        for e in results["errors"]:
            print(f"  - {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
