"""
Contextual Indexer for RAG v2.0
===============================
Implements Anthropic's Contextual Retrieval technique.
Prepends context to each chunk before embedding for 49% better retrieval.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import DATASET_PATH, INDEX_PATH


@dataclass
class ContextualChunk:
    """A chunk with prepended context."""
    chunk_id: str
    file_path: str
    category: str
    start_line: int
    end_line: int
    original_content: str
    context_prefix: str
    contextualized_content: str
    symbols: List[str]


class ContextGenerator:
    """
    Generates contextual prefixes for chunks.
    
    Context includes:
    - File location and type
    - Parent class/function
    - Related protocol commands
    - Symbol definitions
    """
    
    # Category descriptions
    CATEGORY_CONTEXT = {
        "source_code": "This is ActionScript 3 source code from the Evony game client",
        "protocol": "This describes an Evony AMF3 protocol command",
        "exploit": "This documents an exploit or vulnerability",
        "bot": "This is code from an Evony automation bot",
        "cve": "This is CVE vulnerability research",
        "documentation": "This is technical documentation",
        "training": "This is training data for the Evony model",
        "config": "This is configuration data",
    }
    
    # File type patterns
    FILE_PATTERNS = {
        r"\.as$": "ActionScript source file",
        r"Command\.as$": "AMF3 protocol command handler",
        r"Bean\.as$": "data transfer object (DTO)",
        r"Constant\.as$": "constant definitions",
        r"\.py$": "Python script",
        r"\.md$": "Markdown documentation",
        r"\.json$": "JSON data file",
        r"\.jsonl$": "JSON Lines training data",
    }
    
    def generate_context(self, chunk: Dict) -> str:
        """
        Generate contextual prefix for a chunk.
        
        Args:
            chunk: Dictionary with file_path, category, content, symbols
            
        Returns:
            Context prefix string
        """
        parts = []
        
        file_path = chunk.get("file_path", "")
        category = chunk.get("category", "")
        content = chunk.get("content", "")
        symbols = chunk.get("symbols", [])
        start_line = chunk.get("start_line", 0)
        end_line = chunk.get("end_line", 0)
        
        # Category context
        if category in self.CATEGORY_CONTEXT:
            parts.append(self.CATEGORY_CONTEXT[category])
        
        # File type context
        for pattern, description in self.FILE_PATTERNS.items():
            if re.search(pattern, file_path, re.IGNORECASE):
                parts.append(f"from {description}")
                break
        
        # File path
        parts.append(f"'{file_path}' (lines {start_line}-{end_line})")
        
        # Symbol context
        if symbols:
            symbol_str = ", ".join(symbols[:5])  # Limit to 5 symbols
            parts.append(f"defining: {symbol_str}")
        
        # Content-specific context
        content_context = self._analyze_content(content, file_path)
        if content_context:
            parts.append(content_context)
        
        # Combine into context prefix
        context = ". ".join(parts) + "."
        
        return context
    
    def _analyze_content(self, content: str, file_path: str) -> Optional[str]:
        """Analyze content to extract additional context."""
        content_lower = content.lower()
        
        # Command ID detection
        cmd_match = re.search(r'command\s*(?:id|ID)?\s*[=:]\s*(\d+)', content)
        if cmd_match:
            return f"Command ID {cmd_match.group(1)}"
        
        # Class definition
        class_match = re.search(r'(?:public|private)\s+class\s+(\w+)', content)
        if class_match:
            return f"Defines class {class_match.group(1)}"
        
        # Function definition
        func_match = re.search(r'(?:public|private|protected)\s+(?:static\s+)?function\s+(\w+)', content)
        if func_match:
            return f"Defines function {func_match.group(1)}"
        
        # CVE reference
        cve_match = re.search(r'(CVE-\d{4}-\d+)', content, re.IGNORECASE)
        if cve_match:
            return f"References {cve_match.group(1)}"
        
        # Protocol command name
        if "Command" in file_path:
            cmd_name = Path(file_path).stem
            return f"Protocol command: {cmd_name}"
        
        return None
    
    def contextualize_chunk(self, chunk: Dict) -> ContextualChunk:
        """
        Create a contextualized chunk.
        
        Args:
            chunk: Raw chunk dictionary
            
        Returns:
            ContextualChunk with prepended context
        """
        context = self.generate_context(chunk)
        original = chunk.get("content", "")
        
        # Prepend context to content
        contextualized = f"{context}\n\n{original}"
        
        return ContextualChunk(
            chunk_id=chunk.get("chunk_id", ""),
            file_path=chunk.get("file_path", ""),
            category=chunk.get("category", ""),
            start_line=chunk.get("start_line", 0),
            end_line=chunk.get("end_line", 0),
            original_content=original,
            context_prefix=context,
            contextualized_content=contextualized,
            symbols=chunk.get("symbols", [])
        )


class ContextualIndexer:
    """
    Indexes documents with contextual prefixes.
    """
    
    def __init__(self):
        self.context_gen = ContextGenerator()
        self.index_path = INDEX_PATH
        
    def load_existing_chunks(self) -> List[Dict]:
        """Load existing chunks from the index."""
        chunks_file = self.index_path / "chunks.json"
        
        if not chunks_file.exists():
            return []
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def contextualize_all(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add context to all chunks.
        
        Args:
            chunks: List of raw chunks
            
        Returns:
            List of chunks with contextualized content
        """
        contextualized = []
        
        for chunk in chunks:
            ctx_chunk = self.context_gen.contextualize_chunk(chunk)
            
            # Create enhanced chunk dict
            enhanced = chunk.copy()
            enhanced["context_prefix"] = ctx_chunk.context_prefix
            enhanced["contextualized_content"] = ctx_chunk.contextualized_content
            
            contextualized.append(enhanced)
        
        return contextualized
    
    def save_contextualized(self, chunks: List[Dict]):
        """Save contextualized chunks."""
        output_file = self.index_path / "chunks_contextualized.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        
        return output_file
    
    def index_new_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Index new documents with context.
        
        Args:
            docs: List of new documents to index
                  Each doc should have: file_path, content, category
                  
        Returns:
            List of contextualized chunks
        """
        new_chunks = []
        
        for doc in docs:
            # Split into chunks if needed
            content = doc.get("content", "")
            file_path = doc.get("file_path", "")
            category = doc.get("category", "documentation")
            
            # Simple chunking (can be enhanced)
            chunk_size = 500
            lines = content.split("\n")
            
            current_chunk = []
            current_size = 0
            start_line = 1
            
            for i, line in enumerate(lines, 1):
                current_chunk.append(line)
                current_size += len(line)
                
                if current_size >= chunk_size:
                    chunk_content = "\n".join(current_chunk)
                    chunk = {
                        "chunk_id": f"{file_path}:{start_line}-{i}",
                        "file_path": file_path,
                        "category": category,
                        "start_line": start_line,
                        "end_line": i,
                        "content": chunk_content,
                        "symbols": self._extract_symbols(chunk_content)
                    }
                    
                    # Contextualize
                    ctx_chunk = self.context_gen.contextualize_chunk(chunk)
                    chunk["context_prefix"] = ctx_chunk.context_prefix
                    chunk["contextualized_content"] = ctx_chunk.contextualized_content
                    
                    new_chunks.append(chunk)
                    
                    current_chunk = []
                    current_size = 0
                    start_line = i + 1
            
            # Handle remaining content
            if current_chunk:
                chunk_content = "\n".join(current_chunk)
                chunk = {
                    "chunk_id": f"{file_path}:{start_line}-{len(lines)}",
                    "file_path": file_path,
                    "category": category,
                    "start_line": start_line,
                    "end_line": len(lines),
                    "content": chunk_content,
                    "symbols": self._extract_symbols(chunk_content)
                }
                
                ctx_chunk = self.context_gen.contextualize_chunk(chunk)
                chunk["context_prefix"] = ctx_chunk.context_prefix
                chunk["contextualized_content"] = ctx_chunk.contextualized_content
                
                new_chunks.append(chunk)
        
        return new_chunks
    
    def _extract_symbols(self, content: str) -> List[str]:
        """Extract symbol names from content."""
        symbols = []
        
        # Class names
        classes = re.findall(r'class\s+(\w+)', content)
        symbols.extend(classes)
        
        # Function names
        functions = re.findall(r'function\s+(\w+)', content)
        symbols.extend(functions)
        
        # Variable names (public/private)
        variables = re.findall(r'(?:public|private|protected)\s+(?:static\s+)?(?:var|const)\s+(\w+)', content)
        symbols.extend(variables)
        
        # Command IDs
        commands = re.findall(r'COMMAND[_\w]*\s*[=:]\s*(\d+)', content, re.IGNORECASE)
        symbols.extend([f"CMD_{c}" for c in commands])
        
        return list(set(symbols))[:10]  # Limit to 10 unique symbols


# Singleton
_contextual_indexer = None


def get_contextual_indexer() -> ContextualIndexer:
    """Get singleton contextual indexer."""
    global _contextual_indexer
    if _contextual_indexer is None:
        _contextual_indexer = ContextualIndexer()
    return _contextual_indexer
