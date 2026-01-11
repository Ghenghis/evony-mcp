#!/usr/bin/env python3
"""
AS3 Code Chunk Extractor - Creates real code chunks for RAG citations
Extracts actual code from AS3 files with proper context for useful search results.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Source directories containing AS3 code
AS3_SOURCES = [
    r"C:\Users\Admin\Downloads\Evony_Decrypted\AS3_Scripts_(AutoEvony2_NEW.swf)",
    r"C:\Users\Admin\Downloads\Evony_Decrypted\AS3_Scripts_(EvonyClient1921.swf)",
]

# Additional source directories
ADDITIONAL_SOURCES = [
    r"C:\Users\Admin\Downloads\Evony_Decrypted\Scripts",
    r"C:\Users\Admin\Downloads\Evony_Decrypted\exploits",
    r"C:\Users\Admin\Downloads\Evony_Decrypted\keys",
]

# Output path
OUTPUT_DIR = Path(__file__).parent / "index"
CHUNKS_PATH = OUTPUT_DIR / "chunks.json"
BACKUP_PATH = OUTPUT_DIR / f"chunks_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Chunk settings
MAX_CHUNK_LINES = 50
MIN_CHUNK_LINES = 5
OVERLAP_LINES = 10


def extract_as3_chunks(file_path: str) -> List[Dict]:
    """Extract code chunks from an AS3 file with proper context."""
    chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return []
    
    if len(lines) < MIN_CHUNK_LINES:
        # Small file - keep as single chunk
        chunks.append(create_chunk(file_path, lines, 1, len(lines)))
        return chunks
    
    # Extract function/class boundaries for smarter chunking
    boundaries = find_code_boundaries(lines)
    
    if boundaries:
        # Chunk by function/class boundaries
        for start, end, chunk_type in boundaries:
            chunk_lines = lines[start-1:end]
            if len(chunk_lines) >= MIN_CHUNK_LINES:
                chunks.append(create_chunk(file_path, chunk_lines, start, end, chunk_type))
    else:
        # Fall back to sliding window chunking
        for i in range(0, len(lines), MAX_CHUNK_LINES - OVERLAP_LINES):
            end_idx = min(i + MAX_CHUNK_LINES, len(lines))
            chunk_lines = lines[i:end_idx]
            if len(chunk_lines) >= MIN_CHUNK_LINES:
                chunks.append(create_chunk(file_path, chunk_lines, i + 1, end_idx))
    
    return chunks


def find_code_boundaries(lines: List[str]) -> List[tuple]:
    """Find function and class boundaries in AS3 code."""
    boundaries = []
    current_start = None
    current_type = None
    brace_depth = 0
    
    # Patterns for AS3 code structures
    class_pattern = re.compile(r'^\s*(public|private|internal|protected)?\s*(class|interface)\s+(\w+)')
    func_pattern = re.compile(r'^\s*(public|private|protected|internal|override)?\s*(static)?\s*function\s+(\w+)')
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Check for class/interface start
        class_match = class_pattern.match(line)
        if class_match and current_start is None:
            current_start = line_num
            current_type = "class"
        
        # Check for function start
        func_match = func_pattern.match(line)
        if func_match and current_start is None:
            current_start = line_num
            current_type = "function"
        
        # Track braces
        brace_depth += line.count('{') - line.count('}')
        
        # End of block
        if current_start is not None and brace_depth == 0 and '{' in ''.join(lines[current_start-1:line_num]):
            boundaries.append((current_start, line_num, current_type))
            current_start = None
            current_type = None
    
    # Handle unclosed blocks
    if current_start is not None:
        boundaries.append((current_start, len(lines), current_type))
    
    return boundaries


def create_chunk(file_path: str, lines: List[str], start: int, end: int, chunk_type: str = "code") -> Dict:
    """Create a chunk dictionary with proper metadata."""
    content = '\n'.join(lines)
    
    # Extract symbols from content
    symbols = extract_symbols(content)
    
    # Determine category from path
    category = determine_category(file_path)
    
    # Create unique ID
    rel_path = os.path.relpath(file_path, r"C:\Users\Admin\Downloads\Evony_Decrypted")
    chunk_id = f"{rel_path}:{start}-{end}"
    
    return {
        "id": chunk_id,
        "file_path": rel_path,
        "category": category,
        "start_line": start,
        "end_line": end,
        "content": content,
        "chunk_type": chunk_type,
        "symbols": symbols,
        "line_count": end - start + 1
    }


def extract_symbols(content: str) -> List[str]:
    """Extract function names, class names, and important identifiers."""
    symbols = []
    
    # Function names
    func_matches = re.findall(r'function\s+(\w+)', content)
    symbols.extend(func_matches)
    
    # Class names
    class_matches = re.findall(r'class\s+(\w+)', content)
    symbols.extend(class_matches)
    
    # Variable assignments (important ones)
    var_matches = re.findall(r'(?:var|const)\s+(\w+)\s*:', content)
    symbols.extend(var_matches[:10])  # Limit variables
    
    # AMF commands
    amf_matches = re.findall(r'["\'](\w+\.\w+)["\']', content)
    symbols.extend(amf_matches)
    
    return list(set(symbols))[:20]  # Unique, limited


def determine_category(file_path: str) -> str:
    """Determine the category based on file path."""
    path_lower = file_path.lower()
    
    if 'autoevony' in path_lower:
        return 'bot_client'
    elif 'evonyclient' in path_lower:
        return 'game_client'
    elif 'exploit' in path_lower:
        return 'exploits'
    elif 'script' in path_lower:
        return 'scripts'
    elif 'key' in path_lower:
        return 'keys'
    elif 'protocol' in path_lower:
        return 'protocol'
    else:
        return 'as3_code'


def process_directory(dir_path: str, extensions: List[str] = ['.as', '.txt', '.json']) -> List[Dict]:
    """Process all files in a directory."""
    chunks = []
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        print(f"  Directory not found: {dir_path}")
        return []
    
    file_count = 0
    for ext in extensions:
        for file_path in dir_path.rglob(f'*{ext}'):
            file_chunks = extract_as3_chunks(str(file_path))
            chunks.extend(file_chunks)
            file_count += 1
            if file_count % 50 == 0:
                print(f"    Processed {file_count} files...")
    
    return chunks


def main():
    """Main extraction process."""
    print("=" * 60)
    print("AS3 CODE CHUNK EXTRACTOR")
    print("=" * 60)
    
    all_chunks = []
    
    # Process AS3 source directories
    print("\n[1/2] Processing AS3 source directories...")
    for source_dir in AS3_SOURCES:
        print(f"  Processing: {source_dir}")
        chunks = process_directory(source_dir, ['.as'])
        print(f"    Extracted {len(chunks)} chunks")
        all_chunks.extend(chunks)
    
    # Process additional sources
    print("\n[2/2] Processing additional sources...")
    for source_dir in ADDITIONAL_SOURCES:
        print(f"  Processing: {source_dir}")
        chunks = process_directory(source_dir, ['.as', '.txt', '.json', '.py'])
        print(f"    Extracted {len(chunks)} chunks")
        all_chunks.extend(chunks)
    
    # Backup existing chunks
    if CHUNKS_PATH.exists():
        print(f"\n[Backup] Saving backup to {BACKUP_PATH}")
        import shutil
        shutil.copy(CHUNKS_PATH, BACKUP_PATH)
    
    # Save new chunks
    print(f"\n[Save] Writing {len(all_chunks)} chunks to {CHUNKS_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {len(all_chunks)}")
    
    # Category breakdown
    categories = {}
    for chunk in all_chunks:
        cat = chunk.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nBy category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # Sample chunk
    if all_chunks:
        print("\nSample chunk:")
        sample = all_chunks[0]
        print(f"  ID: {sample['id']}")
        print(f"  File: {sample['file_path']}")
        print(f"  Lines: {sample['start_line']}-{sample['end_line']}")
        print(f"  Symbols: {sample['symbols'][:5]}")
        print(f"  Content preview: {sample['content'][:100]}...")
    
    return len(all_chunks)


if __name__ == "__main__":
    main()
