#!/usr/bin/env python3
"""
Repair corrupted chunks.json
============================
Attempts to repair JSON corruption by:
1. Finding the corruption point
2. Recovering valid chunks before corruption
3. Attempting to parse and fix the corrupted section
4. Rebuilding the file
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evony_rag.config import INDEX_PATH


def find_corruption_point(content: str) -> tuple:
    """Find where JSON parsing fails."""
    try:
        json.loads(content)
        return None, "File is valid JSON"
    except json.JSONDecodeError as e:
        return e.pos, f"Line {e.lineno}, Col {e.colno}: {e.msg}"


def repair_chunks(chunks_path: Path, output_path: Path = None):
    """
    Repair corrupted chunks.json.
    """
    print(f"Reading {chunks_path}...")
    
    with open(chunks_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    print(f"File size: {len(content):,} bytes")
    
    # Find corruption point
    pos, msg = find_corruption_point(content)
    
    if pos is None:
        print("File is already valid JSON!")
        return True
    
    print(f"Corruption found at position {pos:,}: {msg}")
    
    # Show context around corruption
    start = max(0, pos - 200)
    end = min(len(content), pos + 200)
    print(f"\nContext around corruption (pos {pos}):")
    print("-" * 60)
    print(content[start:pos] + ">>>HERE<<<" + content[pos:end])
    print("-" * 60)
    
    # Strategy 1: Try to recover chunks before corruption
    print("\nAttempting recovery...")
    
    # Find the last complete chunk before corruption
    # Look for }, pattern before corruption point
    search_start = max(0, pos - 5000)
    search_content = content[search_start:pos]
    
    # Find last complete object end
    last_complete = None
    for match in re.finditer(r'\},\s*\{', search_content):
        last_complete = search_start + match.end() - 1  # Position of the {
    
    if last_complete:
        print(f"Found last complete chunk boundary at position {last_complete:,}")
        
        # Try to parse up to that point
        truncated = content[:last_complete - 1] + "]"  # Close the array
        
        try:
            chunks = json.loads(truncated)
            print(f"Successfully recovered {len(chunks):,} chunks!")
            
            # Save recovered chunks
            output = output_path or chunks_path.parent / "chunks_recovered.json"
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f)
            
            print(f"Saved to {output}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Recovery failed: {e}")
    
    # Strategy 2: Try line-by-line recovery
    print("\nTrying line-by-line recovery...")
    
    chunks = []
    current_chunk = []
    brace_depth = 0
    in_array = False
    
    for i, line in enumerate(content.split('\n')):
        try:
            stripped = line.strip()
            
            if stripped == '[':
                in_array = True
                continue
            if stripped == ']':
                break
            
            if in_array:
                # Track brace depth
                brace_depth += line.count('{') - line.count('}')
                current_chunk.append(line)
                
                if brace_depth == 0 and current_chunk:
                    # Try to parse this chunk
                    chunk_str = ''.join(current_chunk).rstrip(',').strip()
                    if chunk_str:
                        try:
                            chunk = json.loads(chunk_str)
                            chunks.append(chunk)
                        except:
                            pass
                    current_chunk = []
                    
        except Exception as e:
            if i < pos // 50:  # Only report errors near corruption
                continue
            print(f"Error at line {i}: {e}")
            break
        
        if i % 10000 == 0:
            print(f"  Processed {i:,} lines, recovered {len(chunks):,} chunks...")
    
    if chunks:
        print(f"\nRecovered {len(chunks):,} chunks via line-by-line parsing")
        
        output = output_path or chunks_path.parent / "chunks_recovered.json"
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)
        
        print(f"Saved to {output}")
        return True
    
    print("Recovery failed - no chunks could be recovered")
    return False


def validate_and_fix(chunks_path: Path):
    """Validate chunks and attempt fixes."""
    print("=" * 60)
    print("CHUNKS.JSON REPAIR UTILITY")
    print("=" * 60)
    
    if not chunks_path.exists():
        print(f"ERROR: File not found: {chunks_path}")
        return False
    
    # Try to repair
    success = repair_chunks(chunks_path)
    
    if success:
        # Check the recovered file
        recovered_path = chunks_path.parent / "chunks_recovered.json"
        if recovered_path.exists():
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)
            
            try:
                with open(recovered_path) as f:
                    chunks = json.load(f)
                print(f"Recovered file is valid JSON with {len(chunks):,} chunks")
                
                # Show sample
                if chunks:
                    print(f"\nSample chunk:")
                    sample = chunks[0]
                    print(f"  chunk_id: {sample.get('chunk_id', 'N/A')}")
                    print(f"  file_path: {sample.get('file_path', 'N/A')}")
                    print(f"  category: {sample.get('category', 'N/A')}")
                
                # Ask to replace original
                print(f"\nRecovered file saved to: {recovered_path}")
                print("To use, rename chunks_recovered.json to chunks.json")
                
                return True
                
            except Exception as e:
                print(f"Verification failed: {e}")
                return False
    
    return False


if __name__ == "__main__":
    chunks_path = Path(INDEX_PATH) / "chunks.json"
    validate_and_fix(chunks_path)
