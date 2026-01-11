#!/usr/bin/env python3
"""
Repair corrupted chunks.json - Version 2
=========================================
Uses regex-based recovery to extract complete JSON objects.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from evony_rag.config import INDEX_PATH


def repair_with_regex(chunks_path: Path):
    """
    Repair using regex to find complete chunk objects.
    """
    print(f"Reading {chunks_path}...")
    
    with open(chunks_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    print(f"File size: {len(content):,} bytes")
    
    # Pattern to match complete chunk objects
    # Chunks have: id, file_path, content, start_line, end_line, category
    chunk_pattern = re.compile(
        r'\{\s*"id"\s*:\s*"[^"]*"\s*,\s*'
        r'"file_path"\s*:\s*"[^"]*"\s*,\s*'
        r'"content"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*'
        r'"start_line"\s*:\s*\d+\s*,\s*'
        r'"end_line"\s*:\s*\d+\s*,\s*'
        r'"category"\s*:\s*"[^"]*"\s*\}',
        re.DOTALL
    )
    
    print("Searching for complete chunks...")
    matches = chunk_pattern.findall(content)
    print(f"Found {len(matches)} potential chunks")
    
    if not matches:
        # Try simpler pattern
        print("Trying simpler pattern...")
        chunk_pattern = re.compile(r'\{[^{}]*"id"[^{}]*"content"[^{}]*\}', re.DOTALL)
        matches = chunk_pattern.findall(content)
        print(f"Found {len(matches)} with simpler pattern")
    
    # Parse each match
    chunks = []
    errors = 0
    
    for i, match in enumerate(matches):
        try:
            chunk = json.loads(match)
            # Validate it has required fields
            if 'content' in chunk and ('id' in chunk or 'file_path' in chunk):
                chunks.append(chunk)
        except json.JSONDecodeError:
            errors += 1
        
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,}, valid: {len(chunks):,}, errors: {errors}")
    
    print(f"\nRecovered {len(chunks):,} valid chunks ({errors} errors)")
    
    if chunks:
        output = chunks_path.parent / "chunks_recovered.json"
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)
        print(f"Saved to {output}")
        return True
    
    return False


def repair_by_truncation(chunks_path: Path):
    """
    Simply truncate at the last valid point.
    """
    print(f"\nTrying truncation repair...")
    
    with open(chunks_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find the position of the corruption
    # Error was at line 135700 - the file ends abruptly
    # Look for the last complete chunk ending: }
    
    # Find all positions of },{ which indicate chunk boundaries
    positions = []
    for match in re.finditer(r'\}\s*,\s*\{', content):
        positions.append(match.start())
    
    print(f"Found {len(positions)} chunk boundaries")
    
    if positions:
        # Use the last safe position
        last_safe = positions[-1]
        print(f"Last safe boundary at position {last_safe:,}")
        
        # Truncate and close the array
        truncated = content[:last_safe + 1] + "\n]"
        
        try:
            chunks = json.loads(truncated)
            print(f"SUCCESS! Recovered {len(chunks):,} chunks")
            
            output = chunks_path.parent / "chunks_recovered.json"
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(chunks, f)
            print(f"Saved to {output}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Still invalid: {e}")
            
            # Try earlier positions
            for i in range(len(positions) - 1, max(0, len(positions) - 100), -1):
                pos = positions[i]
                truncated = content[:pos + 1] + "\n]"
                try:
                    chunks = json.loads(truncated)
                    print(f"SUCCESS at position {i}! Recovered {len(chunks):,} chunks")
                    
                    output = chunks_path.parent / "chunks_recovered.json"
                    with open(output, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f)
                    print(f"Saved to {output}")
                    return True
                except:
                    continue
    
    return False


def repair_streaming(chunks_path: Path):
    """
    Stream-based repair - process file character by character.
    """
    print(f"\nTrying streaming repair...")
    
    with open(chunks_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    chunks = []
    current = ""
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(content):
        if i % 1000000 == 0 and i > 0:
            print(f"  Position {i:,}, found {len(chunks)} chunks...")
        
        if escape_next:
            current += char
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            current += char
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
        
        if not in_string:
            if char == '{':
                if brace_count == 0:
                    current = ""
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    current += char
                    # Try to parse this chunk
                    try:
                        chunk = json.loads(current)
                        if isinstance(chunk, dict) and 'content' in chunk:
                            chunks.append(chunk)
                    except:
                        pass
                    current = ""
                    continue
        
        if brace_count > 0:
            current += char
    
    print(f"\nRecovered {len(chunks):,} chunks via streaming")
    
    if chunks:
        output = chunks_path.parent / "chunks_recovered.json"
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(chunks, f)
        print(f"Saved to {output}")
        return True
    
    return False


def main():
    chunks_path = Path(INDEX_PATH) / "chunks.json"
    
    print("=" * 60)
    print("CHUNKS.JSON REPAIR UTILITY v2")
    print("=" * 60)
    
    if not chunks_path.exists():
        print(f"ERROR: File not found: {chunks_path}")
        return
    
    # Try truncation first (fastest)
    if repair_by_truncation(chunks_path):
        return
    
    # Try streaming (most reliable)
    if repair_streaming(chunks_path):
        return
    
    # Try regex (fallback)
    if repair_with_regex(chunks_path):
        return
    
    print("\nAll repair methods failed!")


if __name__ == "__main__":
    main()
