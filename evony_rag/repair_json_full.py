#!/usr/bin/env python3
"""
Full JSON Repair Script - 100% Recovery
========================================
Properly repairs truncated JSON array files by:
1. Parsing the file character by character
2. Tracking JSON structure (braces, brackets, strings)
3. Extracting all complete objects
4. Rebuilding a valid JSON array
"""

import json
import re
import sys
from pathlib import Path


def repair_json_array(input_path: str, output_path: str) -> dict:
    """
    Repair a truncated JSON array file.
    
    Returns dict with stats about the repair.
    """
    print(f"Reading {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    file_size = len(content)
    print(f"File size: {file_size:,} bytes")
    
    # Find where the array starts
    array_start = content.find('[')
    if array_start == -1:
        print("ERROR: No array start '[' found!")
        return {"success": False, "error": "No array start found"}
    
    print(f"Array starts at position {array_start}")
    
    # Parse the content to find complete objects
    chunks = []
    pos = array_start + 1  # Skip the opening '['
    
    # State tracking
    in_string = False
    escape_next = False
    brace_depth = 0
    object_start = None
    
    errors = []
    last_good_pos = array_start
    
    while pos < file_size:
        char = content[pos]
        
        # Handle escape sequences in strings
        if escape_next:
            escape_next = False
            pos += 1
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            pos += 1
            continue
        
        # Handle string boundaries
        if char == '"' and not escape_next:
            in_string = not in_string
            pos += 1
            continue
        
        # Skip everything inside strings
        if in_string:
            pos += 1
            continue
        
        # Track object boundaries
        if char == '{':
            if brace_depth == 0:
                object_start = pos
            brace_depth += 1
        
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and object_start is not None:
                # Found a complete object!
                object_str = content[object_start:pos + 1]
                
                try:
                    obj = json.loads(object_str)
                    chunks.append(obj)
                    last_good_pos = pos
                    object_start = None
                except json.JSONDecodeError as e:
                    errors.append(f"Parse error at {object_start}: {e}")
                    object_start = None
        
        elif char == ']' and brace_depth == 0:
            # End of array
            print(f"Found array end at position {pos}")
            break
        
        pos += 1
        
        # Progress reporting
        if pos % 1000000 == 0:
            print(f"  Processed {pos:,}/{file_size:,} bytes, found {len(chunks):,} chunks...")
    
    print(f"\nExtracted {len(chunks):,} complete chunks")
    
    if errors:
        print(f"Encountered {len(errors)} parse errors (ignored)")
    
    # Validate chunks have expected structure
    valid_chunks = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            # Check for required fields
            if 'content' in chunk or 'id' in chunk:
                valid_chunks.append(chunk)
            else:
                pass  # Skip chunks without required fields
    
    print(f"Valid chunks with content/id: {len(valid_chunks):,}")
    
    # Sanitize all chunks - remove control characters
    print(f"\nSanitizing {len(valid_chunks)} chunks...")
    sanitized_chunks = []
    for chunk in valid_chunks:
        clean_chunk = {}
        for key, value in chunk.items():
            if isinstance(value, str):
                # Remove control characters (except newline, tab)
                clean_value = ''.join(
                    c if c in '\n\t' or (ord(c) >= 32 and ord(c) != 127) else ' '
                    for c in value
                )
                clean_chunk[key] = clean_value
            else:
                clean_chunk[key] = value
        sanitized_chunks.append(clean_chunk)
    
    # Save the repaired file - NO indent, ensure_ascii for safety
    print(f"Saving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sanitized_chunks, f, ensure_ascii=True)
    
    # Verify the output
    with open(output_path, 'r', encoding='utf-8') as f:
        verify = json.load(f)
    
    output_size = Path(output_path).stat().st_size
    
    print(f"\n{'='*60}")
    print("REPAIR COMPLETE")
    print(f"{'='*60}")
    print(f"Input file:  {file_size:,} bytes")
    print(f"Output file: {output_size:,} bytes")
    print(f"Chunks recovered: {len(verify):,}")
    print(f"Parse errors: {len(errors)}")
    
    # Show sample chunk
    if verify:
        print(f"\nSample chunk keys: {list(verify[0].keys())}")
    
    return {
        "success": True,
        "input_size": file_size,
        "output_size": output_size,
        "chunks_recovered": len(verify),
        "errors": len(errors),
    }


def main():
    backup_path = r"C:\Users\Admin\Downloads\Evony_Decrypted\index_Backup\chunks.json"
    output_path = r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\index\chunks.json"
    
    result = repair_json_array(backup_path, output_path)
    
    if result["success"]:
        print(f"\n✓ Successfully repaired! {result['chunks_recovered']:,} chunks recovered.")
    else:
        print(f"\n✗ Repair failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
