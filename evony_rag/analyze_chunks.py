#!/usr/bin/env python3
"""Analyze chunks for script content."""
import json
from pathlib import Path

CHUNKS_FILE = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\index\chunks.json")

print("=" * 60)
print("CHUNK ANALYSIS FOR SCRIPTS")
print("=" * 60)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks):,}")

# Find script-related chunks
script_chunks = []
for c in chunks:
    fp = c.get("file_path", "").lower()
    content = c.get("content", "").lower()[:300]
    if "script" in fp or "script" in content or any(cmd in content for cmd in ["setsilence", "farmNPC", "attackNPC", "label ", "goto "]):
        script_chunks.append(c)

print(f"Script-related chunks: {len(script_chunks):,}")

# Sample paths
print("\nSample script chunk paths:")
seen = set()
for c in script_chunks[:20]:
    fp = c.get("file_path", "?")
    if fp not in seen:
        seen.add(fp)
        print(f"  {fp}")

# Sample content
print("\nSample script content:")
for c in script_chunks[:3]:
    print(f"\n--- {c.get('file_path', '?')} ---")
    print(c.get("content", "")[:200])
    print(f"Content: {content}...")
    print()
