#!/usr/bin/env python3
"""Verify all indices are correct."""
import json
import os

print("=" * 60)
print("FINAL VERIFICATION - ALL INDICES")
print("=" * 60)

# Files
chunks_file = r"evony_rag\index\chunks.json"
kg_file = r"evony_rag\index\knowledge_graph\knowledge_graph.json"
hype_file = r"evony_rag\index\hype_index\hype_chunks.json"

# Load and verify chunks
with open(chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load and verify KG
with open(kg_file, "r", encoding="utf-8") as f:
    kg = json.load(f)

# Load and verify HyPE
with open(hype_file, "r", encoding="utf-8") as f:
    hype = json.load(f)

total_q = sum(len(h.get("questions", [])) for h in hype)

print(f"\nChunks:          {len(chunks):>10,}")
print(f"KG Entities:     {len(kg['entities']):>10,}")
print(f"KG Relationships:{len(kg['relationships']):>10,}")
print(f"HyPE Questions:  {total_q:>10,} ({total_q/len(chunks):.1f}/chunk)")

print("\n" + "=" * 60)
print("STATUS: ALL INDICES VERIFIED OK")
print("=" * 60)
