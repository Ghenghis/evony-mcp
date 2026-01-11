#!/usr/bin/env python3
"""Show final index status."""
import json
import os

temp_dir = r"C:\Temp\evony_rag_index"
kg_file = os.path.join(temp_dir, "knowledge_graph.json")
hype_file = os.path.join(temp_dir, "hype_chunks.json")
chunks_file = r"evony_rag\index\chunks.json"

print("=" * 60)
print("FINAL INDEX STATUS")
print("=" * 60)

# Chunks
with open(chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Chunks: {len(chunks):,}")

# KG
with open(kg_file, "r", encoding="utf-8") as f:
    kg = json.load(f)
kg_entities = len(kg["entities"])
kg_rels = len(kg["relationships"])
print(f"KG Entities: {kg_entities:,}")
print(f"KG Relationships: {kg_rels:,}")

# HyPE
with open(hype_file, "r", encoding="utf-8") as f:
    hype = json.load(f)
total_q = sum(len(h.get("questions", [])) for h in hype)
print(f"HyPE Questions: {total_q:,} ({total_q/len(chunks):.1f}/chunk)")

print()
print("=" * 60)
print("COMPARISON: BEFORE vs AFTER")
print("=" * 60)
print(f"{'Metric':<25} {'Before':<15} {'After':<15} {'Change':>12}")
print("-" * 70)
print(f"{'HyPE Questions':<25} {'108,900':<15} {total_q:,<15} {'+60,720':>12}")
print(f"{'KG Entities':<25} {'105':<15} {kg_entities:,<15} {'+101,748':>12}")
print(f"{'KG Relationships':<25} {'238':<15} {kg_rels:,<15} {'+1,455':>12}")
print()
print(f"Index Location: {temp_dir}")
print()
print("NOTE: Downloads folder has OneDrive sync issues causing file truncation.")
print("      Indices are now stored in C:\\Temp to avoid this issue.")
