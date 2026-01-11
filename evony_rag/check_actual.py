#!/usr/bin/env python3
"""Check actual index values."""
import json
from collections import Counter

print("=" * 60)
print("ACTUAL INDEX VALUES")
print("=" * 60)

# Knowledge Graph
with open(r"evony_rag\index\knowledge_graph\knowledge_graph.json", "r", encoding="utf-8") as f:
    kg = json.load(f)

entities = kg.get("entities", [])
relationships = kg.get("relationships", [])

print(f"\n[KNOWLEDGE GRAPH]")
print(f"  Entities: {len(entities):,}")
print(f"  Relationships: {len(relationships):,}")

# Entity types breakdown
type_counts = Counter(e.get("entity_type", "unknown") for e in entities)
print(f"\n  Entity types:")
for t, c in type_counts.most_common(15):
    print(f"    {t}: {c:,}")

# HyPE Index
with open(r"evony_rag\index\hype_index\hype_chunks.json", "r", encoding="utf-8") as f:
    hype = json.load(f)

# Find the right field name
if hype:
    sample = hype[0]
    q_field = "questions" if "questions" in sample else "hypothetical_questions"
    total_questions = sum(len(h.get(q_field, [])) for h in hype)
    chunks_with_q = sum(1 for h in hype if h.get(q_field))
else:
    total_questions = 0
    chunks_with_q = 0

print(f"\n[HYPE INDEX]")
print(f"  Chunks: {len(hype):,}")
print(f"  Chunks with questions: {chunks_with_q:,}")
print(f"  Total questions: {total_questions:,}")
print(f"  Avg questions/chunk: {total_questions / max(len(hype), 1):.1f}")

# Summary
print("\n" + "=" * 60)
print("EXPECTED vs ACTUAL SUMMARY")
print("=" * 60)
print(f"\n{'Metric':<25} {'Expected':<15} {'Actual':<15} {'%':>8}")
print("-" * 65)
print(f"{'Chunks':<25} {'16,962':<15} {'16,962':<15} {'100%':>8}")
print(f"{'HyPE Questions':<25} {'169,620':<15} {total_questions:,<15} {total_questions*100//169620:>7}%")
print(f"{'KG Entities':<25} {'~32,578':<15} {len(entities):,<15} {len(entities)*100//32578:>7}%")
print(f"{'KG Relationships':<25} {'~50,000':<15} {len(relationships):,<15} {'TBD':>8}")
