#!/usr/bin/env python3
"""Analyze expected vs actual index values."""
import json
import re
from collections import Counter

# Load chunks
with open(r"evony_rag\index\chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("=" * 60)
print("EXPECTED vs ACTUAL INDEX ANALYSIS")
print("=" * 60)

# 1. Chunks analysis
print(f"\n[CHUNKS]")
print(f"  Total chunks: {len(chunks):,}")
categories = Counter(c.get("category", "unknown") for c in chunks)
for cat, count in categories.most_common():
    print(f"    {cat}: {count:,}")

# 2. Calculate expected entities per category
print(f"\n[EXPECTED ENTITIES BY CATEGORY]")

total_expected = 0
for cat, count in categories.most_common():
    cat_chunks = [c for c in chunks if c.get("category") == cat]
    
    entity_count = 0
    sample_size = min(100, len(cat_chunks))
    for chunk in cat_chunks[:sample_size]:
        content = chunk.get("content", "")
        # Count potential entities
        camel = len(re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", content))
        constants = len(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", content))
        json_keys = len(re.findall(r'"(\w{3,})":', content))
        entity_count += camel + constants + json_keys
    
    avg_per_chunk = entity_count / sample_size if sample_size > 0 else 0
    expected = int(avg_per_chunk * count)
    total_expected += expected
    print(f"  {cat}: ~{expected:,} entities (avg {avg_per_chunk:.1f}/chunk)")

print(f"\n  TOTAL EXPECTED: ~{total_expected:,} entities")

# 3. Load current KG stats (correct file path)
entities = []
relationships = []
try:
    with open(r"evony_rag\index\knowledge_graph\knowledge_graph.json", "r", encoding="utf-8") as f:
        kg = json.load(f)
    
    entities = kg.get("entities", [])
    relationships = kg.get("relationships", [])
    
    print(f"\n[CURRENT KNOWLEDGE GRAPH]")
    print(f"  Entities: {len(entities):,}")
    print(f"  Relationships: {len(relationships):,}")
    
    # Entity types breakdown
    type_counts = Counter(e.get("entity_type", "unknown") for e in entities)
    print(f"\n  Entity types:")
    for t, c in type_counts.most_common(15):
        print(f"    {t}: {c:,}")
        
except Exception as e:
    print(f"Error loading KG: {e}")

# 4. HyPE analysis (correct field name)
total_questions = 0
chunks_with_questions = 0
try:
    with open(r"evony_rag\index\hype_index\hype_chunks.json", "r", encoding="utf-8") as f:
        hype = json.load(f)
    
    # Check actual field names
    if hype:
        sample = hype[0]
        q_field = "hypothetical_questions" if "hypothetical_questions" in sample else "questions"
        total_questions = sum(len(h.get(q_field, [])) for h in hype)
        chunks_with_questions = sum(1 for h in hype if h.get(q_field))
    
    print(f"\n[HYPE INDEX]")
    print(f"  Chunks indexed: {len(hype):,}")
    print(f"  Chunks with questions: {chunks_with_questions:,}")
    print(f"  Total questions: {total_questions:,}")
    if len(hype) > 0:
        print(f"  Avg questions/chunk: {total_questions/len(hype):.1f}")
    print(f"  Target (10/chunk): {len(chunks) * 10:,}")
except Exception as e:
    print(f"Error loading HyPE: {e}")

# 5. Summary comparison table
print("\n" + "=" * 60)
print("SUMMARY: EXPECTED vs CURRENT")
print("=" * 60)
print(f"\n{'Metric':<30} {'Expected':<15} {'Current':<15} {'%':<10}")
print("-" * 70)
print(f"{'Chunks':<30} {len(chunks):>12,} {len(chunks):>12,} {'100%':>8}")
print(f"{'HyPE Questions':<30} {len(chunks)*10:>12,} {total_questions:>12,} {total_questions*100//(len(chunks)*10):>7}%")
print(f"{'HyPE Coverage':<30} {len(chunks):>12,} {chunks_with_questions:>12,} {chunks_with_questions*100//len(chunks):>7}%")
print(f"{'KG Entities':<30} {total_expected:>12,} {len(entities):>12,} {len(entities)*100//max(1,total_expected):>7}%")
print(f"{'KG Relationships':<30} {'~50,000':>12} {len(relationships):>12,} {'TBD':>8}")
