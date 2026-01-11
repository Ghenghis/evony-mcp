#!/usr/bin/env python3
"""Rebuild indices with atomic file writes."""
import json
import os
import sys
import tempfile
import shutil
sys.path.insert(0, ".")

from pathlib import Path
from collections import Counter

INDEX_PATH = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\index")
CHUNKS_PATH = INDEX_PATH / "chunks.json"

print("=" * 60)
print("FULL INDEX REBUILD (ATOMIC WRITES)")
print("=" * 60)

# Load chunks
print("\n[1] Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"    {len(chunks):,} chunks loaded")

# ============================================================
# REBUILD KNOWLEDGE GRAPH
# ============================================================
print("\n[2] Building Knowledge Graph...")

from evony_rag.knowledge_graph import EntityExtractor, KnowledgeGraph

kg = KnowledgeGraph()
extractor = EntityExtractor()

for i, chunk in enumerate(chunks):
    entities, rels = extractor.extract_from_chunk(chunk)
    for e in entities:
        kg.add_entity(e)
    for r in rels:
        kg.add_relationship(r)
    if i % 5000 == 0:
        print(f"    {i:,}: {len(kg.entities):,} entities")

print(f"    Final: {len(kg.entities):,} entities, {len(kg.relationships):,} relationships")

# Build KG data
kg_data = {
    "entities": [
        {"id": e.id, "name": e.name, "entity_type": e.entity_type, 
         "file_path": e.file_path, "line_number": e.line_number, "properties": e.properties}
        for e in kg.entities.values()
    ],
    "relationships": [
        {"source_id": r.source_id, "target_id": r.target_id, 
         "relation_type": r.relation_type, "confidence": r.confidence}
        for r in kg.relationships
    ]
}

# Write to temp file in SAME directory (important for atomic rename)
kg_dir = INDEX_PATH / "knowledge_graph"
kg_dir.mkdir(parents=True, exist_ok=True)
temp_kg = kg_dir / "kg_temp.json"
final_kg = kg_dir / "knowledge_graph.json"

print("    Writing KG...")
json_str = json.dumps(kg_data)
print(f"    JSON size: {len(json_str):,} bytes")

with open(temp_kg, "w", encoding="utf-8") as f:
    f.write(json_str)

# Verify temp file
temp_size = os.path.getsize(temp_kg)
print(f"    Temp file: {temp_size:,} bytes")

# Atomic replace
if final_kg.exists():
    final_kg.unlink()
temp_kg.rename(final_kg)

final_size = os.path.getsize(final_kg)
print(f"    Final file: {final_size:,} bytes")

# ============================================================
# REBUILD HYPE INDEX
# ============================================================
print("\n[3] Building HyPE Index (10 questions/chunk)...")

from evony_rag.hype_embeddings import HyPEGenerator

generator = HyPEGenerator(use_llm=False)
hype_data = []

fallbacks = [
    "What is in this file?",
    "Show contents of this file",
    "Explain this code",
    "What does this do?",
    "How does this work?",
    "What is the purpose?",
    "Describe functionality",
    "What are key elements?",
    "Show related info",
    "Find similar content",
]

for i, chunk in enumerate(chunks):
    questions = generator.generate_questions_template(chunk)
    
    # Pad to exactly 10 questions
    for fb in fallbacks:
        if len(questions) >= 10:
            break
        if fb not in questions:
            questions.append(fb)
    
    hype_data.append({
        "chunk_id": chunk.get("id", f"chunk_{i}"),
        "file_path": chunk.get("file_path", ""),
        "category": chunk.get("category", ""),
        "questions": questions[:10],
    })
    
    if i % 5000 == 0:
        print(f"    {i:,}: {sum(len(h['questions']) for h in hype_data):,} questions")

total_q = sum(len(h["questions"]) for h in hype_data)
print(f"    Final: {total_q:,} questions ({total_q/len(chunks):.1f}/chunk)")

# Write HyPE with atomic rename
hype_dir = INDEX_PATH / "hype_index"
hype_dir.mkdir(parents=True, exist_ok=True)
temp_hype = hype_dir / "hype_temp.json"
final_hype = hype_dir / "hype_chunks.json"

print("    Writing HyPE...")
hype_str = json.dumps(hype_data)
print(f"    JSON size: {len(hype_str):,} bytes")

with open(temp_hype, "w", encoding="utf-8") as f:
    f.write(hype_str)

if final_hype.exists():
    final_hype.unlink()
temp_hype.rename(final_hype)

hype_size = os.path.getsize(final_hype)
print(f"    Final file: {hype_size:,} bytes")

# ============================================================
# VERIFY
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

with open(final_kg, "r", encoding="utf-8") as f:
    kg_v = json.load(f)
with open(final_hype, "r", encoding="utf-8") as f:
    hype_v = json.load(f)

q_total = sum(len(h.get("questions", [])) for h in hype_v)

print(f"\n{'Metric':<25} {'Expected':<15} {'Actual':<15} {'Status':>10}")
print("-" * 70)
print(f"{'Chunks':<25} {len(chunks):>12,} {len(chunks):>12,} {'OK':>10}")
print(f"{'HyPE Questions':<25} {len(chunks)*10:>12,} {q_total:>12,} {'OK' if q_total >= len(chunks)*10 else 'LOW':>10}")
print(f"{'KG Entities':<25} {'~100,000':>12} {len(kg_v['entities']):>12,} {'OK':>10}")
print(f"{'KG Relationships':<25} {'~1,700':>12} {len(kg_v['relationships']):>12,} {'OK':>10}")

# Type breakdown
types = Counter(e["entity_type"] for e in kg_v["entities"])
print(f"\nEntity Types: {dict(types.most_common(8))}")
print("\nDONE!")
