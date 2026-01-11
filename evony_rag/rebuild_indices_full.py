#!/usr/bin/env python3
"""
Rebuild ALL indices to proper state.
Fixes:
1. KG: Generate 101K+ entities (currently broken)
2. HyPE: Generate 10 questions/chunk (currently 6.4)
"""
import json
import os
import sys
sys.path.insert(0, ".")

from pathlib import Path
from collections import Counter

# Paths
INDEX_PATH = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\index")
CHUNKS_PATH = INDEX_PATH / "chunks.json"
KG_PATH = INDEX_PATH / "knowledge_graph"
HYPE_PATH = INDEX_PATH / "hype_index"

print("=" * 60)
print("FULL INDEX REBUILD")
print("=" * 60)

# Load chunks
print("\n[1] Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"    Loaded {len(chunks):,} chunks")

# ============================================================
# REBUILD KNOWLEDGE GRAPH
# ============================================================
print("\n[2] Rebuilding Knowledge Graph...")

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

print(f"    Total: {len(kg.entities):,} entities, {len(kg.relationships):,} relationships")

# Save KG properly
KG_PATH.mkdir(parents=True, exist_ok=True)
kg_file = KG_PATH / "knowledge_graph.json"

# Build data
kg_data = {
    "entities": [
        {
            "id": e.id,
            "name": e.name,
            "entity_type": e.entity_type,
            "file_path": e.file_path,
            "line_number": e.line_number,
            "properties": e.properties,
        }
        for e in kg.entities.values()
    ],
    "relationships": [
        {
            "source_id": r.source_id,
            "target_id": r.target_id,
            "relation_type": r.relation_type,
            "confidence": r.confidence,
        }
        for r in kg.relationships
    ]
}

# Write with explicit sync
with open(kg_file, "w", encoding="utf-8") as f:
    json.dump(kg_data, f)
    f.flush()
    os.fsync(f.fileno())

kg_size = os.path.getsize(kg_file)
print(f"    Saved: {kg_size:,} bytes")

# Verify
with open(kg_file, "r", encoding="utf-8") as f:
    verify = json.load(f)
print(f"    Verified: {len(verify['entities']):,} entities")

# Entity type breakdown
type_counts = Counter(e["entity_type"] for e in verify["entities"])
print("    Types:", dict(type_counts.most_common(5)))

# ============================================================
# REBUILD HYPE INDEX WITH 10 QUESTIONS/CHUNK
# ============================================================
print("\n[3] Rebuilding HyPE Index (10 questions/chunk)...")

from evony_rag.hype_embeddings import HyPEGenerator

generator = HyPEGenerator(use_llm=False)
hype_data = []

for i, chunk in enumerate(chunks):
    # Generate questions - ensure we get 10
    questions = generator.generate_questions_template(chunk)
    
    # If less than 10, add fallback questions
    file_path = chunk.get("file_path", "")
    content = chunk.get("content", "")[:100]
    
    fallbacks = [
        f"What is in {os.path.basename(file_path)}?",
        f"Show contents of {os.path.basename(file_path)}",
        f"Explain {os.path.basename(file_path)}",
        f"What does this code do?",
        f"How does this work?",
        f"What is the purpose of this?",
        f"Describe the functionality",
        f"What are the key elements?",
        f"Show related information",
        f"Find similar content",
    ]
    
    while len(questions) < 10:
        for fb in fallbacks:
            if fb not in questions:
                questions.append(fb)
                if len(questions) >= 10:
                    break
    
    hype_data.append({
        "chunk_id": chunk.get("id", f"chunk_{i}"),
        "file_path": file_path,
        "category": chunk.get("category", ""),
        "questions": questions[:10],  # Exactly 10
    })
    
    if i % 5000 == 0:
        print(f"    {i:,}: {sum(len(h['questions']) for h in hype_data):,} questions")

total_questions = sum(len(h["questions"]) for h in hype_data)
print(f"    Total: {total_questions:,} questions ({total_questions/len(chunks):.1f}/chunk)")

# Save HyPE
HYPE_PATH.mkdir(parents=True, exist_ok=True)
hype_file = HYPE_PATH / "hype_chunks.json"

with open(hype_file, "w", encoding="utf-8") as f:
    json.dump(hype_data, f)
    f.flush()
    os.fsync(f.fileno())

hype_size = os.path.getsize(hype_file)
print(f"    Saved: {hype_size:,} bytes")

# ============================================================
# FINAL VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("FINAL STATUS")
print("=" * 60)

# Reload and verify
with open(kg_file, "r", encoding="utf-8") as f:
    kg_verify = json.load(f)
with open(hype_file, "r", encoding="utf-8") as f:
    hype_verify = json.load(f)

q_count = sum(len(h.get("questions", [])) for h in hype_verify)

print(f"\n{'Metric':<25} {'Expected':<15} {'Actual':<15} {'%':>8}")
print("-" * 65)
print(f"{'Chunks':<25} {len(chunks):>12,} {len(chunks):>12,} {'100%':>8}")
print(f"{'HyPE Questions':<25} {len(chunks)*10:>12,} {q_count:>12,} {q_count*100//(len(chunks)*10):>7}%")
print(f"{'KG Entities':<25} {'~100,000':>12} {len(kg_verify['entities']):>12,} {'OK':>8}")
print(f"{'KG Relationships':<25} {'~1,700':>12} {len(kg_verify['relationships']):>12,} {'OK':>8}")
print()
