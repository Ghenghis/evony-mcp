#!/usr/bin/env python3
"""Rebuild indices to C:\Temp to avoid OneDrive/sync issues."""
import json
import os
import sys
sys.path.insert(0, ".")

# Fresh directory outside Downloads (avoids OneDrive sync issues)
FRESH_DIR = r"C:\Temp\evony_rag_index"
os.makedirs(FRESH_DIR, exist_ok=True)

kg_file = os.path.join(FRESH_DIR, "knowledge_graph.json")
hype_file = os.path.join(FRESH_DIR, "hype_chunks.json")

print("Loading chunks...")
with open("evony_rag/index/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"{len(chunks):,} chunks loaded")

# Build KG
print("\nBuilding Knowledge Graph...")
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
        print(f"  {i:,}: {len(kg.entities):,} entities")

print(f"  Final: {len(kg.entities):,} entities, {len(kg.relationships):,} relationships")

kg_data = {
    "entities": [
        {"id": e.id, "name": e.name, "entity_type": e.entity_type,
         "file_path": e.file_path, "line_number": e.line_number,
         "properties": e.properties}
        for e in kg.entities.values()
    ],
    "relationships": [
        {"source_id": r.source_id, "target_id": r.target_id,
         "relation_type": r.relation_type, "confidence": r.confidence}
        for r in kg.relationships
    ]
}

with open(kg_file, "w", encoding="utf-8") as f:
    json.dump(kg_data, f)
print(f"  Saved: {os.path.getsize(kg_file):,} bytes")

# Build HyPE with 10 questions per chunk
print("\nBuilding HyPE Index (10 questions/chunk)...")
from evony_rag.hype_embeddings import HyPEGenerator

generator = HyPEGenerator(use_llm=False)
hype_data = []

fallbacks = [
    "What is this?", "Show contents", "Explain this code",
    "How does it work?", "What is the purpose?", "Describe it",
    "Key elements?", "Related info", "Similar content", "More details"
]

for i, chunk in enumerate(chunks):
    questions = generator.generate_questions_template(chunk)
    
    # Pad to 10 questions
    for fb in fallbacks:
        if len(questions) >= 10:
            break
        if fb not in questions:
            questions.append(fb)
    
    hype_data.append({
        "chunk_id": chunk.get("id", f"chunk_{i}"),
        "file_path": chunk.get("file_path", ""),
        "category": chunk.get("category", ""),
        "questions": questions[:10]
    })
    
    if i % 5000 == 0:
        print(f"  {i:,}: {sum(len(h['questions']) for h in hype_data):,} questions")

total_q = sum(len(h["questions"]) for h in hype_data)
print(f"  Final: {total_q:,} questions ({total_q/len(chunks):.1f}/chunk)")

with open(hype_file, "w", encoding="utf-8") as f:
    json.dump(hype_data, f)
print(f"  Saved: {os.path.getsize(hype_file):,} bytes")

# Verify
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

with open(kg_file, "r", encoding="utf-8") as f:
    kg_v = json.load(f)
with open(hype_file, "r", encoding="utf-8") as f:
    hype_v = json.load(f)

q_total = sum(len(h.get("questions", [])) for h in hype_v)

print(f"\nKG Entities: {len(kg_v['entities']):,}")
print(f"KG Relationships: {len(kg_v['relationships']):,}")
print(f"HyPE Questions: {q_total:,} ({q_total/len(chunks):.1f}/chunk)")
print(f"\nFiles saved to: {FRESH_DIR}")
print("\nTo use these indices, copy them to evony_rag/index/ or update config.py")
