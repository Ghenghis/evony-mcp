#!/usr/bin/env python3
"""Rebuild KG directly in project folder."""
import json
import os
import sys
sys.path.insert(0, ".")

kg_file = r"evony_rag\index\knowledge_graph\kg_fresh.json"

# Delete existing file completely
if os.path.exists(kg_file):
    os.remove(kg_file)
    print(f"Deleted old file")

print("Loading chunks...")
with open(r"evony_rag\index\chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"{len(chunks):,} chunks")

print("Building KG...")
from evony_rag.knowledge_graph import EntityExtractor, KnowledgeGraph

kg = KnowledgeGraph()
extractor = EntityExtractor()

for i, chunk in enumerate(chunks):
    entities, rels = extractor.extract_from_chunk(chunk)
    for e in entities:
        kg.add_entity(e)
    for r in rels:
        kg.add_relationship(r)

print(f"Entities: {len(kg.entities):,}")

# Build data
data = {
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

# Write directly
print("Writing...")
with open(kg_file, "w", encoding="utf-8") as f:
    json.dump(data, f)
    f.flush()
    os.fsync(f.fileno())

print(f"File size: {os.path.getsize(kg_file):,} bytes")

# Verify
print("Verifying...")
with open(kg_file, "r", encoding="utf-8") as f:
    verify = json.load(f)
print(f"Verified: {len(verify['entities']):,} entities")
print("DONE!")
