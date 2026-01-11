#!/usr/bin/env python3
"""Test Knowledge Graph save/load."""
import json
import sys
import os
sys.path.insert(0, ".")
from evony_rag.knowledge_graph import EntityExtractor, KnowledgeGraph
from pathlib import Path

# Create a test graph
kg = KnowledgeGraph()

# Load chunks and extract entities
with open("evony_rag/index/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Processing {len(chunks)} chunks...")
extractor = EntityExtractor()

for i, chunk in enumerate(chunks):
    entities, rels = extractor.extract_from_chunk(chunk)
    for e in entities:
        kg.add_entity(e)
    for r in rels:
        kg.add_relationship(r)
    if i % 5000 == 0:
        print(f"  {i}: {len(kg.entities):,} entities")

print(f"\nTotal entities in graph: {len(kg.entities):,}")
print(f"Total relationships: {len(kg.relationships):,}")

# Save directly using absolute path
print("\nSaving...")
abs_path = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\index\knowledge_graph")
output_file = abs_path / "knowledge_graph.json"

# Build data manually
data = {
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

# Write with explicit flush and close
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f)
    f.flush()
    os.fsync(f.fileno())

# Verify file
file_size = os.path.getsize(output_file)
print(f"File size: {file_size:,} bytes")

# Reload and verify
with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded entities: {len(data['entities']):,}")
print(f"Loaded relationships: {len(data['relationships']):,}")
