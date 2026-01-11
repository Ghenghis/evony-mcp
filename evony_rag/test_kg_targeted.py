#!/usr/bin/env python3
"""Targeted KG query tests."""
import json

# Load KG
with open(r"G:\evony_rag_index\knowledge_graph.json") as f:
    kg = json.load(f)

entities = {e["id"]: e for e in kg["entities"]}
relationships = kg["relationships"]

# Build indexes
by_name = {}
outgoing = {}
for e in kg["entities"]:
    name = e["name"].lower()
    if name not in by_name:
        by_name[name] = []
    by_name[name].append(e)

for r in relationships:
    if r["source_id"] not in outgoing:
        outgoing[r["source_id"]] = []
    outgoing[r["source_id"]].append(r)

print("=" * 60)
print("TARGETED KG QUERIES")
print("=" * 60)

# Test specific entity queries
tests = [
    ("troopcommands", "Troop-related"),
    ("sendmsg", "Message sending"),
    ("command", "Commands"),
    ("server", "Server"),
    ("password", "Password"),
    ("attack", "Attack"),
]

for name, desc in tests:
    matches = by_name.get(name, [])
    print(f"\n{desc} ('{name}'): {len(matches)} entities")
    for e in matches[:3]:
        rels = outgoing.get(e["id"], [])
        print(f"  - {e['name']} ({e['entity_type']}) -> {len(rels)} rels")

print("\n" + "=" * 60)
print("TOP CONNECTED ENTITIES")
print("=" * 60)

# Find entities with most relationships
rel_counts = {}
for r in relationships:
    src = r["source_id"]
    rel_counts[src] = rel_counts.get(src, 0) + 1

top = sorted(rel_counts.items(), key=lambda x: -x[1])[:10]
for eid, count in top:
    e = entities.get(eid)
    if e:
        print(f"\n{e['name']} ({e['entity_type']}): {count} outgoing rels")
        # Show sample relationships
        sample_rels = outgoing.get(eid, [])[:3]
        for r in sample_rels:
            tgt = entities.get(r["target_id"])
            tgt_name = tgt["name"] if tgt else r["target_id"][:30]
            print(f"  --[{r['relation_type']}]--> {tgt_name}")

print("\n" + "=" * 60)
print("RELATIONSHIP TYPE EXAMPLES")
print("=" * 60)

# Show examples of each relationship type
rel_types = {}
for r in relationships:
    rt = r["relation_type"]
    if rt not in rel_types:
        rel_types[rt] = []
    if len(rel_types[rt]) < 2:
        rel_types[rt].append(r)

for rt, samples in sorted(rel_types.items()):
    print(f"\n{rt}:")
    for r in samples:
        src = entities.get(r["source_id"])
        tgt = entities.get(r["target_id"])
        src_name = src["name"] if src else "?"
        tgt_name = tgt["name"] if tgt else "?"
        print(f"  {src_name} -> {tgt_name}")

print("\n" + "=" * 60)
print("KG ENHANCEMENT SUMMARY")
print("=" * 60)
print(f"Total entities: {len(entities):,}")
print(f"Total relationships: {len(relationships):,}")
print(f"Unique names: {len(by_name):,}")
print(f"Entities with outgoing rels: {len(outgoing):,}")
print(f"Avg rels per connected entity: {len(relationships)/len(outgoing):.1f}")
