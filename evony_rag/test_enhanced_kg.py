#!/usr/bin/env python3
"""
Test Enhanced Knowledge Graph Performance
==========================================
Tests relationship queries, entity lookups, and graph traversals.
"""
import json
import time
import sys
sys.path.insert(0, ".")

from collections import Counter

# Load KG
print("=" * 60)
print("ENHANCED KNOWLEDGE GRAPH TEST")
print("=" * 60)

kg_file = r"G:\evony_rag_index\knowledge_graph.json"
print(f"\nLoading KG from {kg_file}...")

with open(kg_file, "r", encoding="utf-8") as f:
    kg_data = json.load(f)

entities = {e["id"]: e for e in kg_data["entities"]}
relationships = kg_data["relationships"]

print(f"Entities: {len(entities):,}")
print(f"Relationships: {len(relationships):,}")

# Build indexes for fast lookup
print("\nBuilding indexes...")
by_name = {}
by_type = {}
outgoing = {}
incoming = {}

for eid, e in entities.items():
    name = e["name"].lower()
    if name not in by_name:
        by_name[name] = []
    by_name[name].append(eid)
    
    etype = e["entity_type"]
    if etype not in by_type:
        by_type[etype] = []
    by_type[etype].append(eid)

for r in relationships:
    src = r["source_id"]
    tgt = r["target_id"]
    if src not in outgoing:
        outgoing[src] = []
    outgoing[src].append(r)
    if tgt not in incoming:
        incoming[tgt] = []
    incoming[tgt].append(r)

print(f"Name index: {len(by_name):,} unique names")
print(f"Type index: {len(by_type):,} types")

# Test 1: Entity Type Distribution
print("\n" + "=" * 60)
print("TEST 1: Entity Type Distribution")
print("=" * 60)
type_counts = Counter(e["entity_type"] for e in entities.values())
for etype, count in type_counts.most_common(10):
    print(f"  {etype:<20} {count:>8,}")

# Test 2: Relationship Type Distribution
print("\n" + "=" * 60)
print("TEST 2: Relationship Type Distribution")
print("=" * 60)
rel_counts = Counter(r["relation_type"] for r in relationships)
for rtype, count in rel_counts.most_common(10):
    print(f"  {rtype:<20} {count:>8,}")

# Test 3: Entity Lookups
print("\n" + "=" * 60)
print("TEST 3: Entity Lookups")
print("=" * 60)

test_names = ["TroopCommand", "server", "port", "key", "token", "password", "config", "attack"]
for name in test_names:
    start = time.time()
    matches = by_name.get(name.lower(), [])
    elapsed = (time.time() - start) * 1000
    if matches:
        sample = entities[matches[0]]
        print(f"  '{name}': {len(matches)} matches ({elapsed:.2f}ms) - type: {sample['entity_type']}")
    else:
        # Try partial match
        partial = [n for n in by_name.keys() if name.lower() in n][:3]
        print(f"  '{name}': 0 exact, {len(partial)} partial - {partial}")

# Test 4: Relationship Queries
print("\n" + "=" * 60)
print("TEST 4: Relationship Queries")
print("=" * 60)

def find_related(entity_name, rel_type=None, direction="out"):
    """Find entities related to the given entity."""
    matches = by_name.get(entity_name.lower(), [])
    if not matches:
        return []
    
    results = []
    for eid in matches[:5]:  # Limit to first 5 matches
        rels = outgoing.get(eid, []) if direction == "out" else incoming.get(eid, [])
        for r in rels:
            if rel_type is None or r["relation_type"] == rel_type:
                target_id = r["target_id"] if direction == "out" else r["source_id"]
                target = entities.get(target_id)
                if target:
                    results.append((target["name"], r["relation_type"], r.get("confidence", 1.0)))
    return results[:10]

# Test relationship queries
queries = [
    ("server", None, "out"),
    ("port", "configures", "in"),
    ("command", "calls", "out"),
    ("token", "references", "out"),
    ("attack", None, "out"),
]

for name, rel_type, direction in queries:
    start = time.time()
    results = find_related(name, rel_type, direction)
    elapsed = (time.time() - start) * 1000
    rel_desc = rel_type or "any"
    print(f"\n  '{name}' --[{rel_desc}]--> ({elapsed:.2f}ms)")
    for target, rtype, conf in results[:5]:
        print(f"    -> {target} ({rtype}, conf={conf:.2f})")

# Test 5: Graph Traversal (2-hop)
print("\n" + "=" * 60)
print("TEST 5: Graph Traversal (2-hop)")
print("=" * 60)

def traverse_2hop(start_name):
    """Find entities 2 hops away."""
    hop1_ids = set()
    hop2_entities = []
    
    start_ids = by_name.get(start_name.lower(), [])[:3]
    
    # Hop 1
    for sid in start_ids:
        for r in outgoing.get(sid, [])[:10]:
            hop1_ids.add(r["target_id"])
    
    # Hop 2
    for h1id in list(hop1_ids)[:20]:
        for r in outgoing.get(h1id, [])[:5]:
            target = entities.get(r["target_id"])
            if target and target["id"] not in start_ids:
                hop2_entities.append(target["name"])
    
    return list(set(hop2_entities))[:10]

traverse_tests = ["server", "command", "config"]
for name in traverse_tests:
    start = time.time()
    hop2 = traverse_2hop(name)
    elapsed = (time.time() - start) * 1000
    print(f"\n  '{name}' (2-hop, {elapsed:.2f}ms): {len(hop2)} entities")
    if hop2:
        print(f"    Sample: {hop2[:5]}")

# Test 6: High-Connectivity Entities
print("\n" + "=" * 60)
print("TEST 6: Most Connected Entities (Hub Detection)")
print("=" * 60)

connectivity = {}
for eid in entities:
    out_count = len(outgoing.get(eid, []))
    in_count = len(incoming.get(eid, []))
    connectivity[eid] = out_count + in_count

top_connected = sorted(connectivity.items(), key=lambda x: -x[1])[:10]
print("\nTop 10 most connected entities:")
for eid, count in top_connected:
    e = entities[eid]
    print(f"  {e['name']:<30} ({e['entity_type']:<15}) connections: {count}")

# Test 7: Query Performance Summary
print("\n" + "=" * 60)
print("TEST 7: Performance Summary")
print("=" * 60)

# Benchmark 100 random lookups
import random
sample_names = random.sample(list(by_name.keys()), min(100, len(by_name)))

start = time.time()
for name in sample_names:
    _ = by_name.get(name, [])
lookup_time = (time.time() - start) * 1000

start = time.time()
for name in sample_names[:50]:
    ids = by_name.get(name, [])
    for eid in ids[:2]:
        _ = outgoing.get(eid, [])
rel_time = (time.time() - start) * 1000

print(f"  100 entity lookups: {lookup_time:.2f}ms ({lookup_time/100:.3f}ms/lookup)")
print(f"  50 relationship queries: {rel_time:.2f}ms ({rel_time/50:.3f}ms/query)")

print("\n" + "=" * 60)
print("ENHANCED KG TEST COMPLETE")
print("=" * 60)
