#!/usr/bin/env python3
"""
Export Knowledge Graph to Neo4j
================================
Creates Cypher import scripts for Neo4j visualization.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, ".")

print("=" * 60)
print("EXPORT KNOWLEDGE GRAPH TO NEO4J")
print("=" * 60)

# Load KG
KG_FILE = Path(r"G:\evony_rag_index\knowledge_graph.json")
OUTPUT_DIR = Path(r"G:\evony_rag_index\neo4j_export")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\nLoading KG from {KG_FILE}...")
with open(KG_FILE, "r", encoding="utf-8") as f:
    kg = json.load(f)

entities = kg["entities"]
relationships = kg["relationships"]

print(f"Entities: {len(entities):,}")
print(f"Relationships: {len(relationships):,}")

# Escape for Cypher
def escape_cypher(s):
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

# Export entities
print("\nExporting entities...")
entity_cypher = OUTPUT_DIR / "01_create_entities.cypher"

with open(entity_cypher, "w", encoding="utf-8") as f:
    f.write("// Create Entity nodes\n")
    f.write("// Run with: cypher-shell < 01_create_entities.cypher\n\n")
    
    # Create constraints
    f.write("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;\n\n")
    
    # Batch entities
    batch_size = 1000
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        f.write(f"// Batch {i // batch_size + 1}\n")
        
        for e in batch:
            name = escape_cypher(e.get("name", ""))
            etype = escape_cypher(e.get("entity_type", ""))
            file_path = escape_cypher(e.get("file_path", ""))
            line = e.get("line_number", 0)
            eid = escape_cypher(e.get("id", ""))
            
            f.write(f"CREATE (:Entity:_{etype} {{id: '{eid}', name: '{name}', type: '{etype}', file: '{file_path}', line: {line}}});\n")
        
        f.write("\n")

print(f"  Written: {entity_cypher}")

# Export relationships
print("Exporting relationships...")
rel_cypher = OUTPUT_DIR / "02_create_relationships.cypher"

with open(rel_cypher, "w", encoding="utf-8") as f:
    f.write("// Create Relationships\n")
    f.write("// Run with: cypher-shell < 02_create_relationships.cypher\n\n")
    
    # Sample relationships (Neo4j import can be slow)
    sample_size = min(50000, len(relationships))
    sample = relationships[:sample_size]
    
    f.write(f"// Exporting {sample_size:,} of {len(relationships):,} relationships\n\n")
    
    for i in range(0, len(sample), batch_size):
        batch = sample[i:i + batch_size]
        f.write(f"// Batch {i // batch_size + 1}\n")
        
        for r in batch:
            source = escape_cypher(r.get("source_id", ""))
            target = escape_cypher(r.get("target_id", ""))
            rel_type = escape_cypher(r.get("relation_type", "RELATED")).upper().replace(" ", "_")
            confidence = r.get("confidence", 1.0)
            
            f.write(f"MATCH (a:Entity {{id: '{source}'}}), (b:Entity {{id: '{target}'}}) CREATE (a)-[:{rel_type} {{confidence: {confidence}}}]->(b);\n")
        
        f.write("\n")

print(f"  Written: {rel_cypher}")

# Export as JSON for Neo4j import tool
print("Exporting JSON for neo4j-admin import...")

nodes_json = OUTPUT_DIR / "nodes.json"
rels_json = OUTPUT_DIR / "relationships.json"

# Nodes
with open(nodes_json, "w", encoding="utf-8") as f:
    for e in entities:
        node = {
            "id": e.get("id", ""),
            "name": e.get("name", ""),
            "type": e.get("entity_type", ""),
            "file": e.get("file_path", ""),
            "line": e.get("line_number", 0)
        }
        f.write(json.dumps(node) + "\n")

print(f"  Written: {nodes_json}")

# Relationships
with open(rels_json, "w", encoding="utf-8") as f:
    for r in relationships:
        rel = {
            "source": r.get("source_id", ""),
            "target": r.get("target_id", ""),
            "type": r.get("relation_type", "RELATED"),
            "confidence": r.get("confidence", 1.0)
        }
        f.write(json.dumps(rel) + "\n")

print(f"  Written: {rels_json}")

# Create visualization HTML using vis.js
print("\nCreating interactive visualization...")
vis_html = OUTPUT_DIR / "visualization.html"

# Sample for visualization (too many nodes crashes browser)
sample_entities = entities[:500]
sample_rels = [r for r in relationships[:2000] 
               if any(e["id"] == r["source_id"] for e in sample_entities) and 
                  any(e["id"] == r["target_id"] for e in sample_entities)][:500]

html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Evony Knowledge Graph</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial; }}
        #network {{ width: 100%; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }}
    </style>
</head>
<body>
    <div id="info">
        <h3>Evony Knowledge Graph</h3>
        <p>Nodes: {len(sample_entities)} (of {len(entities):,})</p>
        <p>Edges: {len(sample_rels)} (of {len(relationships):,})</p>
        <p>Click nodes to see details</p>
    </div>
    <div id="network"></div>
    <script>
        var nodes = new vis.DataSet([
'''

# Add nodes
for e in sample_entities:
    name = e.get("name", "").replace("'", "\\'")[:30]
    etype = e.get("entity_type", "")
    color = {
        "class": "#ff6b6b",
        "function": "#4ecdc4",
        "command": "#45b7d1",
        "config": "#96ceb4",
        "constant": "#ffeaa7",
    }.get(etype, "#dfe6e9")
    html_content += f"            {{id: '{e['id']}', label: '{name}', color: '{color}', title: '{etype}'}},\n"

html_content += '''        ]);
        var edges = new vis.DataSet([
'''

# Add edges
for r in sample_rels:
    rel_type = r.get("relation_type", "")[:20]
    html_content += f"            {{from: '{r['source_id']}', to: '{r['target_id']}', arrows: 'to', title: '{rel_type}'}},\n"

html_content += '''        ]);
        var container = document.getElementById('network');
        var data = { nodes: nodes, edges: edges };
        var options = {
            nodes: { shape: 'dot', size: 10 },
            edges: { smooth: { type: 'continuous' } },
            physics: { stabilization: { iterations: 100 } }
        };
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
'''

with open(vis_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"  Written: {vis_html}")

print("\n" + "=" * 60)
print("EXPORT COMPLETE")
print("=" * 60)
print(f"""
Files created in {OUTPUT_DIR}:
  - 01_create_entities.cypher   (for Neo4j cypher-shell)
  - 02_create_relationships.cypher
  - nodes.json                  (for neo4j-admin import)
  - relationships.json
  - visualization.html          (open in browser for preview)

To import into Neo4j:
  1. Start Neo4j Desktop
  2. Open a terminal to your database
  3. Run: cypher-shell < 01_create_entities.cypher
  4. Run: cypher-shell < 02_create_relationships.cypher

Or open visualization.html in a browser for instant preview!
""")
