#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Relationship Extraction
================================================
Adds:
1. More relationship patterns (calls, references, contains, uses)
2. Co-occurrence relationships (entities in same chunk)
3. LLM-based semantic relationship extraction
"""
import re
import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
import hashlib

sys.path.insert(0, ".")

from evony_rag.knowledge_graph import Entity, Relationship, EntityExtractor, KnowledgeGraph


class EnhancedRelationshipExtractor:
    """
    Enhanced relationship extraction with multiple strategies.
    """
    
    # Additional relationship patterns
    RELATIONSHIP_PATTERNS = {
        # Function/method calls
        "calls": [
            r'(\w+)\s*\.\s*(\w+)\s*\(',  # obj.method()
            r'(\w+)\s*\(\s*(\w+)',  # func(arg)
            r'new\s+(\w+)',  # new ClassName
        ],
        
        # References/uses
        "references": [
            r'(\w+)\s*=\s*(\w+)',  # assignment
            r'return\s+(\w+)',  # return value
            r'(\w+)\s*:\s*(\w+)',  # type annotation
            r'\$\{(\w+)\}',  # template variable
            r'\$(\w+)',  # variable reference
        ],
        
        # Contains/parent-child
        "contains": [
            r'<(\w+)[^>]*>.*?</\1>',  # XML parent-child
            r'"(\w+)":\s*\{[^}]*"(\w+)"',  # nested JSON
            r'class\s+(\w+).*?function\s+(\w+)',  # class contains function
        ],
        
        # Uses/depends on
        "uses": [
            r'import\s+(\w+)',  # import
            r'require\s*\(\s*["\'](\w+)',  # require
            r'include\s+["\']([^"\']+)',  # include
            r'from\s+(\w+)\s+import',  # from X import
        ],
        
        # Sends/receives (for protocol)
        "sends": [
            r'send\s*\(\s*(\w+)',  # send(msg)
            r'write\s*\(\s*(\w+)',  # write(data)
            r'emit\s*\(\s*["\'](\w+)',  # emit('event')
        ],
        
        # Configures
        "configures": [
            r'"(\w+)":\s*"([^"]+)"',  # config key-value
            r'(\w+)\s*=\s*["\']([^"\']+)',  # setting assignment
            r'set\s+(\w+)\s+(\S+)',  # set command
        ],
    }
    
    # Words to skip as entities
    SKIP_WORDS = {
        'if', 'for', 'while', 'switch', 'return', 'new', 'var', 'function',
        'class', 'public', 'private', 'static', 'const', 'let', 'this',
        'true', 'false', 'null', 'undefined', 'void', 'int', 'string',
        'number', 'boolean', 'object', 'array', 'the', 'and', 'or', 'not'
    }
    
    def __init__(self, use_llm: bool = False, lmstudio_url: str = "http://localhost:1234/v1", model: str = "evony-7b-3800"):
        self.use_llm = use_llm
        self.lmstudio_url = lmstudio_url
        self.model = model
        self.base_extractor = EntityExtractor()
    
    def extract_enhanced_relationships(
        self, 
        content: str, 
        entities: List[Entity],
        file_path: str = "",
        category: str = ""
    ) -> List[Relationship]:
        """
        Extract relationships using multiple strategies.
        """
        relationships = []
        
        # 1. Pattern-based relationships
        pattern_rels = self._extract_pattern_relationships(content, entities, file_path)
        relationships.extend(pattern_rels)
        
        # 2. Co-occurrence relationships
        cooccur_rels = self._extract_cooccurrence_relationships(entities, file_path)
        relationships.extend(cooccur_rels)
        
        # 3. LLM-based relationships (if enabled)
        if self.use_llm:
            llm_rels = self._extract_llm_relationships(content, entities)
            relationships.extend(llm_rels)
        
        return relationships
    
    def _extract_pattern_relationships(
        self, 
        content: str, 
        entities: List[Entity],
        file_path: str
    ) -> List[Relationship]:
        """Extract relationships using regex patterns."""
        relationships = []
        entity_names = {e.name.lower(): e for e in entities}
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        
        for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, content, re.DOTALL | re.MULTILINE):
                        groups = match.groups()
                        if len(groups) >= 1:
                            source_name = groups[0].lower() if groups[0] else None
                            target_name = groups[1].lower() if len(groups) > 1 and groups[1] else None
                            
                            # Skip common words
                            if source_name in self.SKIP_WORDS:
                                continue
                            if target_name and target_name in self.SKIP_WORDS:
                                continue
                            
                            # Find matching entities
                            source_entity = entity_names.get(source_name)
                            target_entity = entity_names.get(target_name) if target_name else None
                            
                            if source_entity and target_entity:
                                relationships.append(Relationship(
                                    source_id=source_entity.id,
                                    target_id=target_entity.id,
                                    relation_type=rel_type,
                                    confidence=0.8
                                ))
                            elif source_entity and target_name:
                                # Create relationship to inferred entity
                                relationships.append(Relationship(
                                    source_id=source_entity.id,
                                    target_id=f"inferred:{file_hash}:{target_name}",
                                    relation_type=rel_type,
                                    confidence=0.6
                                ))
                except re.error:
                    continue
        
        return relationships
    
    def _extract_cooccurrence_relationships(
        self, 
        entities: List[Entity],
        file_path: str
    ) -> List[Relationship]:
        """
        Create relationships between entities that appear in the same chunk.
        These represent semantic proximity/association.
        """
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Group entities by type to create meaningful relationships
        type_groups = {}
        for e in entities:
            if e.entity_type not in type_groups:
                type_groups[e.entity_type] = []
            type_groups[e.entity_type].append(e)
        
        # Create "related_to" relationships between different types
        types = list(type_groups.keys())
        for i, type1 in enumerate(types):
            for type2 in types[i+1:]:
                # Limit combinations to avoid explosion
                entities1 = type_groups[type1][:5]  # Max 5 per type
                entities2 = type_groups[type2][:5]
                
                for e1 in entities1:
                    for e2 in entities2:
                        relationships.append(Relationship(
                            source_id=e1.id,
                            target_id=e2.id,
                            relation_type="co_occurs_with",
                            confidence=0.5
                        ))
        
        # Create "same_file" relationships within same type (limited)
        for entity_type, type_entities in type_groups.items():
            if len(type_entities) >= 2:
                # Only first 3 entities to limit explosion
                for e1, e2 in combinations(type_entities[:3], 2):
                    relationships.append(Relationship(
                        source_id=e1.id,
                        target_id=e2.id,
                        relation_type="same_context",
                        confidence=0.4
                    ))
        
        return relationships
    
    def _extract_llm_relationships(
        self, 
        content: str, 
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Use LLM to extract semantic relationships.
        """
        if not entities or len(entities) < 2:
            return []
        
        try:
            import requests
            
            # Prepare entity list for prompt
            entity_list = [f"- {e.name} ({e.entity_type})" for e in entities[:20]]  # Limit to 20
            entity_text = "\n".join(entity_list)
            
            prompt = f"""Analyze this code/text and identify relationships between the entities listed.

TEXT:
{content[:2000]}

ENTITIES:
{entity_text}

List relationships in format: source_entity -> relationship_type -> target_entity
Only use these relationship types: calls, references, contains, uses, configures, extends, implements, sends, receives

Output only the relationships, one per line. If no clear relationships, output "NONE"."""

            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result["choices"][0]["message"]["content"]
                return self._parse_llm_relationships(llm_output, entities)
            
        except Exception as e:
            pass  # Silently fail, return empty
        
        return []
    
    def _parse_llm_relationships(
        self, 
        llm_output: str, 
        entities: List[Entity]
    ) -> List[Relationship]:
        """Parse LLM output into relationships."""
        relationships = []
        entity_map = {e.name.lower(): e for e in entities}
        
        if "NONE" in llm_output:
            return relationships
        
        # Parse lines like "EntityA -> calls -> EntityB"
        pattern = r'(\w+)\s*->\s*(\w+)\s*->\s*(\w+)'
        for match in re.finditer(pattern, llm_output):
            source_name = match.group(1).lower()
            rel_type = match.group(2).lower()
            target_name = match.group(3).lower()
            
            source = entity_map.get(source_name)
            target = entity_map.get(target_name)
            
            if source and target:
                relationships.append(Relationship(
                    source_id=source.id,
                    target_id=target.id,
                    relation_type=rel_type,
                    confidence=0.7
                ))
        
        return relationships


def rebuild_kg_with_enhanced_relationships(use_llm: bool = False, llm_sample_rate: float = 0.1):
    """
    Rebuild the knowledge graph with enhanced relationship extraction.
    
    Args:
        use_llm: Enable LLM-based extraction
        llm_sample_rate: Fraction of chunks to process with LLM (0.1 = 10%)
    """
    import random
    chunks_file = r"evony_rag\index\chunks.json"
    kg_file = r"G:\evony_rag_index\knowledge_graph.json"
    
    print("=" * 60)
    print("ENHANCED KNOWLEDGE GRAPH BUILD")
    print("=" * 60)
    print(f"LLM extraction: {'ENABLED' if use_llm else 'DISABLED'}")
    if use_llm:
        print(f"LLM sample rate: {llm_sample_rate*100:.0f}% of chunks")
    
    # Load chunks
    print("\nLoading chunks...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"{len(chunks):,} chunks loaded")
    
    # Initialize extractors - disable LLM in extractor, we'll call it selectively
    base_extractor = EntityExtractor()
    enhanced_extractor = EnhancedRelationshipExtractor(use_llm=False)  # Pattern-based only
    llm_extractor = EnhancedRelationshipExtractor(use_llm=True) if use_llm else None
    
    # Build graph
    kg = KnowledgeGraph()
    total_base_rels = 0
    total_enhanced_rels = 0
    total_llm_rels = 0
    llm_processed = 0
    
    print("\nExtracting entities and relationships...")
    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        file_path = chunk.get("file_path", "")
        category = chunk.get("category", "")
        
        # Extract entities with base extractor
        entities, base_relationships = base_extractor.extract_from_chunk(chunk)
        
        # Add entities to graph
        for e in entities:
            kg.add_entity(e)
        
        # Add base relationships
        for r in base_relationships:
            kg.add_relationship(r)
        total_base_rels += len(base_relationships)
        
        # Extract enhanced relationships (pattern + co-occurrence)
        enhanced_rels = enhanced_extractor.extract_enhanced_relationships(
            content, entities, file_path, category
        )
        
        # Add enhanced relationships
        for r in enhanced_rels:
            kg.add_relationship(r)
        total_enhanced_rels += len(enhanced_rels)
        
        # LLM extraction on sampled chunks
        if llm_extractor and random.random() < llm_sample_rate and len(entities) >= 2:
            llm_rels = llm_extractor._extract_llm_relationships(content, entities)
            for r in llm_rels:
                kg.add_relationship(r)
            total_llm_rels += len(llm_rels)
            llm_processed += 1
        
        if i % 2000 == 0:
            llm_info = f", LLM: {total_llm_rels:,}" if use_llm else ""
            print(f"  {i:,}: {len(kg.entities):,} entities, {len(kg.relationships):,} rels{llm_info}")
    
    print(f"\nFinal: {len(kg.entities):,} entities")
    print(f"  Base relationships: {total_base_rels:,}")
    print(f"  Enhanced relationships: {total_enhanced_rels:,}")
    if use_llm:
        print(f"  LLM relationships: {total_llm_rels:,} (from {llm_processed:,} chunks)")
    print(f"  Total relationships: {len(kg.relationships):,}")
    
    # Build data for saving
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
    
    # Count relationship types
    rel_types = {}
    for r in kg.relationships:
        rel_types[r.relation_type] = rel_types.get(r.relation_type, 0) + 1
    
    print("\nRelationship types:")
    for rt, count in sorted(rel_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {rt}: {count:,}")
    
    # Save
    print(f"\nSaving to {kg_file}...")
    with open(kg_file, "w", encoding="utf-8") as f:
        json.dump(kg_data, f)
        f.flush()
        os.fsync(f.fileno())
    
    print(f"Saved: {os.path.getsize(kg_file):,} bytes")
    
    # Verify
    print("\nVerifying...")
    with open(kg_file, "r", encoding="utf-8") as f:
        verify = json.load(f)
    print(f"Verified: {len(verify['entities']):,} entities, {len(verify['relationships']):,} relationships")
    
    return len(kg.entities), len(kg.relationships)


if __name__ == "__main__":
    # Check for --llm flag
    use_llm = "--llm" in sys.argv
    rebuild_kg_with_enhanced_relationships(use_llm=use_llm)
