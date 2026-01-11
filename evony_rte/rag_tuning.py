"""
Evony RTE - RAG Semantic Search Tuning
=======================================
Optimized RAG configuration for Evony game analysis.
Tuned for protocol documentation, exploit research, and code analysis.

Key Optimizations:
- Evony-specific stopwords
- Category weighting (protocol > source_code > docs)
- Improved chunking for AS3/AMF3 content
- Query expansion for game terminology
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# EVONY-SPECIFIC STOPWORDS
# ============================================================================

EVONY_STOPWORDS = {
    # Generic Python/code stopwords to filter
    'import', 'from', 'def', 'class', 'return', 'self', 'none', 'true', 'false',
    'the', 'and', 'or', 'is', 'in', 'to', 'for', 'of', 'a', 'an', 'as', 'by',
    'with', 'that', 'this', 'it', 'be', 'are', 'was', 'were', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'use', 'used', 'using',
    
    # Generic technical terms (less useful for Evony-specific queries)
    'function', 'method', 'variable', 'parameter', 'argument', 'value', 'type',
    'string', 'integer', 'number', 'array', 'object', 'list', 'dict', 'tuple',
    'file', 'path', 'directory', 'module', 'package', 'library',
}

# Evony-specific terms to BOOST (not filter)
EVONY_BOOST_TERMS = {
    # Game mechanics
    'troop', 'troops', 'army', 'march', 'attack', 'defend', 'scout', 'reinforce',
    'castle', 'city', 'building', 'resource', 'gold', 'food', 'lumber', 'stone', 'iron',
    'hero', 'general', 'alliance', 'war', 'battle', 'combat',
    
    # Troop types
    'worker', 'warrior', 'scout', 'pikeman', 'swordsman', 'archer', 'cavalry',
    'cataphract', 'ballista', 'ram', 'catapult', 'transporter',
    
    # Protocol/technical
    'amf', 'amf3', 'packet', 'command', 'request', 'response', 'handler',
    'encode', 'decode', 'serialize', 'deserialize', 'payload', 'header',
    'socket', 'connection', 'session', 'token', 'auth', 'login',
    
    # Exploit-related
    'overflow', 'underflow', 'bypass', 'exploit', 'vulnerability', 'injection',
    'validation', 'sanitize', 'boundary', 'limit', 'negative', 'integer',
    
    # Flash/SWF
    'swf', 'flash', 'actionscript', 'as3', 'abc', 'bytecode', 'decompile',
    'ffdec', 'jpexs', 'rabcdasm', 'ghidra',
    
    # Commands
    'produceTroop', 'disbandTroop', 'getInfo', 'cityInfo', 'marchArmy',
    'buildBuilding', 'upgradeBuilding', 'trainHero', 'equipHero',
}

# Query expansion mappings
QUERY_EXPANSIONS = {
    'troops': ['troop', 'army', 'soldier', 'unit', 'produceTroop'],
    'archer': ['archer', 'bowman', 'ranged', 'troop_type_6'],
    'overflow': ['overflow', 'integer_overflow', 'int32', 'boundary', 'exploit'],
    'packet': ['packet', 'amf3', 'message', 'request', 'response', 'payload'],
    'resource': ['resource', 'gold', 'food', 'lumber', 'stone', 'iron', 'getResources'],
    'attack': ['attack', 'march', 'battle', 'combat', 'marchArmy', 'assault'],
    'hero': ['hero', 'general', 'commander', 'heroInfo', 'trainHero'],
    'building': ['building', 'structure', 'upgrade', 'buildBuilding', 'construct'],
    'login': ['login', 'auth', 'authentication', 'session', 'token', 'connect'],
    'decompile': ['decompile', 'disassemble', 'reverse', 'ffdec', 'jpexs', 'swf'],
}

# Category weights for ranking
CATEGORY_WEIGHTS = {
    'protocol': 2.0,        # AMF3 protocol docs - highest priority
    'exploit': 1.8,         # Exploit documentation
    'source_code': 1.5,     # AS3 source code
    'command': 1.5,         # Game commands
    'api': 1.3,             # API documentation
    'client': 1.2,          # Client analysis
    'server': 1.2,          # Server analysis
    'documentation': 1.0,   # General docs
    'default': 0.8,         # Unknown category
}


# ============================================================================
# QUERY PREPROCESSING
# ============================================================================

def preprocess_query(query: str) -> Dict[str, Any]:
    """
    Preprocess query for optimal Evony RAG search.
    
    Returns:
        {
            'original': str,
            'cleaned': str,
            'terms': List[str],
            'expanded_terms': List[str],
            'boost_terms': List[str],
            'category_hint': str
        }
    """
    original = query
    
    # Lowercase and clean
    cleaned = query.lower().strip()
    cleaned = re.sub(r'[^\w\s\-_.]', ' ', cleaned)
    
    # Tokenize
    terms = [t for t in cleaned.split() if t and t not in EVONY_STOPWORDS]
    
    # Expand terms
    expanded = set(terms)
    for term in terms:
        if term in QUERY_EXPANSIONS:
            expanded.update(QUERY_EXPANSIONS[term])
    
    # Find boost terms
    boost = [t for t in expanded if t in EVONY_BOOST_TERMS]
    
    # Detect category hint from query
    category_hint = detect_category(query, terms)
    
    return {
        'original': original,
        'cleaned': cleaned,
        'terms': terms,
        'expanded_terms': list(expanded),
        'boost_terms': boost,
        'category_hint': category_hint,
        'term_count': len(terms),
        'expansion_count': len(expanded) - len(terms)
    }


def detect_category(query: str, terms: List[str]) -> str:
    """Detect likely category from query."""
    query_lower = query.lower()
    
    # Protocol indicators
    if any(t in query_lower for t in ['amf', 'packet', 'encode', 'decode', 'protocol']):
        return 'protocol'
    
    # Exploit indicators
    if any(t in query_lower for t in ['overflow', 'exploit', 'vulnerability', 'bypass', 'injection']):
        return 'exploit'
    
    # Command indicators
    if any(t in query_lower for t in ['command', 'request', 'response', 'handler']):
        return 'command'
    
    # Client analysis
    if any(t in query_lower for t in ['decompile', 'swf', 'flash', 'as3', 'client']):
        return 'client'
    
    # Game mechanics
    if any(t in query_lower for t in ['troop', 'resource', 'building', 'hero', 'march']):
        return 'source_code'
    
    return 'default'


# ============================================================================
# RESULT RANKING
# ============================================================================

def rank_results(results: List[Dict], query_info: Dict) -> List[Dict]:
    """
    Re-rank search results with Evony-specific scoring.
    
    Scoring factors:
    - Category weight
    - Boost term matches
    - Exact phrase matches
    - Recency (for exploits)
    """
    category_hint = query_info.get('category_hint', 'default')
    boost_terms = set(query_info.get('boost_terms', []))
    original_query = query_info.get('original', '').lower()
    
    scored_results = []
    
    for result in results:
        score = result.get('score', 0.5)
        
        # Apply category weight
        result_category = result.get('category', result.get('type', 'default'))
        category_weight = CATEGORY_WEIGHTS.get(result_category, CATEGORY_WEIGHTS['default'])
        
        # Bonus if category matches hint
        if result_category == category_hint:
            category_weight *= 1.3
        
        score *= category_weight
        
        # Boost term bonus
        content = str(result.get('content', '')).lower()
        title = str(result.get('title', result.get('name', ''))).lower()
        
        boost_matches = sum(1 for t in boost_terms if t in content or t in title)
        if boost_matches > 0:
            score *= (1 + 0.1 * boost_matches)
        
        # Exact phrase bonus
        if original_query in content or original_query in title:
            score *= 1.5
        
        # Penalize generic Python docs
        if 'python' in title and 'evony' not in title:
            score *= 0.5
        
        result['adjusted_score'] = round(score, 4)
        result['ranking_factors'] = {
            'category_weight': category_weight,
            'boost_matches': boost_matches,
            'exact_match': original_query in content or original_query in title
        }
        
        scored_results.append(result)
    
    # Sort by adjusted score
    scored_results.sort(key=lambda x: x['adjusted_score'], reverse=True)
    
    return scored_results


# ============================================================================
# CHUNKING STRATEGY
# ============================================================================

def chunk_evony_content(content: str, source_type: str = 'default', 
                        max_chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
    """
    Chunk content with Evony-specific strategy.
    
    Strategies by source type:
    - as3: Chunk by class/function boundaries
    - protocol: Chunk by command/handler definitions
    - documentation: Chunk by section headers
    - default: Fixed-size with overlap
    """
    if source_type == 'as3':
        return _chunk_as3_code(content, max_chunk_size)
    elif source_type == 'protocol':
        return _chunk_protocol_docs(content, max_chunk_size)
    elif source_type == 'documentation':
        return _chunk_by_headers(content, max_chunk_size)
    else:
        return _chunk_fixed_size(content, max_chunk_size, overlap)


def _chunk_as3_code(content: str, max_size: int) -> List[Dict]:
    """Chunk AS3 code by class/function boundaries."""
    chunks = []
    
    # Split by class definitions
    class_pattern = r'((?:public|private|internal)\s+class\s+\w+[^{]*\{)'
    parts = re.split(class_pattern, content)
    
    current_chunk = ""
    current_class = "global"
    
    for i, part in enumerate(parts):
        if re.match(class_pattern, part):
            # Save previous chunk
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip(),
                    'type': 'as3_class',
                    'class_name': current_class,
                    'size': len(current_chunk)
                })
            # Extract class name
            match = re.search(r'class\s+(\w+)', part)
            current_class = match.group(1) if match else "unknown"
            current_chunk = part
        else:
            current_chunk += part
            
            # Split large chunks by function
            if len(current_chunk) > max_size:
                func_pattern = r'((?:public|private|protected|internal)\s+function\s+\w+[^{]*\{)'
                func_parts = re.split(func_pattern, current_chunk)
                
                for fp in func_parts:
                    if fp.strip():
                        chunks.append({
                            'content': fp.strip()[:max_size],
                            'type': 'as3_function',
                            'class_name': current_class,
                            'size': len(fp.strip()[:max_size])
                        })
                current_chunk = ""
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip(),
            'type': 'as3_class',
            'class_name': current_class,
            'size': len(current_chunk)
        })
    
    return chunks


def _chunk_protocol_docs(content: str, max_size: int) -> List[Dict]:
    """Chunk protocol documentation by command definitions."""
    chunks = []
    
    # Split by command patterns like "city.getInfo" or "Command: "
    cmd_pattern = r'(?:^|\n)((?:\w+\.\w+)|(?:Command:\s*\w+))'
    parts = re.split(cmd_pattern, content)
    
    current_chunk = ""
    current_cmd = "overview"
    
    for part in parts:
        if re.match(r'\w+\.\w+', part) or part.startswith('Command:'):
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip()[:max_size],
                    'type': 'protocol_command',
                    'command': current_cmd,
                    'size': len(current_chunk.strip()[:max_size])
                })
            current_cmd = part.replace('Command:', '').strip()
            current_chunk = part
        else:
            current_chunk += part
    
    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip()[:max_size],
            'type': 'protocol_command',
            'command': current_cmd,
            'size': len(current_chunk.strip()[:max_size])
        })
    
    return chunks


def _chunk_by_headers(content: str, max_size: int) -> List[Dict]:
    """Chunk documentation by markdown headers."""
    chunks = []
    
    # Split by markdown headers
    header_pattern = r'(^#{1,3}\s+.+$)'
    parts = re.split(header_pattern, content, flags=re.MULTILINE)
    
    current_chunk = ""
    current_header = "Introduction"
    
    for part in parts:
        if re.match(r'^#{1,3}\s+', part):
            if current_chunk.strip():
                chunks.append({
                    'content': current_chunk.strip()[:max_size],
                    'type': 'documentation',
                    'section': current_header,
                    'size': len(current_chunk.strip()[:max_size])
                })
            current_header = part.strip('#').strip()
            current_chunk = part
        else:
            current_chunk += part
    
    if current_chunk.strip():
        chunks.append({
            'content': current_chunk.strip()[:max_size],
            'type': 'documentation',
            'section': current_header,
            'size': len(current_chunk.strip()[:max_size])
        })
    
    return chunks


def _chunk_fixed_size(content: str, max_size: int, overlap: int) -> List[Dict]:
    """Fixed-size chunking with overlap."""
    chunks = []
    
    for i in range(0, len(content), max_size - overlap):
        chunk = content[i:i + max_size]
        if chunk.strip():
            chunks.append({
                'content': chunk.strip(),
                'type': 'text',
                'offset': i,
                'size': len(chunk)
            })
    
    return chunks


# ============================================================================
# RAG HANDLER (MCP Integration)
# ============================================================================

def handle_rag_config(args: Dict) -> Dict:
    """
    Configure RAG semantic search for Evony.
    
    Parameters:
        action: 'status' | 'preprocess' | 'expand' | 'weights' | 'test'
        query: Query string for preprocess/test actions
        category: Category to get/set weight
        weight: New weight value
    
    Returns:
        Configuration status or processed query info
    """
    action = args.get('action', 'status')
    
    if action == 'status':
        return {
            'stopwords_count': len(EVONY_STOPWORDS),
            'boost_terms_count': len(EVONY_BOOST_TERMS),
            'expansions_count': len(QUERY_EXPANSIONS),
            'category_weights': CATEGORY_WEIGHTS,
            'supported_source_types': ['as3', 'protocol', 'documentation', 'default']
        }
    
    if action == 'preprocess':
        query = args.get('query', '')
        if not query:
            return {'error': 'Query required for preprocess action'}
        return preprocess_query(query)
    
    if action == 'expand':
        query = args.get('query', '')
        if not query:
            return {'error': 'Query required for expand action'}
        info = preprocess_query(query)
        return {
            'original': info['original'],
            'expanded': info['expanded_terms'],
            'boost_terms': info['boost_terms'],
            'expansion_count': info['expansion_count']
        }
    
    if action == 'weights':
        return {
            'category_weights': CATEGORY_WEIGHTS,
            'boost_terms_sample': list(EVONY_BOOST_TERMS)[:20],
            'note': 'Higher weight = higher priority in results'
        }
    
    if action == 'test':
        query = args.get('query', 'troop overflow exploit')
        info = preprocess_query(query)
        
        # Create mock results for testing
        mock_results = [
            {'content': 'AMF3 packet encoding for troop commands', 'category': 'protocol', 'score': 0.8},
            {'content': 'Integer overflow in produceTroop handler', 'category': 'exploit', 'score': 0.7},
            {'content': 'Python string manipulation tutorial', 'category': 'default', 'score': 0.9},
            {'content': 'TroopCommands.as3 source code', 'category': 'source_code', 'score': 0.75},
        ]
        
        ranked = rank_results(mock_results, info)
        
        return {
            'query_info': info,
            'mock_results': ranked,
            'note': 'Generic Python doc was demoted, exploit/protocol boosted'
        }
    
    return {'error': f'Unknown action: {action}'}


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'EVONY_STOPWORDS',
    'EVONY_BOOST_TERMS',
    'QUERY_EXPANSIONS',
    'CATEGORY_WEIGHTS',
    'preprocess_query',
    'detect_category',
    'rank_results',
    'chunk_evony_content',
    'handle_rag_config'
]
