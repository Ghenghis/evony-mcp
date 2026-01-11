"""
Evony RAG - Query Router
=========================
Routes queries to appropriate handlers and applies safety filters.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import BLOCKED_PATTERNS, QUERY_INTENTS, CATEGORIES


@dataclass
class QueryAnalysis:
    """Analysis of a user query."""
    query: str
    intent: str
    categories: List[str]
    is_blocked: bool
    block_reason: Optional[str]
    keywords: List[str]


class QueryRouter:
    """Routes queries and applies safety filters."""
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        "code_explain": [
            r"what does .* (do|mean|function)",
            r"explain .* (code|function|class|method)",
            r"how does .* work",
            r"what is the purpose of",
        ],
        "protocol_info": [
            r"(amf|protocol) (command|request|message)",
            r"what (parameters|args)",
            r"how (do i|to) send",
            r"\w+\.\w+",  # command.action pattern
        ],
        "find_files": [
            r"find .* (file|code|snippet)",
            r"show me .* (files|code)",
            r"where is .* (defined|located)",
            r"search for",
        ],
        "howto": [
            r"how (do i|can i|to)",
            r"implement",
            r"create .* (script|bot|tool)",
        ],
        "exploit_info": [
            r"(glitch|exploit|overflow|bug)",
            r"(vulnerability|vuln)",
            r"integer overflow",
        ],
    }
    
    # Category keywords
    CATEGORY_KEYWORDS = {
        "source_code": ["class", "function", "def", "public", "private", ".as", ".py"],
        "protocol": ["amf", "command", "request", "response", "packet", "socket"],
        "keys": ["key", "encrypt", "decrypt", "hash", "md5", "salt", "signature"],
        "exploits": ["glitch", "exploit", "overflow", "bug", "vulnerability"],
        "scripts": ["bot", "autoevony", "script", "automation", "includeurl"],
        "documentation": ["doc", "readme", "guide", "tutorial"],
    }
    
    def __init__(self):
        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]
    
    def _check_blocked(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if query matches blocked patterns."""
        for pattern in self.blocked_patterns:
            if pattern.search(query):
                return True, "Operational exploit requests are not supported. Ask about the mechanics instead."
        return False, None
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        query_lower = query.lower()
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return "general"
    
    def _detect_categories(self, query: str) -> List[str]:
        """Detect relevant categories from query."""
        query_lower = query.lower()
        categories = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    categories.append(category)
                    break
        
        # Default to searching all safe categories
        if not categories:
            categories = [cat for cat, info in CATEGORIES.items() if info["safe"]]
        
        return list(set(categories))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common words
        stopwords = {
            "what", "how", "does", "the", "a", "an", "is", "are", "can", "do",
            "to", "in", "for", "of", "and", "or", "this", "that", "it", "i",
            "me", "my", "show", "find", "explain", "tell", "about", "with"
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords[:10]
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query and return routing information."""
        is_blocked, block_reason = self._check_blocked(query)
        
        return QueryAnalysis(
            query=query,
            intent=self._detect_intent(query),
            categories=self._detect_categories(query),
            is_blocked=is_blocked,
            block_reason=block_reason,
            keywords=self._extract_keywords(query),
        )
    
    def should_include_exploits(self, query: str) -> bool:
        """Determine if exploit category should be included (educational only)."""
        query_lower = query.lower()
        
        # Educational patterns are OK
        educational = [
            "how does .* work",
            "explain .* (mechanism|mechanics)",
            "what is .* (overflow|glitch)",
            "why does .* happen",
            "technical (details|explanation)",
        ]
        
        for pattern in educational:
            if re.search(pattern, query_lower):
                return True
        
        return False
