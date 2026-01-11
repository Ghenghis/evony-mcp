#!/usr/bin/env python3
"""Query Classifier for RAG-Native Routing"""

import re
from typing import Tuple

class QueryClassifier:
    """Classifies queries into task types for optimal routing"""
    
    TASK_PATTERNS = {
        "gui_map": [
            r"screen|button|ui|interface|click|menu|panel|widget",
            r"what.*visible|layout|position|element"
        ],
        "gui_action": [
            r"how (do i|to)|steps|navigate|open|access|find where",
            r"click.*to|guide me|show me how"
        ],
        "code_explain": [
            r"explain|what (is|does)|describe|tell me about",
            r"how does.*work|meaning of|purpose of"
        ],
        "patch_plan": [
            r"fix|patch|modify|change|update|bug|error",
            r"refactor|improve|optimize"
        ],
        "query_synthesis": [
            r"find|search|locate|where is|look for|retrieve",
            r"what files|which class|source of"
        ],
        "evidence_cite": [
            r"cite|reference|source|proof|evidence",
            r"show me.*code|actual implementation"
        ]
    }
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query and return (task_type, confidence)
        """
        query_lower = query.lower()
        scores = {}
        
        for task_type, patterns in self.TASK_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[task_type] = score
        
        if max(scores.values()) == 0:
            return ("general", 0.5)
        
        best_task = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_task] / 2)
        
        return (best_task, confidence)
    
    def get_search_strategy(self, task_type: str) -> dict:
        """Get optimal search strategy for task type"""
        
        strategies = {
            "gui_map": {
                "k_lexical": 10,
                "k_semantic": 20,
                "prefer": "semantic",
                "file_types": [".as", ".xml", ".json"]
            },
            "gui_action": {
                "k_lexical": 15,
                "k_semantic": 15,
                "prefer": "hybrid",
                "file_types": [".as", ".md"]
            },
            "code_explain": {
                "k_lexical": 20,
                "k_semantic": 10,
                "prefer": "lexical",
                "file_types": [".as", ".py"]
            },
            "patch_plan": {
                "k_lexical": 25,
                "k_semantic": 10,
                "prefer": "lexical",
                "file_types": [".as", ".py", ".js"]
            },
            "query_synthesis": {
                "k_lexical": 10,
                "k_semantic": 10,
                "prefer": "hybrid",
                "file_types": ["*"]
            },
            "evidence_cite": {
                "k_lexical": 30,
                "k_semantic": 5,
                "prefer": "lexical",
                "file_types": [".as", ".py"]
            },
            "general": {
                "k_lexical": 15,
                "k_semantic": 15,
                "prefer": "hybrid",
                "file_types": ["*"]
            }
        }
        
        return strategies.get(task_type, strategies["general"])


# Singleton instance
_classifier = None

def get_classifier() -> QueryClassifier:
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier
