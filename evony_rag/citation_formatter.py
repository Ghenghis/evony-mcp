#!/usr/bin/env python3
"""Citation Formatter for RAG-Native Responses"""

import re
from typing import List, Dict
from pathlib import Path

class CitationFormatter:
    """Formats citations in RAG-native style"""
    
    def __init__(self, base_path: str = ""):
        self.base_path = base_path
        
    def format_citation(self, file_path: str, line_start: int, line_end: int = None) -> str:
        """Format a single citation"""
        # Clean path
        clean_path = file_path.replace("\\", "/")
        if self.base_path and clean_path.startswith(self.base_path):
            clean_path = clean_path[len(self.base_path):]
        if clean_path.startswith("/"):
            clean_path = clean_path[1:]
            
        if line_end and line_end != line_start:
            return f"`@{clean_path}:{line_start}-{line_end}`"
        return f"`@{clean_path}:{line_start}`"
    
    def format_evidence_block(self, results: List[Dict]) -> str:
        """Format multiple search results as evidence block"""
        if not results:
            return "No evidence found."
        
        output = "**Evidence:**\n"
        for i, result in enumerate(results[:5], 1):
            file_path = result.get("file", result.get("path", "unknown"))
            line = result.get("line", result.get("line_start", 1))
            snippet = result.get("snippet", result.get("content", ""))[:200]
            
            citation = self.format_citation(file_path, line)
            output += f"\n{i}. {citation}\n```\n{snippet}\n```\n"
        
        return output
    
    def inject_citations(self, text: str, results: List[Dict]) -> str:
        """Inject citations into response text"""
        # Find file references and add citations
        file_pattern = r"(\w+\.as|\w+\.py|\w+\.json)"
        
        def add_citation(match):
            filename = match.group(1)
            for result in results:
                if filename in result.get("file", ""):
                    line = result.get("line", 1)
                    return f"`@source_code/{filename}:{line}`"
            return match.group(0)
        
        return re.sub(file_pattern, add_citation, text)
    
    def format_uncertainty_response(self, query: str) -> str:
        """Format response when evidence is insufficient"""
        # Extract key terms
        words = query.lower().split()
        key_terms = [w for w in words if len(w) > 4][:3]
        
        suggestions = []
        if key_terms:
            suggestions.append(f"{key_terms[0]} definition source_code")
            suggestions.append(f"{key_terms[0]} implementation")
            if len(key_terms) > 1:
                suggestions.append(f"{key_terms[0]} {key_terms[1]} handler")
        else:
            suggestions = [
                "command implementation",
                "bean class definition",
                "response handler"
            ]
        
        return f"""**Need Evidence**

To answer this accurately, please retrieve:
1. {suggestions[0]}
2. {suggestions[1]}
{f"3. {suggestions[2]}" if len(suggestions) > 2 else ""}

I cannot provide a reliable answer without evidence from the codebase."""


# Singleton
_formatter = None

def get_formatter(base_path: str = "") -> CitationFormatter:
    global _formatter
    if _formatter is None:
        _formatter = CitationFormatter(base_path)
    return _formatter
