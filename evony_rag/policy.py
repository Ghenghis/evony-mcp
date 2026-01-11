"""
Evony RAG - Policy Engine
==========================
Controls query modes, category access, and safety filters.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .config import DATASET_PATH


@dataclass
class QueryPolicy:
    """Policy for a specific query."""
    mode: str
    include_categories: Set[str]
    exclude_categories: Set[str]
    keys_allowed: bool
    is_blocked: bool
    block_reason: Optional[str]
    evidence_level: str
    final_k: int
    min_score: float


class PolicyEngine:
    """Manages query policies and access control."""
    
    DEFAULT_POLICY = {
        'modes': {
            'research': {
                'include': ['source_code', 'protocol', 'documentation', 'scripts', 'game_data', 'tools'],
                'exclude': ['exploits'],
                'keys_allowed': True,
            },
            'forensics': {
                'include': ['source_code', 'protocol', 'documentation'],
                'exclude': ['exploits', 'keys', 'scripts'],
                'keys_allowed': False,
            },
            'full_access': {
                'include': ['source_code', 'protocol', 'documentation', 'scripts', 'game_data', 'tools', 'keys', 'exploits'],
                'exclude': [],
                'keys_allowed': True,
            }
        },
        'blocked_patterns': [
            r"how (do i|can i|to) (use|exploit|abuse) .* (glitch|bug|exploit)",
            r"give me .* (exploit|hack|cheat)",
            r"step.?by.?step .* (glitch|exploit)",
            r"working (exploit|hack|cheat)",
        ],
        'allowed_patterns': [
            r"how does .* work",
            r"explain .* (mechanism|mechanics)",
            r"what is .* (overflow|glitch)",
            r"where is .* defined",
            r"what parameters",
        ],
        'retrieval': {
            'k_lexical': 20,
            'k_vector': 20,
            'final_k': 8,
            'min_score': 0.01,  # RRF scores are typically low
            'evidence_level': 'normal',
        },
        'evidence_levels': {
            'brief': {'max_sources': 3, 'show_snippets': False},
            'normal': {'max_sources': 5, 'show_snippets': True},
            'verbose': {'max_sources': 10, 'show_snippets': True},
        }
    }
    
    def __init__(self, policy_path: Path = None):
        self.policy_path = policy_path or (DATASET_PATH / 'metadata' / 'policy.yaml')
        self.policy = self._load_policy()
        self.current_mode = 'research'
        self._compile_patterns()
    
    def _load_policy(self) -> Dict:
        """Load policy from YAML file."""
        if self.policy_path.exists():
            try:
                with open(self.policy_path, 'r') as f:
                    return yaml.safe_load(f)
            except (yaml.YAMLError, IOError) as e:
                import logging
                logging.getLogger(__name__).warning(f"Policy load failed, using defaults: {e}")
        return self.DEFAULT_POLICY
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.policy.get('blocked_patterns', [])
        ]
        self.allowed_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.policy.get('allowed_patterns', [])
        ]
    
    def set_mode(self, mode: str) -> bool:
        """Set the current query mode."""
        if mode in self.policy.get('modes', {}):
            self.current_mode = mode
            return True
        return False
    
    def get_modes(self) -> List[str]:
        """Get available modes."""
        return list(self.policy.get('modes', {}).keys())
    
    def _check_blocked(self, query: str) -> tuple[bool, Optional[str]]:
        """Check if query is blocked."""
        query_lower = query.lower()
        
        # Check if explicitly allowed (educational)
        for pattern in self.allowed_patterns:
            if pattern.search(query_lower):
                return False, None
        
        # Check if blocked (operational)
        for pattern in self.blocked_patterns:
            if pattern.search(query_lower):
                return True, "Operational requests are blocked. Ask about mechanics instead."
        
        return False, None
    
    def evaluate(self, query: str,
                 mode: str = None,
                 include: List[str] = None,
                 exclude: List[str] = None,
                 evidence_level: str = None,
                 final_k: int = None) -> QueryPolicy:
        """Evaluate query against policy and return access rules."""
        
        mode = mode or self.current_mode
        mode_config = self.policy.get('modes', {}).get(mode, {})
        retrieval = self.policy.get('retrieval', {})
        
        # Get base categories from mode
        base_include = set(mode_config.get('include', []))
        base_exclude = set(mode_config.get('exclude', []))
        
        # Apply overrides
        if include:
            base_include = set(include)
        if exclude:
            base_exclude.update(exclude)
        
        # Remove excluded from included
        final_include = base_include - base_exclude
        
        # Check if blocked
        is_blocked, block_reason = self._check_blocked(query)
        
        return QueryPolicy(
            mode=mode,
            include_categories=final_include,
            exclude_categories=base_exclude,
            keys_allowed=mode_config.get('keys_allowed', True),
            is_blocked=is_blocked,
            block_reason=block_reason,
            evidence_level=evidence_level or retrieval.get('evidence_level', 'normal'),
            final_k=final_k or retrieval.get('final_k', 8),
            min_score=retrieval.get('min_score', 0.25),
        )
    
    def get_evidence_config(self, level: str) -> Dict:
        """Get evidence display configuration."""
        levels = self.policy.get('evidence_levels', {})
        return levels.get(level, levels.get('normal', {}))
    
    def get_retrieval_config(self) -> Dict:
        """Get retrieval configuration."""
        return self.policy.get('retrieval', {})


# Singleton
_policy_engine = None

def get_policy() -> PolicyEngine:
    """Get singleton policy engine."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
    return _policy_engine
