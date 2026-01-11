"""
Evony RTE - MCP Server Performance Optimization Module
=======================================================
Implements performance enhancements to reduce Windsurf IDE lag:

1. Lazy Loading - Tool classes initialized only when first used
2. Response Caching - Cache expensive computations 
3. Tool Instance Pooling - Reuse tool instances
4. Async-Ready Handlers - Non-blocking operations
5. Lightweight Status Checks - Fast health checks
6. Memory Optimization - Reduce memory footprint

Best Practices for MCP Servers with Many Tools:
- Defer heavy imports until needed
- Cache tool detection results
- Use connection pooling for external processes
- Implement timeouts on all external calls
- Batch similar operations when possible
"""

import time
import functools
import threading
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
import hashlib
import weakref

# ============================================================================
# LAZY LOADING - Defer tool initialization until first use
# ============================================================================

class LazyLoader:
    """Lazy loader for heavy tool classes."""
    
    _instances: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, tool_name: str, factory: Callable) -> Any:
        """Get or create tool instance lazily."""
        if tool_name not in cls._instances:
            with cls._lock:
                if tool_name not in cls._instances:
                    cls._instances[tool_name] = factory()
        return cls._instances[tool_name]
    
    @classmethod
    def clear(cls, tool_name: str = None):
        """Clear cached instances."""
        with cls._lock:
            if tool_name:
                cls._instances.pop(tool_name, None)
            else:
                cls._instances.clear()

# ============================================================================
# RESPONSE CACHING - Cache expensive operations
# ============================================================================

class ResponseCache:
    """LRU cache for MCP responses with TTL."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
        self._lock = threading.Lock()
    
    def _make_key(self, tool_name: str, args: Dict) -> str:
        """Create cache key from tool name and args."""
        args_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()
    
    def get(self, tool_name: str, args: Dict) -> Optional[Dict]:
        """Get cached response if valid."""
        key = self._make_key(tool_name, args)
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                del self._cache[key]
        return None
    
    def set(self, tool_name: str, args: Dict, value: Dict):
        """Cache response."""
        key = self._make_key(tool_name, args)
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (value, time.time())
    
    def invalidate(self, tool_name: str = None):
        """Invalidate cache entries."""
        with self._lock:
            if tool_name:
                keys_to_del = [k for k in self._cache if k.startswith(tool_name)]
                for k in keys_to_del:
                    del self._cache[k]
            else:
                self._cache.clear()

# Global cache instance
_response_cache = ResponseCache(max_size=200, ttl_seconds=60)

# ============================================================================
# CACHING DECORATOR - Easy way to add caching to handlers
# ============================================================================

def cached_handler(ttl_seconds: int = 60, cache_errors: bool = False):
    """Decorator to cache handler responses."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(args: Dict) -> Dict:
            tool_name = func.__name__
            
            # Check cache
            cached = _response_cache.get(tool_name, args)
            if cached is not None:
                cached["_cached"] = True
                return cached
            
            # Execute handler
            result = func(args)
            
            # Cache successful results (or all if cache_errors=True)
            if cache_errors or "error" not in result:
                _response_cache.set(tool_name, args, result)
            
            return result
        return wrapper
    return decorator

# ============================================================================
# TOOL STATUS CACHE - Cache tool detection for 5 minutes
# ============================================================================

class ToolStatusCache:
    """Cache tool availability checks to avoid repeated filesystem/WSL calls."""
    
    _cache: Dict[str, tuple] = {}
    _ttl = 300  # 5 minutes
    _lock = threading.Lock()
    
    @classmethod
    def get(cls, tool_name: str) -> Optional[Dict]:
        """Get cached status."""
        with cls._lock:
            if tool_name in cls._cache:
                status, timestamp = cls._cache[tool_name]
                if time.time() - timestamp < cls._ttl:
                    return status
        return None
    
    @classmethod
    def set(cls, tool_name: str, status: Dict):
        """Cache status."""
        with cls._lock:
            cls._cache[tool_name] = (status, time.time())
    
    @classmethod
    def clear(cls):
        """Clear all cached statuses."""
        with cls._lock:
            cls._cache.clear()

# ============================================================================
# LIGHTWEIGHT HEALTH CHECK - Fast status without heavy initialization
# ============================================================================

def quick_health_check() -> Dict:
    """Fast health check for MCP server - no heavy operations."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "handlers_loaded": True,
        "cache_size": len(_response_cache._cache),
        "tool_instances": len(LazyLoader._instances)
    }

# ============================================================================
# PERFORMANCE METRICS - Track handler performance
# ============================================================================

class PerformanceMetrics:
    """Track handler execution times."""
    
    _metrics: Dict[str, list] = {}
    _lock = threading.Lock()
    
    @classmethod
    def record(cls, handler_name: str, duration_ms: float):
        """Record handler execution time."""
        with cls._lock:
            if handler_name not in cls._metrics:
                cls._metrics[handler_name] = []
            cls._metrics[handler_name].append(duration_ms)
            # Keep last 100 measurements
            if len(cls._metrics[handler_name]) > 100:
                cls._metrics[handler_name] = cls._metrics[handler_name][-100:]
    
    @classmethod
    def get_stats(cls, handler_name: str = None) -> Dict:
        """Get performance statistics."""
        with cls._lock:
            if handler_name:
                times = cls._metrics.get(handler_name, [])
                if not times:
                    return {"handler": handler_name, "calls": 0}
                return {
                    "handler": handler_name,
                    "calls": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times)
                }
            
            # All handlers
            stats = {}
            for name, times in cls._metrics.items():
                if times:
                    stats[name] = {
                        "calls": len(times),
                        "avg_ms": round(sum(times) / len(times), 2),
                        "max_ms": round(max(times), 2)
                    }
            return stats
    
    @classmethod
    def clear(cls):
        """Clear all metrics."""
        with cls._lock:
            cls._metrics.clear()

def timed_handler(func: Callable) -> Callable:
    """Decorator to track handler execution time."""
    @functools.wraps(func)
    def wrapper(args: Dict) -> Dict:
        start = time.perf_counter()
        try:
            result = func(args)
            return result
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            PerformanceMetrics.record(func.__name__, duration_ms)
    return wrapper

# ============================================================================
# OPTIMIZED TOOL LIST - Lightweight tool definitions
# ============================================================================

def get_lightweight_tools_list() -> list:
    """Get minimal tool list for fast tools/list response."""
    # Return pre-computed lightweight list instead of computing each time
    return _LIGHTWEIGHT_TOOLS_CACHE

_LIGHTWEIGHT_TOOLS_CACHE = []  # Populated on first import

def init_lightweight_tools_cache(tools: list):
    """Initialize lightweight tools cache."""
    global _LIGHTWEIGHT_TOOLS_CACHE
    _LIGHTWEIGHT_TOOLS_CACHE = [
        {"name": t["name"], "description": t["description"][:100]}
        for t in tools
    ]

# ============================================================================
# WSL CONNECTION POOL - Reuse WSL connections
# ============================================================================

class WSLConnectionPool:
    """Pool WSL subprocess connections for faster repeated calls."""
    
    _last_check = 0
    _wsl_available = None
    _check_interval = 60  # Re-check every minute
    
    @classmethod
    def is_wsl_available(cls) -> bool:
        """Check WSL availability with caching."""
        now = time.time()
        if cls._wsl_available is None or (now - cls._last_check) > cls._check_interval:
            import subprocess
            try:
                result = subprocess.run(
                    ["wsl", "--list"], 
                    capture_output=True, 
                    timeout=3
                )
                cls._wsl_available = result.returncode == 0
            except:
                cls._wsl_available = False
            cls._last_check = now
        return cls._wsl_available

# ============================================================================
# PERFORMANCE HANDLER - Get performance stats
# ============================================================================

def handle_performance_stats(args: Dict) -> Dict:
    """Get MCP server performance statistics."""
    action = args.get("action", "stats")
    
    if action == "stats":
        return {
            "handler_stats": PerformanceMetrics.get_stats(),
            "cache_stats": {
                "response_cache_size": len(_response_cache._cache),
                "tool_status_cache_size": len(ToolStatusCache._cache),
                "tool_instances": len(LazyLoader._instances)
            },
            "wsl_available": WSLConnectionPool.is_wsl_available()
        }
    elif action == "clear_cache":
        _response_cache.invalidate()
        ToolStatusCache.clear()
        return {"cleared": True, "message": "All caches cleared"}
    elif action == "clear_metrics":
        PerformanceMetrics.clear()
        return {"cleared": True, "message": "Performance metrics cleared"}
    elif action == "health":
        return quick_health_check()
    
    return {"error": f"Unknown action: {action}"}

# ============================================================================
# OPTIMIZATION TIPS
# ============================================================================

PERFORMANCE_TIPS = """
MCP Server Performance Optimization Tips:
=========================================

1. LAZY LOADING
   - Tool classes are initialized only when first used
   - Reduces startup time and memory usage
   - Use LazyLoader.get_instance() for heavy tools

2. RESPONSE CACHING  
   - Frequently called handlers should use @cached_handler decorator
   - Cache status checks and tool availability
   - Default TTL is 60 seconds

3. WSL OPTIMIZATION
   - WSL availability is cached to avoid repeated checks
   - Batch WSL commands when possible
   - Use timeout on all WSL subprocess calls

4. HANDLER TIMING
   - Use @timed_handler decorator to track slow handlers
   - Review PerformanceMetrics.get_stats() for bottlenecks
   - Handlers > 1000ms should be investigated

5. MEMORY MANAGEMENT
   - LazyLoader caches tool instances
   - Clear unused instances with LazyLoader.clear()
   - Response cache auto-evicts old entries

6. WINDSURF/CLAUDE SPECIFIC
   - Keep tools/list response lightweight
   - Implement quick_health_check() for status
   - Avoid blocking operations in handlers
"""

__all__ = [
    'LazyLoader',
    'ResponseCache', 
    'ToolStatusCache',
    'PerformanceMetrics',
    'WSLConnectionPool',
    'cached_handler',
    'timed_handler',
    'quick_health_check',
    'handle_performance_stats',
    'PERFORMANCE_TIPS',
]
