"""
Evony RTE - Metrics Dashboard
==============================
Real-time metrics and monitoring for Evony reverse engineering.
Designed for both Claude Desktop and Windsurf MCP integration.

Metrics Categories:
- Handler performance (response times, success rates)
- Exploit testing metrics (dry runs, verifications)
- Packet analysis statistics
- RAG query performance
- Bot server health
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# METRICS STORAGE
# ============================================================================

class MetricsStore:
    """Thread-safe metrics storage with persistence."""
    
    def __init__(self, persist_file: str = None):
        self._lock = threading.RLock()
        self._metrics = {
            'handlers': defaultdict(lambda: {
                'calls': 0,
                'successes': 0,
                'failures': 0,
                'total_time_ms': 0,
                'last_call': None,
                'last_error': None
            }),
            'exploits': {
                'dry_runs': 0,
                'live_tests': 0,
                'verifications': 0,
                'rollbacks': 0,
                'by_type': defaultdict(int)
            },
            'packets': {
                'encoded': 0,
                'decoded': 0,
                'captured': 0,
                'injected': 0,
                'bytes_processed': 0
            },
            'rag': {
                'queries': 0,
                'cache_hits': 0,
                'avg_results': 0,
                'by_category': defaultdict(int)
            },
            'bot_server': {
                'connections': 0,
                'commands_sent': 0,
                'responses_received': 0,
                'errors': 0,
                'uptime_start': None
            },
            'system': {
                'start_time': datetime.now().isoformat(),
                'total_requests': 0,
                'errors': 0
            }
        }
        self._persist_file = persist_file or str(Path(__file__).parent / "metrics_data.json")
        self._load_metrics()
    
    def _load_metrics(self):
        """Load persisted metrics if available."""
        try:
            if Path(self._persist_file).exists():
                with open(self._persist_file, 'r') as f:
                    saved = json.load(f)
                    # Merge with defaults (keep structure, update values)
                    for category in saved:
                        if category in self._metrics:
                            if isinstance(self._metrics[category], dict):
                                self._metrics[category].update(saved[category])
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
    
    def _save_metrics(self):
        """Persist metrics to file."""
        try:
            # Convert defaultdicts to regular dicts for JSON
            data = {}
            for k, v in self._metrics.items():
                if isinstance(v, defaultdict):
                    data[k] = dict(v)
                elif isinstance(v, dict):
                    data[k] = {kk: dict(vv) if isinstance(vv, defaultdict) else vv 
                              for kk, vv in v.items()}
                else:
                    data[k] = v
            with open(self._persist_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save metrics: {e}")
    
    def record_handler_call(self, handler_name: str, success: bool, duration_ms: float, error: str = None):
        """Record a handler call."""
        with self._lock:
            h = self._metrics['handlers'][handler_name]
            h['calls'] += 1
            h['total_time_ms'] += duration_ms
            h['last_call'] = datetime.now().isoformat()
            if success:
                h['successes'] += 1
            else:
                h['failures'] += 1
                h['last_error'] = error
            self._metrics['system']['total_requests'] += 1
            if not success:
                self._metrics['system']['errors'] += 1
    
    def record_exploit_test(self, exploit_type: str, dry_run: bool, verified: bool = False, rolled_back: bool = False):
        """Record exploit testing activity."""
        with self._lock:
            e = self._metrics['exploits']
            if dry_run:
                e['dry_runs'] += 1
            else:
                e['live_tests'] += 1
            if verified:
                e['verifications'] += 1
            if rolled_back:
                e['rollbacks'] += 1
            e['by_type'][exploit_type] += 1
    
    def record_packet_activity(self, action: str, byte_count: int = 0):
        """Record packet processing activity."""
        with self._lock:
            p = self._metrics['packets']
            if action in p:
                p[action] += 1
            p['bytes_processed'] += byte_count
    
    def record_rag_query(self, category: str, result_count: int, cache_hit: bool = False):
        """Record RAG query activity."""
        with self._lock:
            r = self._metrics['rag']
            r['queries'] += 1
            if cache_hit:
                r['cache_hits'] += 1
            r['by_category'][category] += 1
            # Update rolling average
            r['avg_results'] = (r['avg_results'] * (r['queries'] - 1) + result_count) / r['queries']
    
    def record_bot_activity(self, action: str):
        """Record bot server activity."""
        with self._lock:
            b = self._metrics['bot_server']
            if action == 'connect':
                b['connections'] += 1
                if not b['uptime_start']:
                    b['uptime_start'] = datetime.now().isoformat()
            elif action == 'send':
                b['commands_sent'] += 1
            elif action == 'receive':
                b['responses_received'] += 1
            elif action == 'error':
                b['errors'] += 1
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics."""
        with self._lock:
            self._save_metrics()
            return json.loads(json.dumps(self._metrics, default=str))
    
    def get_handler_metrics(self) -> Dict:
        """Get handler-specific metrics."""
        with self._lock:
            handlers = {}
            for name, data in self._metrics['handlers'].items():
                avg_time = data['total_time_ms'] / data['calls'] if data['calls'] > 0 else 0
                success_rate = (data['successes'] / data['calls'] * 100) if data['calls'] > 0 else 0
                handlers[name] = {
                    **data,
                    'avg_time_ms': round(avg_time, 2),
                    'success_rate': round(success_rate, 1)
                }
            return handlers
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.__init__(self._persist_file)


# Global metrics store
_metrics_store = None

def get_metrics_store() -> MetricsStore:
    """Get the global metrics store."""
    global _metrics_store
    if _metrics_store is None:
        _metrics_store = MetricsStore()
    return _metrics_store


# ============================================================================
# EVONY-SPECIFIC METRICS
# ============================================================================

class EvonyMetrics:
    """Evony-specific metrics for game analysis."""
    
    def __init__(self):
        self.store = get_metrics_store()
        self._evony_data = {
            'commands_analyzed': defaultdict(int),  # city.getInfo: 5
            'protocols_traced': defaultdict(int),   # AMF3: 100
            'vulnerabilities': {
                'found': 0,
                'tested': 0,
                'confirmed': 0,
                'by_category': defaultdict(int)
            },
            'client_analysis': {
                'classes_decompiled': 0,
                'functions_analyzed': 0,
                'strings_extracted': 0,
                'callgraphs_generated': 0
            },
            'game_state': {
                'resources_tracked': 0,
                'troops_monitored': 0,
                'cities_analyzed': 0
            }
        }
    
    def record_command_analysis(self, command: str):
        """Record analysis of a game command."""
        self._evony_data['commands_analyzed'][command] += 1
    
    def record_protocol_trace(self, protocol: str = 'AMF3'):
        """Record protocol tracing."""
        self._evony_data['protocols_traced'][protocol] += 1
    
    def record_vulnerability(self, category: str, tested: bool = False, confirmed: bool = False):
        """Record vulnerability discovery."""
        v = self._evony_data['vulnerabilities']
        v['found'] += 1
        v['by_category'][category] += 1
        if tested:
            v['tested'] += 1
        if confirmed:
            v['confirmed'] += 1
    
    def record_client_analysis(self, analysis_type: str, count: int = 1):
        """Record client code analysis."""
        if analysis_type in self._evony_data['client_analysis']:
            self._evony_data['client_analysis'][analysis_type] += count
    
    def record_game_state(self, state_type: str, count: int = 1):
        """Record game state tracking."""
        if state_type in self._evony_data['game_state']:
            self._evony_data['game_state'][state_type] += count
    
    def get_evony_metrics(self) -> Dict:
        """Get all Evony-specific metrics."""
        return {
            'commands_analyzed': dict(self._evony_data['commands_analyzed']),
            'protocols_traced': dict(self._evony_data['protocols_traced']),
            'vulnerabilities': {
                **self._evony_data['vulnerabilities'],
                'by_category': dict(self._evony_data['vulnerabilities']['by_category'])
            },
            'client_analysis': self._evony_data['client_analysis'],
            'game_state': self._evony_data['game_state']
        }


# Global Evony metrics
_evony_metrics = None

def get_evony_metrics() -> EvonyMetrics:
    """Get the global Evony metrics."""
    global _evony_metrics
    if _evony_metrics is None:
        _evony_metrics = EvonyMetrics()
    return _evony_metrics


# ============================================================================
# DASHBOARD HANDLERS (MCP Integration)
# ============================================================================

def handle_metrics_dashboard(args: Dict) -> Dict:
    """
    Get metrics dashboard for Evony RTE.
    
    Parameters:
        action: 'summary' | 'handlers' | 'exploits' | 'packets' | 'rag' | 'evony' | 'all' | 'reset'
        handler: Optional handler name for detailed metrics
        format: 'json' | 'text' | 'html' (default: json)
    
    Returns:
        Metrics data based on action
    """
    action = args.get('action', 'summary')
    handler_name = args.get('handler')
    output_format = args.get('format', 'json')
    
    store = get_metrics_store()
    evony = get_evony_metrics()
    
    if action == 'reset':
        store.reset_metrics()
        return {'status': 'reset', 'message': 'All metrics have been reset'}
    
    if action == 'handlers':
        metrics = store.get_handler_metrics()
        if handler_name:
            return metrics.get(handler_name, {'error': f'Handler {handler_name} not found'})
        return {'handlers': metrics, 'total': len(metrics)}
    
    if action == 'exploits':
        return store.get_all_metrics()['exploits']
    
    if action == 'packets':
        return store.get_all_metrics()['packets']
    
    if action == 'rag':
        return store.get_all_metrics()['rag']
    
    if action == 'evony':
        return evony.get_evony_metrics()
    
    if action == 'all':
        return {
            'system': store.get_all_metrics(),
            'evony': evony.get_evony_metrics(),
            'timestamp': datetime.now().isoformat()
        }
    
    # Default: summary
    all_metrics = store.get_all_metrics()
    handler_metrics = store.get_handler_metrics()
    
    # Calculate top handlers
    top_handlers = sorted(
        handler_metrics.items(),
        key=lambda x: x[1]['calls'],
        reverse=True
    )[:10]
    
    # Calculate slowest handlers
    slowest_handlers = sorted(
        [(k, v) for k, v in handler_metrics.items() if v['calls'] > 0],
        key=lambda x: x[1]['avg_time_ms'],
        reverse=True
    )[:5]
    
    summary = {
        'overview': {
            'total_requests': all_metrics['system']['total_requests'],
            'total_errors': all_metrics['system']['errors'],
            'error_rate': round(all_metrics['system']['errors'] / max(all_metrics['system']['total_requests'], 1) * 100, 2),
            'uptime_since': all_metrics['system']['start_time'],
            'handlers_active': len([h for h in handler_metrics.values() if h['calls'] > 0])
        },
        'top_handlers': [
            {'name': name, 'calls': data['calls'], 'success_rate': data['success_rate']}
            for name, data in top_handlers
        ],
        'slowest_handlers': [
            {'name': name, 'avg_time_ms': data['avg_time_ms'], 'calls': data['calls']}
            for name, data in slowest_handlers
        ],
        'exploit_activity': {
            'dry_runs': all_metrics['exploits']['dry_runs'],
            'live_tests': all_metrics['exploits']['live_tests'],
            'verifications': all_metrics['exploits']['verifications']
        },
        'packet_stats': {
            'encoded': all_metrics['packets']['encoded'],
            'decoded': all_metrics['packets']['decoded'],
            'total_bytes': all_metrics['packets']['bytes_processed']
        },
        'evony_analysis': evony.get_evony_metrics()['client_analysis']
    }
    
    if output_format == 'text':
        return _format_summary_text(summary)
    
    return summary


def _format_summary_text(summary: Dict) -> Dict:
    """Format summary as readable text."""
    lines = [
        "=" * 60,
        "EVONY RTE METRICS DASHBOARD",
        "=" * 60,
        "",
        "📊 OVERVIEW",
        f"  Total Requests: {summary['overview']['total_requests']}",
        f"  Error Rate: {summary['overview']['error_rate']}%",
        f"  Active Handlers: {summary['overview']['handlers_active']}",
        f"  Uptime Since: {summary['overview']['uptime_since']}",
        "",
        "🔝 TOP HANDLERS",
    ]
    
    for h in summary['top_handlers']:
        lines.append(f"  {h['name']}: {h['calls']} calls ({h['success_rate']}% success)")
    
    lines.extend([
        "",
        "🐢 SLOWEST HANDLERS",
    ])
    
    for h in summary['slowest_handlers']:
        lines.append(f"  {h['name']}: {h['avg_time_ms']}ms avg ({h['calls']} calls)")
    
    lines.extend([
        "",
        "💥 EXPLOIT ACTIVITY",
        f"  Dry Runs: {summary['exploit_activity']['dry_runs']}",
        f"  Live Tests: {summary['exploit_activity']['live_tests']}",
        f"  Verifications: {summary['exploit_activity']['verifications']}",
        "",
        "📦 PACKET STATS",
        f"  Encoded: {summary['packet_stats']['encoded']}",
        f"  Decoded: {summary['packet_stats']['decoded']}",
        f"  Bytes Processed: {summary['packet_stats']['total_bytes']:,}",
        "",
        "=" * 60,
    ])
    
    return {'text': '\n'.join(lines), 'summary': summary}


# ============================================================================
# METRICS DECORATOR (Auto-instrument handlers)
# ============================================================================

def metrics_tracked(handler_name: str = None):
    """Decorator to auto-track metrics for handlers."""
    def decorator(func):
        def wrapper(args):
            name = handler_name or func.__name__
            store = get_metrics_store()
            start = time.time()
            
            try:
                result = func(args)
                duration = (time.time() - start) * 1000
                success = 'error' not in result if isinstance(result, dict) else True
                store.record_handler_call(name, success, duration)
                
                # Track Evony-specific metrics
                evony = get_evony_metrics()
                if 'cmd' in args:
                    evony.record_command_analysis(args['cmd'])
                if name.startswith('exploit'):
                    evony.record_vulnerability(
                        args.get('exploit_id', 'unknown'),
                        tested=name == 'exploit_test',
                        confirmed=result.get('verified', False) if isinstance(result, dict) else False
                    )
                
                return result
            except Exception as e:
                duration = (time.time() - start) * 1000
                store.record_handler_call(name, False, duration, str(e))
                raise
        
        return wrapper
    return decorator


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'MetricsStore',
    'get_metrics_store',
    'EvonyMetrics', 
    'get_evony_metrics',
    'handle_metrics_dashboard',
    'metrics_tracked'
]
