"""
Evony RAG v2 - MCP Server (stdio) with Debug Logging
=====================================================
MCP server using stdio for Windsurf IDE integration.
"""

import sys
import json
import logging
import traceback
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Setup logging to file
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"mcp_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# NO stderr output - Windsurf marks server as Error if ANY stderr output occurs

logger.info("=" * 60)
logger.info("EVONY MCP SERVER STARTING")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Python: {sys.version}")
logger.info(f"Working dir: {Path.cwd()}")
logger.info("=" * 60)


class ProgressIndicator:
    """Progress indicator - logs only, no stderr (Windsurf treats stderr as error)."""
    
    def __init__(self):
        self._task = ""
        self._start_time = 0
        
    def start(self, task: str):
        """Start tracking a task."""
        self._task = task
        self._start_time = time.time()
        logger.info(f"[PROGRESS] Started: {task}")
        
    def stop(self, result: str = "done"):
        """Stop tracking."""
        elapsed = time.time() - self._start_time
        logger.info(f"[PROGRESS] Completed: {self._task} in {elapsed:.1f}s - {result}")
        
    def error(self, msg: str):
        """Log error."""
        elapsed = time.time() - self._start_time
        logger.error(f"[PROGRESS] Error: {self._task} ({elapsed:.1f}s) - {msg}")


# Global progress indicator
progress = ProgressIndicator()

# Lazy imports to speed up startup
_rag = None
_policy = None
_init_error = None

def get_rag():
    """Lazy load RAG engine."""
    global _rag, _init_error
    if _init_error:
        raise _init_error
    if _rag is None:
        try:
            progress.start("Loading RAG engine (first use)")
            logger.info("Loading RAG engine...")
            from .rag_v2 import get_rag_v2
            _rag = get_rag_v2()
            progress.stop(f"{_rag.get_stats().get('chunks', 0)} chunks loaded")
            logger.info("RAG engine loaded successfully")
        except Exception as e:
            _init_error = e
            logger.error(f"Failed to load RAG engine: {e}")
            logger.error(traceback.format_exc())
            raise
    return _rag

def get_policy():
    """Lazy load policy."""
    global _policy
    if _policy is None:
        try:
            logger.info("Loading policy engine...")
            from .policy import get_policy as _get_policy
            _policy = _get_policy()
            logger.info("Policy engine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            logger.error(traceback.format_exc())
            raise
    return _policy


def get_tools() -> List[Dict]:
    """Get MCP tools manifest."""
    logger.debug("get_tools() called")
    return [
        {
            "name": "evony_search",
            "description": "Search Evony knowledge base with hybrid lexical+semantic search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "include": {"type": "array", "items": {"type": "string"}, "description": "Categories to include"},
                    "exclude": {"type": "array", "items": {"type": "string"}, "description": "Categories to exclude"},
                    "k": {"type": "integer", "description": "Number of results (default: 10)"},
                },
                "required": ["query"]
            }
        },
        {
            "name": "evony_answer",
            "description": "Answer question with RAG context and citations.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Question to answer"},
                    "mode": {"type": "string", "enum": ["research", "forensics", "full_access"], "description": "Query mode"},
                    "include": {"type": "array", "items": {"type": "string"}, "description": "Categories to include"},
                    "exclude": {"type": "array", "items": {"type": "string"}, "description": "Categories to exclude"},
                    "k": {"type": "integer", "description": "Number of sources"},
                    "evidence_level": {"type": "string", "enum": ["brief", "normal", "verbose"]},
                    "use_llm": {"type": "boolean", "description": "Use LLM (default: true)"},
                },
                "required": ["question"]
            }
        },
        {
            "name": "evony_open",
            "description": "Open file from knowledge base.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path"},
                    "start_line": {"type": "integer", "description": "Start line"},
                    "end_line": {"type": "integer", "description": "End line"},
                },
                "required": ["path"]
            }
        },
        {
            "name": "evony_symbol",
            "description": "Find symbol definitions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name"},
                },
                "required": ["name"]
            }
        },
        {
            "name": "evony_trace",
            "description": "Multi-hop concept tracing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Starting topic"},
                    "depth": {"type": "integer", "description": "Trace depth (default: 3)"},
                },
                "required": ["topic"]
            }
        },
        {
            "name": "evony_mode",
            "description": "Get or set query mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["research", "forensics", "full_access"]},
                }
            }
        },
        {
            "name": "evony_stats",
            "description": "Get knowledge base statistics.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "evony_reload",
            "description": "Force reload the knowledge base index. Use after updating chunks.json.",
            "inputSchema": {"type": "object", "properties": {}}
        },
    ]


def handle_tool(name: str, args: Dict) -> Dict:
    """Handle MCP tool calls."""
    logger.info(f"Tool call: {name} with args: {json.dumps(args)[:200]}")
    
    try:
        if name == "evony_search":
            progress.start(f"Searching: {args.get('query', '')[:30]}")
            rag = get_rag()
            results = rag.search_only(
                query=args.get("query", ""),
                include=args.get("include"),
                exclude=args.get("exclude"),
                k=args.get("k", 10),
            )
            progress.stop(f"{len(results)} results")
            logger.info(f"Search returned {len(results)} results")
            return {
                "results": [
                    {
                        "file": r.file_path,
                        "lines": f"{r.start_line}-{r.end_line}",
                        "category": r.category,
                        "score": round(r.combined_score, 3),
                        "lexical": round(r.lexical_score, 3),
                        "semantic": round(r.semantic_score, 3),
                        "snippet": r.content[:300],
                    }
                    for r in results[:20]
                ]
            }
        
        elif name == "evony_answer":
            progress.start(f"Answering: {args.get('question', '')[:30]}")
            rag = get_rag()
            response = rag.query(
                query=args.get("question", ""),
                mode=args.get("mode"),
                include=args.get("include"),
                exclude=args.get("exclude"),
                evidence_level=args.get("evidence_level"),
                final_k=args.get("k"),
                use_llm=args.get("use_llm", True),
            )
            progress.stop(f"{len(response.citations)} citations")
            logger.info(f"Answer generated, {len(response.citations)} citations")
            return {
                "answer": response.answer,
                "citations": [
                    {
                        "file": c.file_path,
                        "lines": f"{c.start_line}-{c.end_line}",
                        "category": c.category,
                        "score": round(c.combined_score, 3),
                    }
                    for c in response.citations
                ],
                "mode": response.policy.mode,
                "model": response.model_used,
            }
        
        elif name == "evony_open":
            progress.start(f"Opening: {args.get('path', '')[:40]}")
            rag = get_rag()
            content = rag.get_file(
                path=args.get("path", ""),
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
            )
            if content is None:
                progress.error("File not found")
                return {"error": f"File not found: {args.get('path')}"}
            progress.stop("loaded")
            return {
                "path": args.get("path"),
                "start_line": args.get("start_line", 1),
                "content": content,
            }
        
        elif name == "evony_symbol":
            progress.start(f"Finding symbol: {args.get('name', '')}")
            rag = get_rag()
            results = rag.find_symbol(args.get("name", ""))
            progress.stop(f"{len(results)} occurrences")
            logger.info(f"Symbol lookup: {len(results)} occurrences")
            return {"symbol": args.get("name"), "occurrences": results[:20]}
        
        elif name == "evony_trace":
            progress.start(f"Tracing: {args.get('topic', '')}")
            rag = get_rag()
            results = rag.trace(
                topic=args.get("topic", ""),
                depth=args.get("depth", 3),
            )
            progress.stop(f"{len(results)} hops")
            logger.info(f"Trace: {len(results)} hops")
            return {"topic": args.get("topic"), "trace": results}
        
        elif name == "evony_mode":
            policy = get_policy()
            if "mode" in args:
                success = policy.set_mode(args["mode"])
                logger.info(f"Mode set to: {args['mode']}, success: {success}")
            return {"mode": policy.current_mode, "available": policy.get_modes()}
        
        elif name == "evony_stats":
            progress.start("Getting stats")
            rag = get_rag()
            stats = rag.get_stats()
            progress.stop(f"{stats.get('chunks', 0)} chunks")
            logger.info(f"Stats: {stats}")
            return stats
        
        elif name == "evony_reload":
            progress.start("Reloading index")
            global _rag
            from .hybrid_search import reload_hybrid_search
            import evony_rag.rag_v2 as rag_v2_module
            # Force reload hybrid search
            hs = reload_hybrid_search()
            # Reset RAG singletons to pick up new search
            _rag = None
            rag_v2_module._rag_v2 = None
            rag = get_rag()
            stats = rag.get_stats()
            progress.stop(f"Reloaded {stats.get('chunks', 0)} chunks")
            logger.info(f"Index reloaded: {stats}")
            return {"reloaded": True, "stats": stats}
        
        else:
            logger.warning(f"Unknown tool: {name}")
            return {"error": f"Unknown tool: {name}"}
            
    except Exception as e:
        progress.error(str(e)[:50])
        logger.error(f"Tool error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def handle_request(request: Dict) -> Dict:
    """Handle MCP JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")
    
    logger.debug(f"Request: method={method}, id={req_id}")
    
    try:
        if method == "initialize":
            logger.info("Initialize request received")
            # Match client protocol version
            client_version = params.get("protocolVersion", "2025-03-26")
            logger.info(f"Client protocol: {client_version}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": client_version,
                    "serverInfo": {"name": "evony-knowledge", "version": "2.0.0"},
                    "capabilities": {
                        "tools": {"listChanged": True}
                    }
                }
            }
        
        elif method == "notifications/initialized":
            logger.info("Client initialized notification")
            return None  # No response for notifications
        
        elif method == "tools/list":
            logger.info("Tools list request")
            tools = get_tools()
            logger.info(f"Returning {len(tools)} tools")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools}
            }
        
        elif method == "tools/call":
            logger.info(f"Tool call: {params.get('name')}")
            result = handle_tool(
                params.get("name", ""),
                params.get("arguments", {})
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                }
            }
        
        elif method == "ping":
            logger.debug("Ping received")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            }
        
        else:
            logger.warning(f"Unknown method: {method}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            }
            
    except Exception as e:
        logger.error(f"Request error: {e}")
        logger.error(traceback.format_exc())
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": str(e)}
        }


def run_stdio():
    """Run MCP server with stdio."""
    logger.info("Starting stdio loop...")
    
    request_count = 0
    try:
        # Use explicit readline loop
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    # EOF - stdin closed, exit cleanly
                    logger.info("stdin closed (EOF), shutting down")
                    break
                    
                request_count += 1
                logger.info(f"=== REQUEST #{request_count} ===")
                line = line.strip()
                if not line:
                    continue
            except Exception as read_err:
                logger.error(f"Read error: {read_err}")
                time.sleep(0.1)
                continue
            
            # Strip UTF-8 BOM if present (Windsurf sends this on first message)
            # BOM can appear as \ufeff or as raw bytes ï»¿
            if line.startswith('\ufeff'):
                line = line[1:]
                logger.debug("Stripped unicode BOM from input")
            elif line.startswith('ï»¿'):
                line = line[3:]
                logger.debug("Stripped raw BOM bytes from input")
            # Also try stripping any non-JSON prefix
            if not line.startswith('{'):
                idx = line.find('{')
                if idx > 0:
                    logger.debug(f"Stripped {idx} chars before JSON")
                    line = line[idx:]
            
            logger.debug(f"Received: {line[:200]}...")
            
            try:
                request = json.loads(line)
                response = handle_request(request)
                
                if response is not None:
                    response_str = json.dumps(response)
                    logger.info(f"SENDING response for id={response.get('id')}, len={len(response_str)}")
                    sys.stdout.write(response_str + "\n")
                    sys.stdout.flush()
                    logger.info(f"SENT and FLUSHED response #{request_count}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}, line: {line[:100]}")
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                logger.error(traceback.format_exc())
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": str(e)}
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        raise
    
    logger.info("Server stopped")


def preload_rag_background():
    """Pre-load RAG engine in background thread."""
    try:
        logger.info("Background preload starting...")
        get_rag()
        logger.info("Background preload complete")
    except Exception as e:
        logger.error(f"Failed to preload RAG: {e}")


def main():
    """Main entry point."""
    logger.info("Main entry point")
    # Start RAG preload in background so we can respond to initialize immediately
    preload_thread = threading.Thread(target=preload_rag_background, daemon=True)
    preload_thread.start()
    # Start stdio loop immediately - it will wait for RAG on first tool call
    run_stdio()


if __name__ == "__main__":
    main()
