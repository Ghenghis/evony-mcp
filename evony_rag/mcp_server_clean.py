"""
Evony RAG v2 - Clean MCP Server (stdio)
=======================================
CRITICAL: No stderr output - Windsurf treats stderr as error.
"""

import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Setup logging to FILE ONLY - NO STDERR
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"mcp_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8')]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("EVONY MCP SERVER STARTING (clean)")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Python: {sys.version}")
logger.info("=" * 60)

# Lazy load RAG
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        logger.info("Loading RAG engine...")
        from .rag_v2 import get_rag_v2
        _rag = get_rag_v2()
        logger.info(f"RAG loaded: {_rag.get_stats().get('chunks', 0)} chunks")
    return _rag

def get_tools() -> List[Dict]:
    """Return tool definitions."""
    return [
        {
            "name": "evony_search",
            "description": "Search Evony knowledge base with hybrid lexical+semantic search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Number of results (default: 10)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "evony_stats",
            "description": "Get knowledge base statistics.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "evony_mode",
            "description": "Get or set query mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["research", "forensics", "full_access"]}
                }
            }
        }
    ]

def handle_tool_call(name: str, args: Dict) -> Dict:
    """Execute a tool call."""
    logger.info(f"Tool call: {name} with args: {args}")
    
    try:
        if name == "evony_stats":
            rag = get_rag()
            stats = rag.get_stats()
            return {"content": [{"type": "text", "text": json.dumps(stats, indent=2)}]}
        
        elif name == "evony_search":
            rag = get_rag()
            query = args.get("query", "")
            k = args.get("k", 10)
            results = rag.search_only(query, k=k)
            # Format results for output
            output = []
            for r in results:
                output.append({
                    "file": r.file_path,
                    "lines": f"{r.start_line}-{r.end_line}",
                    "category": r.category,
                    "score": round(r.combined_score, 3),
                    "snippet": r.content[:200] if r.content else ""
                })
            return {"content": [{"type": "text", "text": json.dumps(output, indent=2)}]}
        
        elif name == "evony_mode":
            rag = get_rag()
            mode = args.get("mode")
            if mode:
                success = rag.policy.set_mode(mode)
                if success:
                    return {"content": [{"type": "text", "text": f"Mode set to: {mode}"}]}
                else:
                    return {"content": [{"type": "text", "text": f"Invalid mode: {mode}"}], "isError": True}
            else:
                return {"content": [{"type": "text", "text": f"Current mode: {rag.policy.current_mode}"}]}
        
        else:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}
    
    except Exception as e:
        logger.error(f"Tool error: {e}\n{traceback.format_exc()}")
        return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}

def send_response(response: Dict):
    """Send JSON-RPC response to stdout."""
    line = json.dumps(response)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    logger.debug(f"Sent: {line[:200]}...")

def handle_request(request: Dict) -> Dict:
    """Process a JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})
    
    logger.info(f"Request: method={method}, id={req_id}")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "serverInfo": {"name": "evony-knowledge", "version": "2.1.0"},
                "capabilities": {"tools": {"listChanged": True}}
            }
        }
    
    elif method == "notifications/initialized":
        return None  # No response for notifications
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": get_tools()}
        }
    
    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        result = handle_tool_call(tool_name, tool_args)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result
        }
    
    else:
        logger.warning(f"Unknown method: {method}")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }

def main():
    """Main stdio loop."""
    logger.info("Starting stdio loop...")
    request_num = 0
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            # Remove BOM if present (unicode or raw bytes)
            if line.startswith('\ufeff'):
                line = line[1:]
            elif line.startswith('ï»¿'):
                line = line[3:]
            # Strip any non-JSON prefix
            if line and not line.startswith('{'):
                idx = line.find('{')
                if idx > 0:
                    line = line[idx:]
            
            request_num += 1
            logger.info(f"=== REQUEST #{request_num} ===")
            logger.debug(f"Received: {line[:200]}...")
            
            try:
                request = json.loads(line)
                response = handle_request(request)
                if response:
                    send_response(response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                send_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                })
    
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
    
    logger.info("Server stopped")

if __name__ == "__main__":
    main()
