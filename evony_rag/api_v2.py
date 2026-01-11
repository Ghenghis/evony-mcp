"""
Evony RAG v2 - Enhanced API Server
===================================
OpenAI-compatible + dedicated RAG endpoints.
"""

import json
import time
import uuid
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from .rag_v2 import get_rag_v2
from .policy import get_policy


class EvonyAPIv2Handler(BaseHTTPRequestHandler):
    """HTTP handler for v2 API."""
    
    rag = None
    policy = None
    
    def log_message(self, format, *args):
        pass
    
    def send_json(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [{"id": "evony-rag-v2", "object": "model", "created": int(time.time())}]
            })
        elif path == "/health":
            self.send_json({"status": "healthy", "version": "2.0"})
        elif path == "/stats":
            self.send_json(self.rag.get_stats())
        elif path == "/modes":
            self.send_json({
                "current": self.policy.current_mode,
                "available": self.policy.get_modes(),
            })
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"
        
        try:
            data = json.loads(body)
        except:
            self.send_json({"error": "Invalid JSON"}, 400)
            return
        
        if path == "/v1/chat/completions":
            self._handle_chat(data)
        elif path == "/v1/rag/search":
            self._handle_search(data)
        elif path == "/v1/rag/answer":
            self._handle_answer(data)
        elif path == "/v1/rag/symbol":
            self._handle_symbol(data)
        elif path == "/v1/rag/trace":
            self._handle_trace(data)
        elif path == "/v1/rag/open":
            self._handle_open(data)
        elif path == "/mode":
            self._handle_mode(data)
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def _handle_chat(self, data: Dict):
        """OpenAI-compatible chat completion."""
        messages = data.get("messages", [])
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
        
        if not user_msg:
            self.send_json({"error": "No user message"}, 400)
            return
        
        response = self.rag.query(
            query=user_msg,
            mode=data.get("mode"),
            include=data.get("include"),
            exclude=data.get("exclude"),
            use_llm=True,
        )
        
        # Format citations
        citations = "\n\n---\n**Sources:**\n" if response.citations else ""
        for c in response.citations:
            citations += f"- `{c.file_path}:{c.start_line}-{c.end_line}` ({c.combined_score:.0%})\n"
        
        full_answer = response.answer + citations
        
        self.send_json({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "evony-rag-v2",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_answer},
                "finish_reason": "stop"
            }],
        })
    
    def _handle_search(self, data: Dict):
        """Retrieval-only endpoint."""
        results = self.rag.search_only(
            query=data.get("query", ""),
            include=data.get("include"),
            exclude=data.get("exclude"),
            k=data.get("k", 10),
        )
        
        self.send_json({
            "results": [
                {
                    "file": r.file_path,
                    "lines": f"{r.start_line}-{r.end_line}",
                    "category": r.category,
                    "score": round(r.combined_score, 3),
                    "lexical": round(r.lexical_score, 3),
                    "semantic": round(r.semantic_score, 3),
                    "snippet": r.content[:500],
                }
                for r in results
            ]
        })
    
    def _handle_answer(self, data: Dict):
        """Answer with citations."""
        response = self.rag.query(
            query=data.get("question", data.get("query", "")),
            mode=data.get("mode"),
            include=data.get("include"),
            exclude=data.get("exclude"),
            evidence_level=data.get("evidence_level"),
            final_k=data.get("k"),
            use_llm=data.get("use_llm", True),
        )
        
        self.send_json({
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
        })
    
    def _handle_symbol(self, data: Dict):
        """Symbol lookup."""
        results = self.rag.find_symbol(data.get("name", ""))
        self.send_json({"symbol": data.get("name"), "occurrences": results[:20]})
    
    def _handle_trace(self, data: Dict):
        """Multi-hop trace."""
        results = self.rag.trace(
            topic=data.get("topic", ""),
            depth=data.get("depth", 3),
        )
        self.send_json({"topic": data.get("topic"), "trace": results})
    
    def _handle_open(self, data: Dict):
        """Get file content."""
        content = self.rag.get_file(
            path=data.get("path", ""),
            start_line=data.get("start_line"),
            end_line=data.get("end_line"),
        )
        if content:
            self.send_json({"path": data.get("path"), "content": content})
        else:
            self.send_json({"error": "File not found"}, 404)
    
    def _handle_mode(self, data: Dict):
        """Set mode."""
        if "mode" in data:
            self.policy.set_mode(data["mode"])
        self.send_json({
            "mode": self.policy.current_mode,
            "available": self.policy.get_modes(),
        })


def run_api(host: str = "localhost", port: int = 8766):
    """Run API server."""
    print("Initializing Evony Knowledge API v2...")
    
    rag = get_rag_v2()
    policy = get_policy()
    
    EvonyAPIv2Handler.rag = rag
    EvonyAPIv2Handler.policy = policy
    
    server = HTTPServer((host, port), EvonyAPIv2Handler)
    stats = rag.get_stats()
    
    print(f"\n{'='*60}")
    print("EVONY KNOWLEDGE API v2")
    print(f"{'='*60}")
    print(f"URL: http://{host}:{port}")
    print(f"Chunks: {stats.get('chunks', 0)}")
    print(f"Symbols: {stats.get('symbols', 0)}")
    print(f"Mode: {stats.get('mode', 'research')}")
    print(f"\nEndpoints:")
    print(f"  POST /v1/chat/completions  - OpenAI-compatible")
    print(f"  POST /v1/rag/search        - Retrieval only")
    print(f"  POST /v1/rag/answer        - Answer + citations")
    print(f"  POST /v1/rag/symbol        - Symbol lookup")
    print(f"  POST /v1/rag/trace         - Multi-hop trace")
    print(f"  POST /v1/rag/open          - Get file content")
    print(f"  GET  /modes                - List modes")
    print(f"  POST /mode                 - Set mode")
    print(f"\nLM Studio: Set API base to http://{host}:{port}/v1")
    print(f"{'='*60}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_api()
