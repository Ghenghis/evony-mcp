"""
Evony RAG - LM Studio Compatible API
=====================================
Provides OpenAI-compatible API that integrates RAG with LM Studio.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .config import LMSTUDIO_URL
from .rag_engine import get_rag


class EvonyAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible API with RAG."""
    
    rag = None
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        
        if path == "/v1/models":
            self.send_json({
                "object": "list",
                "data": [
                    {
                        "id": "evony-rag",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "evony-knowledge"
                    }
                ]
            })
        
        elif path == "/health":
            stats = self.rag.get_stats() if self.rag else {}
            self.send_json({
                "status": "healthy",
                "index_loaded": self.rag.index_loaded if self.rag else False,
                "chunks": stats.get("num_chunks", 0)
            })
        
        elif path == "/stats":
            stats = self.rag.get_stats() if self.rag else {}
            self.send_json(stats)
        
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path
        
        # Read body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return
        
        if path == "/v1/chat/completions":
            self.handle_chat_completion(data)
        
        elif path == "/v1/completions":
            self.handle_completion(data)
        
        elif path == "/query":
            self.handle_query(data)
        
        elif path == "/find":
            self.handle_find(data)
        
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def handle_chat_completion(self, data: Dict):
        """Handle chat completion request (OpenAI-compatible)."""
        messages = data.get("messages", [])
        
        # Extract user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            self.send_json({"error": "No user message"}, 400)
            return
        
        # Query RAG
        response = self.rag.query(user_message, use_llm=True)
        
        # Format citations
        citations_text = ""
        if response.citations:
            citations_text = "\n\n---\n**Sources:**\n"
            for cit in response.citations:
                citations_text += f"- `{cit.file_path}:{cit.start_line}-{cit.end_line}` ({cit.relevance:.0%})\n"
        
        full_answer = response.answer + citations_text
        
        # OpenAI-compatible response
        self.send_json({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "evony-rag",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(full_answer.split()),
                "total_tokens": len(user_message.split()) + len(full_answer.split())
            }
        })
    
    def handle_completion(self, data: Dict):
        """Handle legacy completion request."""
        prompt = data.get("prompt", "")
        
        response = self.rag.query(prompt, use_llm=True)
        
        self.send_json({
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "evony-rag",
            "choices": [
                {
                    "text": response.answer,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        })
    
    def handle_query(self, data: Dict):
        """Handle direct RAG query."""
        query = data.get("query", "")
        use_llm = data.get("use_llm", True)
        
        response = self.rag.query(query, use_llm=use_llm)
        
        self.send_json({
            "answer": response.answer,
            "citations": [
                {
                    "file": c.file_path,
                    "lines": f"{c.start_line}-{c.end_line}",
                    "category": c.category,
                    "relevance": c.relevance,
                    "snippet": c.snippet
                }
                for c in response.citations
            ],
            "intent": response.query_analysis.intent,
            "model": response.model_used
        })
    
    def handle_find(self, data: Dict):
        """Handle file search."""
        query = data.get("query", "")
        top_k = data.get("top_k", 10)
        
        citations = self.rag.find_files(query, top_k=top_k)
        
        self.send_json({
            "files": [
                {
                    "file": c.file_path,
                    "lines": f"{c.start_line}-{c.end_line}",
                    "category": c.category,
                    "relevance": c.relevance
                }
                for c in citations
            ]
        })


def run_api_server(host: str = "localhost", port: int = 8766):
    """Run the API server."""
    # Initialize RAG
    rag = get_rag()
    print("Loading Evony knowledge index...")
    
    if not rag.load_index():
        print("Building index (first run)...")
        rag.build_index()
    
    EvonyAPIHandler.rag = rag
    
    server = HTTPServer((host, port), EvonyAPIHandler)
    
    print(f"\n{'='*60}")
    print("EVONY KNOWLEDGE API SERVER")
    print(f"{'='*60}")
    print(f"Listening on http://{host}:{port}")
    print(f"Index: {rag.get_stats().get('num_chunks', 0)} chunks")
    print(f"\nEndpoints:")
    print(f"  POST /v1/chat/completions  - OpenAI-compatible chat")
    print(f"  POST /query                - Direct RAG query")
    print(f"  POST /find                 - Find relevant files")
    print(f"  GET  /health               - Health check")
    print(f"  GET  /stats                - Index statistics")
    print(f"\nLM Studio: Set API base to http://{host}:{port}/v1")
    print(f"{'='*60}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_api_server()
