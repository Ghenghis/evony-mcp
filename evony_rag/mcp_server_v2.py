"""
Evony RAG v2 - Enhanced MCP Server
===================================
Full-featured MCP server with all tools for Windsurf IDE.
"""

import json
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from .config import MCP_HOST, MCP_PORT, DATASET_PATH
from .rag_v2 import get_rag_v2, EvonyRAGv2
from .policy import get_policy
from .precision_rag import get_precision_rag


class EvonyMCPServerV2:
    """Enhanced MCP Server for Windsurf."""
    
    def __init__(self, host: str = MCP_HOST, port: int = MCP_PORT):
        self.host = host
        self.port = port
        self.rag = None
        self.precision_rag = None
        self.policy = get_policy()
        
    def _init_rag(self):
        """Initialize RAG on first use."""
        if self.rag is None:
            self.rag = get_rag_v2()
    
    async def handle_tool(self, name: str, args: Dict) -> Dict:
        """Handle MCP tool calls."""
        self._init_rag()
        
        if name == "evony.search":
            # evony.search(query, include?, exclude?, k?)
            results = self.rag.search_only(
                query=args.get("query", ""),
                include=args.get("include"),
                exclude=args.get("exclude"),
                k=args.get("k", 10),
            )
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
                    for r in results
                ]
            }
        
        elif name == "evony.answer":
            # evony.answer(question, include?, exclude?, k?, evidence_level?, mode?)
            response = self.rag.query(
                query=args.get("question", ""),
                mode=args.get("mode"),
                include=args.get("include"),
                exclude=args.get("exclude"),
                evidence_level=args.get("evidence_level"),
                final_k=args.get("k"),
                use_llm=args.get("use_llm", True),
            )
            
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
                "categories": list(response.policy.include_categories),
                "model": response.model_used,
            }
        
        elif name == "evony.open":
            # evony.open(path, start_line?, end_line?)
            content = self.rag.get_file(
                path=args.get("path", ""),
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
            )
            
            if content is None:
                return {"error": f"File not found: {args.get('path')}"}
            
            return {
                "path": args.get("path"),
                "start_line": args.get("start_line", 1),
                "content": content,
            }
        
        elif name == "evony.symbol":
            # evony.symbol(name)
            results = self.rag.find_symbol(args.get("name", ""))
            return {
                "symbol": args.get("name"),
                "occurrences": results[:20],
            }
        
        elif name == "evony.trace":
            # evony.trace(topic, depth?)
            results = self.rag.trace(
                topic=args.get("topic", ""),
                depth=args.get("depth", 3),
            )
            return {
                "topic": args.get("topic"),
                "trace": results,
            }
        
        elif name == "evony.mode":
            # evony.mode(mode?)
            if "mode" in args:
                success = self.policy.set_mode(args["mode"])
                return {
                    "mode": self.policy.current_mode,
                    "success": success,
                    "available": self.policy.get_modes(),
                }
            else:
                return {
                    "mode": self.policy.current_mode,
                    "available": self.policy.get_modes(),
                }
        
        elif name == "evony.stats":
            # evony.stats()
            return self.rag.get_stats()
        
        elif name == "evony.precision":
            # evony.precision(question, use_cache?) - Verified answer with confidence
            if self.precision_rag is None:
                self.precision_rag = get_precision_rag()
            
            result = self.precision_rag.query(
                question=args.get("question", ""),
                use_cache=args.get("use_cache", True)
            )
            
            return {
                "answer": result.answer,
                "confidence": round(result.confidence, 2),
                "is_grounded": result.is_grounded,
                "citations": [
                    {
                        "file": c.file_path,
                        "lines": f"{c.line_start}-{c.line_end}",
                        "snippet": c.content_snippet[:150],
                        "relevance": round(c.relevance_score, 2),
                    }
                    for c in result.citations
                ],
                "verification": result.verification_notes,
                "stats": {
                    "time_ms": result.retrieval_stats.get("total_time_ms", 0),
                    "strategies": result.retrieval_stats.get("strategies", []),
                }
            }
        
        elif name == "evony.feedback":
            # evony.feedback(question, answer, rating, correction?) - Submit feedback
            if self.precision_rag is None:
                self.precision_rag = get_precision_rag()
            
            success = self.precision_rag.add_feedback(
                query=args.get("question", ""),
                answer=args.get("answer", ""),
                rating=args.get("rating", "partial"),
                correction=args.get("correction")
            )
            
            stats = self.precision_rag.get_feedback_stats()
            return {
                "success": success,
                "stats": stats
            }
        
        else:
            return {"error": f"Unknown tool: {name}"}
    
    def get_tools(self) -> List[Dict]:
        """Get MCP tools manifest."""
        return [
            {
                "name": "evony.search",
                "description": "Search Evony knowledge base. Returns ranked chunks with hybrid lexical+semantic scores.",
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
                "name": "evony.answer",
                "description": "Answer question with RAG context and citations. Uses LM Studio if available.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to answer"},
                        "mode": {"type": "string", "enum": ["research", "forensics", "full_access"], "description": "Query mode"},
                        "include": {"type": "array", "items": {"type": "string"}, "description": "Categories to include"},
                        "exclude": {"type": "array", "items": {"type": "string"}, "description": "Categories to exclude"},
                        "k": {"type": "integer", "description": "Number of sources"},
                        "evidence_level": {"type": "string", "enum": ["brief", "normal", "verbose"]},
                        "use_llm": {"type": "boolean", "description": "Use LM Studio for answer (default: true)"},
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "evony.open",
                "description": "Open file from knowledge base. Returns content with optional line range.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path (from citation)"},
                        "start_line": {"type": "integer", "description": "Start line (optional)"},
                        "end_line": {"type": "integer", "description": "End line (optional)"},
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "evony.symbol",
                "description": "Find symbol (class, function, constant) definitions and usages.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Symbol name (e.g., ACTION_KEY, army.newArmy)"},
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "evony.trace",
                "description": "Multi-hop trace: follow connections between concepts (protocol → handler → key).",
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
                "name": "evony.mode",
                "description": "Get or set query mode (research, forensics, full_access).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["research", "forensics", "full_access"]},
                    }
                }
            },
            {
                "name": "evony.stats",
                "description": "Get knowledge base statistics.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "evony.precision",
                "description": "Get verified answer with confidence score, citations, and hallucination detection. Best for accurate answers.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to answer with verification"},
                        "use_cache": {"type": "boolean", "description": "Use cached answers (default: true)"},
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "evony.feedback",
                "description": "Submit feedback on an answer to improve the system. Use 'correct', 'partial', or 'incorrect' rating.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The original question"},
                        "answer": {"type": "string", "description": "The answer that was given"},
                        "rating": {"type": "string", "enum": ["correct", "partial", "incorrect"], "description": "Rating of the answer"},
                        "correction": {"type": "string", "description": "Correct answer if rating is 'incorrect'"},
                    },
                    "required": ["question", "answer", "rating"]
                }
            },
        ]
    
    async def handle_request(self, request: Dict) -> Dict:
        """Handle MCP JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")
        
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "evony-knowledge-v2", "version": "2.0.0"},
                        "capabilities": {"tools": {}}
                    }
                }
            
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": self.get_tools()}
                }
            
            elif method == "tools/call":
                result = await self.handle_tool(
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
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"}
                }
                
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)}
            }
    
    async def handle_client(self, reader, writer):
        """Handle client connection."""
        addr = writer.get_extra_info('peername')
        print(f"Client connected: {addr}")
        
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                
                try:
                    request = json.loads(line.decode())
                    response = await self.handle_request(request)
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                except json.JSONDecodeError:
                    continue
                    
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"Client disconnected: {addr}")
    
    async def start(self):
        """Start MCP server."""
        print("Initializing Evony Knowledge System v2...")
        self._init_rag()
        
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addr = server.sockets[0].getsockname()
        
        stats = self.rag.get_stats()
        
        print(f"\n{'='*60}")
        print("EVONY KNOWLEDGE MCP SERVER v2")
        print(f"{'='*60}")
        print(f"Address: {addr[0]}:{addr[1]}")
        print(f"Chunks: {stats.get('chunks', 0)}")
        print(f"Symbols: {stats.get('symbols', 0)}")
        print(f"Mode: {stats.get('mode', 'research')}")
        print(f"\nTools: {len(self.get_tools())}")
        for tool in self.get_tools():
            print(f"  - {tool['name']}")
        print(f"\nWindsurf config:")
        print(f'  "evony-knowledge": {{"command": "python", "args": ["-m", "evony_rag.mcp_server_v2"]}}')
        print(f"{'='*60}\n")
        
        async with server:
            await server.serve_forever()


def run_server():
    """Run MCP server."""
    server = EvonyMCPServerV2()
    asyncio.run(server.start())


if __name__ == "__main__":
    run_server()
