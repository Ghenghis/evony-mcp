"""
Evony RAG - MCP Server
=======================
Model Context Protocol server for Windsurf IDE integration.
"""

import json
import asyncio
from typing import Dict, Any, List
from dataclasses import asdict

from .config import MCP_HOST, MCP_PORT, CATEGORIES
from .rag_engine import get_rag, RAGResponse


class EvonyMCPServer:
    """MCP Server for Windsurf IDE integration."""
    
    def __init__(self, host: str = MCP_HOST, port: int = MCP_PORT):
        self.host = host
        self.port = port
        self.rag = get_rag()
        
    def _format_response(self, response: RAGResponse) -> Dict[str, Any]:
        """Format RAG response for MCP."""
        citations = []
        for cit in response.citations:
            citations.append({
                "file": cit.file_path,
                "lines": f"{cit.start_line}-{cit.end_line}",
                "category": cit.category,
                "relevance": f"{cit.relevance:.0%}",
                "snippet": cit.snippet,
            })
        
        return {
            "answer": response.answer,
            "citations": citations,
            "intent": response.query_analysis.intent,
            "categories_searched": response.query_analysis.categories,
            "model": response.model_used,
        }
    
    async def handle_tool_call(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls."""
        
        if tool == "evony_query":
            query = args.get("query", "")
            use_llm = args.get("use_llm", True)
            
            response = self.rag.query(query, use_llm=use_llm)
            return self._format_response(response)
        
        elif tool == "evony_find_files":
            query = args.get("query", "")
            top_k = args.get("top_k", 10)
            
            citations = self.rag.find_files(query, top_k=top_k)
            return {
                "files": [
                    {
                        "file": cit.file_path,
                        "lines": f"{cit.start_line}-{cit.end_line}",
                        "category": cit.category,
                        "relevance": f"{cit.relevance:.0%}",
                    }
                    for cit in citations
                ]
            }
        
        elif tool == "evony_get_file":
            file_path = args.get("file_path", "")
            from .config import DATASET_PATH
            
            full_path = DATASET_PATH / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                return {"content": content, "path": str(full_path)}
            else:
                return {"error": f"File not found: {file_path}"}
        
        elif tool == "evony_stats":
            stats = self.rag.get_stats()
            return {"stats": stats, "categories": CATEGORIES}
        
        elif tool == "evony_categories":
            return {"categories": CATEGORIES}
        
        else:
            return {"error": f"Unknown tool: {tool}"}
    
    def get_tools_manifest(self) -> List[Dict[str, Any]]:
        """Return MCP tools manifest."""
        return [
            {
                "name": "evony_query",
                "description": "Query the Evony knowledge base with RAG. Returns answer with citations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question about Evony (protocol, code, exploits, etc.)"
                        },
                        "use_llm": {
                            "type": "boolean",
                            "description": "Use LM Studio for generation (default: true)",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "evony_find_files",
                "description": "Find relevant files in the Evony knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant files"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "evony_get_file",
                "description": "Get the full content of a file from the Evony knowledge base.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path to file (from citation)"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "evony_stats",
                "description": "Get statistics about the Evony knowledge index.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "evony_categories",
                "description": "List available categories in the knowledge base.",
                "inputSchema": {
                    "type": "object", 
                    "properties": {}
                }
            }
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
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
                        "serverInfo": {
                            "name": "evony-knowledge",
                            "version": "1.0.0"
                        },
                        "capabilities": {
                            "tools": {}
                        }
                    }
                }
            
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": self.get_tools_manifest()
                    }
                }
            
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                
                result = await self.handle_tool_call(tool_name, tool_args)
                
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
                
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def handle_client(self, reader: asyncio.StreamReader, 
                           writer: asyncio.StreamWriter):
        """Handle a client connection."""
        addr = writer.get_extra_info('peername')
        print(f"Client connected: {addr}")
        
        try:
            while True:
                # Read JSON-RPC message
                line = await reader.readline()
                if not line:
                    break
                
                try:
                    request = json.loads(line.decode())
                    response = await self.handle_request(request)
                    
                    response_bytes = (json.dumps(response) + "\n").encode()
                    writer.write(response_bytes)
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
        """Start the MCP server."""
        # Load index first
        print("Loading Evony knowledge index...")
        if not self.rag.load_index():
            print("Building index (first run)...")
            self.rag.build_index()
        
        server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )
        
        addr = server.sockets[0].getsockname()
        print(f"\n{'='*60}")
        print("EVONY KNOWLEDGE MCP SERVER")
        print(f"{'='*60}")
        print(f"Listening on {addr[0]}:{addr[1]}")
        print(f"Tools: {len(self.get_tools_manifest())}")
        print(f"Index: {self.rag.get_stats().get('num_chunks', 0)} chunks")
        print(f"\nAdd to Windsurf mcp_config.json:")
        print(f'  "evony-knowledge": {{"command": "python", "args": ["-m", "evony_rag.mcp_server"]}}')
        print(f"{'='*60}\n")
        
        async with server:
            await server.serve_forever()


def run_server():
    """Run the MCP server."""
    server = EvonyMCPServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    run_server()
