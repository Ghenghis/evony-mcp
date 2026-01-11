# Evony MCP Servers

Two MCP (Model Context Protocol) servers for Evony reverse engineering and knowledge base queries.

## Servers

### evony-knowledge (RAG)
Hybrid search knowledge base with 339,160 code chunks from decompiled Evony AS3 source.

**Tools:**
- `evony_search` - Hybrid lexical+semantic search
- `evony_answer` - RAG-powered Q&A with citations
- `evony_trace` - Multi-hop concept tracing
- `evony_symbol` - Find symbol definitions
- `evony_open` - Open file from knowledge base
- `evony_stats` - Knowledge base statistics
- `evony_mode` - Set query mode (research/forensics/full_access)
- `evony_reload` - Force reload index

### evony-rte (Real-Time Engine)
62-handler toolkit for protocol analysis, exploit research, and reverse engineering.

**Categories:**
- Protocol analysis (AMF3, packets)
- Exploit scanning and testing
- LM Studio integration
- Bot server management
- Advanced RE tools

## Installation

```bash
pip install -r requirements.txt
```

## Claude Desktop Config

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "evony-knowledge": {
      "command": "cmd",
      "args": ["/c", "cd", "/d", "C:\\path\\to\\evony-mcp", "&&", "python", "-m", "evony_rag.mcp_server_v2_stdio"],
      "timeout": 60000
    },
    "evony-rte": {
      "command": "cmd", 
      "args": ["/c", "cd", "/d", "C:\\path\\to\\evony-mcp", "&&", "python", "-m", "evony_rte.mcp_server"],
      "timeout": 60000
    }
  }
}
```

## Building the Index

The RAG server requires an index. To build:

```bash
python -m evony_rag.extract_code_chunks
python -m evony_rag.build_indices
```

## Requirements

- Python 3.10+
- See requirements.txt for dependencies

## License

MIT
