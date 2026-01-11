# Evony MCP Servers v2.0

This repository contains two refactored and production-ready MCP (Model Context Protocol) servers for Evony reverse engineering, knowledge base queries, and real-time engineering tasks. The codebase has been audited, stabilized, and documented to ensure reliability and maintainability.

## Architecture Overview

The system consists of two independent but complementary MCP servers that can be used with compatible clients like Windsurf IDE or Claude.

![Architecture Diagram](architecture.svg)

### Key Components

*   **`evony_rag` (RAG Server):** A Retrieval-Augmented Generation server that provides a knowledge base over the decompiled Evony source code. It uses a hybrid search approach (lexical + semantic) to answer questions, find symbols, and trace concepts through the codebase.
*   **`evony_rte` (Real-Time Engine):** A comprehensive toolkit for real-time protocol analysis, exploit research, and reverse engineering. It integrates with over 60 tools for packet manipulation, client analysis, and game interaction.

## RAG Server Data Flow

The `evony_rag` server follows a sophisticated pipeline to process user queries and generate accurate, cited answers.

![RAG Data Flow](rag_flow.svg)

## Features

### `evony_rag` (RAG Server)

*   **Hybrid Search:** Combines BM25 lexical search and embedding-based semantic search for superior retrieval accuracy.
*   **Policy Engine:** Enforces query modes (`research`, `forensics`, `full_access`) and safety filters to prevent misuse.
*   **RAG-Powered Q&A:** Generates answers to natural language questions with citations pointing to the exact file and line in the source code.
*   **Symbol & Concept Tracing:** Provides tools to find symbol definitions (`evony_symbol`) and trace concepts across multiple files (`evony_trace`).

### `evony_rte` (Real-Time Engine)

*   **Comprehensive Tooling:** Over 60 handlers for packet capture, AMF3 decoding/encoding, exploit testing, and client analysis.
*   **Tool Integration:** Integrates with essential reverse engineering tools like FFDec, Ghidra, Wireshark, and Radare2.
*   **Game Interaction:** Provides tools to connect to the game server, send commands, and monitor game state in real time.
*   **Robust Validation:** Includes a multi-layered validation system for all handlers to ensure stability.

## Installation

1.  **Clone the repository:**

    ```bash
    gh repo clone Ghenghis/evony-mcp
    cd evony-mcp
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the RAG Index (First-time setup):**

    The RAG server requires a pre-built index of the Evony source code. This can be a time-consuming process.

    ```bash
    python -m evony_rag.extract_code_chunks
    python -m evony_rag.build_indices
    ```

## Usage

Both servers are designed to be used with an MCP-compatible client. The following configuration can be used to connect to the servers from the Claude Desktop App.

Add the following to your `%APPDATA%\Claude\claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "evony-knowledge": {
      "command": "python",
      "args": ["-m", "evony_rag"],
      "workingDirectory": "C:\\path\\to\\evony-mcp",
      "timeout": 60000
    },
    "evony-rte": {
      "command": "python", 
      "args": ["-m", "evony_rte"],
      "workingDirectory": "C:\\path\\to\\evony-mcp",
      "timeout": 60000
    }
  }
}
```

## License

This project is licensed under the MIT License.
