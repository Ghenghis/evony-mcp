# Evony MCP Servers: Code Audit and Action Plan

## 1. Introduction

This document outlines a comprehensive action plan to address the findings from a detailed code audit of the `evony-mcp` repository. The audit covered both the `evony_rag` (RAG) and `evony_rte` (RTE) MCP servers, identifying issues related to stability, code quality, maintainability, and consistency. The primary goal of this plan is to create a stable, production-ready codebase with robust error handling, consistent design patterns, and comprehensive documentation.

## 2. High-Priority Issues (Must-Fix for Stability)

These issues directly impact the stability and functionality of the servers and should be addressed first.

| Issue ID | Description | Server(s) | Recommended Action |
|---|---|---|---|
| **HP-1** | **Handler Mismatches** | `evony_rte` | Reconcile the `HANDLERS` dictionary with the defined `handle_*` functions. Ensure all registered handlers have a corresponding implementation and that all implemented handlers are registered. |
| **HP-2** | **Hardcoded Paths** | Both | Replace all hardcoded local paths (e.g., `C:\Users\Admin`, `G:\`) with relative paths or a centralized configuration system. This is critical for portability and preventing runtime errors. |
| **HP-3** | **Inconsistent MCP Protocol** | Both | Standardize the MCP protocol version across both servers. The `evony_rte` server uses `2025-06-18` while `evony_rag` uses `2024-11-05`. A single, current version should be used. |
| **HP-4** | **Bare `except` Clauses** | Both | Replace all bare `except:` clauses with specific exception types (e.g., `except Exception as e:`). This will prevent the code from catching system-exiting exceptions and will improve error logging. |

## 3. Medium-Priority Issues (Code Quality & Maintainability)

These issues affect the long-term health of the codebase, making it harder to debug, maintain, and extend.

| Issue ID | Description | Server(s) | Recommended Action |
|---|---|---|---|
| **MP-1** | **Inconsistent Logging** | Both | Replace all `print()` statements with a structured logger (like the one already used in parts of the codebase). This will provide consistent, filterable, and production-ready logging. |
| **MP-2** | **Missing `__all__` Exports** | Both | Add `__all__` to all modules that are part of the public API of each package. This will prevent unintended symbols from being imported and will make the module APIs explicit. |
| **MP-3** | **Low Type Hint Coverage** | `evony_rag` | Add type hints to all function signatures in the `evony_rag` server. The `evony_rte` server has excellent coverage (94%), and `evony_rag` should be brought to the same standard (currently 53%). |
| **MP-4** | **Duplicate Functionality** | Both | Refactor common functionality (e.g., `handle_tool`, `main`) into a shared module to avoid code duplication and ensure consistency. |

## 4. Low-Priority Issues (Best Practices & Future-Proofing)

These are minor issues that should be addressed to improve the overall quality and professionalism of the project.

| Issue ID | Description | Server(s) | Recommended Action |
|---|---|---|---|
| **LP-1** | **Missing `__main__.py`** | `evony_rag` | Create a `__main__.py` file for the `evony_rag` package to provide a standard entry point for running the server. |
| **LP-2** | **Inconsistent Error Responses** | Both | Standardize the error response format for all tool handlers. Both servers use a mix of `{"error": ...}` and `raise Exception`, which should be unified. |

## 5. Proposed Refactoring & Unification

To address the systemic issues of code duplication and inconsistency, the following refactoring is proposed:

1.  **Create a `common` module:** A new top-level `evony_common` package should be created to house shared code, such as:
    *   Configuration management (loading from a single `config.toml` file).
    *   MCP server base class (handling the request/response loop).
    *   Logging setup.
    *   Utility functions.

2.  **Unify Configuration:** Both servers should load their configuration from a single, shared `config.toml` file. This will eliminate hardcoded paths and make configuration management easier.

3.  **Standardize MCP Server Implementation:** Both `evony_rag` and `evony_rte` should inherit from the same `MCPServer` base class from the `evony_common` module.

## 6. Documentation & Diagramming Plan

To improve the project's clarity and maintainability, the following documentation and diagrams will be created:

1.  **Architecture Diagram:** An SVG diagram showing the high-level architecture of the two MCP servers, their key components, and their interactions.
2.  **Data Flow Diagram:** An SVG diagram illustrating the data flow for a typical RAG query, from the initial request to the final response.
3.  **README Updates:** The main `README.md` will be updated to include:
    *   A clear project overview.
    *   Detailed setup and installation instructions.
    *   Usage examples for both servers.
    *   The generated architecture and data flow diagrams.

## 7. Conclusion

By implementing this action plan, the `evony-mcp` project will be transformed into a stable, robust, and maintainable codebase. The proposed changes will improve reliability, reduce technical debt, and make it easier for developers to contribute to the project in the future.
