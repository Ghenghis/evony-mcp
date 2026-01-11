#!/usr/bin/env python3
"""Analyze gaps in the RAG system."""
import json
from pathlib import Path

print("=" * 60)
print("RAG SYSTEM GAP ANALYSIS")
print("=" * 60)

# Read main files
precision_rag = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\precision_rag.py").read_text(encoding="utf-8")
mcp = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\mcp_server_v2.py").read_text(encoding="utf-8")
gradio = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\gradio_ui.py").read_text(encoding="utf-8")

gaps = []

# 1. Feedback loop integration
print("\n1. FEEDBACK LOOP INTEGRATION")
fb_in_precision = "feedback_loop" in precision_rag
fb_in_gradio = "feedback" in gradio.lower()
fb_in_mcp = "feedback" in mcp.lower()
print(f"   precision_rag.py: {fb_in_precision}")
print(f"   gradio_ui.py: {fb_in_gradio}")
print(f"   mcp_server_v2.py: {fb_in_mcp}")
if not fb_in_precision:
    gaps.append("Feedback loop not integrated in precision_rag.py")
if not fb_in_gradio:
    gaps.append("Feedback buttons not in Gradio UI")

# 2. Cross-encoder reranking
print("\n2. CROSS-ENCODER RERANKING")
ce_integrated = "CrossEncoder" in precision_rag or "cross_encoder" in precision_rag
print(f"   In precision_rag.py: {ce_integrated}")
if not ce_integrated:
    gaps.append("Cross-encoder reranking not integrated")

# 3. Question formatter
print("\n3. QUESTION FORMATTER")
qf_integrated = "question_formatter" in precision_rag
print(f"   In precision_rag.py: {qf_integrated}")
if not qf_integrated:
    gaps.append("Question formatter not integrated")

# 4. MCP server tools
print("\n4. MCP SERVER TOOLS")
has_precision = "evony.precision" in mcp
has_feedback = "evony.feedback" in mcp
print(f"   evony.precision tool: {has_precision}")
print(f"   evony.feedback tool: {has_feedback}")
if not has_feedback:
    gaps.append("No evony.feedback tool in MCP server")

# 5. Index stats
print("\n5. INDEX COVERAGE")
chunks = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\evony_rag\index\chunks.json")
if chunks.exists():
    with open(chunks) as f:
        data = json.load(f)
    print(f"   Chunks indexed: {len(data):,}")
else:
    print("   chunks.json NOT FOUND")
    gaps.append("Chunks index missing")

# 6. Embeddings
print("\n6. EMBEDDINGS")
emb = Path(r"G:\evony_rag_index\embeddings.npy")
print(f"   embeddings.npy exists: {emb.exists()}")
if not emb.exists():
    gaps.append("Embeddings file missing")

# 7. Knowledge Graph
print("\n7. KNOWLEDGE GRAPH")
kg = Path(r"G:\evony_rag_index\knowledge_graph.json")
print(f"   knowledge_graph.json exists: {kg.exists()}")
if not kg.exists():
    gaps.append("Knowledge graph missing")

# 8. Answer cache
print("\n8. ANSWER CACHE")
cache = Path(r"G:\evony_rag_index\answer_cache.json")
print(f"   answer_cache.json exists: {cache.exists()}")

# 9. Self-consistency (multi-answer)
print("\n9. SELF-CONSISTENCY")
self_consist = "self_consist" in precision_rag or "multi_answer" in precision_rag or "consensus" in precision_rag
print(f"   Implemented: {self_consist}")
if not self_consist:
    gaps.append("Self-consistency (multi-answer consensus) not implemented")

# Summary
print("\n" + "=" * 60)
print("GAPS FOUND:")
print("=" * 60)
if gaps:
    for i, gap in enumerate(gaps, 1):
        print(f"  {i}. {gap}")
else:
    print("  No major gaps found!")
