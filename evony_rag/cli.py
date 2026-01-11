"""
Evony RAG - Command Line Interface
===================================
Interactive CLI for querying the Evony knowledge base.
"""

import sys
import json
from pathlib import Path

from .rag_engine import get_rag
from .config import CATEGORIES


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("EVONY KNOWLEDGE SYSTEM - RAG CLI")
    print("="*60)
    print("Commands:")
    print("  /find <query>     - Find relevant files")
    print("  /stats            - Show index statistics")
    print("  /cats             - List categories")
    print("  /rebuild          - Rebuild the index")
    print("  /nollm            - Toggle LLM (answer without LM Studio)")
    print("  /quit             - Exit")
    print("="*60 + "\n")


def format_citation(cit) -> str:
    """Format a citation for display."""
    return f"  📄 {cit.file_path}:{cit.start_line}-{cit.end_line} ({cit.relevance:.0%})"


def run_cli():
    """Run the interactive CLI."""
    print_banner()
    
    rag = get_rag()
    use_llm = True
    
    print("Loading knowledge index...")
    if not rag.load_index():
        print("Building index (first run - this takes a minute)...")
        rag.build_index()
    
    stats = rag.get_stats()
    print(f"Loaded {stats.get('num_chunks', 0)} knowledge chunks")
    print(f"LLM mode: {'ON (uses LM Studio)' if use_llm else 'OFF (retrieval only)'}\n")
    
    while True:
        try:
            query = input("[You]: ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.startswith("/"):
                cmd = query.lower().split()[0]
                args = query[len(cmd):].strip()
                
                if cmd in ["/quit", "/exit", "/q"]:
                    print("Goodbye!")
                    break
                
                elif cmd == "/find":
                    if not args:
                        print("Usage: /find <query>")
                        continue
                    
                    citations = rag.find_files(args, top_k=10)
                    print("\nRelevant files:")
                    for cit in citations:
                        print(format_citation(cit))
                    print()
                
                elif cmd == "/stats":
                    stats = rag.get_stats()
                    print(f"\nIndex Statistics:")
                    print(f"  Total chunks: {stats.get('num_chunks', 0)}")
                    print(f"  Model: {stats.get('model', 'unknown')}")
                    if 'categories' in stats:
                        print(f"  By category:")
                        for cat, count in stats['categories'].items():
                            print(f"    {cat}: {count}")
                    print()
                
                elif cmd == "/cats":
                    print("\nCategories:")
                    for cat, info in CATEGORIES.items():
                        safe = "✅" if info["safe"] else "⚠️"
                        print(f"  {safe} {cat}: {info['description']}")
                    print()
                
                elif cmd == "/rebuild":
                    print("Rebuilding index...")
                    rag.build_index()
                    print("Done!")
                
                elif cmd == "/nollm":
                    use_llm = not use_llm
                    print(f"LLM mode: {'ON' if use_llm else 'OFF'}")
                
                else:
                    print(f"Unknown command: {cmd}")
                
                continue
            
            # Query the RAG
            print("\n[Evony Knowledge]:")
            response = rag.query(query, use_llm=use_llm)
            
            # Print answer
            print(response.answer)
            
            # Print citations
            if response.citations:
                print("\n📚 Sources:")
                for cit in response.citations:
                    print(format_citation(cit))
            
            # Print metadata
            print(f"\n[intent: {response.query_analysis.intent} | model: {response.model_used}]\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run_cli()
