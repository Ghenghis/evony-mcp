"""
Evony RAG v2 - Enhanced CLI
============================
Interactive CLI with mode switching and advanced features.
"""

from .rag_v2 import get_rag_v2
from .policy import get_policy


def print_banner():
    print("\n" + "="*60)
    print("EVONY KNOWLEDGE SYSTEM v2 - Hybrid RAG")
    print("="*60)
    print("Commands:")
    print("  /mode [name]       - Get/set mode (research/forensics/full_access)")
    print("  /find <query>      - Search without generation")
    print("  /symbol <name>     - Find symbol definitions")
    print("  /trace <topic>     - Multi-hop trace")
    print("  /open <path>       - View file content")
    print("  /stats             - System statistics")
    print("  /nollm             - Toggle LLM generation")
    print("  /evidence [level]  - Set evidence level (brief/normal/verbose)")
    print("  /include <cats>    - Include categories (comma-separated)")
    print("  /exclude <cats>    - Exclude categories")
    print("  /quit              - Exit")
    print("="*60 + "\n")


def format_result(r):
    return f"  📄 {r.file_path}:{r.start_line}-{r.end_line} (score:{r.combined_score:.2f} lex:{r.lexical_score:.2f} sem:{r.semantic_score:.2f})"


def run_cli():
    print_banner()
    
    print("Loading knowledge system...")
    rag = get_rag_v2()
    policy = get_policy()
    
    use_llm = True
    evidence_level = "normal"
    include_cats = None
    exclude_cats = None
    
    stats = rag.get_stats()
    print(f"Loaded: {stats.get('chunks', 0)} chunks, {stats.get('symbols', 0)} symbols")
    print(f"Mode: {policy.current_mode} | LLM: {'ON' if use_llm else 'OFF'}\n")
    
    while True:
        try:
            query = input(f"[{policy.current_mode}] > ").strip()
            
            if not query:
                continue
            
            if query.startswith("/"):
                parts = query.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd in ["/quit", "/exit", "/q"]:
                    print("Goodbye!")
                    break
                
                elif cmd == "/mode":
                    if args:
                        if policy.set_mode(args):
                            print(f"Mode set to: {policy.current_mode}")
                        else:
                            print(f"Invalid mode. Available: {policy.get_modes()}")
                    else:
                        print(f"Current: {policy.current_mode}")
                        print(f"Available: {policy.get_modes()}")
                
                elif cmd == "/find":
                    if not args:
                        print("Usage: /find <query>")
                        continue
                    results = rag.search_only(args, include=include_cats, exclude=exclude_cats, k=10)
                    print(f"\nFound {len(results)} results:")
                    for r in results:
                        print(format_result(r))
                    print()
                
                elif cmd == "/symbol":
                    if not args:
                        print("Usage: /symbol <name>")
                        continue
                    results = rag.find_symbol(args)
                    print(f"\nSymbol '{args}' found {len(results)} times:")
                    for r in results[:10]:
                        print(f"  📍 {r['file']}:{r['line']} - {r['symbol']}")
                    print()
                
                elif cmd == "/trace":
                    if not args:
                        print("Usage: /trace <topic>")
                        continue
                    results = rag.trace(args, depth=3)
                    print(f"\nTrace from '{args}':")
                    for r in results:
                        print(f"  [hop {r['hop']}] {r['topic']} → {r['file']}:{r['lines']}")
                    print()
                
                elif cmd == "/open":
                    if not args:
                        print("Usage: /open <path> [start:end]")
                        continue
                    path_parts = args.split()
                    path = path_parts[0]
                    start = end = None
                    if len(path_parts) > 1 and ':' in path_parts[1]:
                        start, end = map(int, path_parts[1].split(':'))
                    content = rag.get_file(path, start, end)
                    if content:
                        print(f"\n--- {path} ---")
                        print(content[:2000])
                        if len(content) > 2000:
                            print(f"\n... ({len(content)} chars total)")
                    else:
                        print(f"File not found: {path}")
                    print()
                
                elif cmd == "/stats":
                    stats = rag.get_stats()
                    print(f"\nStatistics:")
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                    print()
                
                elif cmd == "/nollm":
                    use_llm = not use_llm
                    print(f"LLM mode: {'ON' if use_llm else 'OFF'}")
                
                elif cmd == "/evidence":
                    if args in ["brief", "normal", "verbose"]:
                        evidence_level = args
                        print(f"Evidence level: {evidence_level}")
                    else:
                        print(f"Current: {evidence_level}")
                        print("Options: brief, normal, verbose")
                
                elif cmd == "/include":
                    if args:
                        include_cats = [c.strip() for c in args.split(',')]
                        print(f"Include: {include_cats}")
                    else:
                        include_cats = None
                        print("Include filter cleared")
                
                elif cmd == "/exclude":
                    if args:
                        exclude_cats = [c.strip() for c in args.split(',')]
                        print(f"Exclude: {exclude_cats}")
                    else:
                        exclude_cats = None
                        print("Exclude filter cleared")
                
                else:
                    print(f"Unknown command: {cmd}")
                
                continue
            
            # Regular query
            print("\n[Evony Knowledge]:")
            response = rag.query(
                query=query,
                include=include_cats,
                exclude=exclude_cats,
                evidence_level=evidence_level,
                use_llm=use_llm,
            )
            
            print(response.answer)
            
            if response.citations:
                print("\n📚 Sources:")
                for c in response.citations:
                    print(f"  📄 {c.file_path}:{c.start_line}-{c.end_line} ({c.combined_score:.0%})")
            
            if response.symbols_found:
                print("\n🔍 Related symbols:")
                for s in response.symbols_found[:3]:
                    print(f"  {s['symbol']} @ {s['file']}:{s['line']}")
            
            print(f"\n[mode: {response.policy.mode} | model: {response.model_used}]\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    run_cli()
