#!/usr/bin/env python3
"""
Comprehensive handler test - tests ALL 55 handlers with REAL parameters
Tests multiple actions/functions for each handler
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evony_rte.mcp_server import handle_tool, HANDLERS

# Real test file
TEST_SWF = r"C:\Users\Admin\Downloads\Evony_Decrypted\AutoEvony.swf"

# Comprehensive test cases with REAL parameters
TEST_CASES = {
    # Core packet tools
    "packet_capture": [
        {"interface": "eth0", "duration": 1},
        {"filter": "tcp port 443"},
    ],
    "packet_decode": [
        {"data": "00"},
        {"data": "0a0b0c0d"},
        {"hex": "414d4633"},
    ],
    "packet_encode": [
        {"data": {"test": "value"}},
        {"cmd": "city.getInfo", "data": {"castleId": 123}},
    ],
    
    # Client analysis
    "client_search": [
        {"pattern": "socket", "scope": "client"},
        {"pattern": "encrypt", "scope": "all"},
        {"pattern": "sendMessage"},
    ],
    "client_strings": [
        {"pattern": "socket"},
        {"pattern": "http"},
        {"filter": "url"},
    ],
    "client_callgraph": [
        {"function": "sendCommand"},
        {"function": "connect", "depth": 2},
    ],
    
    # Exploit tools
    "exploit_list": [
        {},
        {"category": "overflow"},
        {"status": "tested"},
    ],
    "exploit_test": [
        {"exploit_id": "troop_overflow", "dry_run": True},
        {"exploit_id": "resource_underflow"},
        {},  # Should return available exploits
    ],
    "exploit_scan": [
        {"target": "client", "type": "overflow"},
        {"scope": "all"},
    ],
    "exploit_create": [
        {},  # Should return template
        {"name": "test", "cmd": "test.cmd", "payload": {}},
    ],
    
    # Game status
    "game_status": [
        {"server": "na45"},
        {},
    ],
    "game_connect": [
        {"server": "na45"},
        {"host": "evony.com", "port": 443},
    ],
    "game_send": [
        {"cmd": "city.getInfo", "data": {"castleId": 123}},
        {},
    ],
    "game_resources": [
        {},
        {"castle_id": 12345},
    ],
    "game_troops": [
        {},
        {"castle_id": 12345},
    ],
    
    # RTE stats
    "rte_stats": [{}],
    
    # Cross-reference
    "xref_client_server": [
        {},
        {"client_function": "sendTroops"},
    ],
    "xref_packet_handler": [
        {"cmd": "troop.produceTroop"},
        {"cmd": "city.getInfo"},
    ],
    "xref_validation": [
        {"cmd": "troop.produceTroop"},
        {"cmd": "resource.collect"},
    ],
    
    # Protocol analysis
    "diff_versions": [
        {"v1": "1.0", "v2": "2.0"},
        {"focus": "battle"},
    ],
    "analyze_protocol": [
        {"data": "00", "type": "amf"},
        {"type": "amf3"},
    ],
    "find_vulnerabilities": [
        {"scope": "client"},
        {"patterns": ["overflow"]},
    ],
    
    # Calculators
    "overflow_calculator": [
        {"value": 2147483647, "operation": "add", "amount": 1},
        {"value": 0, "operation": "subtract", "amount": 100},
    ],
    "troop_repair": [
        {"troop_type": "archer", "count": 100},
        {"troop_type": "cavalry", "count": -1000},
    ],
    
    # Account tools
    "account_scan": [
        {},  # Should return capabilities
        {"account_data": {"troops": {}}},
    ],
    "stealth_config": [
        {"action": "get"},
        {},
    ],
    "traffic_intercept": [
        {"action": "status"},
        {},
    ],
    "failsafe_status": [{}],
    
    # Tools status
    "tools_status": [{}],
    "advanced_tools_status": [{}],
    
    # RABCDAsm tools
    "rabcdasm_export": [
        {"swf_file": TEST_SWF},
    ],
    "rabcdasm_disasm": [
        {"abc_file": "test.abc"},
        {},
    ],
    "rabcdasm_patch": [
        {"swf_file": TEST_SWF},
    ],
    
    # SWF tools
    "swf_extract": [
        {"swf_file": TEST_SWF, "type": "all"},
    ],
    "flasm_disasm": [
        {"swf_file": TEST_SWF},
    ],
    
    # Diagram tools
    "diagram_generate": [
        {"type": "packet_flow"},
        {"type": "class_hierarchy"},
    ],
    
    # Proxy tools
    "proxy_start": [
        {"port": 8888},
        {},
    ],
    "proxy_stop": [{}],
    "proxy_status": [{}],
    "proxy_capture": [
        {"limit": 10},
        {},
    ],
    "proxy_inject": [
        {},  # Should return ready status
        {"cmd": "test", "data": {}},
    ],
    "proxy_session": [{}],
    
    # Packet tools
    "packet_replay": [
        {},  # Should return ready status
        {"packet_index": 0},
    ],
    "packet_inject": [
        {},  # Should return ready status
        {"cmd": "test", "data": {}},
    ],
    
    # Advanced RE tools
    "scapy": [
        {"action": "status"},
        {"action": "interfaces"},
    ],
    "ngrep": [
        {"action": "status"},
    ],
    "tcpflow": [
        {"action": "status"},
    ],
    "radare2": [
        {"action": "strings", "file": TEST_SWF},
        {"action": "status"},
    ],
    "ghidra": [
        {"action": "status"},
    ],
    "swfmill": [
        {"action": "status"},
        {"action": "to_xml", "swf_file": TEST_SWF},
    ],
    "swftools": [
        {"action": "status"},
        {"action": "extract", "swf_file": TEST_SWF},
    ],
    "flasm_tool": [
        {"action": "status"},
        {"action": "disasm", "swf_file": TEST_SWF},
    ],
    "zeek": [
        {"action": "status"},
    ],
    "ffdec": [
        {"action": "status"},
        {"action": "decompile", "swf_file": TEST_SWF},
    ],
    
    # Performance
    "performance_stats": [
        {"action": "stats"},
        {},
    ],
}

def run_comprehensive_tests():
    print("=" * 70)
    print("COMPREHENSIVE HANDLER TEST - REAL PARAMETERS")
    print("=" * 70)
    print(f"Total Handlers: {len(HANDLERS)}")
    print(f"Test Cases Defined: {len(TEST_CASES)}")
    print("=" * 70)
    
    results = {"success": 0, "error": 0, "handlers": {}}
    
    for handler_name in sorted(HANDLERS.keys()):
        test_cases = TEST_CASES.get(handler_name, [{}])
        handler_results = []
        
        for i, args in enumerate(test_cases):
            try:
                result = handle_tool(handler_name, args)
                has_error = "error" in result
                handler_results.append({
                    "args": args,
                    "success": not has_error,
                    "result": result
                })
            except Exception as e:
                handler_results.append({
                    "args": args,
                    "success": False,
                    "exception": str(e)
                })
        
        # Count successes
        successes = sum(1 for r in handler_results if r["success"])
        total = len(handler_results)
        
        if successes == total:
            print(f"  {handler_name:35} OK ({total} tests)")
            results["success"] += 1
        elif successes > 0:
            print(f"  {handler_name:35} PARTIAL ({successes}/{total} passed)")
            results["error"] += 1
        else:
            print(f"  {handler_name:35} FAILED (0/{total} passed)")
            results["error"] += 1
        
        results["handlers"][handler_name] = handler_results
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Handlers with ALL tests passing: {results['success']}")
    print(f"  Handlers with errors:            {results['error']}")
    print(f"  Total handlers:                  {len(HANDLERS)}")
    print("=" * 70)
    
    # Show failed handlers
    if results["error"] > 0:
        print("\nHandlers with failures:")
        for name, tests in results["handlers"].items():
            failures = [t for t in tests if not t["success"]]
            if failures:
                print(f"\n  {name}:")
                for f in failures:
                    if "exception" in f:
                        print(f"    Args: {f['args']} -> EXCEPTION: {f['exception'][:50]}")
                    else:
                        err = f["result"].get("error", "unknown")
                        print(f"    Args: {f['args']} -> ERROR: {str(err)[:50]}")
    
    return results

if __name__ == "__main__":
    run_comprehensive_tests()
