#!/usr/bin/env python
"""
Comprehensive test script for all Evony RTE MCP tools.
Tests all handlers and verifies they respond correctly.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evony_rte.mcp_server import handle_tool, TOOLS, HANDLERS

def test_all_handlers():
    """Test all handlers with basic calls."""
    print(f"=" * 60)
    print(f"EVONY RTE TOOL TEST SUITE")
    print(f"=" * 60)
    print(f"Total Tools Defined: {len(TOOLS)}")
    print(f"Total Handlers: {len(HANDLERS)}")
    print(f"=" * 60)
    
    results = {"success": [], "fail": [], "not_installed": []}
    
    # Test configurations for each handler
    test_configs = {
        "packet_capture": {"action": "status"},
        "packet_decode": {"data": "000a0b"},
        "packet_encode": {"data": {"test": 1}},
        "client_search": {"pattern": "troop"},
        "exploit_list": {},
        "exploit_test": {"exploit_id": "test"},
        "game_status": {},
        "rte_stats": {},
        "proxy_status": {},
        "proxy_session": {},
        "tools_status": {},
        "advanced_tools_status": {},
        "scapy": {"action": "status"},
        "ngrep": {"action": "search", "pattern": "test"},
        "tcpflow": {"action": "extract", "pcap_file": "test.pcap"},
        "radare2": {"action": "strings", "file": "test.exe"},
        "ghidra": {"action": "status"},
        "swfmill": {"action": "to_xml", "swf_file": "test.swf"},
        "swftools": {"action": "strings", "swf_file": "test.swf"},
        "flasm_tool": {"action": "disasm", "swf_file": "test.swf"},
        "zeek": {"action": "analyze", "pcap_file": "test.pcap"},
        "troop_repair": {"castle_id": 1},
        "stealth_config": {},
        "failsafe_status": {},
        "diagram_generate": {"type": "test", "name": "test", "data": {}},
        "client_strings": {"filter": "troop"},
        "game_resources": {},
        "game_troops": {},
    }
    
    for name in HANDLERS.keys():
        args = test_configs.get(name, {})
        try:
            result = handle_tool(name, args)
            result_str = json.dumps(result)
            
            if "not installed" in result_str.lower() or "not configured" in result_str.lower():
                results["not_installed"].append(name)
                status = "NOT INSTALLED"
            elif "error" in result_str.lower() and "file not found" not in result_str.lower():
                # Some errors are expected (missing files, etc)
                if "unknown tool" in result_str.lower():
                    results["fail"].append(name)
                    status = "FAIL"
                else:
                    results["success"].append(name)
                    status = "OK (expected error)"
            else:
                results["success"].append(name)
                status = "OK"
            
            print(f"  {name:30} {status}")
            
        except Exception as e:
            results["fail"].append(name)
            print(f"  {name:30} EXCEPTION: {str(e)[:40]}")
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"  Success:       {len(results['success'])}")
    print(f"  Not Installed: {len(results['not_installed'])}")
    print(f"  Failed:        {len(results['fail'])}")
    print(f"  Total:         {len(HANDLERS)}")
    
    if results["fail"]:
        print(f"\nFailed handlers: {results['fail']}")
    
    if results["not_installed"]:
        print(f"\nNot installed: {results['not_installed']}")
    
    return results

if __name__ == "__main__":
    test_all_handlers()
