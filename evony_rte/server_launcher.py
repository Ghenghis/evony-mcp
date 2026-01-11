#!/usr/bin/env python3
"""
Evony RTE Server Launcher - CLI for Windsurf and Claude Desktop
Integrates MCP Server with Bot Server - REAL connections only
No mocked, simulated, or fake data - strictly production mode

Usage:
    python -m evony_rte.server_launcher start     # Start integrated server
    python -m evony_rte.server_launcher status    # Check server status
    python -m evony_rte.server_launcher stop      # Stop server
    python -m evony_rte.server_launcher test      # Run handler tests
    python -m evony_rte.server_launcher cli       # Interactive CLI mode
"""
import argparse
import subprocess
import sys
import os
import json
import time
import threading
import socket
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class EvonyRTEServer:
    """Integrated Evony RTE Server with Bot Server."""
    
    BOT_SERVER_PORTS = [9999, 9998, 9997, 9996, 9995]
    
    def __init__(self):
        self.bot_server_process = None
        self.bot_server_port = None
        self.mcp_active = False
        self.server_thread = None
        
    def find_available_port(self) -> Optional[int]:
        """Find first available port."""
        for port in self.BOT_SERVER_PORTS:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result != 0:
                    return port
            except:
                pass
        return None
    
    def check_bot_server(self) -> Dict:
        """Check if bot server is running."""
        for port in self.BOT_SERVER_PORTS:
            try:
                import urllib.request
                req = urllib.request.Request(f"http://localhost:{port}/api/status")
                resp = urllib.request.urlopen(req, timeout=2)
                data = json.loads(resp.read().decode())
                return {"running": True, "port": port, "data": data}
            except:
                continue
        return {"running": False, "port": None}
    
    def start_bot_server(self) -> Dict:
        """Start the bot server."""
        # Check if already running
        status = self.check_bot_server()
        if status["running"]:
            self.bot_server_port = status["port"]
            return {"status": "already_running", "port": status["port"]}
        
        # Find available port
        port = self.find_available_port()
        if not port:
            return {"error": "No available ports"}
        
        # Start bot server
        bot_server_path = Path(__file__).parent.parent / "evony_bot" / "server.py"
        if not bot_server_path.exists():
            return {"error": f"Bot server not found at {bot_server_path}"}
        
        try:
            self.bot_server_process = subprocess.Popen(
                [sys.executable, str(bot_server_path), "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(bot_server_path.parent)
            )
            self.bot_server_port = port
            
            # Wait for server to start
            time.sleep(2)
            
            # Verify it's running
            status = self.check_bot_server()
            if status["running"]:
                return {"status": "started", "port": port, "pid": self.bot_server_process.pid}
            else:
                return {"status": "started_unverified", "port": port, "pid": self.bot_server_process.pid}
        except Exception as e:
            return {"error": str(e)}
    
    def stop_bot_server(self) -> Dict:
        """Stop the bot server."""
        if self.bot_server_process:
            self.bot_server_process.terminate()
            self.bot_server_process = None
            return {"status": "stopped"}
        return {"status": "not_running"}
    
    def get_status(self) -> Dict:
        """Get full server status."""
        bot_status = self.check_bot_server()
        
        # Check MCP handlers
        try:
            from evony_rte.mcp_server import HANDLERS
            handler_count = len(HANDLERS)
        except:
            handler_count = 0
        
        return {
            "bot_server": bot_status,
            "mcp_handlers": handler_count,
            "server_pid": self.bot_server_process.pid if self.bot_server_process else None
        }
    
    def run_tests(self) -> Dict:
        """Run comprehensive handler tests."""
        from evony_rte.mcp_server import HANDLERS, handle_tool
        
        results = {"passed": 0, "failed": 0, "handlers": {}}
        
        for name in sorted(HANDLERS.keys()):
            try:
                result = handle_tool(name, {})
                has_error = "error" in result
                results["handlers"][name] = {
                    "success": not has_error,
                    "result_keys": list(result.keys())
                }
                if has_error:
                    results["failed"] += 1
                else:
                    results["passed"] += 1
            except Exception as e:
                results["handlers"][name] = {"success": False, "exception": str(e)}
                results["failed"] += 1
        
        results["total"] = results["passed"] + results["failed"]
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evony RTE Server Launcher - CLI for Windsurf and Claude Desktop"
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "test", "cli", "help"],
        help="Command to execute"
    )
    parser.add_argument("--port", type=int, default=9999, help="Bot server port")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    args = parser.parse_args()
    server = EvonyRTEServer()
    
    if args.command == "start":
        result = server.start_bot_server()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            elif result.get("status") == "already_running":
                print(f"✅ Bot server already running on port {result['port']}")
            else:
                print(f"✅ Bot server started on port {result.get('port')}")
                print(f"   PID: {result.get('pid')}")
                print(f"   URL: http://localhost:{result.get('port')}")
    
    elif args.command == "stop":
        result = server.stop_bot_server()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"✅ Bot server {result['status']}")
    
    elif args.command == "status":
        result = server.get_status()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            bot = result["bot_server"]
            if bot["running"]:
                print(f"✅ Bot Server: Running on port {bot['port']}")
            else:
                print("❌ Bot Server: Not running")
            print(f"📊 MCP Handlers: {result['mcp_handlers']}")
    
    elif args.command == "test":
        print("Running comprehensive handler tests...")
        result = server.run_tests()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"RESULTS: {result['passed']}/{result['total']} passed")
            print(f"{'='*60}")
            if result["failed"] > 0:
                print("\nFailed handlers:")
                for name, info in result["handlers"].items():
                    if not info["success"]:
                        print(f"  ❌ {name}")
    
    elif args.command == "cli":
        print("Evony RTE Interactive CLI")
        print("Type 'help' for commands, 'exit' to quit")
        print("-" * 40)
        
        while True:
            try:
                cmd = input("evony> ").strip()
                if cmd == "exit" or cmd == "quit":
                    break
                elif cmd == "help":
                    print("Commands: status, start, stop, test, handlers, call <handler> [json_args]")
                elif cmd == "status":
                    print(json.dumps(server.get_status(), indent=2))
                elif cmd == "start":
                    print(json.dumps(server.start_bot_server(), indent=2))
                elif cmd == "stop":
                    print(json.dumps(server.stop_bot_server(), indent=2))
                elif cmd == "test":
                    result = server.run_tests()
                    print(f"Passed: {result['passed']}/{result['total']}")
                elif cmd == "handlers":
                    from evony_rte.mcp_server import HANDLERS
                    for name in sorted(HANDLERS.keys()):
                        print(f"  {name}")
                elif cmd.startswith("call "):
                    parts = cmd[5:].split(" ", 1)
                    handler_name = parts[0]
                    args = json.loads(parts[1]) if len(parts) > 1 else {}
                    from evony_rte.mcp_server import handle_tool
                    result = handle_tool(handler_name, args)
                    print(json.dumps(result, indent=2))
                elif cmd:
                    print(f"Unknown command: {cmd}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.command == "help":
        parser.print_help()


if __name__ == "__main__":
    main()
