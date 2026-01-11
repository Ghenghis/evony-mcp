#!/usr/bin/env python3
"""
Integrated Bot Server for Evony RTE MCP
Auto-starts with MCP server, auto-switches ports if needed
Provides HTTP API for game commands
"""
import socket
import threading
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Optional
import sys
import os

class IntegratedBotServer:
    """Bot server that integrates with MCP and auto-switches ports."""
    
    DEFAULT_PORTS = [9999, 9998, 9997, 9996, 9995, 8999, 8888]
    
    def __init__(self):
        self.server = None
        self.port = None
        self.running = False
        self.thread = None
        self.session_data = {
            "connected": False,
            "server": None,
            "resources": {},
            "troops": {},
            "commands_sent": 0,
            "last_response": None
        }
    
    def find_available_port(self) -> Optional[int]:
        """Find first available port from the list."""
        for port in self.DEFAULT_PORTS:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                if result != 0:  # Port is available
                    return port
            except:
                pass
        return None
    
    def start(self, preferred_port: int = 9999) -> Dict:
        """Start the bot server, auto-switch ports if needed."""
        if self.running:
            return {"status": "already_running", "port": self.port}
        
        # Try preferred port first
        ports_to_try = [preferred_port] + [p for p in self.DEFAULT_PORTS if p != preferred_port]
        
        for port in ports_to_try:
            try:
                self.server = HTTPServer(('localhost', port), BotRequestHandler)
                self.server.bot = self
                self.port = port
                self.running = True
                
                # Start in background thread
                self.thread = threading.Thread(target=self._run_server, daemon=True)
                self.thread.start()
                
                return {
                    "status": "started",
                    "port": port,
                    "url": f"http://localhost:{port}",
                    "endpoints": ["/api/status", "/api/send", "/api/connect"]
                }
            except OSError as e:
                if "Address already in use" in str(e) or "10048" in str(e):
                    continue
                raise
        
        return {"error": "No available ports found", "tried": ports_to_try}
    
    def _run_server(self):
        """Run server in background."""
        try:
            self.server.serve_forever()
        except Exception as e:
            self.running = False
    
    def stop(self) -> Dict:
        """Stop the bot server."""
        if not self.running:
            return {"status": "not_running"}
        
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server = None
        
        return {"status": "stopped", "port": self.port}
    
    def get_status(self) -> Dict:
        """Get server status."""
        return {
            "running": self.running,
            "port": self.port,
            "session": self.session_data,
            "uptime": "active" if self.running else "stopped"
        }
    
    def send_command(self, cmd: str, data: Dict) -> Dict:
        """Send game command (simulated when not connected)."""
        self.session_data["commands_sent"] += 1
        
        # Encode packet
        from .amf3_tools import encode_evony_packet
        encoded = encode_evony_packet(cmd, data)
        
        response = {
            "status": "command_processed",
            "cmd": cmd,
            "data": data,
            "amf3_hex": encoded.get("amf3_hex", ""),
            "packet_size": len(encoded.get("amf3_hex", "")) // 2,
            "command_count": self.session_data["commands_sent"]
        }
        
        self.session_data["last_response"] = response
        return response


class BotRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for bot API."""
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_GET(self):
        """Handle GET requests."""
        path = self.path.split('?')[0]
        
        if path == '/api/status':
            self._send_json(self.server.bot.get_status())
        elif path == '/api/health':
            self._send_json({"healthy": True, "port": self.server.bot.port})
        else:
            self._send_json({"error": "Unknown endpoint", "endpoints": ["/api/status", "/api/send", "/api/connect"]}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        path = self.path.split('?')[0]
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length else '{}'
        
        try:
            data = json.loads(body) if body else {}
        except:
            data = {}
        
        if path == '/api/send':
            cmd = data.get('cmd', '')
            payload = data.get('data', {})
            result = self.server.bot.send_command(cmd, payload)
            self._send_json(result)
        elif path == '/api/connect':
            server = data.get('server', 'na45')
            self.server.bot.session_data["server"] = server
            self.server.bot.session_data["connected"] = True
            self._send_json({"status": "connected", "server": server})
        else:
            self._send_json({"error": "Unknown endpoint"}, 404)


# Global bot server instance
_bot_server = None

def get_bot_server() -> IntegratedBotServer:
    """Get or create global bot server instance."""
    global _bot_server
    if _bot_server is None:
        _bot_server = IntegratedBotServer()
    return _bot_server

def ensure_bot_server_running() -> int:
    """Ensure bot server is running, start if not. Returns port number."""
    bot = get_bot_server()
    if not bot.running:
        result = bot.start()
        return result.get("port", 9999)
    return bot.port or 9999

def stop_bot_server() -> Dict:
    """Stop the bot server."""
    bot = get_bot_server()
    return bot.stop()
