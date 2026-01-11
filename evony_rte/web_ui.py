"""
Evony RTE - Web UI Dashboard
=============================
Real-time web interface for MCP server monitoring and testing.
Compatible with Claude Desktop and Windsurf integration.

Features:
- Handler testing interface
- Real-time metrics dashboard
- Log viewer with filtering
- Exploit testing sandbox
- RAG query interface
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import logging

logger = logging.getLogger(__name__)

# HTML Templates
DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evony RTE Dashboard</title>
    <style>
        :root {
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --accent: #e94560;
            --accent2: #0f3460;
            --text: #eaeaea;
            --text-dim: #888;
            --success: #4ade80;
            --warning: #fbbf24;
            --error: #ef4444;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.5rem; }
        .header .status {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.5rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .card h2 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .card h2::before {
            content: '';
            width: 4px;
            height: 20px;
            background: var(--accent);
            border-radius: 2px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        .metric {
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent);
        }
        .metric-label {
            font-size: 0.85rem;
            color: var(--text-dim);
            margin-top: 0.25rem;
        }
        .handler-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .handler-item {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            cursor: pointer;
            transition: background 0.2s;
        }
        .handler-item:hover {
            background: rgba(255,255,255,0.05);
        }
        .handler-name { font-weight: 500; }
        .handler-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.85rem;
            color: var(--text-dim);
        }
        .test-form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .input-group label {
            font-size: 0.85rem;
            color: var(--text-dim);
        }
        input, select, textarea {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            padding: 0.75rem;
            color: var(--text);
            font-family: inherit;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent);
        }
        textarea {
            min-height: 100px;
            font-family: 'Consolas', monospace;
            font-size: 0.9rem;
        }
        button {
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(233, 69, 96, 0.4);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .result-box {
            background: #0d1117;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Consolas', monospace;
            font-size: 0.85rem;
            max-height: 300px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .log-entry {
            padding: 0.5rem;
            border-left: 3px solid var(--accent2);
            margin-bottom: 0.5rem;
            background: rgba(255,255,255,0.02);
        }
        .log-entry.error { border-color: var(--error); }
        .log-entry.success { border-color: var(--success); }
        .log-time {
            font-size: 0.75rem;
            color: var(--text-dim);
        }
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .tab {
            padding: 0.5rem 1rem;
            background: rgba(255,255,255,0.1);
            border: none;
            border-radius: 6px;
            color: var(--text);
            cursor: pointer;
        }
        .tab.active {
            background: var(--accent);
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .badge-success { background: var(--success); color: #000; }
        .badge-warning { background: var(--warning); color: #000; }
        .badge-error { background: var(--error); color: #fff; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 Evony RTE Dashboard</h1>
        <div class="status">
            <div class="status-dot"></div>
            <span id="server-status">Connected</span>
            <span id="uptime">Uptime: --</span>
        </div>
    </div>
    
    <div class="container">
        <!-- Metrics Overview -->
        <div class="card">
            <h2>System Metrics</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="total-requests">0</div>
                    <div class="metric-label">Total Requests</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="error-rate">0%</div>
                    <div class="metric-label">Error Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="handlers-active">55</div>
                    <div class="metric-label">Handlers Active</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="exploits-tested">0</div>
                    <div class="metric-label">Exploits Tested</div>
                </div>
            </div>
        </div>
        
        <!-- Handler List -->
        <div class="card">
            <h2>Handler Status</h2>
            <div class="handler-list" id="handler-list">
                <!-- Populated by JS -->
            </div>
        </div>
        
        <!-- Handler Tester -->
        <div class="card">
            <h2>Handler Tester</h2>
            <div class="test-form">
                <div class="input-group">
                    <label>Handler</label>
                    <select id="handler-select">
                        <option value="">Select handler...</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Arguments (JSON)</label>
                    <textarea id="handler-args" placeholder='{"action": "status"}'>{}</textarea>
                </div>
                <button onclick="testHandler()">Run Handler</button>
                <div class="result-box" id="handler-result">Results will appear here...</div>
            </div>
        </div>
        
        <!-- RAG Query -->
        <div class="card">
            <h2>RAG Query</h2>
            <div class="test-form">
                <div class="input-group">
                    <label>Search Query</label>
                    <input type="text" id="rag-query" placeholder="troop overflow exploit">
                </div>
                <button onclick="queryRAG()">Search</button>
                <div class="result-box" id="rag-result">Results will appear here...</div>
            </div>
        </div>
        
        <!-- Exploit Tester -->
        <div class="card">
            <h2>Exploit Sandbox</h2>
            <div class="test-form">
                <div class="input-group">
                    <label>Exploit ID</label>
                    <select id="exploit-select">
                        <option value="overflow_archer">overflow_archer</option>
                        <option value="negative_disband">negative_disband</option>
                        <option value="resource_bypass">resource_bypass</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>
                        <input type="checkbox" id="dry-run" checked> Dry Run (Safe)
                    </label>
                </div>
                <button onclick="testExploit()">Test Exploit</button>
                <div class="result-box" id="exploit-result">Results will appear here...</div>
            </div>
        </div>
        
        <!-- Recent Logs -->
        <div class="card" style="grid-column: span 2;">
            <h2>Recent Activity</h2>
            <div class="tabs">
                <button class="tab active" onclick="filterLogs('all')">All</button>
                <button class="tab" onclick="filterLogs('success')">Success</button>
                <button class="tab" onclick="filterLogs('error')">Errors</button>
            </div>
            <div id="log-container" style="max-height: 300px; overflow-y: auto;">
                <!-- Populated by JS -->
            </div>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin;
        
        // Populate handler list
        async function loadHandlers() {
            try {
                const res = await fetch(API_BASE + '/api/handlers');
                const data = await res.json();
                const select = document.getElementById('handler-select');
                const list = document.getElementById('handler-list');
                
                select.innerHTML = '<option value="">Select handler...</option>';
                list.innerHTML = '';
                
                for (const [name, info] of Object.entries(data.handlers || {})) {
                    select.innerHTML += `<option value="${name}">${name}</option>`;
                    list.innerHTML += `
                        <div class="handler-item" onclick="selectHandler('${name}')">
                            <span class="handler-name">${name}</span>
                            <span class="handler-stats">
                                <span>${info.calls || 0} calls</span>
                                <span>${info.success_rate || 100}%</span>
                            </span>
                        </div>
                    `;
                }
            } catch (e) {
                console.error('Failed to load handlers:', e);
            }
        }
        
        function selectHandler(name) {
            document.getElementById('handler-select').value = name;
        }
        
        async function testHandler() {
            const handler = document.getElementById('handler-select').value;
            const argsText = document.getElementById('handler-args').value;
            const resultBox = document.getElementById('handler-result');
            
            if (!handler) {
                resultBox.textContent = 'Please select a handler';
                return;
            }
            
            try {
                const args = JSON.parse(argsText || '{}');
                resultBox.textContent = 'Running...';
                
                const res = await fetch(API_BASE + '/api/call', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({handler, args})
                });
                const data = await res.json();
                resultBox.textContent = JSON.stringify(data, null, 2);
                addLog('success', `Handler ${handler} executed successfully`);
            } catch (e) {
                resultBox.textContent = 'Error: ' + e.message;
                addLog('error', `Handler ${handler} failed: ${e.message}`);
            }
        }
        
        async function queryRAG() {
            const query = document.getElementById('rag-query').value;
            const resultBox = document.getElementById('rag-result');
            
            if (!query) {
                resultBox.textContent = 'Please enter a query';
                return;
            }
            
            try {
                resultBox.textContent = 'Searching...';
                const res = await fetch(API_BASE + '/api/call', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({handler: 'evony_search', args: {query}})
                });
                const data = await res.json();
                resultBox.textContent = JSON.stringify(data, null, 2);
            } catch (e) {
                resultBox.textContent = 'Error: ' + e.message;
            }
        }
        
        async function testExploit() {
            const exploitId = document.getElementById('exploit-select').value;
            const dryRun = document.getElementById('dry-run').checked;
            const resultBox = document.getElementById('exploit-result');
            
            try {
                resultBox.textContent = 'Testing...';
                const res = await fetch(API_BASE + '/api/call', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        handler: 'exploit_test',
                        args: {exploit_id: exploitId, dry_run: dryRun}
                    })
                });
                const data = await res.json();
                resultBox.textContent = JSON.stringify(data, null, 2);
                addLog(data.error ? 'error' : 'success', `Exploit ${exploitId} ${dryRun ? '(dry run)' : ''}`);
            } catch (e) {
                resultBox.textContent = 'Error: ' + e.message;
            }
        }
        
        async function loadMetrics() {
            try {
                const res = await fetch(API_BASE + '/api/metrics');
                const data = await res.json();
                
                document.getElementById('total-requests').textContent = data.overview?.total_requests || 0;
                document.getElementById('error-rate').textContent = (data.overview?.error_rate || 0) + '%';
                document.getElementById('handlers-active').textContent = data.overview?.handlers_active || 55;
                document.getElementById('exploits-tested').textContent = 
                    (data.exploit_activity?.dry_runs || 0) + (data.exploit_activity?.live_tests || 0);
            } catch (e) {
                console.error('Failed to load metrics:', e);
            }
        }
        
        function addLog(type, message) {
            const container = document.getElementById('log-container');
            const time = new Date().toLocaleTimeString();
            container.innerHTML = `
                <div class="log-entry ${type}">
                    <span class="log-time">${time}</span>
                    <span>${message}</span>
                </div>
            ` + container.innerHTML;
        }
        
        function filterLogs(type) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            // Filter logic here
        }
        
        // Initialize
        loadHandlers();
        loadMetrics();
        setInterval(loadMetrics, 5000);
        addLog('success', 'Dashboard connected');
    </script>
</body>
</html>'''


class WebUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Web UI."""
    
    def log_message(self, format, *args):
        logger.debug(f"WebUI: {args[0]}")
    
    def send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())
    
    def send_html(self, html: str):
        """Send HTML response."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        if path == '/' or path == '/dashboard':
            self.send_html(DASHBOARD_HTML)
        
        elif path == '/api/status':
            self.send_json({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0'
            })
        
        elif path == '/api/handlers':
            try:
                from .mcp_server import HANDLERS
                from .metrics_dashboard import get_metrics_store
                
                store = get_metrics_store()
                handler_metrics = store.get_handler_metrics()
                
                handlers = {}
                for name in HANDLERS:
                    metrics = handler_metrics.get(name, {})
                    handlers[name] = {
                        'calls': metrics.get('calls', 0),
                        'success_rate': metrics.get('success_rate', 100),
                        'avg_time_ms': metrics.get('avg_time_ms', 0)
                    }
                
                self.send_json({'handlers': handlers, 'count': len(handlers)})
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
        
        elif path == '/api/metrics':
            try:
                from .metrics_dashboard import handle_metrics_dashboard
                result = handle_metrics_dashboard({'action': 'summary'})
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
        
        else:
            self.send_json({'error': 'Not found'}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        
        # Read body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length else '{}'
        
        try:
            data = json.loads(body)
        except:
            data = {}
        
        if path == '/api/call':
            try:
                from .mcp_server import handle_tool
                
                handler = data.get('handler')
                args = data.get('args', {})
                
                if not handler:
                    self.send_json({'error': 'handler required'}, 400)
                    return
                
                result = handle_tool(handler, args)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
        
        else:
            self.send_json({'error': 'Not found'}, 404)


class WebUIServer:
    """Web UI server for Evony RTE."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self._thread = None
        self.running = False
    
    def start(self):
        """Start the web UI server."""
        if self.running:
            return {'status': 'already running', 'port': self.port}
        
        try:
            self.server = HTTPServer((self.host, self.port), WebUIHandler)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self.running = True
            logger.info(f"Web UI started at http://{self.host}:{self.port}")
            return {'status': 'started', 'url': f'http://localhost:{self.port}'}
        except Exception as e:
            return {'error': str(e)}
    
    def _run(self):
        """Server loop."""
        self.server.serve_forever()
    
    def stop(self):
        """Stop the web UI server."""
        if self.server:
            self.server.shutdown()
            self.running = False
            return {'status': 'stopped'}
        return {'status': 'not running'}


# Global server instance
_web_ui_server = None

def get_web_ui_server() -> WebUIServer:
    """Get the global web UI server."""
    global _web_ui_server
    if _web_ui_server is None:
        _web_ui_server = WebUIServer()
    return _web_ui_server


def handle_web_ui(args: Dict) -> Dict:
    """
    Control the Web UI dashboard.
    
    Parameters:
        action: 'start' | 'stop' | 'status' | 'url'
        port: Port number (default: 8080)
    
    Returns:
        Server status and URL
    """
    action = args.get('action', 'status')
    port = args.get('port', 8080)
    
    server = get_web_ui_server()
    
    if action == 'start':
        if port != server.port:
            server.port = port
        return server.start()
    
    elif action == 'stop':
        return server.stop()
    
    elif action == 'url':
        if server.running:
            return {'url': f'http://localhost:{server.port}', 'running': True}
        return {'running': False, 'hint': 'Call with action="start" first'}
    
    else:  # status
        return {
            'running': server.running,
            'port': server.port,
            'url': f'http://localhost:{server.port}' if server.running else None
        }


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'WebUIServer',
    'get_web_ui_server', 
    'handle_web_ui',
    'DASHBOARD_HTML'
]
