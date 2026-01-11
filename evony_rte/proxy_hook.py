"""
Evony RTE - AutoEvony Proxy Hook System
========================================
CRITICAL: Allows simultaneous access - user plays while we capture/inject.

Architecture:
  [AutoEvony] <---> [Proxy Hook] <---> [Evony Server]
                         |
                    [MCP Tools]
                         |
                   [Windsurf/Claude]

Features:
- MITM proxy captures all traffic
- Real-time packet decoding
- Packet injection without disrupting user
- Shared session access
- Live action capture (troops, items, marches)
"""

import socket
import ssl
import threading
import queue
import struct
import time
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"proxy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProxyHook")

# ============================================================================
# EVONY SERVER CONFIGURATION
# ============================================================================

EVONY_SERVERS = {
    # Format: "server_name": ("host", port)
    "na1": ("na1.evony.com", 443),
    "na45": ("na45.evony.com", 443),
    "na50": ("na50.evony.com", 443),
    "eu1": ("eu1.evony.com", 443),
    "s1": ("s1.evony.com", 443),
    # Policy servers
    "policy": ("policy.evony.com", 843),
    # Add more as discovered
}

DEFAULT_PROXY_PORT = 9998  # Local proxy port

# ============================================================================
# PACKET STRUCTURES
# ============================================================================

@dataclass
class CapturedPacket:
    """Captured packet with metadata."""
    timestamp: float
    direction: str  # "client_to_server" or "server_to_client"
    raw_data: bytes
    decoded: Optional[Dict] = None
    command: str = ""
    modified: bool = False
    original_data: Optional[bytes] = None

@dataclass
class SessionState:
    """Current session state extracted from traffic."""
    connected: bool = False
    logged_in: bool = False
    username: str = ""
    player_id: int = 0
    castles: List[Dict] = field(default_factory=list)
    resources: Dict[int, Dict] = field(default_factory=dict)  # castle_id -> resources
    troops: Dict[int, Dict] = field(default_factory=dict)  # castle_id -> troops
    last_update: float = 0

# ============================================================================
# AMF3 REAL-TIME DECODER
# ============================================================================

class RealTimeAMF3Decoder:
    """Fast AMF3 decoder for real-time traffic."""
    
    AMF3_TYPES = {
        0x00: "undefined", 0x01: "null", 0x02: "false", 0x03: "true",
        0x04: "integer", 0x05: "double", 0x06: "string", 0x07: "xml_doc",
        0x08: "date", 0x09: "array", 0x0A: "object", 0x0B: "xml",
        0x0C: "bytearray"
    }
    
    def __init__(self, data: bytes):
        self.stream = BytesIO(data)
        self.string_refs = []
        self.object_refs = []
        self.traits_refs = []
    
    def read_u29(self) -> int:
        result = 0
        for i in range(4):
            b = self.stream.read(1)
            if not b:
                break
            b = b[0]
            if i < 3:
                result = (result << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            else:
                result = (result << 8) | b
        return result
    
    def read_string(self) -> str:
        ref = self.read_u29()
        if (ref & 1) == 0:
            idx = ref >> 1
            return self.string_refs[idx] if idx < len(self.string_refs) else ""
        length = ref >> 1
        if length == 0:
            return ""
        s = self.stream.read(length).decode('utf-8', errors='replace')
        self.string_refs.append(s)
        return s
    
    def read_value(self) -> Any:
        type_byte = self.stream.read(1)
        if not type_byte:
            return None
        type_marker = type_byte[0]
        
        if type_marker == 0x00 or type_marker == 0x01:
            return None
        elif type_marker == 0x02:
            return False
        elif type_marker == 0x03:
            return True
        elif type_marker == 0x04:
            n = self.read_u29()
            if n & 0x10000000:
                n -= 0x20000000
            return n
        elif type_marker == 0x05:
            return struct.unpack('>d', self.stream.read(8))[0]
        elif type_marker == 0x06:
            return self.read_string()
        elif type_marker == 0x09:
            return self.read_array()
        elif type_marker == 0x0A:
            return self.read_object()
        elif type_marker == 0x0C:
            ref = self.read_u29()
            length = ref >> 1
            return {"__bytearray": self.stream.read(length).hex()[:100]}
        else:
            return {"__unknown_type": type_marker}
    
    def read_array(self) -> List:
        ref = self.read_u29()
        if (ref & 1) == 0:
            idx = ref >> 1
            return self.object_refs[idx] if idx < len(self.object_refs) else []
        length = ref >> 1
        result = []
        self.object_refs.append(result)
        while True:
            key = self.read_string()
            if key == "":
                break
            # Associative portion - skip for now
            self.read_value()
        for _ in range(length):
            result.append(self.read_value())
        return result
    
    def read_object(self) -> Dict:
        ref = self.read_u29()
        if (ref & 1) == 0:
            idx = ref >> 1
            return self.object_refs[idx] if idx < len(self.object_refs) else {}
        
        traits_ref = ref >> 1
        if (traits_ref & 1) == 0:
            traits_idx = traits_ref >> 1
            traits = self.traits_refs[traits_idx] if traits_idx < len(self.traits_refs) else {}
        else:
            is_dynamic = (traits_ref >> 2) & 1
            sealed_count = traits_ref >> 3
            class_name = self.read_string()
            sealed_names = [self.read_string() for _ in range(sealed_count)]
            traits = {"class": class_name, "dynamic": is_dynamic, "sealed": sealed_names}
            self.traits_refs.append(traits)
        
        result = {}
        if traits.get("class"):
            result["__class"] = traits["class"]
        self.object_refs.append(result)
        
        for name in traits.get("sealed", []):
            result[name] = self.read_value()
        
        if traits.get("dynamic", True):
            while True:
                key = self.read_string()
                if key == "":
                    break
                result[key] = self.read_value()
        
        return result
    
    def decode(self) -> Dict:
        try:
            return self.read_value()
        except Exception as e:
            return {"__error": str(e)}

# ============================================================================
# PROXY HOOK CORE
# ============================================================================

class ProxyHook:
    """
    MITM Proxy that hooks into AutoEvony traffic.
    
    Usage:
    1. Start proxy on local port
    2. Configure AutoEvony to connect through proxy
    3. User plays normally while we capture/inject
    """
    
    def __init__(self, local_port: int = DEFAULT_PROXY_PORT):
        self.local_port = local_port
        self.running = False
        self.server_socket = None
        
        # Traffic capture
        self.captured_packets: deque = deque(maxlen=10000)
        self.packet_queue = queue.Queue()
        
        # Session state
        self.session = SessionState()
        
        # Injection queue
        self.injection_queue = queue.Queue()
        
        # Callbacks for real-time hooks
        self.on_packet_captured: Optional[Callable] = None
        self.on_command_detected: Optional[Callable] = None
        self.packet_modifier: Optional[Callable] = None  # (packet) -> modified_packet
        
        # Active connections
        self.client_socket = None
        self.server_socket_conn = None
        self.target_server = None
        
        # Statistics
        self.stats = {
            "packets_captured": 0,
            "packets_injected": 0,
            "bytes_captured": 0,
            "commands_detected": {}
        }
    
    def start(self, target_server: str = "na45") -> Dict:
        """Start the proxy hook."""
        if self.running:
            return {"error": "Proxy already running"}
        
        if target_server not in EVONY_SERVERS:
            return {"error": f"Unknown server: {target_server}", "available": list(EVONY_SERVERS.keys())}
        
        self.target_server = EVONY_SERVERS[target_server]
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('127.0.0.1', self.local_port))
            self.server_socket.listen(5)
            self.running = True
            
            # Start accept thread
            threading.Thread(target=self._accept_loop, daemon=True).start()
            
            logger.info(f"Proxy started on port {self.local_port} -> {target_server}")
            
            return {
                "status": "running",
                "local_port": self.local_port,
                "target": f"{self.target_server[0]}:{self.target_server[1]}",
                "configure": f"Set AutoEvony to connect to 127.0.0.1:{self.local_port}"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def stop(self) -> Dict:
        """Stop the proxy."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket_conn:
            self.server_socket_conn.close()
        
        return {
            "status": "stopped",
            "stats": self.stats
        }
    
    def _accept_loop(self):
        """Accept incoming connections from AutoEvony."""
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_sock, client_addr = self.server_socket.accept()
                logger.info(f"Client connected: {client_addr}")
                
                self.client_socket = client_sock
                
                # Connect to real server
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                # Wrap with SSL if needed
                if self.target_server[1] == 443:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    server_sock = context.wrap_socket(server_sock, server_hostname=self.target_server[0])
                
                server_sock.connect(self.target_server)
                self.server_socket_conn = server_sock
                
                logger.info(f"Connected to server: {self.target_server}")
                self.session.connected = True
                
                # Start relay threads
                threading.Thread(target=self._relay_client_to_server, daemon=True).start()
                threading.Thread(target=self._relay_server_to_client, daemon=True).start()
                threading.Thread(target=self._injection_loop, daemon=True).start()
                
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Accept error: {e}")
    
    def _relay_client_to_server(self):
        """Relay data from AutoEvony client to Evony server."""
        while self.running and self.client_socket and self.server_socket_conn:
            try:
                data = self.client_socket.recv(65536)
                if not data:
                    break
                
                # Capture and decode
                packet = self._capture_packet(data, "client_to_server")
                
                # Allow modification
                if self.packet_modifier:
                    modified = self.packet_modifier(packet)
                    if modified and modified.raw_data != data:
                        data = modified.raw_data
                        packet.modified = True
                
                # Forward to server
                self.server_socket_conn.sendall(data)
                
            except Exception as e:
                logger.error(f"Client relay error: {e}")
                break
    
    def _relay_server_to_client(self):
        """Relay data from Evony server to AutoEvony client."""
        while self.running and self.client_socket and self.server_socket_conn:
            try:
                data = self.server_socket_conn.recv(65536)
                if not data:
                    break
                
                # Capture and decode
                packet = self._capture_packet(data, "server_to_client")
                
                # Update session state from server responses
                self._update_session_state(packet)
                
                # Forward to client
                self.client_socket.sendall(data)
                
            except Exception as e:
                logger.error(f"Server relay error: {e}")
                break
    
    def _injection_loop(self):
        """Process injection queue - send our packets to server."""
        while self.running:
            try:
                packet_data = self.injection_queue.get(timeout=1.0)
                if self.server_socket_conn:
                    self.server_socket_conn.sendall(packet_data)
                    self.stats["packets_injected"] += 1
                    logger.info(f"Injected packet: {len(packet_data)} bytes")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Injection error: {e}")
    
    def _capture_packet(self, data: bytes, direction: str) -> CapturedPacket:
        """Capture and decode a packet."""
        packet = CapturedPacket(
            timestamp=time.time(),
            direction=direction,
            raw_data=data
        )
        
        # Try to decode
        try:
            # Check for length prefix
            if len(data) >= 4:
                length = struct.unpack('>I', data[:4])[0]
                is_compressed = bool(length & 0x80000000)
                actual_length = length & 0x7FFFFFFF
                
                payload = data[4:]
                
                # Decompress if needed
                if is_compressed:
                    import zlib
                    try:
                        payload = zlib.decompress(payload)
                    except:
                        pass
                
                # Decode AMF3
                decoder = RealTimeAMF3Decoder(payload)
                decoded = decoder.decode()
                
                if isinstance(decoded, dict):
                    packet.decoded = decoded
                    packet.command = decoded.get("cmd", "")
                    
                    # Track command statistics
                    if packet.command:
                        self.stats["commands_detected"][packet.command] = \
                            self.stats["commands_detected"].get(packet.command, 0) + 1
                        
                        # Call callback
                        if self.on_command_detected:
                            self.on_command_detected(packet)
        except Exception as e:
            logger.debug(f"Decode error: {e}")
        
        # Store packet
        self.captured_packets.append(packet)
        self.stats["packets_captured"] += 1
        self.stats["bytes_captured"] += len(data)
        
        # Call callback
        if self.on_packet_captured:
            self.on_packet_captured(packet)
        
        return packet
    
    def _update_session_state(self, packet: CapturedPacket):
        """Update session state from server responses."""
        if not packet.decoded:
            return
        
        decoded = packet.decoded
        
        # Check for login response
        if "player" in decoded:
            self.session.logged_in = True
            player = decoded.get("player", {})
            self.session.player_id = player.get("id", 0)
            self.session.username = player.get("name", "")
        
        # Check for castle data
        if "castles" in decoded:
            self.session.castles = decoded.get("castles", [])
        
        # Check for resources
        if "resources" in decoded:
            castle_id = decoded.get("castleId", 0)
            self.session.resources[castle_id] = decoded.get("resources", {})
        
        # Check for troops
        if "troops" in decoded:
            castle_id = decoded.get("castleId", 0)
            self.session.troops[castle_id] = decoded.get("troops", {})
        
        self.session.last_update = time.time()
    
    def inject_packet(self, packet_data: bytes) -> Dict:
        """Inject a packet into the server stream."""
        if not self.running:
            return {"error": "Proxy not running"}
        if not self.server_socket_conn:
            return {"error": "Not connected to server"}
        
        self.injection_queue.put(packet_data)
        return {"status": "queued", "size": len(packet_data)}
    
    def inject_command(self, cmd: str, data: Dict) -> Dict:
        """Inject a command (encodes to AMF3 automatically)."""
        from .amf3_tools import encode_evony_packet
        
        result = encode_evony_packet(cmd, data)
        packet_bytes = bytes.fromhex(result["with_prefix_hex"])
        
        return self.inject_packet(packet_bytes)
    
    def get_captured_packets(self, limit: int = 100, 
                              command_filter: str = None) -> List[Dict]:
        """Get recent captured packets."""
        packets = list(self.captured_packets)[-limit:]
        
        if command_filter:
            packets = [p for p in packets if command_filter.lower() in p.command.lower()]
        
        return [{
            "timestamp": p.timestamp,
            "direction": p.direction,
            "command": p.command,
            "decoded": p.decoded,
            "size": len(p.raw_data),
            "modified": p.modified
        } for p in packets]
    
    def get_session_state(self) -> Dict:
        """Get current session state."""
        return {
            "connected": self.session.connected,
            "logged_in": self.session.logged_in,
            "username": self.session.username,
            "player_id": self.session.player_id,
            "castles": len(self.session.castles),
            "resources": self.session.resources,
            "troops": self.session.troops,
            "last_update": self.session.last_update
        }
    
    def get_stats(self) -> Dict:
        """Get proxy statistics."""
        return {
            **self.stats,
            "running": self.running,
            "session": self.get_session_state()
        }

# ============================================================================
# GLOBAL PROXY INSTANCE
# ============================================================================

_proxy_instance: Optional[ProxyHook] = None

def get_proxy() -> ProxyHook:
    """Get or create global proxy instance."""
    global _proxy_instance
    if _proxy_instance is None:
        _proxy_instance = ProxyHook()
    return _proxy_instance

# ============================================================================
# MCP HANDLER FUNCTIONS
# ============================================================================

def handle_proxy_start(args: Dict) -> Dict:
    """Start the proxy hook."""
    server = args.get("server", "na45")
    port = args.get("port", DEFAULT_PROXY_PORT)
    
    proxy = get_proxy()
    proxy.local_port = port
    return proxy.start(server)

def handle_proxy_stop(args: Dict) -> Dict:
    """Stop the proxy hook."""
    proxy = get_proxy()
    return proxy.stop()

def handle_proxy_status(args: Dict) -> Dict:
    """Get proxy status and stats."""
    proxy = get_proxy()
    return proxy.get_stats()

def handle_proxy_capture(args: Dict) -> Dict:
    """Get captured packets."""
    proxy = get_proxy()
    limit = args.get("limit", 100)
    command_filter = args.get("filter", None)
    return {
        "packets": proxy.get_captured_packets(limit, command_filter),
        "total_captured": proxy.stats["packets_captured"]
    }

def handle_proxy_inject(args: Dict) -> Dict:
    """Inject a command through proxy."""
    proxy = get_proxy()
    cmd = args.get("cmd", "")
    data = args.get("data", {})
    
    if not cmd:
        return {"error": "Command required"}
    
    return proxy.inject_command(cmd, data)

def handle_proxy_session(args: Dict) -> Dict:
    """Get current session state (resources, troops, etc)."""
    proxy = get_proxy()
    return proxy.get_session_state()

# Handler registry
PROXY_HANDLERS = {
    "proxy_start": handle_proxy_start,
    "proxy_stop": handle_proxy_stop,
    "proxy_status": handle_proxy_status,
    "proxy_capture": handle_proxy_capture,
    "proxy_inject": handle_proxy_inject,
    "proxy_session": handle_proxy_session,
}

__all__ = [
    'ProxyHook',
    'get_proxy',
    'PROXY_HANDLERS',
    'handle_proxy_start',
    'handle_proxy_stop',
    'handle_proxy_status',
    'handle_proxy_capture',
    'handle_proxy_inject',
    'handle_proxy_session',
]
