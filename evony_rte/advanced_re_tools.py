"""
Evony RTE - Advanced Reverse Engineering Tools Integration
============================================================
Integrates critical RE tools for deep analysis:
- Zeek: Network security monitoring
- Ghidra: NSA RE framework
- ngrep: Network pattern grep
- tcpflow: TCP stream extraction  
- scapy: Packet crafting/manipulation
- radare2: RE framework with SWF support
- swfmill: SWF to XML conversion
- swftools: SWF manipulation suite
- flasm: Flash assembler/disassembler

All tools CLI-accessible by Windsurf/Claude.
"""

import os
import subprocess
import shutil
import json
import tempfile
import struct
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Base paths
EVONY_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "re_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Platform detection
IS_WINDOWS = os.name == 'nt'
WSL_DISTRO = "Ubuntu"  # Default WSL distro

# ============================================================================
# PROCESS MANAGEMENT - PREVENT DUPLICATE LAUNCHES
# ============================================================================

# Process names to check before launching (prevents 10-50 copies being opened)
# Use EXACT process names to avoid false positives
PROCESS_NAMES = {
    "ffdec": ["ffdec.exe"],  # FFDec specific process
    "ghidra": ["ghidraRun.exe", "ghidra.exe"],  # Ghidra specific processes  
    "wireshark": ["Wireshark.exe"],
    "radare2": ["radare2.exe"],  # Removed r2.exe - too many false positives
    "tshark": ["tshark.exe"],
}

# Tool paths for safe launching
TOOL_PATHS = {
    "ffdec": r"C:\Program Files (x86)\FFDec\ffdec.exe",
    "ghidra": r"C:\ProgramData\chocolatey\lib\ghidra\tools\ghidra_12.0_PUBLIC\ghidraRun.bat",
    "wireshark": r"C:\Program Files\Wireshark\Wireshark.exe",
}

# Global launch guard - tracks what we've launched this session
_LAUNCHED_THIS_SESSION = set()

def is_process_running(process_names: List[str]) -> Dict:
    """
    Check if any of the given process names are currently running.
    Returns dict with running status and process info.
    """
    if not IS_WINDOWS:
        return {"running": False, "processes": []}
    
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10
        )
        running_processes = []
        for line in result.stdout.split('\n'):
            if not line.strip():
                continue
            parts = line.strip().strip('"').split('","')
            if len(parts) >= 2:
                actual_name = parts[0].strip('"')
                # EXACT match only - prevents false positives like LEDKeeper2.exe matching r2.exe
                for proc_name in process_names:
                    if actual_name.lower() == proc_name.lower():
                        running_processes.append({
                            "name": actual_name,
                            "pid": parts[1] if len(parts) > 1 else "unknown"
                        })
                        break
        
        return {
            "running": len(running_processes) > 0,
            "processes": running_processes,
            "count": len(running_processes)
        }
    except Exception as e:
        return {"running": False, "error": str(e), "processes": []}

def check_tool_running(tool_id: str) -> Dict:
    """Check if a specific tool is already running."""
    if tool_id not in PROCESS_NAMES:
        return {"running": False, "tool": tool_id, "note": "No process check defined"}
    
    return is_process_running(PROCESS_NAMES[tool_id])

def launch_if_not_running(tool_id: str, launch_path: str = None, launch_args: List[str] = None) -> Dict:
    """
    Launch a tool only if it's not already running.
    Prevents duplicate processes from being spawned.
    
    FAILSAFE CHECKS:
    1. Check if process is already running via tasklist
    2. Check if we already launched it this session
    3. Verify the executable exists before launching
    """
    global _LAUNCHED_THIS_SESSION
    
    # Use default path if not provided
    if not launch_path and tool_id in TOOL_PATHS:
        launch_path = TOOL_PATHS[tool_id]
    
    if not launch_path:
        return {
            "launched": False,
            "error": f"No launch path for {tool_id}",
            "hint": "Provide launch_path or add to TOOL_PATHS"
        }
    
    # FAILSAFE 1: Check if process is already running
    status = check_tool_running(tool_id)
    if status.get("running"):
        return {
            "launched": False,
            "already_running": True,
            "failsafe": "process_check",
            "processes": status.get("processes", []),
            "message": f"{tool_id} is already running ({status.get('count', 1)} instance(s)) - NOT launching duplicate"
        }
    
    # FAILSAFE 2: Check if we launched it this session (even if process check missed it)
    if tool_id in _LAUNCHED_THIS_SESSION:
        return {
            "launched": False,
            "already_launched_this_session": True,
            "failsafe": "session_guard",
            "message": f"{tool_id} was already launched this session - NOT launching again"
        }
    
    # FAILSAFE 3: Verify executable exists
    if not Path(launch_path).exists():
        return {
            "launched": False,
            "error": f"Executable not found: {launch_path}",
            "failsafe": "path_check"
        }
    
    # All failsafes passed - launch the tool
    try:
        args = [launch_path] + (launch_args or [])
        subprocess.Popen(args, shell=True)
        _LAUNCHED_THIS_SESSION.add(tool_id)  # Mark as launched
        return {
            "launched": True,
            "path": launch_path,
            "args": launch_args,
            "message": f"{tool_id} launched successfully",
            "failsafes_passed": ["process_check", "session_guard", "path_check"]
        }
    except Exception as e:
        return {
            "launched": False,
            "error": str(e),
            "path": launch_path
        }

def safe_tool_launch(tool_id: str, file_to_open: str = None) -> Dict:
    """
    MASTER FAILSAFE FUNCTION - Use this for all tool launches.
    Handles FFDec, Ghidra, Wireshark with full duplicate prevention.
    """
    tool_id = tool_id.lower()
    
    # Get tool path
    if tool_id not in TOOL_PATHS:
        return {
            "error": f"Unknown tool: {tool_id}",
            "available_tools": list(TOOL_PATHS.keys())
        }
    
    launch_path = TOOL_PATHS[tool_id]
    launch_args = [file_to_open] if file_to_open else None
    
    return launch_if_not_running(tool_id, launch_path, launch_args)

def get_all_running_tools() -> Dict:
    """Get status of all monitored tools - whether they're running or not."""
    result = {}
    for tool_id in PROCESS_NAMES:
        result[tool_id] = check_tool_running(tool_id)
    result["launched_this_session"] = list(_LAUNCHED_THIS_SESSION)
    return result

def reset_session_guard():
    """Reset the session launch guard (use with caution)."""
    global _LAUNCHED_THIS_SESSION
    _LAUNCHED_THIS_SESSION = set()
    return {"reset": True, "message": "Session guard cleared"}

# ============================================================================
# WSL INTEGRATION FOR WINDOWS
# ============================================================================

def check_wsl_available() -> bool:
    """Check if WSL is available on Windows."""
    if not IS_WINDOWS:
        return False
    try:
        result = subprocess.run(["wsl", "--list"], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def check_wsl_tool(tool_name: str) -> Optional[str]:
    """Check if a tool is available in WSL."""
    if not IS_WINDOWS:
        return None
    try:
        result = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "-e", "which", tool_name],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"wsl:{result.stdout.strip()}"
        # Check common install paths (e.g., zeek at /opt/zeek/bin/)
        alt_paths = [f"/opt/{tool_name}/bin/{tool_name}", f"/usr/local/bin/{tool_name}"]
        for path in alt_paths:
            result = subprocess.run(
                ["wsl", "-d", WSL_DISTRO, "-e", "test", "-x", path],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return f"wsl:{path}"
    except:
        pass
    return None

def run_wsl_command(cmd: List[str], timeout: int = 60, cwd: str = None) -> Dict:
    """Run a command in WSL and return result."""
    if not IS_WINDOWS:
        return {"error": "WSL only available on Windows"}
    
    wsl_cmd = ["wsl", "-d", WSL_DISTRO, "-e"] + cmd
    
    try:
        result = subprocess.run(
            wsl_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out", "timeout": timeout}
    except Exception as e:
        return {"error": str(e)}

def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    if not win_path:
        return win_path
    # Convert C:\path\to\file to /mnt/c/path/to/file
    path = win_path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path

def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path."""
    if not wsl_path:
        return wsl_path
    if wsl_path.startswith("/mnt/") and len(wsl_path) > 6:
        drive = wsl_path[5].upper()
        return f"{drive}:{wsl_path[6:]}".replace("/", "\\")
    return wsl_path

WSL_AVAILABLE = check_wsl_available() if IS_WINDOWS else False

# ============================================================================
# TOOL DETECTION AND STATUS
# ============================================================================

ADVANCED_TOOLS = {
    # Network Analysis
    "zeek": {
        "name": "Zeek",
        "commands": ["zeek", "zeek-cut"],
        "description": "Network security monitor for protocol analysis",
        "install_hint": "choco install zeek OR apt install zeek",
        "category": "network"
    },
    "ngrep": {
        "name": "ngrep",
        "commands": ["ngrep"],
        "description": "Network grep for pattern searching in packets",
        "install_hint": "choco install ngrep OR apt install ngrep",
        "category": "network"
    },
    "tcpflow": {
        "name": "tcpflow",
        "commands": ["tcpflow"],
        "description": "TCP stream extraction and reassembly",
        "install_hint": "apt install tcpflow (Linux) OR build from source",
        "category": "network"
    },
    "scapy": {
        "name": "Scapy",
        "commands": ["scapy"],
        "description": "Python packet manipulation library",
        "install_hint": "pip install scapy",
        "category": "network",
        "python_module": "scapy"
    },
    
    # Reverse Engineering
    "ghidra": {
        "name": "Ghidra",
        "commands": ["ghidraRun", "analyzeHeadless"],
        "description": "NSA reverse engineering framework",
        "install_hint": "Download from https://ghidra-sre.org/",
        "category": "re",
        "env_var": "GHIDRA_HOME",
        "windows_paths": [
            Path("C:/ProgramData/chocolatey/lib/ghidra/tools/ghidra_12.0_PUBLIC"),
            Path("C:/ProgramData/chocolatey/lib/ghidra/tools/ghidra_11.2_PUBLIC"),
            Path("C:/Tools/ghidra"),
            Path("C:/ghidra"),
        ]
    },
    "radare2": {
        "name": "radare2",
        "commands": ["r2", "radare2", "rabin2", "rasm2"],
        "description": "RE framework with SWF/ABC support",
        "install_hint": "choco install radare2 OR apt install radare2",
        "category": "re"
    },
    
    # Flash/SWF Tools
    "swfmill": {
        "name": "swfmill",
        "commands": ["swfmill"],
        "description": "SWF to XML conversion and manipulation",
        "install_hint": "apt install swfmill OR brew install swfmill",
        "category": "flash"
    },
    "swftools": {
        "name": "SWFTools",
        "commands": ["swfextract", "swfdump", "swfcombine", "swfstrings"],
        "description": "SWF manipulation suite",
        "install_hint": "Build from https://github.com/swftools/swftools",
        "category": "flash",
        "windows_paths": [
            Path("C:/Tools/swftools"),
            Path("C:/Program Files/SWFTools"),
            Path("C:/Program Files (x86)/SWFTools"),
        ],
        "source_repo": "C:/Tools/swftools",
        "fallback_tool": "ffdec"
    },
    "ffdec": {
        "name": "FFDec/JPEXS",
        "commands": ["ffdec", "ffdec-cli"],
        "description": "Flash decompiler with deobfuscation",
        "install_hint": "Download from https://github.com/jindrapetrik/jpexs-decompiler",
        "category": "flash",
        "jar_file": True,
        "windows_paths": [
            Path("C:/Program Files/FFDec"),
            Path("C:/Program Files (x86)/FFDec"),
            Path("C:/Tools/FFDec"),
        ]
    }
}

def find_executable(name: str, tool_id: str = None) -> Optional[str]:
    """Find executable in PATH, common locations, or tool-specific Windows paths."""
    # Check PATH first
    path = shutil.which(name)
    if path:
        return path
    
    # Check tool-specific Windows paths first
    if tool_id and tool_id in ADVANCED_TOOLS:
        tool_info = ADVANCED_TOOLS[tool_id]
        windows_paths = tool_info.get("windows_paths", [])
        for win_path in windows_paths:
            if win_path.exists():
                # Check for .exe, .bat, .cmd extensions
                for ext in [".exe", ".bat", ".cmd", ""]:
                    exe = win_path / f"{name}{ext}"
                    if exe.exists():
                        return str(exe)
                # Check bin subdirectory
                for ext in [".exe", ".bat", ".cmd", ""]:
                    exe = win_path / "bin" / f"{name}{ext}"
                    if exe.exists():
                        return str(exe)
                # Check support subdirectory (Ghidra)
                for ext in [".exe", ".bat", ".cmd", ""]:
                    exe = win_path / "support" / f"{name}{ext}"
                    if exe.exists():
                        return str(exe)
                # Check src subdirectory (for source builds)
                for ext in [".exe", ".bat", ".cmd", ""]:
                    exe = win_path / "src" / f"{name}{ext}"
                    if exe.exists():
                        return str(exe)
    
    # Check common Windows locations
    common_paths = [
        Path("C:/Tools"),
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
        Path(os.environ.get("LOCALAPPDATA", "")),
        Path(os.environ.get("PROGRAMDATA", "")),
        Path(os.environ.get("CHOCOLATEY_LIB", "C:/ProgramData/chocolatey/lib")),
        Path.home() / "bin",
        Path.home() / ".local/bin",
    ]
    
    for base in common_paths:
        if not base or not base.exists():
            continue
        # Direct check
        exe = base / f"{name}.exe"
        if exe.exists():
            return str(exe)
        # Check subdirectories
        for subdir in ["bin", name, f"{name}/bin", f"{name}/tools"]:
            exe = base / subdir / f"{name}.exe"
            if exe.exists():
                return str(exe)
    
    return None

def get_fallback_tool(tool_id: str) -> Optional[str]:
    """Get fallback tool if primary tool not available."""
    tool_info = ADVANCED_TOOLS.get(tool_id, {})
    fallback = tool_info.get("fallback_tool")
    if fallback and fallback in ADVANCED_TOOLS:
        fallback_status = get_tool_status(fallback)
        if fallback_status.get("installed"):
            return fallback
    return None

def check_python_module(module: str) -> bool:
    """Check if Python module is installed."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False

def get_tool_status(tool_id: str) -> Dict:
    """Get status of a specific tool."""
    tool_info = ADVANCED_TOOLS.get(tool_id, {})
    if not tool_info:
        return {"error": f"Unknown tool: {tool_id}"}
    
    result = {
        "name": tool_info["name"],
        "description": tool_info["description"],
        "category": tool_info["category"],
        "installed": False,
        "path": None,
        "version": None,
        "install_hint": tool_info["install_hint"],
        "wsl": False
    }
    
    # Check Python module if applicable
    if "python_module" in tool_info:
        if check_python_module(tool_info["python_module"]):
            result["installed"] = True
            result["path"] = "Python module"
            try:
                mod = __import__(tool_info["python_module"])
                result["version"] = getattr(mod, "__version__", "unknown")
            except:
                pass
        return result
    
    # Check executables (with tool-specific Windows paths)
    # Skip --version for tools that crash or launch GUI instead of returning version
    # FFDec: jansi library crash on Windows
    # Ghidra: ghidraRun --version launches the GUI instead of returning version
    # Wireshark: can launch GUI on some systems
    SKIP_VERSION_CHECK = {"ffdec", "ffdec-cli", "ghidraRun", "analyzeHeadless", "wireshark", "Wireshark"}
    
    for cmd in tool_info["commands"]:
        path = find_executable(cmd, tool_id)
        if path:
            result["installed"] = True
            result["path"] = path
            # Try to get version (skip for problematic tools)
            if cmd not in SKIP_VERSION_CHECK:
                try:
                    proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
                    version_output = proc.stdout or proc.stderr
                    result["version"] = version_output.strip()[:100]
                except:
                    pass
            else:
                result["version"] = "installed (version check skipped)"
            break
    
    # Check for source repo (needs building)
    if not result["installed"] and "source_repo" in tool_info:
        source_path = Path(tool_info["source_repo"])
        if source_path.exists():
            result["source_available"] = True
            result["source_path"] = str(source_path)
            result["install_hint"] = f"Source at {source_path} - needs building"
    
    # Add fallback info if tool has one
    if not result["installed"] and "fallback_tool" in tool_info:
        fallback = tool_info["fallback_tool"]
        result["fallback_tool"] = fallback
        result["fallback_note"] = f"Use '{fallback}' as alternative (FFDec covers most SWF functionality)"
    
    # Check environment variable
    if not result["installed"] and "env_var" in tool_info:
        env_path = os.environ.get(tool_info["env_var"])
        if env_path and Path(env_path).exists():
            result["installed"] = True
            result["path"] = env_path
    
    # Check WSL for Linux tools (on Windows)
    if not result["installed"] and IS_WINDOWS and WSL_AVAILABLE:
        for cmd in tool_info["commands"]:
            wsl_path = check_wsl_tool(cmd)
            if wsl_path:
                result["installed"] = True
                result["path"] = wsl_path
                result["wsl"] = True
                # Try to get version via WSL
                try:
                    wsl_result = run_wsl_command([cmd, "--version"], timeout=10)
                    if wsl_result.get("success"):
                        result["version"] = wsl_result.get("stdout", "").strip()[:100]
                except:
                    pass
                break
    
    return result

def get_all_advanced_tools_status() -> Dict:
    """Get status of all advanced RE tools."""
    tools = {}
    installed = 0
    
    for tool_id in ADVANCED_TOOLS:
        status = get_tool_status(tool_id)
        tools[tool_id] = status
        if status["installed"]:
            installed += 1
    
    return {
        "tools": tools,
        "installed": installed,
        "total": len(ADVANCED_TOOLS),
        "categories": {
            "network": [t for t, i in tools.items() if ADVANCED_TOOLS[t]["category"] == "network"],
            "re": [t for t, i in tools.items() if ADVANCED_TOOLS[t]["category"] == "re"],
            "flash": [t for t, i in tools.items() if ADVANCED_TOOLS[t]["category"] == "flash"]
        }
    }

# ============================================================================
# SCAPY - PACKET CRAFTING (Python-based, always available)
# ============================================================================

class ScapyTools:
    """Scapy-based packet crafting for AMF3 exploitation."""
    
    def __init__(self):
        self.scapy_available = check_python_module("scapy")
    
    def craft_amf_packet(self, target_ip: str, target_port: int, 
                         amf_payload: bytes, src_port: int = 0) -> Dict:
        """Craft TCP packet with AMF3 payload."""
        if not self.scapy_available:
            return {"error": "Scapy not installed. Run: pip install scapy"}
        
        try:
            from scapy.all import IP, TCP, Raw, conf
            conf.verb = 0  # Quiet mode
            
            # Build packet
            if src_port == 0:
                import random
                src_port = random.randint(49152, 65535)
            
            pkt = IP(dst=target_ip) / TCP(sport=src_port, dport=target_port) / Raw(load=amf_payload)
            
            return {
                "success": True,
                "packet_summary": pkt.summary(),
                "packet_hex": bytes(pkt).hex()[:500],
                "payload_size": len(amf_payload),
                "target": f"{target_ip}:{target_port}",
                "note": "Use send_packet to transmit"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def send_packet(self, packet_hex: str) -> Dict:
        """Send a crafted packet."""
        if not self.scapy_available:
            return {"error": "Scapy not installed"}
        
        try:
            from scapy.all import IP, send, conf
            conf.verb = 0
            
            packet_bytes = bytes.fromhex(packet_hex)
            pkt = IP(packet_bytes)
            send(pkt)
            
            return {"sent": True, "size": len(packet_bytes)}
        except Exception as e:
            return {"error": str(e)}
    
    def sniff_amf(self, interface: str = None, count: int = 10, 
                  filter_expr: str = "tcp port 80 or tcp port 443") -> Dict:
        """Sniff network for AMF3 packets."""
        if not self.scapy_available:
            return {"error": "Scapy not installed"}
        
        try:
            from scapy.all import sniff, Raw
            
            packets = []
            
            def packet_callback(pkt):
                if pkt.haslayer(Raw):
                    payload = bytes(pkt[Raw].load)
                    # Check for AMF3 markers
                    if b'\x00' in payload[:10]:  # AMF often starts with type markers
                        packets.append({
                            "src": pkt.sprintf("%IP.src%:%TCP.sport%"),
                            "dst": pkt.sprintf("%IP.dst%:%TCP.dport%"),
                            "payload_preview": payload[:100].hex(),
                            "size": len(payload)
                        })
            
            sniff(iface=interface, filter=filter_expr, prn=packet_callback, 
                  count=count, timeout=30, store=False)
            
            return {
                "captured": len(packets),
                "packets": packets,
                "filter": filter_expr
            }
        except Exception as e:
            return {"error": str(e)}
    
    def parse_pcap(self, pcap_file: str) -> Dict:
        """Parse pcap file for AMF3 content."""
        if not self.scapy_available:
            return {"error": "Scapy not installed"}
        
        if not Path(pcap_file).exists():
            return {"error": f"File not found: {pcap_file}"}
        
        try:
            from scapy.all import rdpcap, Raw
            
            packets = rdpcap(pcap_file)
            amf_packets = []
            
            for i, pkt in enumerate(packets):
                if pkt.haslayer(Raw):
                    payload = bytes(pkt[Raw].load)
                    # Look for AMF3 type markers (0x00-0x0C common)
                    if len(payload) > 4 and payload[4] in range(0, 13):
                        amf_packets.append({
                            "index": i,
                            "size": len(payload),
                            "type_marker": hex(payload[4]) if len(payload) > 4 else None,
                            "preview": payload[:50].hex()
                        })
            
            return {
                "total_packets": len(packets),
                "amf_candidates": len(amf_packets),
                "packets": amf_packets[:50]  # Limit output
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# NGREP - NETWORK PATTERN GREP
# ============================================================================

class NgrepTools:
    """Network grep for searching patterns in packets."""
    
    def __init__(self):
        self.ngrep_path = find_executable("ngrep")
        self.use_wsl = False
        # Check WSL if not found natively
        if not self.ngrep_path and IS_WINDOWS and WSL_AVAILABLE:
            wsl_path = check_wsl_tool("ngrep")
            if wsl_path:
                self.ngrep_path = "ngrep"
                self.use_wsl = True
    
    def search_pattern(self, pattern: str, interface: str = None,
                       filter_expr: str = "", timeout: int = 30) -> Dict:
        """Search for pattern in live traffic."""
        if not self.ngrep_path:
            return {"error": "ngrep not installed", "install": "apt install ngrep (WSL)"}
        
        try:
            cmd = ["ngrep", "-q", "-W", "byline"]
            if interface:
                cmd.extend(["-d", interface])
            cmd.append(pattern)
            if filter_expr:
                cmd.append(filter_expr)
            
            if self.use_wsl:
                result = run_wsl_command(cmd, timeout=timeout)
                if "error" in result:
                    return result
                proc_stdout = result.get("stdout", "")
            else:
                proc = subprocess.run([self.ngrep_path] + cmd[1:], capture_output=True, text=True, timeout=timeout)
                proc_stdout = proc.stdout
            
            matches = []
            for line in proc_stdout.split('\n'):
                if line.strip():
                    matches.append(line.strip())
            
            return {
                "pattern": pattern,
                "matches": matches[:100],
                "match_count": len(matches),
                "wsl": self.use_wsl
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "note": "No matches in timeout period"}
        except Exception as e:
            return {"error": str(e)}
    
    def search_amf_markers(self, interface: str = None, timeout: int = 30) -> Dict:
        """Search for AMF3 protocol markers in traffic."""
        # AMF3 type markers: 0x0A (object), 0x09 (array), 0x06 (string)
        return self.search_pattern("\\x0a|\\x09|\\x06", interface, "tcp", timeout)
    
    def search_pcap(self, pcap_file: str, pattern: str) -> Dict:
        """Search pattern in pcap file."""
        if not self.ngrep_path:
            return {"error": "ngrep not installed"}
        
        if not Path(pcap_file).exists():
            return {"error": f"File not found: {pcap_file}"}
        
        try:
            cmd = [self.ngrep_path, "-I", pcap_file, "-q", pattern]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            return {
                "file": pcap_file,
                "pattern": pattern,
                "output": proc.stdout[:5000],
                "matches_found": len(proc.stdout.strip().split('\n')) if proc.stdout.strip() else 0
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# TCPFLOW - TCP STREAM EXTRACTION
# ============================================================================

class TcpflowTools:
    """TCP stream extraction for AMF analysis."""
    
    def __init__(self):
        self.tcpflow_path = find_executable("tcpflow")
        self.use_wsl = False
        if not self.tcpflow_path and IS_WINDOWS and WSL_AVAILABLE:
            wsl_path = check_wsl_tool("tcpflow")
            if wsl_path:
                self.tcpflow_path = "tcpflow"
                self.use_wsl = True
    
    def extract_streams(self, pcap_file: str, output_dir: str = None) -> Dict:
        """Extract TCP streams from pcap."""
        if not self.tcpflow_path:
            return {"error": "tcpflow not installed", "install": "apt install tcpflow (WSL)"}
        
        if not Path(pcap_file).exists():
            return {"error": f"File not found: {pcap_file}"}
        
        if not output_dir:
            output_dir = str(OUTPUT_DIR / f"tcpflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            if self.use_wsl:
                wsl_pcap = windows_to_wsl_path(pcap_file)
                wsl_output = windows_to_wsl_path(output_dir)
                cmd = ["tcpflow", "-r", wsl_pcap, "-o", wsl_output]
                result = run_wsl_command(cmd, timeout=120)
                stderr = result.get("stderr", "")[:500] if result.get("stderr") else None
            else:
                cmd = [self.tcpflow_path, "-r", pcap_file, "-o", output_dir]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                stderr = proc.stderr[:500] if proc.stderr else None
            
            # List extracted streams
            streams = list(Path(output_dir).glob("*"))
            
            return {
                "output_dir": output_dir,
                "streams_extracted": len(streams),
                "stream_files": [str(s.name) for s in streams[:50]],
                "stderr": stderr,
                "wsl": self.use_wsl
            }
        except Exception as e:
            return {"error": str(e)}
    
    def capture_live(self, interface: str, duration: int = 30, 
                     output_dir: str = None) -> Dict:
        """Capture live TCP streams."""
        if not self.tcpflow_path:
            return {"error": "tcpflow not installed", "install": "apt install tcpflow (WSL)"}
        
        if not output_dir:
            output_dir = str(OUTPUT_DIR / f"tcpflow_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            cmd = [self.tcpflow_path, "-i", interface, "-o", output_dir]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            import time
            time.sleep(duration)
            proc.terminate()
            
            streams = list(Path(output_dir).glob("*"))
            
            return {
                "output_dir": output_dir,
                "duration": duration,
                "streams_captured": len(streams),
                "stream_files": [str(s.name) for s in streams[:50]]
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# RADARE2 - REVERSE ENGINEERING FRAMEWORK
# ============================================================================

class Radare2Tools:
    """Radare2 integration for binary/SWF analysis."""
    
    def __init__(self):
        self.r2_path = find_executable("r2") or find_executable("radare2")
        self.rabin2_path = find_executable("rabin2")
    
    def analyze_binary(self, file_path: str) -> Dict:
        """Analyze binary with radare2."""
        if not self.r2_path:
            return {"error": "radare2 not installed", "install": "choco install radare2"}
        
        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            # Run radare2 analysis commands
            cmd = [self.r2_path, "-q", "-c", "aa;afl;q", file_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            functions = []
            for line in proc.stdout.split('\n'):
                if line.strip():
                    functions.append(line.strip())
            
            return {
                "file": file_path,
                "functions_found": len(functions),
                "functions": functions[:100],
                "analysis": "basic"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_strings(self, file_path: str, min_length: int = 4) -> Dict:
        """Extract strings from binary."""
        if not self.rabin2_path:
            if not self.r2_path:
                return {"error": "radare2 not installed"}
            # Fall back to r2
            try:
                cmd = [self.r2_path, "-q", "-c", f"izz~[2:]", file_path]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                strings = [s.strip() for s in proc.stdout.split('\n') if len(s.strip()) >= min_length]
                return {"strings": strings[:500], "total": len(strings)}
            except Exception as e:
                return {"error": str(e)}
        
        try:
            cmd = [self.rabin2_path, "-z", file_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            strings = []
            for line in proc.stdout.split('\n'):
                if len(line) >= min_length:
                    strings.append(line.strip())
            
            return {
                "file": file_path,
                "strings": strings[:500],
                "total": len(strings),
                "min_length": min_length
            }
        except Exception as e:
            return {"error": str(e)}
    
    def disassemble(self, file_path: str, address: str = "main", 
                    count: int = 50) -> Dict:
        """Disassemble at address."""
        if not self.r2_path:
            return {"error": "radare2 not installed"}
        
        try:
            cmd = [self.r2_path, "-q", "-c", f"aa;s {address};pd {count};q", file_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            return {
                "file": file_path,
                "address": address,
                "disassembly": proc.stdout[:5000]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def search_pattern(self, file_path: str, pattern: str) -> Dict:
        """Search for pattern in binary."""
        if not self.r2_path:
            return {"error": "radare2 not installed"}
        
        try:
            # Search hex pattern
            cmd = [self.r2_path, "-q", "-c", f"/ {pattern};q", file_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            matches = []
            for line in proc.stdout.split('\n'):
                if "0x" in line:
                    matches.append(line.strip())
            
            return {
                "pattern": pattern,
                "matches": matches[:100],
                "match_count": len(matches)
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# GHIDRA - NSA RE FRAMEWORK
# ============================================================================

class GhidraTools:
    """Ghidra integration for advanced RE."""
    
    # Known Ghidra installation paths
    GHIDRA_PATHS = [
        Path(os.environ.get("GHIDRA_HOME", "")) if os.environ.get("GHIDRA_HOME") else None,
        Path(r"C:\ProgramData\chocolatey\lib\ghidra\tools\ghidra_12.0_PUBLIC"),
        Path(r"C:\ProgramData\chocolatey\lib\ghidra\tools\ghidra_11.2_PUBLIC"),
        Path(r"C:\ghidra"),
        Path(r"C:\Tools\ghidra"),
        Path.home() / "ghidra",
    ]
    
    def __init__(self):
        self.ghidra_home = None
        self.analyze_headless = None
        
        # Auto-detect Ghidra installation
        for path in self.GHIDRA_PATHS:
            if path and path.exists():
                self.ghidra_home = str(path)
                # Check for headless analyzer
                headless_bat = path / "support" / "analyzeHeadless.bat"
                if headless_bat.exists():
                    self.analyze_headless = str(headless_bat)
                    break
                headless = path / "support" / "analyzeHeadless"
                if headless.exists():
                    self.analyze_headless = str(headless)
                    break
    
    def analyze_file(self, file_path: str, project_name: str = "evony_analysis") -> Dict:
        """Analyze file with Ghidra headless."""
        if not self.analyze_headless:
            return {
                "error": "Ghidra not configured",
                "install": "Set GHIDRA_HOME environment variable",
                "download": "https://ghidra-sre.org/"
            }
        
        if not Path(file_path).exists():
            return {"error": f"File not found: {file_path}"}
        
        project_dir = OUTPUT_DIR / "ghidra_projects"
        project_dir.mkdir(exist_ok=True)
        
        try:
            cmd = [
                self.analyze_headless,
                str(project_dir),
                project_name,
                "-import", file_path,
                "-overwrite",
                "-scriptPath", str(Path(__file__).parent),
                "-postScript", "ExportFunctions.java"
            ]
            
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return {
                "status": "analyzed",
                "project": project_name,
                "project_dir": str(project_dir),
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:1000] if proc.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {"error": "Analysis timeout (5 min limit)"}
        except Exception as e:
            return {"error": str(e)}
    
    def export_functions(self, project_name: str) -> Dict:
        """Export function list from Ghidra project."""
        # This would require a Ghidra script - placeholder
        return {
            "status": "not_implemented",
            "note": "Requires Ghidra script integration",
            "project": project_name
        }

# ============================================================================
# SWFMILL - SWF TO XML CONVERSION
# ============================================================================

class SwfmillTools:
    """SWF to XML conversion for editing."""
    
    def __init__(self):
        self.swfmill_path = find_executable("swfmill")
        self.use_wsl = False
        if not self.swfmill_path and IS_WINDOWS and WSL_AVAILABLE:
            wsl_path = check_wsl_tool("swfmill")
            if wsl_path:
                self.swfmill_path = "swfmill"
                self.use_wsl = True
    
    def swf_to_xml(self, swf_file: str, output_xml: str = None) -> Dict:
        """Convert SWF to editable XML."""
        if not self.swfmill_path:
            return {"error": "swfmill not installed", "install": "apt install swfmill (WSL)"}
        
        if not Path(swf_file).exists():
            return {"error": f"File not found: {swf_file}"}
        
        if not output_xml:
            output_xml = str(OUTPUT_DIR / f"{Path(swf_file).stem}.xml")
        
        try:
            if self.use_wsl:
                wsl_swf = windows_to_wsl_path(swf_file)
                wsl_xml = windows_to_wsl_path(output_xml)
                cmd = ["swfmill", "swf2xml", wsl_swf, wsl_xml]
                result = run_wsl_command(cmd, timeout=120)
                stderr = result.get("stderr", "")
            else:
                cmd = [self.swfmill_path, "swf2xml", swf_file, output_xml]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                stderr = proc.stderr
            
            if Path(output_xml).exists():
                size = Path(output_xml).stat().st_size
                # Read first 1000 chars as preview
                with open(output_xml, 'r', encoding='utf-8', errors='ignore') as f:
                    preview = f.read(1000)
                
                return {
                    "success": True,
                    "input": swf_file,
                    "output": output_xml,
                    "xml_size": size,
                    "preview": preview,
                    "wsl": self.use_wsl
                }
            else:
                return {"error": "Conversion failed", "stderr": stderr}
        except Exception as e:
            return {"error": str(e)}
    
    def xml_to_swf(self, xml_file: str, output_swf: str = None) -> Dict:
        """Convert XML back to SWF."""
        if not self.swfmill_path:
            return {"error": "swfmill not installed"}
        
        if not Path(xml_file).exists():
            return {"error": f"File not found: {xml_file}"}
        
        if not output_swf:
            output_swf = str(OUTPUT_DIR / f"{Path(xml_file).stem}_patched.swf")
        
        try:
            cmd = [self.swfmill_path, "xml2swf", xml_file, output_swf]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if Path(output_swf).exists():
                return {
                    "success": True,
                    "input": xml_file,
                    "output": output_swf,
                    "swf_size": Path(output_swf).stat().st_size
                }
            else:
                return {"error": "Conversion failed", "stderr": proc.stderr}
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# SWFTOOLS - SWF MANIPULATION SUITE
# ============================================================================

class SwftoolsIntegration:
    """SWFTools suite integration."""
    
    def __init__(self):
        self.swfextract = find_executable("swfextract")
        self.swfdump = find_executable("swfdump")
        self.swfstrings = find_executable("swfstrings")
        self.swfcombine = find_executable("swfcombine")
    
    def extract(self, swf_file: str, extract_type: str = "all") -> Dict:
        """Extract components from SWF."""
        if not self.swfextract:
            return {"error": "swfextract not installed", "install": "Build from swftools source"}
        
        if not Path(swf_file).exists():
            return {"error": f"File not found: {swf_file}"}
        
        output_dir = OUTPUT_DIR / f"swfextract_{Path(swf_file).stem}"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # First get info
            cmd = [self.swfextract, swf_file]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            extracted = []
            
            if extract_type in ["all", "shapes"]:
                # Extract shapes
                shapes_cmd = [self.swfextract, "-a", "-o", str(output_dir / "shapes"), swf_file]
                subprocess.run(shapes_cmd, capture_output=True, timeout=60)
                extracted.append("shapes")
            
            if extract_type in ["all", "images"]:
                # Extract images
                img_cmd = [self.swfextract, "-P", "-o", str(output_dir / "images"), swf_file]
                subprocess.run(img_cmd, capture_output=True, timeout=60)
                extracted.append("images")
            
            if extract_type in ["all", "sounds"]:
                # Extract sounds
                snd_cmd = [self.swfextract, "-m", "-o", str(output_dir / "sounds"), swf_file]
                subprocess.run(snd_cmd, capture_output=True, timeout=60)
                extracted.append("sounds")
            
            return {
                "info": proc.stdout[:2000],
                "output_dir": str(output_dir),
                "extracted": extracted
            }
        except Exception as e:
            return {"error": str(e)}
    
    def dump_info(self, swf_file: str) -> Dict:
        """Dump SWF structure info."""
        if not self.swfdump:
            return {"error": "swfdump not installed"}
        
        try:
            cmd = [self.swfdump, "-a", swf_file]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            return {
                "file": swf_file,
                "dump": proc.stdout[:5000],
                "tags_found": proc.stdout.count("TAG")
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_strings(self, swf_file: str) -> Dict:
        """Extract strings from SWF."""
        if not self.swfstrings:
            return {"error": "swfstrings not installed"}
        
        try:
            cmd = [self.swfstrings, swf_file]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            strings = [s.strip() for s in proc.stdout.split('\n') if s.strip()]
            
            return {
                "file": swf_file,
                "strings": strings[:500],
                "total": len(strings)
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# FLASM - FLASH ASSEMBLER/DISASSEMBLER
# ============================================================================

class FlasmTools:
    """Flasm integration for Flash bytecode."""
    
    def __init__(self):
        self.flasm_path = find_executable("flasm")
    
    def disassemble(self, swf_file: str, output_file: str = None) -> Dict:
        """Disassemble SWF to flasm format."""
        if not self.flasm_path:
            return {"error": "flasm not installed", "install": "Build from matthiasbock/flasm"}
        
        if not Path(swf_file).exists():
            return {"error": f"File not found: {swf_file}"}
        
        if not output_file:
            output_file = str(OUTPUT_DIR / f"{Path(swf_file).stem}.flm")
        
        try:
            cmd = [self.flasm_path, "-d", swf_file]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Save output
            with open(output_file, 'w') as f:
                f.write(proc.stdout)
            
            return {
                "success": True,
                "input": swf_file,
                "output": output_file,
                "size": len(proc.stdout),
                "preview": proc.stdout[:2000]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def assemble(self, flasm_file: str, output_swf: str = None) -> Dict:
        """Assemble flasm to SWF."""
        if not self.flasm_path:
            return {"error": "flasm not installed"}
        
        if not Path(flasm_file).exists():
            return {"error": f"File not found: {flasm_file}"}
        
        if not output_swf:
            output_swf = str(OUTPUT_DIR / f"{Path(flasm_file).stem}_assembled.swf")
        
        try:
            cmd = [self.flasm_path, "-a", flasm_file]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            return {
                "success": proc.returncode == 0,
                "input": flasm_file,
                "output": output_swf if proc.returncode == 0 else None,
                "stderr": proc.stderr[:1000] if proc.stderr else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def update_swf(self, swf_file: str, flasm_file: str) -> Dict:
        """Update SWF with modified flasm."""
        if not self.flasm_path:
            return {"error": "flasm not installed"}
        
        try:
            cmd = [self.flasm_path, "-u", swf_file, flasm_file]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            return {
                "success": proc.returncode == 0,
                "swf": swf_file,
                "flasm": flasm_file,
                "stderr": proc.stderr[:500] if proc.stderr else None
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# ZEEK - NETWORK SECURITY MONITOR
# ============================================================================

class ZeekTools:
    """Zeek network security monitoring."""
    
    def __init__(self):
        self.zeek_path = find_executable("zeek")
        self.zeek_cut = find_executable("zeek-cut")
        self.use_wsl = False
        if not self.zeek_path and IS_WINDOWS and WSL_AVAILABLE:
            wsl_path = check_wsl_tool("zeek")
            if wsl_path:
                self.zeek_path = wsl_path.replace("wsl:", "")
                self.use_wsl = True
    
    def analyze_pcap(self, pcap_file: str, output_dir: str = None) -> Dict:
        """Analyze pcap with Zeek."""
        if not self.zeek_path:
            return {"error": "Zeek not installed", "install": "apt install zeek (WSL)"}
        
        if not Path(pcap_file).exists():
            return {"error": f"File not found: {pcap_file}"}
        
        if not output_dir:
            output_dir = str(OUTPUT_DIR / f"zeek_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            if self.use_wsl:
                wsl_pcap = windows_to_wsl_path(pcap_file)
                wsl_output = windows_to_wsl_path(output_dir)
                cmd = [self.zeek_path, "-r", wsl_pcap]
                result = run_wsl_command(["bash", "-c", f"cd {wsl_output} && {self.zeek_path} -r {wsl_pcap}"], timeout=120)
            else:
                cmd = [self.zeek_path, "-r", pcap_file]
                proc = subprocess.run(cmd, capture_output=True, text=True, 
                                      timeout=120, cwd=output_dir)
            
            # List generated log files
            logs = list(Path(output_dir).glob("*.log"))
            
            log_previews = {}
            for log in logs[:10]:
                try:
                    with open(log, 'r') as f:
                        log_previews[log.name] = f.read(500)
                except:
                    pass
            
            return {
                "output_dir": output_dir,
                "logs_generated": [l.name for l in logs],
                "log_previews": log_previews,
                "stderr": proc.stderr[:500] if proc.stderr else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    def extract_connections(self, log_dir: str) -> Dict:
        """Extract connection info from Zeek logs."""
        conn_log = Path(log_dir) / "conn.log"
        
        if not conn_log.exists():
            return {"error": "conn.log not found"}
        
        try:
            connections = []
            with open(conn_log, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 5:
                            connections.append({
                                "src_ip": parts[2] if len(parts) > 2 else "",
                                "src_port": parts[3] if len(parts) > 3 else "",
                                "dst_ip": parts[4] if len(parts) > 4 else "",
                                "dst_port": parts[5] if len(parts) > 5 else "",
                                "proto": parts[6] if len(parts) > 6 else ""
                            })
            
            return {
                "connections": connections[:100],
                "total": len(connections)
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# HANDLER FUNCTIONS FOR MCP
# ============================================================================

def handle_advanced_tools_status(args: Dict) -> Dict:
    """Get status of all advanced RE tools."""
    return get_all_advanced_tools_status()

def handle_scapy(args: Dict) -> Dict:
    """Scapy packet crafting operations."""
    action = args.get("action", "status")
    tools = ScapyTools()
    
    if action == "status":
        return {"installed": tools.scapy_available, "actions": ["status", "craft", "sniff", "parse_pcap", "interfaces"]}
    elif action == "interfaces":
        return {"installed": tools.scapy_available, "note": "Use scapy to list interfaces", "actions": ["status", "craft", "sniff"]}
    elif action == "craft":
        return tools.craft_amf_packet(
            args.get("target_ip", ""),
            args.get("target_port", 80),
            bytes.fromhex(args.get("payload_hex", ""))
        )
    elif action == "sniff":
        return tools.sniff_amf(
            args.get("interface"),
            args.get("count", 10),
            args.get("filter", "tcp")
        )
    elif action == "parse_pcap":
        return tools.parse_pcap(args.get("pcap_file", ""))
    
    return {"error": f"Unknown action: {action}"}

def handle_ngrep(args: Dict) -> Dict:
    """ngrep network pattern search."""
    action = args.get("action", "status")
    tools = NgrepTools()
    
    if action == "status":
        return {
            "installed": tools.ngrep_path is not None,
            "path": tools.ngrep_path,
            "wsl": tools.use_wsl,
            "actions": ["search", "amf", "pcap", "status"]
        }
    elif action == "search":
        pattern = args.get("pattern", "")
        if not pattern:
            return {
                "status": "ready",
                "installed": tools.ngrep_path is not None,
                "usage": "Provide pattern parameter"
            }
        return tools.search_pattern(
            pattern,
            args.get("interface"),
            args.get("filter", ""),
            args.get("timeout", 30)
        )
    elif action == "amf":
        return tools.search_amf_markers(args.get("interface"), args.get("timeout", 30))
    elif action == "pcap":
        return tools.search_pcap(args.get("pcap_file", ""), args.get("pattern", ""))
    
    return {"error": f"Unknown action: {action}"}

def handle_tcpflow(args: Dict) -> Dict:
    """tcpflow stream extraction."""
    action = args.get("action", "status")
    tools = TcpflowTools()
    
    if action == "status":
        return {"installed": tools.tcpflow_path is not None, "path": tools.tcpflow_path, "wsl": tools.use_wsl, "actions": ["status", "extract", "capture"]}
    elif action == "extract":
        return tools.extract_streams(args.get("pcap_file", ""), args.get("output_dir"))
    elif action == "capture":
        return tools.capture_live(
            args.get("interface", ""),
            args.get("duration", 30),
            args.get("output_dir")
        )
    
    return {"error": f"Unknown action: {action}"}

def handle_radare2(args: Dict) -> Dict:
    """radare2 RE operations."""
    action = args.get("action", "status")
    tools = Radare2Tools()
    
    if action == "status":
        return {"installed": tools.r2_path is not None, "path": tools.r2_path, "actions": ["status", "analyze", "strings", "disasm", "search"]}
    elif action == "analyze":
        return tools.analyze_binary(args.get("file", ""))
    elif action == "strings":
        return tools.get_strings(args.get("file", ""), args.get("min_length", 4))
    elif action == "disasm":
        return tools.disassemble(
            args.get("file", ""),
            args.get("address", "main"),
            args.get("count", 50)
        )
    elif action == "search":
        return tools.search_pattern(args.get("file", ""), args.get("pattern", ""))
    
    return {"error": f"Unknown action: {action}"}

def handle_ghidra(args: Dict) -> Dict:
    """Ghidra RE operations."""
    action = args.get("action", "analyze")
    tools = GhidraTools()
    
    if action == "analyze":
        return tools.analyze_file(args.get("file", ""), args.get("project", "evony"))
    elif action == "export":
        return tools.export_functions(args.get("project", ""))
    elif action == "status":
        return {
            "ghidra_home": tools.ghidra_home,
            "headless_available": tools.analyze_headless is not None
        }
    
    return {"error": f"Unknown action: {action}"}

def handle_swfmill(args: Dict) -> Dict:
    """swfmill SWF/XML conversion."""
    action = args.get("action", "status")
    tools = SwfmillTools()
    
    if action == "status":
        return {"installed": tools.swfmill_path is not None, "path": tools.swfmill_path, "wsl": tools.use_wsl, "actions": ["status", "to_xml", "to_swf"]}
    elif action == "to_xml":
        return tools.swf_to_xml(args.get("swf_file", ""), args.get("output"))
    elif action == "to_swf":
        return tools.xml_to_swf(args.get("xml_file", ""), args.get("output"))
    
    return {"error": f"Unknown action: {action}"}

def handle_swftools(args: Dict) -> Dict:
    """swftools operations - uses FFDec via WSL as alternative."""
    action = args.get("action", "extract")
    swf_file = args.get("swf_file", "")
    
    # Use FFDec via WSL as alternative (swftools not available)
    ffdec = FFDecTools()
    if ffdec.ffdec_path:
        if action == "extract":
            result = ffdec.export_all(swf_file) if swf_file else {"error": "swf_file required"}
            if result.get("success"):
                result["note"] = "Used FFDec (WSL) as alternative to swftools"
            return result
        elif action == "dump":
            return ffdec.get_abc_list(swf_file) if swf_file else {"error": "swf_file required"}
        elif action == "strings":
            result = ffdec.decompile(swf_file) if swf_file else {"error": "swf_file required"}
            if result.get("success"):
                result["note"] = "Decompiled scripts contain strings"
            return result
        elif action == "status":
            return {"installed": True, "path": ffdec.ffdec_path, "wsl": ffdec.use_wsl, "alternative": "FFDec"}
    
    # Fallback to swftools if available
    tools = SwftoolsIntegration()
    if action == "extract":
        return tools.extract(swf_file, args.get("type", "all"))
    elif action == "dump":
        return tools.dump_info(swf_file)
    elif action == "strings":
        return tools.get_strings(swf_file)
    
    return {"error": f"Unknown action: {action}"}

def handle_flasm(args: Dict) -> Dict:
    """flasm operations - uses FFDec via WSL as alternative."""
    action = args.get("action", "disasm")
    swf_file = args.get("swf_file", "")
    
    # Use FFDec via WSL as alternative (flasm not available)
    ffdec = FFDecTools()
    if ffdec.ffdec_path:
        if action == "disasm":
            result = ffdec.decompile(swf_file) if swf_file else {"error": "swf_file required"}
            if result.get("success"):
                result["note"] = "Used FFDec (WSL) as alternative to flasm"
            return result
        elif action == "status":
            return {"installed": True, "path": ffdec.ffdec_path, "wsl": ffdec.use_wsl, "alternative": "FFDec"}
        elif action == "asm":
            return {"error": "Assembly not supported with FFDec alternative", "hint": "Install flasm for assembly"}
        elif action == "update":
            return {"error": "SWF update not supported with FFDec alternative", "hint": "Install flasm"}
    
    # Fallback to flasm if available
    tools = FlasmTools()
    if action == "disasm":
        return tools.disassemble(swf_file, args.get("output"))
    elif action == "asm":
        return tools.assemble(args.get("flasm_file", ""), args.get("output"))
    elif action == "update":
        return tools.update_swf(swf_file, args.get("flasm_file", ""))
    
    return {"error": f"Unknown action: {action}"}

def handle_zeek(args: Dict) -> Dict:
    """Zeek network analysis."""
    action = args.get("action", "status")
    tools = ZeekTools()
    
    if action == "status":
        return {"installed": tools.zeek_path is not None, "path": tools.zeek_path, "wsl": tools.use_wsl, "actions": ["status", "analyze", "connections"]}
    elif action == "analyze":
        return tools.analyze_pcap(args.get("pcap_file", ""), args.get("output_dir"))
    elif action == "connections":
        return tools.extract_connections(args.get("log_dir", ""))
    
    return {"error": f"Unknown action: {action}"}

# ============================================================================
# FFDEC/JPEXS - FLASH DECOMPILER (WSL CLI for performance)
# ============================================================================

class FFDecTools:
    """FFDec/JPEXS Flash decompiler - uses WSL CLI for better performance."""
    
    def __init__(self):
        self.use_wsl = False
        self.ffdec_path = None
        
        # ONLY use WSL for performance (Windows GUI is too slow - 2-3 min load time)
        if IS_WINDOWS and WSL_AVAILABLE:
            wsl_path = "/opt/ffdec/ffdec.sh"
            try:
                result = subprocess.run(
                    ["wsl", "-d", WSL_DISTRO, "-e", "test", "-x", wsl_path],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    self.ffdec_path = wsl_path
                    self.use_wsl = True
            except:
                pass
        # NO Windows fallback - Windows GUI is too slow (2-3 min load time)
    
    def decompile(self, swf_file: str, output_dir: str = None) -> Dict:
        """Decompile SWF to ActionScript."""
        if not self.ffdec_path:
            return {"error": "FFDec not installed", "install": "Copy ffdec to /opt/ffdec/ in WSL"}
        
        if not Path(swf_file).exists():
            return {"error": f"File not found: {swf_file}"}
        
        if not output_dir:
            output_dir = str(OUTPUT_DIR / f"ffdec_{Path(swf_file).stem}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            if self.use_wsl:
                wsl_swf = windows_to_wsl_path(swf_file)
                wsl_out = windows_to_wsl_path(output_dir)
                cmd = [self.ffdec_path, "-export", "script", wsl_out, wsl_swf]
                result = run_wsl_command(cmd, timeout=180)
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
            else:
                cmd = [self.ffdec_path, "-export", "script", output_dir, swf_file]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                stdout = proc.stdout
                stderr = proc.stderr
            
            # Count exported files
            scripts = list(Path(output_dir).rglob("*.as"))
            
            return {
                "success": True,
                "input": swf_file,
                "output_dir": output_dir,
                "scripts_exported": len(scripts),
                "sample_files": [str(s.relative_to(output_dir)) for s in scripts[:20]],
                "wsl": self.use_wsl,
                "note": "WSL CLI mode - faster than Windows GUI"
            }
        except subprocess.TimeoutExpired:
            return {"error": "Decompilation timed out (3 min limit)"}
        except Exception as e:
            return {"error": str(e)}
    
    def export_all(self, swf_file: str, output_dir: str = None) -> Dict:
        """Export all resources from SWF."""
        if not self.ffdec_path:
            return {"error": "FFDec not installed"}
        
        if not Path(swf_file).exists():
            return {"error": f"File not found: {swf_file}"}
        
        if not output_dir:
            output_dir = str(OUTPUT_DIR / f"ffdec_all_{Path(swf_file).stem}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            if self.use_wsl:
                wsl_swf = windows_to_wsl_path(swf_file)
                wsl_out = windows_to_wsl_path(output_dir)
                cmd = [self.ffdec_path, "-export", "all", wsl_out, wsl_swf]
                result = run_wsl_command(cmd, timeout=300)
            else:
                cmd = [self.ffdec_path, "-export", "all", output_dir, swf_file]
                subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Count exported files
            all_files = list(Path(output_dir).rglob("*"))
            files_by_type = {}
            for f in all_files:
                if f.is_file():
                    ext = f.suffix.lower() or "no_ext"
                    files_by_type[ext] = files_by_type.get(ext, 0) + 1
            
            return {
                "success": True,
                "input": swf_file,
                "output_dir": output_dir,
                "total_files": len([f for f in all_files if f.is_file()]),
                "by_type": files_by_type,
                "wsl": self.use_wsl
            }
        except subprocess.TimeoutExpired:
            return {"error": "Export timed out (5 min limit)"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_abc_list(self, swf_file: str) -> Dict:
        """List ABC (ActionScript ByteCode) tags in SWF."""
        if not self.ffdec_path:
            return {"error": "FFDec not installed"}
        
        if not Path(swf_file).exists():
            return {"error": f"File not found: {swf_file}"}
        
        try:
            if self.use_wsl:
                wsl_swf = windows_to_wsl_path(swf_file)
                cmd = [self.ffdec_path, "-dumpSWF", wsl_swf]
                result = run_wsl_command(cmd, timeout=60)
                stdout = result.get("stdout", "")
            else:
                cmd = [self.ffdec_path, "-dumpSWF", swf_file]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                stdout = proc.stdout
            
            # Parse ABC info from output
            abc_tags = []
            for line in stdout.split('\n'):
                if 'DoABC' in line or 'abc' in line.lower():
                    abc_tags.append(line.strip())
            
            return {
                "swf_file": swf_file,
                "abc_tags": abc_tags[:50],
                "dump_preview": stdout[:2000],
                "wsl": self.use_wsl
            }
        except Exception as e:
            return {"error": str(e)}

def handle_ffdec(args: Dict) -> Dict:
    """FFDec/JPEXS operations via WSL CLI."""
    action = args.get("action", "decompile")
    tools = FFDecTools()
    
    if action == "decompile":
        return tools.decompile(args.get("swf_file", ""), args.get("output_dir"))
    elif action == "export_all":
        return tools.export_all(args.get("swf_file", ""), args.get("output_dir"))
    elif action == "abc_list":
        return tools.get_abc_list(args.get("swf_file", ""))
    elif action == "status":
        return {
            "installed": tools.ffdec_path is not None,
            "path": tools.ffdec_path,
            "wsl_mode": tools.use_wsl,
            "note": "WSL CLI mode is 10-20x faster than Windows GUI"
        }
    
    return {"error": f"Unknown action: {action}"}

# Handler registry
ADVANCED_RE_HANDLERS = {
    "advanced_tools_status": handle_advanced_tools_status,
    "scapy": handle_scapy,
    "ngrep": handle_ngrep,
    "tcpflow": handle_tcpflow,
    "radare2": handle_radare2,
    "ghidra": handle_ghidra,
    "swfmill": handle_swfmill,
    "swftools": handle_swftools,
    "flasm_tool": handle_flasm,
    "zeek": handle_zeek,
    "ffdec": handle_ffdec,
}

__all__ = [
    'ADVANCED_RE_HANDLERS',
    'get_all_advanced_tools_status',
    'ScapyTools',
    'NgrepTools', 
    'TcpflowTools',
    'Radare2Tools',
    'GhidraTools',
    'SwfmillTools',
    'SwftoolsIntegration',
    'FlasmTools',
    'ZeekTools',
    'FFDecTools',
    'handle_ffdec',
]
