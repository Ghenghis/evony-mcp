"""
Evony RTE - Extended Tools
==========================
Additional tools for tshark, JPEXS, memory analysis, and full game integration.
"""

import subprocess
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Paths
EVONY_ROOT = Path(__file__).parent.parent
CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

# ============================================================================
# TSHARK TOOLS
# ============================================================================

def check_tshark() -> Dict:
    """Check if tshark is installed."""
    try:
        result = subprocess.run(["tshark", "-v"], capture_output=True, text=True, timeout=5)
        version = result.stdout.split('\n')[0] if result.returncode == 0 else None
        return {"installed": result.returncode == 0, "version": version}
    except FileNotFoundError:
        return {"installed": False, "error": "tshark not found. Install Wireshark."}
    except Exception as e:
        return {"installed": False, "error": str(e)}

def tshark_capture(interface: str = None, duration: int = 30, 
                   filter_expr: str = None, output_file: str = None) -> Dict:
    """Capture packets using tshark."""
    if not check_tshark().get("installed"):
        return {"error": "tshark not installed. Run: choco install wireshark"}
    
    if output_file is None:
        output_file = CAPTURES_DIR / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap"
    
    cmd = ["tshark", "-a", f"duration:{duration}", "-w", str(output_file)]
    
    if interface:
        cmd.extend(["-i", interface])
    if filter_expr:
        cmd.extend(["-f", filter_expr])
    
    try:
        # Start capture in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {
            "status": "capturing",
            "pid": process.pid,
            "output_file": str(output_file),
            "duration": duration,
            "cmd": " ".join(cmd)
        }
    except Exception as e:
        return {"error": str(e)}

def tshark_read(pcap_file: str, filter_expr: str = None, 
                format_type: str = "json", limit: int = 100) -> Dict:
    """Read and parse captured packets."""
    if not Path(pcap_file).exists():
        return {"error": f"File not found: {pcap_file}"}
    
    cmd = ["tshark", "-r", pcap_file, "-c", str(limit)]
    
    if filter_expr:
        cmd.extend(["-Y", filter_expr])
    if format_type == "json":
        cmd.extend(["-T", "json"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            if format_type == "json":
                packets = json.loads(result.stdout) if result.stdout.strip() else []
                return {"packets": packets, "count": len(packets)}
            return {"output": result.stdout, "lines": len(result.stdout.split('\n'))}
        return {"error": result.stderr}
    except Exception as e:
        return {"error": str(e)}

def tshark_list_interfaces() -> Dict:
    """List available network interfaces."""
    try:
        result = subprocess.run(["tshark", "-D"], capture_output=True, text=True, timeout=10)
        interfaces = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    interfaces.append({"id": parts[0], "name": parts[1]})
        return {"interfaces": interfaces}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# JPEXS/FFDEC TOOLS - Uses WSL CLI for performance (10-20x faster)
# ============================================================================

WSL_FFDEC_PATH = "/opt/ffdec/ffdec.sh"
WSL_DISTRO = "Ubuntu"

def check_wsl_ffdec() -> Optional[str]:
    """Check if FFDec is available in WSL."""
    if os.name != 'nt':
        return None
    try:
        result = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "-e", "test", "-x", WSL_FFDEC_PATH],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            return WSL_FFDEC_PATH
    except:
        pass
    return None

def run_wsl_ffdec(args: List[str], timeout: int = 180) -> Dict:
    """Run FFDec command via WSL."""
    try:
        cmd = ["wsl", "-d", WSL_DISTRO, "-e", WSL_FFDEC_PATH] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}

def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    if not win_path:
        return win_path
    path = win_path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path

def find_jpexs() -> Optional[str]:
    """Find JPEXS/FFDec - prefers WSL CLI for performance."""
    # ALWAYS prefer WSL for speed (10-20x faster than Windows GUI)
    wsl_path = check_wsl_ffdec()
    if wsl_path:
        return f"wsl:{wsl_path}"
    return None

def check_jpexs() -> Dict:
    """Check if JPEXS/FFDec is installed - uses WSL CLI."""
    wsl_path = check_wsl_ffdec()
    if wsl_path:
        # Quick check via WSL
        result = run_wsl_ffdec(["--help"], timeout=10)
        version = "24.1.1 (WSL CLI)"
        if result.get("stdout"):
            first_line = result["stdout"].split('\n')[0]
            if "JPEXS" in first_line:
                version = first_line
        return {
            "installed": True,
            "path": f"wsl:{wsl_path}",
            "version": version,
            "mode": "WSL CLI (fast)",
            "note": "Using WSL CLI mode - 10-20x faster than Windows GUI"
        }
    
    return {
        "installed": False,
        "error": "FFDec not found in WSL. Run: sudo mkdir -p /opt/ffdec && sudo cp -r /mnt/c/.../ffdec/* /opt/ffdec/",
        "hint": "WSL CLI mode is much faster than Windows GUI"
    }

def jpexs_decompile(swf_path: str, output_dir: str = None) -> Dict:
    """Decompile SWF file using FFDec via WSL CLI (fast mode)."""
    wsl_path = check_wsl_ffdec()
    if not wsl_path:
        return {"error": "FFDec not installed in WSL", "hint": "Copy ffdec to /opt/ffdec/ in WSL"}
    
    if not Path(swf_path).exists():
        return {"error": f"SWF not found: {swf_path}"}
    
    if output_dir is None:
        output_dir = str(Path(swf_path).parent / f"{Path(swf_path).stem}_decompiled")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert paths for WSL
    wsl_swf = windows_to_wsl_path(swf_path)
    wsl_out = windows_to_wsl_path(output_dir)
    
    result = run_wsl_ffdec(["-export", "script", wsl_out, wsl_swf], timeout=300)
    
    if result.get("success") or result.get("returncode") == 0:
        as_files = list(Path(output_dir).rglob("*.as"))
        return {
            "status": "success",
            "output_dir": output_dir,
            "files_decompiled": len(as_files),
            "mode": "WSL CLI (fast)",
            "note": "10-20x faster than Windows GUI"
        }
    return {"error": result.get("stderr") or result.get("error", "Export failed")}

def jpexs_extract_strings(swf_path: str) -> Dict:
    """Extract all strings from SWF using FFDec via WSL CLI."""
    wsl_path = check_wsl_ffdec()
    if not wsl_path:
        return {"error": "FFDec not installed in WSL"}
    
    if not Path(swf_path).exists():
        return {"error": f"SWF not found: {swf_path}"}
    
    wsl_swf = windows_to_wsl_path(swf_path)
    result = run_wsl_ffdec(["-dumpSWF", wsl_swf], timeout=120)
    
    if result.get("success") or result.get("returncode") == 0:
        lines = result.get("stdout", "").strip().split('\n')
        return {
            "status": "success",
            "classes": lines[:100],
            "total_classes": len(lines),
            "mode": "WSL CLI (fast)"
        }
    return {"error": result.get("stderr") or result.get("error", "Dump failed")}

# ============================================================================
# MEMORY TOOLS
# ============================================================================

def check_pymem() -> Dict:
    """Check if pymem is installed."""
    try:
        import pymem
        return {"installed": True, "version": pymem.__version__ if hasattr(pymem, '__version__') else "unknown"}
    except ImportError:
        return {"installed": False, "error": "pymem not installed. Run: pip install pymem"}

def memory_list_processes(filter_name: str = None) -> Dict:
    """List running processes. Filter is optional."""
    try:
        import psutil
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                info = proc.info
                name = info.get('name', '')
                if filter_name and name and filter_name.lower() not in name.lower():
                    continue
                mem_info = info.get('memory_info')
                processes.append({
                    "pid": info.get('pid', 0),
                    "name": name or "unknown",
                    "memory_mb": round(mem_info.rss / 1024 / 1024, 2) if mem_info else 0
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception:
                continue
        # Sort by memory descending
        processes.sort(key=lambda x: x['memory_mb'], reverse=True)
        return {
            "processes": processes[:50],
            "total": len(processes),
            "filter_applied": filter_name
        }
    except ImportError:
        return {"error": "psutil not installed. Run: pip install psutil"}
    except Exception as e:
        return {"error": str(e)}

def memory_scan_value(pid: int, value: int, value_type: str = "int") -> Dict:
    """Scan process memory for a value."""
    try:
        import pymem
        pm = pymem.Pymem(pid)
        # This is a simplified version - full implementation would need pattern scanning
        return {
            "status": "connected",
            "process": pm.process_base.name if hasattr(pm, 'process_base') else str(pid),
            "hint": "Full memory scanning requires admin privileges and specific patterns"
        }
    except ImportError:
        return {"error": "pymem not installed"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# GAME INTEGRATION
# ============================================================================

BOT_SERVER_URL = "http://localhost:9999"

def game_bot_status() -> Dict:
    """Check bot server status."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{BOT_SERVER_URL}/api/status")
        resp = urllib.request.urlopen(req, timeout=5)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"connected": False, "error": str(e), "hint": "Start: python evony_bot/server.py"}

def game_send_command(cmd: str, data: Dict) -> Dict:
    """Send command via bot server."""
    try:
        import urllib.request
        payload = json.dumps({"cmd": cmd, **data}).encode()
        req = urllib.request.Request(
            f"{BOT_SERVER_URL}/api/send",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def game_get_state(castle_id: int = None) -> Dict:
    """Get current game state."""
    try:
        import urllib.request
        url = f"{BOT_SERVER_URL}/api/state"
        if castle_id:
            url += f"?castle_id={castle_id}"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def game_verify_exploit(exploit_id: str, castle_id: int) -> Dict:
    """Verify if exploit worked by comparing before/after state."""
    # Get before state
    before = game_get_state(castle_id)
    if "error" in before:
        return {"error": f"Failed to get before state: {before['error']}"}
    
    # Would execute exploit here
    # after = game_get_state(castle_id)
    
    return {
        "status": "verification_ready",
        "before_state": before,
        "hint": "Execute exploit, then call game_get_state to compare"
    }

# ============================================================================
# TOOL REGISTRY
# ============================================================================

EXTENDED_TOOLS = {
    # Tshark
    "tshark_check": check_tshark,
    "tshark_capture": lambda args: tshark_capture(
        args.get("interface"), args.get("duration", 30),
        args.get("filter"), args.get("output_file")
    ),
    "tshark_read": lambda args: tshark_read(
        args.get("file", ""), args.get("filter"),
        args.get("format", "json"), args.get("limit", 100)
    ) if args.get("file") else {"error": "file parameter required"},
    "tshark_interfaces": tshark_list_interfaces,
    
    # JPEXS
    "jpexs_check": check_jpexs,
    "jpexs_decompile": lambda args: jpexs_decompile(
        args.get("swf_path"), args.get("output_dir")
    ),
    "jpexs_strings": lambda args: jpexs_extract_strings(args.get("swf_path")),
    
    # Memory
    "memory_check": check_pymem,
    "memory_processes": lambda args: memory_list_processes(args.get("filter")),
    "memory_scan": lambda args: memory_scan_value(
        args.get("pid"), args.get("value"), args.get("type", "int")
    ),
    
    # Game
    "game_bot_status": lambda args: game_bot_status(),
    "game_send": lambda args: game_send_command(args.get("cmd"), args.get("data", {})),
    "game_state": lambda args: game_get_state(args.get("castle_id")),
    "game_verify": lambda args: game_verify_exploit(args.get("exploit_id"), args.get("castle_id")),
}

def handle_extended_tool(name: str, args: Dict) -> Dict:
    """Handle extended tool calls."""
    handler = EXTENDED_TOOLS.get(name)
    if handler:
        if callable(handler):
            # Always pass args (even empty dict) to lambdas
            try:
                return handler(args if args else {})
            except TypeError:
                # For functions that take no args
                return handler()
        return handler
    return {"error": f"Unknown extended tool: {name}"}
