"""
Evony RTE - External Tool Integration
======================================
Integration with reverse engineering tools:
- RABCDAsm (ABC bytecode disassembly/assembly)
- swfextract (SWF component extraction)
- flasm (Flash assembler/disassembler)
- graphviz (flow diagram generation)
"""

import subprocess
import shutil
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# ============================================================================
# TOOL PATHS
# ============================================================================

EVONY_ROOT = Path(__file__).parent.parent
RABCDASM_DIR = EVONY_ROOT / "RABCDAsm"
TOOLS_DIR = EVONY_ROOT / "Electron_Browser" / "tools"

@dataclass
class ToolConfig:
    """Configuration for external tool."""
    name: str
    executable: str
    installed: bool = False
    version: str = ""
    path: Optional[Path] = None
    install_hint: str = ""

# ============================================================================
# TOOL DETECTION
# ============================================================================

def find_executable(name: str, search_paths: List[Path] = None) -> Optional[Path]:
    """Find executable in PATH or search paths."""
    # Check PATH first
    path = shutil.which(name)
    if path:
        return Path(path)
    
    # Check common Windows locations
    if search_paths is None:
        search_paths = [
            TOOLS_DIR,
            RABCDASM_DIR,
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
            Path.home() / "bin",
        ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        for ext in ["", ".exe", ".bat", ".cmd"]:
            candidate = search_path / f"{name}{ext}"
            if candidate.exists():
                return candidate
            # Check subdirectories
            for subdir in search_path.iterdir():
                if subdir.is_dir():
                    candidate = subdir / f"{name}{ext}"
                    if candidate.exists():
                        return candidate
    
    return None

def check_tool(name: str) -> ToolConfig:
    """Check if tool is available."""
    configs = {
        "rabcdasm": ToolConfig(
            name="RABCDAsm",
            executable="rabcdasm",
            install_hint="Build from D source in RABCDAsm folder or download binaries"
        ),
        "rabcasm": ToolConfig(
            name="RABCAsm",
            executable="rabcasm",
            install_hint="Part of RABCDAsm package"
        ),
        "abcexport": ToolConfig(
            name="ABCExport",
            executable="abcexport",
            install_hint="Part of RABCDAsm package"
        ),
        "abcreplace": ToolConfig(
            name="ABCReplace",
            executable="abcreplace",
            install_hint="Part of RABCDAsm package"
        ),
        "swfextract": ToolConfig(
            name="swfextract",
            executable="swfextract",
            install_hint="Install swftools: choco install swftools"
        ),
        "flasm": ToolConfig(
            name="flasm",
            executable="flasm",
            install_hint="Download from http://flasm.sourceforge.net/"
        ),
        "dot": ToolConfig(
            name="Graphviz",
            executable="dot",
            install_hint="Install graphviz: choco install graphviz"
        ),
        "tshark": ToolConfig(
            name="TShark",
            executable="tshark",
            install_hint="Install Wireshark: choco install wireshark"
        ),
        "java": ToolConfig(
            name="Java",
            executable="java",
            install_hint="Install JDK: choco install openjdk"
        )
    }
    
    config = configs.get(name.lower(), ToolConfig(name=name, executable=name))
    
    # Try to find the executable
    path = find_executable(config.executable)
    if path:
        config.installed = True
        config.path = path
        # Try to get version
        try:
            result = subprocess.run(
                [str(path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            config.version = result.stdout.split('\n')[0][:50] if result.stdout else ""
        except:
            pass
    
    return config

def get_all_tools_status() -> Dict:
    """Get status of all required tools."""
    tools = ["rabcdasm", "rabcasm", "abcexport", "abcreplace", 
             "swfextract", "flasm", "dot", "tshark", "java"]
    
    status = {}
    for tool in tools:
        config = check_tool(tool)
        status[tool] = {
            "name": config.name,
            "installed": config.installed,
            "version": config.version,
            "path": str(config.path) if config.path else None,
            "install_hint": config.install_hint if not config.installed else ""
        }
    
    installed_count = sum(1 for t in status.values() if t["installed"])
    
    return {
        "tools": status,
        "installed": installed_count,
        "total": len(tools),
        "all_ready": installed_count == len(tools)
    }

# ============================================================================
# RABCDAsm INTEGRATION
# ============================================================================

class RABCDAsmTools:
    """Interface to RABCDAsm for ABC bytecode manipulation."""
    
    def __init__(self):
        self.rabcdasm_dir = RABCDASM_DIR
        self.output_dir = EVONY_ROOT / "evony_rte" / "disasm_output"
        self.output_dir.mkdir(exist_ok=True)
    
    def check_installation(self) -> Dict:
        """Check if RABCDAsm tools are available."""
        tools = ["rabcdasm", "rabcasm", "abcexport", "abcreplace"]
        results = {}
        
        for tool in tools:
            config = check_tool(tool)
            results[tool] = {
                "installed": config.installed,
                "path": str(config.path) if config.path else None
            }
        
        # Check for D source files (can build)
        d_sources = list(self.rabcdasm_dir.glob("*.d"))
        
        return {
            "tools": results,
            "can_build": len(d_sources) > 0,
            "d_sources": len(d_sources),
            "hint": "Run 'dmd -of=rabcdasm rabcdasm.d' to build" if d_sources else ""
        }
    
    def export_abc(self, swf_file: str) -> Dict:
        """Export ABC bytecode from SWF."""
        config = check_tool("abcexport")
        if not config.installed:
            return {"error": f"abcexport not installed. {config.install_hint}"}
        
        swf_path = Path(swf_file)
        if not swf_path.exists():
            return {"error": f"SWF file not found: {swf_file}"}
        
        output_dir = self.output_dir / swf_path.stem
        output_dir.mkdir(exist_ok=True)
        
        try:
            result = subprocess.run(
                [str(config.path), str(swf_path)],
                capture_output=True,
                text=True,
                cwd=str(output_dir),
                timeout=120
            )
            
            abc_files = list(output_dir.glob("*.abc"))
            
            return {
                "success": result.returncode == 0,
                "output_dir": str(output_dir),
                "abc_files": [str(f) for f in abc_files],
                "count": len(abc_files),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"error": "Export timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    def disassemble_abc(self, abc_file: str) -> Dict:
        """Disassemble ABC bytecode to readable format."""
        config = check_tool("rabcdasm")
        if not config.installed:
            return {"error": f"rabcdasm not installed. {config.install_hint}"}
        
        abc_path = Path(abc_file)
        if not abc_path.exists():
            return {"error": f"ABC file not found: {abc_file}"}
        
        output_dir = self.output_dir / abc_path.stem
        
        try:
            result = subprocess.run(
                [str(config.path), str(abc_path)],
                capture_output=True,
                text=True,
                cwd=str(output_dir.parent),
                timeout=300
            )
            
            # Find output files
            asasm_files = list(output_dir.glob("**/*.asasm")) if output_dir.exists() else []
            
            return {
                "success": result.returncode == 0,
                "output_dir": str(output_dir),
                "asasm_files": len(asasm_files),
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:500] if result.stderr else ""
            }
        except subprocess.TimeoutExpired:
            return {"error": "Disassembly timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    def assemble_abc(self, asasm_dir: str) -> Dict:
        """Assemble modified ASASM back to ABC."""
        config = check_tool("rabcasm")
        if not config.installed:
            return {"error": f"rabcasm not installed. {config.install_hint}"}
        
        asasm_path = Path(asasm_dir)
        main_asasm = asasm_path / f"{asasm_path.name}.main.asasm"
        
        if not main_asasm.exists():
            # Find any .main.asasm file
            main_files = list(asasm_path.glob("*.main.asasm"))
            if main_files:
                main_asasm = main_files[0]
            else:
                return {"error": f"No .main.asasm file found in {asasm_dir}"}
        
        try:
            result = subprocess.run(
                [str(config.path), str(main_asasm)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Find output ABC
            abc_file = main_asasm.with_suffix(".abc")
            
            return {
                "success": result.returncode == 0 and abc_file.exists(),
                "abc_file": str(abc_file) if abc_file.exists() else None,
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:500] if result.stderr else ""
            }
        except Exception as e:
            return {"error": str(e)}
    
    def replace_abc(self, swf_file: str, abc_index: int, new_abc: str) -> Dict:
        """Replace ABC in SWF with modified version."""
        config = check_tool("abcreplace")
        if not config.installed:
            return {"error": f"abcreplace not installed. {config.install_hint}"}
        
        swf_path = Path(swf_file)
        new_abc_path = Path(new_abc)
        
        if not swf_path.exists():
            return {"error": f"SWF file not found: {swf_file}"}
        if not new_abc_path.exists():
            return {"error": f"ABC file not found: {new_abc}"}
        
        # Create backup
        backup = swf_path.with_suffix(".swf.backup")
        if not backup.exists():
            shutil.copy(swf_path, backup)
        
        try:
            result = subprocess.run(
                [str(config.path), str(swf_path), str(abc_index), str(new_abc_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                "success": result.returncode == 0,
                "swf_file": str(swf_path),
                "backup": str(backup),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# SWF TOOLS INTEGRATION
# ============================================================================

class SWFTools:
    """Interface to swfextract and other SWF tools."""
    
    def __init__(self):
        self.output_dir = EVONY_ROOT / "evony_rte" / "swf_output"
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_components(self, swf_file: str, component_type: str = "all") -> Dict:
        """Extract components from SWF."""
        config = check_tool("swfextract")
        if not config.installed:
            return {"error": f"swfextract not installed. {config.install_hint}"}
        
        swf_path = Path(swf_file)
        if not swf_path.exists():
            return {"error": f"SWF file not found: {swf_file}"}
        
        output_dir = self.output_dir / swf_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # First, list contents
        try:
            list_result = subprocess.run(
                [str(config.path), str(swf_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse output to find component IDs
            components = self._parse_swfextract_output(list_result.stdout)
            
            extracted = []
            
            if component_type in ["all", "shapes"]:
                for shape_id in components.get("shapes", [])[:10]:
                    out_file = output_dir / f"shape_{shape_id}.swf"
                    subprocess.run(
                        [str(config.path), "-s", str(shape_id), "-o", str(out_file), str(swf_path)],
                        capture_output=True,
                        timeout=30
                    )
                    if out_file.exists():
                        extracted.append(str(out_file))
            
            if component_type in ["all", "sounds"]:
                for sound_id in components.get("sounds", [])[:10]:
                    out_file = output_dir / f"sound_{sound_id}.mp3"
                    subprocess.run(
                        [str(config.path), "-m", str(sound_id), "-o", str(out_file), str(swf_path)],
                        capture_output=True,
                        timeout=30
                    )
                    if out_file.exists():
                        extracted.append(str(out_file))
            
            if component_type in ["all", "images"]:
                for img_id in components.get("images", [])[:20]:
                    out_file = output_dir / f"image_{img_id}.png"
                    subprocess.run(
                        [str(config.path), "-p", str(img_id), "-o", str(out_file), str(swf_path)],
                        capture_output=True,
                        timeout=30
                    )
                    if out_file.exists():
                        extracted.append(str(out_file))
            
            return {
                "success": True,
                "output_dir": str(output_dir),
                "components": components,
                "extracted": extracted,
                "count": len(extracted)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_swfextract_output(self, output: str) -> Dict:
        """Parse swfextract listing output."""
        components = {
            "shapes": [],
            "sounds": [],
            "images": [],
            "fonts": [],
            "frames": []
        }
        
        # Parse patterns like "[-s] 1 Shape"
        shape_pattern = r'\[-s\]\s+(\d+)'
        sound_pattern = r'\[-m\]\s+(\d+)'
        image_pattern = r'\[-p\]\s+(\d+)'
        font_pattern = r'\[-f\]\s+(\d+)'
        
        components["shapes"] = [int(m) for m in re.findall(shape_pattern, output)]
        components["sounds"] = [int(m) for m in re.findall(sound_pattern, output)]
        components["images"] = [int(m) for m in re.findall(image_pattern, output)]
        components["fonts"] = [int(m) for m in re.findall(font_pattern, output)]
        
        return components

# ============================================================================
# FLASM INTEGRATION
# ============================================================================

class FlasmTools:
    """Interface to flasm for Flash bytecode manipulation."""
    
    def __init__(self):
        self.output_dir = EVONY_ROOT / "evony_rte" / "flasm_output"
        self.output_dir.mkdir(exist_ok=True)
    
    def disassemble(self, swf_file: str) -> Dict:
        """Disassemble SWF to flasm format."""
        config = check_tool("flasm")
        if not config.installed:
            return {"error": f"flasm not installed. {config.install_hint}"}
        
        swf_path = Path(swf_file)
        if not swf_path.exists():
            return {"error": f"SWF file not found: {swf_file}"}
        
        output_file = self.output_dir / f"{swf_path.stem}.flm"
        
        try:
            result = subprocess.run(
                [str(config.path), "-d", str(swf_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Flasm outputs to stdout
            if result.stdout:
                output_file.write_text(result.stdout)
            
            return {
                "success": result.returncode == 0,
                "output_file": str(output_file) if output_file.exists() else None,
                "lines": len(result.stdout.split('\n')) if result.stdout else 0,
                "stderr": result.stderr[:500] if result.stderr else ""
            }
        except Exception as e:
            return {"error": str(e)}
    
    def assemble(self, flm_file: str) -> Dict:
        """Assemble flasm source back to SWF."""
        config = check_tool("flasm")
        if not config.installed:
            return {"error": f"flasm not installed. {config.install_hint}"}
        
        flm_path = Path(flm_file)
        if not flm_path.exists():
            return {"error": f"FLM file not found: {flm_file}"}
        
        try:
            result = subprocess.run(
                [str(config.path), "-a", str(flm_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Output SWF has same name but .swf extension
            output_swf = flm_path.with_suffix(".swf")
            
            return {
                "success": result.returncode == 0 and output_swf.exists(),
                "output_swf": str(output_swf) if output_swf.exists() else None,
                "stdout": result.stdout[:500],
                "stderr": result.stderr[:500] if result.stderr else ""
            }
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# GRAPHVIZ INTEGRATION
# ============================================================================

class GraphvizTools:
    """Interface to Graphviz for diagram generation."""
    
    def __init__(self):
        self.output_dir = EVONY_ROOT / "evony_rte" / "diagrams"
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_diagram(self, dot_source: str, output_name: str, 
                         format: str = "svg") -> Dict:
        """Generate diagram from DOT source."""
        config = check_tool("dot")
        if not config.installed:
            return {"error": f"Graphviz not installed. {config.install_hint}"}
        
        # Write DOT source to temp file
        dot_file = self.output_dir / f"{output_name}.dot"
        output_file = self.output_dir / f"{output_name}.{format}"
        
        dot_file.write_text(dot_source)
        
        try:
            result = subprocess.run(
                [str(config.path), f"-T{format}", "-o", str(output_file), str(dot_file)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return {
                "success": result.returncode == 0 and output_file.exists(),
                "output_file": str(output_file) if output_file.exists() else None,
                "dot_file": str(dot_file),
                "format": format,
                "stderr": result.stderr[:500] if result.stderr else ""
            }
        except Exception as e:
            return {"error": str(e)}
    
    def generate_callgraph(self, functions: List[Dict], output_name: str) -> Dict:
        """Generate function call graph."""
        dot_lines = [
            "digraph CallGraph {",
            "  rankdir=LR;",
            "  node [shape=box, style=filled, fillcolor=lightblue];",
            ""
        ]
        
        nodes = set()
        edges = []
        
        for func in functions:
            caller = func.get("caller", "")
            callee = func.get("callee", "")
            if caller and callee:
                nodes.add(caller)
                nodes.add(callee)
                edges.append(f'  "{caller}" -> "{callee}";')
        
        for node in nodes:
            dot_lines.append(f'  "{node}";')
        
        dot_lines.append("")
        dot_lines.extend(edges)
        dot_lines.append("}")
        
        return self.generate_diagram("\n".join(dot_lines), output_name)
    
    def generate_exploit_flow(self, exploit_name: str, steps: List[str]) -> Dict:
        """Generate exploit flow diagram."""
        dot_lines = [
            "digraph ExploitFlow {",
            "  rankdir=TB;",
            "  node [shape=box, style=filled];",
            '  start [label="Start", fillcolor=green];',
            '  end [label="Success", fillcolor=green];',
            ""
        ]
        
        prev_node = "start"
        for i, step in enumerate(steps):
            node_id = f"step{i}"
            dot_lines.append(f'  {node_id} [label="{step}", fillcolor=lightblue];')
            dot_lines.append(f'  {prev_node} -> {node_id};')
            prev_node = node_id
        
        dot_lines.append(f'  {prev_node} -> end;')
        dot_lines.append("}")
        
        return self.generate_diagram("\n".join(dot_lines), f"exploit_{exploit_name}")

# ============================================================================
# HANDLER FUNCTIONS FOR MCP
# ============================================================================

# Global tool instances
_rabcdasm = RABCDAsmTools()
_swftools = SWFTools()
_flasm = FlasmTools()
_graphviz = GraphvizTools()

def handle_tools_status(args: Dict) -> Dict:
    """Get status of all external tools."""
    return get_all_tools_status()

def handle_rabcdasm_export(args: Dict) -> Dict:
    """Export ABC from SWF."""
    swf_file = args.get("swf_file", "")
    return _rabcdasm.export_abc(swf_file)

def handle_rabcdasm_disasm(args: Dict) -> Dict:
    """Disassemble ABC bytecode."""
    abc_file = args.get("abc_file", "")
    return _rabcdasm.disassemble_abc(abc_file)

def handle_rabcdasm_asm(args: Dict) -> Dict:
    """Assemble ASASM to ABC."""
    asasm_dir = args.get("asasm_dir", "")
    return _rabcdasm.assemble_abc(asasm_dir)

def handle_rabcdasm_replace(args: Dict) -> Dict:
    """Replace ABC in SWF."""
    swf_file = args.get("swf_file", "")
    abc_index = args.get("abc_index", 0)
    new_abc = args.get("new_abc", "")
    return _rabcdasm.replace_abc(swf_file, abc_index, new_abc)

def handle_swf_extract(args: Dict) -> Dict:
    """Extract components from SWF."""
    swf_file = args.get("swf_file", "")
    component_type = args.get("type", "all")
    return _swftools.extract_components(swf_file, component_type)

def handle_flasm_disasm(args: Dict) -> Dict:
    """Disassemble SWF with flasm."""
    swf_file = args.get("swf_file", "")
    return _flasm.disassemble(swf_file)

def handle_flasm_asm(args: Dict) -> Dict:
    """Assemble FLM with flasm."""
    flm_file = args.get("flm_file", "")
    return _flasm.assemble(flm_file)

def handle_diagram_generate(args: Dict) -> Dict:
    """Generate diagram from DOT source."""
    dot_source = args.get("dot_source", "")
    name = args.get("name", "diagram")
    format = args.get("format", "svg")
    return _graphviz.generate_diagram(dot_source, name, format)

def handle_diagram_callgraph(args: Dict) -> Dict:
    """Generate call graph diagram."""
    functions = args.get("functions", [])
    name = args.get("name", "callgraph")
    return _graphviz.generate_callgraph(functions, name)

# Export
TOOL_HANDLERS = {
    "tools_status": handle_tools_status,
    "rabcdasm_export": handle_rabcdasm_export,
    "rabcdasm_disasm": handle_rabcdasm_disasm,
    "rabcdasm_asm": handle_rabcdasm_asm,
    "rabcdasm_replace": handle_rabcdasm_replace,
    "swf_extract": handle_swf_extract,
    "flasm_disasm": handle_flasm_disasm,
    "flasm_asm": handle_flasm_asm,
    "diagram_generate": handle_diagram_generate,
    "diagram_callgraph": handle_diagram_callgraph,
}

__all__ = [
    'get_all_tools_status',
    'RABCDAsmTools',
    'SWFTools',
    'FlasmTools',
    'GraphvizTools',
    'TOOL_HANDLERS'
]
