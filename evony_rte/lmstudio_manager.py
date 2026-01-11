"""
Evony LM Studio Model Manager
==============================
Real-time model switching, loading status, presets, and enhanced configuration
for Claude/Windsurf integration with Evony-trained models.

Features:
- Real-time model switching with loading progress
- Evony-specific system prompts and inference presets
- File system access configuration (all drives except Windows)
- Optimized settings for exploit analysis, packet decoding, protocol RE
"""

import os
import json
import time
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

LMSTUDIO_API_URL = "http://localhost:1234"
LMSTUDIO_REST_API = f"{LMSTUDIO_API_URL}/api/v0"
LMSTUDIO_OPENAI_API = f"{LMSTUDIO_API_URL}/v1"

# Models directory from user's LM Studio
MODELS_DIR = Path(r"C:\Users\Admin\.lmstudio\models")

# Evony data locations - accessible to models
EVONY_DATA_PATHS = {
    "primary": Path(r"C:\Users\Admin\Downloads\Evony_Decrypted"),
    "training": Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\Evony_Training_Data"),
    "exploits": Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\Evony_Dataset\exploits"),
    "protocol": Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\Docs"),
    "source_code": Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\Evony_Dataset\source_code"),
    "captures": Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\Evony_Dataset\network_captures"),
}

# File system access - all drives except Windows system
ALLOWED_DRIVES = ["D:", "E:", "F:", "G:", "H:"]  # Add more as needed
BLOCKED_PATHS = [
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
]

# =============================================================================
# EVONY MODEL CATALOG
# =============================================================================

@dataclass
class EvonyModel:
    """Evony model configuration."""
    name: str
    path: str
    quantization: str
    size_gb: float
    context_length: int
    gpu_layers: int
    use_case: str
    speed: str  # fast, balanced, quality
    recommended_tasks: List[str]


# Catalog of Evony models from user's LM Studio (from images)
EVONY_MODELS = {
    # F16 - Highest quality, slowest
    "evony-7b-3800-F16": EvonyModel(
        name="evony-7b-3800-F16",
        path="Borg/evony-7b-3800/evony-7b-3800-F16.gguf",
        quantization="F16",
        size_gb=15.24,
        context_length=32768,
        gpu_layers=28,
        use_case="Maximum accuracy exploit analysis",
        speed="quality",
        recommended_tasks=["exploit_verification", "protocol_deep_analysis", "source_audit"]
    ),
    # F14 variants
    "evony-7b-evony-7b-f16": EvonyModel(
        name="evony-7b-evony-7b-f16",
        path="Borg/evony-7b/evony-7b-f16.gguf",
        quantization="F14",
        size_gb=15.24,
        context_length=32768,
        gpu_layers=28,
        use_case="High quality general purpose",
        speed="quality",
        recommended_tasks=["documentation", "code_review", "exploit_research"]
    ),
    # Q8 - High quality, good speed
    "evony-7b-3800-rtx30801": EvonyModel(
        name="evony-7b-3800-rtx30801",
        path="Borg/evony-7b-3800-rtx30801/evony-7b-3800-RTX30801-Q8.gguf",
        quantization="Q8",
        size_gb=8.10,
        context_length=32768,
        gpu_layers=28,
        use_case="Balanced quality and speed",
        speed="balanced",
        recommended_tasks=["packet_analysis", "amf3_decode", "exploit_scan"]
    ),
    # Q6 - Good quality, faster
    "evony-7b-3800-rx7800xt": EvonyModel(
        name="evony-7b-3800-rx7800xt",
        path="Borg/evony-7b-3800-rx7800xt/evony-7b-3800-RX7800XT-Q6.gguf",
        quantization="Q6",
        size_gb=6.25,
        context_length=32768,
        gpu_layers=28,
        use_case="Fast analysis with good quality",
        speed="balanced",
        recommended_tasks=["quick_scan", "protocol_lookup", "enum_decode"]
    ),
    # Q5 - Fast
    "evony-7b-rtx3080": EvonyModel(
        name="evony-7b-rtx3080",
        path="Borg/evony-7b-rtx3080/evony-7b-RTX3080-Q4.gguf",
        quantization="Q5",
        size_gb=4.68,
        context_length=32768,
        gpu_layers=28,
        use_case="Fast general queries",
        speed="fast",
        recommended_tasks=["quick_lookup", "command_help", "basic_decode"]
    ),
    # Q4 - Fastest
    "evony-7b-3800-rtx3080": EvonyModel(
        name="evony-7b-3800-rtx3080",
        path="Borg/evony-7b-3800-rtx3080/evony-7b-3800-RTX3080-Q4.gguf",
        quantization="Q4",
        size_gb=4.68,
        context_length=32768,
        gpu_layers=28,
        use_case="Rapid iteration and testing",
        speed="fast",
        recommended_tasks=["interactive_chat", "quick_help", "code_completion"]
    ),
    # Q3 - Ultra fast
    "evony-7b-3800-speed": EvonyModel(
        name="evony-7b-3800-speed",
        path="Borg/evony-7b-3800-speed/evony-7b-3800-Speed-Q3.gguf",
        quantization="Q3",
        size_gb=2.81,
        context_length=32768,
        gpu_layers=28,
        use_case="Ultra-fast responses",
        speed="fast",
        recommended_tasks=["autocomplete", "quick_questions", "simple_decode"]
    ),
    # Q2 - Ultrafast (lower quality)
    "evony-7b-3800-ultrafast": EvonyModel(
        name="evony-7b-3800-ultrafast",
        path="Borg/evony-7b-3800-ultrafast/evony-7b-3800-UltraFast-Q2.gguf",
        quantization="Q2",
        size_gb=3.02,
        context_length=32768,
        gpu_layers=28,
        use_case="Instant responses, basic tasks",
        speed="fast",
        recommended_tasks=["simple_queries", "format_help", "quick_reference"]
    ),
    # Phase 2 models (8B)
    "evony-qwen3-8b-phase2k3_8": EvonyModel(
        name="evony-qwen3-8b-phase2k3_8",
        path="Borg/evony-qwen3-8b-phase2k3_8/evony-qwen3-8b-phase2-Q8_0.gguf",
        quantization="Q8_0",
        size_gb=9.10,
        context_length=32768,
        gpu_layers=28,
        use_case="Phase 2 enhanced training",
        speed="balanced",
        recommended_tasks=["advanced_exploit", "protocol_analysis", "cve_research"]
    ),
    "evony-qwen3-8b-phase2k4_k_m": EvonyModel(
        name="evony-qwen3-8b-phase2k4_k_m",
        path="Borg/evony-qwen3-8b-phase2k4_k_m/evony-qwen3-8b-phase2-Q4_K_M.gguf",
        quantization="Q4_K_M",
        size_gb=4.68,
        context_length=32768,
        gpu_layers=28,
        use_case="Phase 2 fast mode",
        speed="fast",
        recommended_tasks=["quick_analysis", "code_gen", "packet_decode"]
    ),
    "evony-qwen3-8b-phase2pf16": EvonyModel(
        name="evony-qwen3-8b-phase2pf16",
        path="Borg/evony-qwen3-8b-phase2pf16/evony-qwen3-8b-phase2-F16.gguf",
        quantization="F16",
        size_gb=15.24,
        context_length=32768,
        gpu_layers=28,
        use_case="Phase 2 maximum quality",
        speed="quality",
        recommended_tasks=["exploit_verification", "deep_analysis", "audit"]
    ),
}

# =============================================================================
# SYSTEM PROMPTS FOR EVONY TASKS
# =============================================================================

EVONY_SYSTEM_PROMPTS = {
    "exploit_analyst": """You are an Evony Age II security researcher specializing in exploit analysis.
Your knowledge includes:
- 100+ verified exploit templates from EVA bots and AutoEvony
- Complete AMF3 protocol specification and packet structures
- Flash AS3 source code analysis and decompilation
- CVE research and vulnerability patterns

When analyzing exploits:
1. Cross-reference with known CVEs and vulnerability databases
2. Identify the attack vector (packet injection, race condition, overflow)
3. Document the exploit chain step-by-step
4. Assess impact on game integrity
5. Suggest detection and mitigation strategies

Be precise, technical, and cite specific packet commands or source code when possible.""",

    "protocol_engineer": """You are an Evony Age II protocol reverse engineer.
Your expertise covers:
- AMF3 binary protocol encoding/decoding
- 301+ documented protocol commands
- Packet capture analysis from PCAP files
- Client-server communication patterns

When analyzing protocols:
1. Decode AMF3 payloads to human-readable format
2. Identify command IDs and parameter structures
3. Map packet flows to game actions
4. Document undocumented or obfuscated commands
5. Explain encryption and obfuscation layers

Provide byte-level analysis when relevant.""",

    "source_auditor": """You are an Evony Age II source code auditor.
Your knowledge spans:
- ActionScript 3 (AS3) codebase from decompiled SWF files
- Python automation scripts from AutoEvony and EVA bots
- Server-side command handlers and validation logic
- Anti-cheat bypass techniques

When auditing code:
1. Identify security vulnerabilities (injection, overflow, race)
2. Trace data flow from user input to server action
3. Document unsafe patterns and their exploitation
4. Cross-reference with known exploits
5. Suggest secure coding fixes

Focus on actionable security findings.""",

    "packet_decoder": """You are an Evony Age II packet decoder specialist.
You can decode and analyze:
- Raw AMF3 binary data
- Captured network traffic
- Encrypted payloads
- Command sequences

When decoding packets:
1. Parse AMF3 markers and data types
2. Extract command IDs and parameters
3. Identify the game action being performed
4. Flag suspicious or malformed packets
5. Reconstruct the full conversation flow

Output structured decoded data.""",

    "general_assistant": """You are an Evony Age II expert assistant trained on:
- 84,627 training examples from game analysis
- Complete protocol documentation (301+ commands)
- Source code from AutoEvony, EVA bots, and decompiled clients
- Security research and exploit documentation

You help with:
- Understanding game mechanics and protocols
- Analyzing network traffic and packets
- Reviewing source code for vulnerabilities
- Documenting findings in clear technical format

Be helpful, accurate, and cite sources when possible.""",

    "fast_helper": """You are a fast Evony assistant for quick queries.
Provide brief, accurate answers about:
- Protocol commands and their IDs
- Enum values and game constants
- Quick packet decoding
- Command syntax help

Keep responses concise and to the point.""",
}

# =============================================================================
# INFERENCE PRESETS
# =============================================================================

INFERENCE_PRESETS = {
    "exploit_analysis": {
        "name": "Exploit Analysis",
        "description": "Deep analysis of security vulnerabilities",
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 4096,
        "system_prompt": "exploit_analyst",
        "recommended_model": "evony-7b-3800-F16",
    },
    "packet_decode": {
        "name": "Packet Decoding",
        "description": "Fast AMF3 packet decoding",
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "max_tokens": 2048,
        "system_prompt": "packet_decoder",
        "recommended_model": "evony-7b-3800-rtx30801",
    },
    "protocol_research": {
        "name": "Protocol Research",
        "description": "In-depth protocol reverse engineering",
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 50,
        "repeat_penalty": 1.05,
        "max_tokens": 4096,
        "system_prompt": "protocol_engineer",
        "recommended_model": "evony-qwen3-8b-phase2pf16",
    },
    "code_review": {
        "name": "Code Review",
        "description": "Source code security audit",
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 4096,
        "system_prompt": "source_auditor",
        "recommended_model": "evony-7b-3800-F16",
    },
    "quick_help": {
        "name": "Quick Help",
        "description": "Fast responses for simple queries",
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.0,
        "max_tokens": 1024,
        "system_prompt": "fast_helper",
        "recommended_model": "evony-7b-3800-speed",
    },
    "interactive": {
        "name": "Interactive Chat",
        "description": "Balanced for conversation",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repeat_penalty": 1.1,
        "max_tokens": 2048,
        "system_prompt": "general_assistant",
        "recommended_model": "evony-7b-3800-rtx3080",
    },
    "creative": {
        "name": "Creative Writing",
        "description": "Documentation and report writing",
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 60,
        "repeat_penalty": 1.15,
        "max_tokens": 4096,
        "system_prompt": "general_assistant",
        "recommended_model": "evony-qwen3-8b-phase2k3_8",
    },
}

# =============================================================================
# LM STUDIO LOAD SETTINGS (Optimal for Evony models)
# =============================================================================

LOAD_CONFIGS = {
    "maximum_quality": {
        "name": "Maximum Quality",
        "context_length": 32768,
        "gpu_offload": "max",  # 28/28 layers
        "flash_attention": True,
        "kv_cache_quant": "F16",
        "eval_batch_size": 512,
        "keep_in_memory": True,
        "mmap": True,
        "description": "Best quality, uses most VRAM"
    },
    "balanced": {
        "name": "Balanced",
        "context_length": 16384,
        "gpu_offload": "max",
        "flash_attention": True,
        "kv_cache_quant": "Q8_0",
        "eval_batch_size": 512,
        "keep_in_memory": True,
        "mmap": True,
        "description": "Good balance of quality and speed"
    },
    "fast": {
        "name": "Fast",
        "context_length": 8192,
        "gpu_offload": "max",
        "flash_attention": True,
        "kv_cache_quant": "Q4_0",
        "eval_batch_size": 256,
        "keep_in_memory": True,
        "mmap": True,
        "description": "Faster responses, lower context"
    },
    "memory_saver": {
        "name": "Memory Saver",
        "context_length": 4096,
        "gpu_offload": 0.5,
        "flash_attention": True,
        "kv_cache_quant": "Q4_0",
        "eval_batch_size": 128,
        "keep_in_memory": False,
        "mmap": True,
        "description": "Minimal VRAM usage"
    },
}

# =============================================================================
# MODEL MANAGER CLASS
# =============================================================================

class ModelLoadStatus(Enum):
    NOT_LOADED = "not-loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class ModelState:
    """Current state of a model."""
    model_id: str
    status: ModelLoadStatus
    load_progress: float = 0.0
    load_time_seconds: float = 0.0
    error_message: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


class LMStudioManager:
    """
    Manager for LM Studio model operations.
    Provides real-time model switching with loading status for Claude/Windsurf.
    """
    
    def __init__(self):
        self.api_url = LMSTUDIO_API_URL
        self.rest_api = LMSTUDIO_REST_API
        self.openai_api = LMSTUDIO_OPENAI_API
        self.current_model: Optional[str] = None
        self.loading_model: Optional[str] = None
        self._load_start_time: Optional[float] = None
    
    # =========================================================================
    # MODEL LISTING AND STATUS
    # =========================================================================
    
    def list_models(self, filter_evony: bool = True) -> Dict[str, Any]:
        """
        List all available models in LM Studio.
        
        Args:
            filter_evony: If True, only show Evony-trained models
            
        Returns:
            Dict with model information including load state
        """
        try:
            response = requests.get(f"{self.rest_api}/models", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("data", []):
                model_id = model.get("id", "")
                
                # Filter for Evony models if requested
                if filter_evony and "evony" not in model_id.lower():
                    continue
                
                # Get Evony model info if available
                evony_info = EVONY_MODELS.get(model_id, None)
                
                models.append({
                    "id": model_id,
                    "state": model.get("state", "unknown"),
                    "quantization": model.get("quantization", "unknown"),
                    "max_context": model.get("max_context_length", 0),
                    "type": model.get("type", "llm"),
                    "evony_info": {
                        "use_case": evony_info.use_case if evony_info else None,
                        "speed": evony_info.speed if evony_info else None,
                        "recommended_tasks": evony_info.recommended_tasks if evony_info else [],
                    } if evony_info else None
                })
            
            # Count loaded models
            loaded = sum(1 for m in models if m["state"] == "loaded")
            
            return {
                "success": True,
                "total_models": len(models),
                "loaded_count": loaded,
                "models": models,
            }
            
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "LM Studio not running. Start LM Studio and enable the API server.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_model_status(self, model_id: str) -> ModelState:
        """Get detailed status of a specific model."""
        try:
            response = requests.get(f"{self.rest_api}/models/{model_id}", timeout=5)
            
            if response.status_code == 404:
                return ModelState(
                    model_id=model_id,
                    status=ModelLoadStatus.NOT_LOADED,
                    error_message="Model not found in LM Studio"
                )
            
            response.raise_for_status()
            data = response.json()
            
            state_str = data.get("state", "not-loaded")
            status = ModelLoadStatus.LOADED if state_str == "loaded" else ModelLoadStatus.NOT_LOADED
            
            return ModelState(
                model_id=model_id,
                status=status,
                config={
                    "quantization": data.get("quantization"),
                    "max_context": data.get("max_context_length"),
                    "type": data.get("type"),
                }
            )
            
        except Exception as e:
            return ModelState(
                model_id=model_id,
                status=ModelLoadStatus.ERROR,
                error_message=str(e)
            )
    
    def get_current_model(self) -> Optional[Dict[str, Any]]:
        """Get the currently loaded model."""
        result = self.list_models(filter_evony=False)
        if not result.get("success"):
            return None
        
        for model in result.get("models", []):
            if model.get("state") == "loaded":
                return model
        
        return None
    
    # =========================================================================
    # MODEL LOADING AND SWITCHING
    # =========================================================================
    
    def load_model(
        self,
        model_id: str,
        config: str = "balanced",
        identifier: Optional[str] = None,
        wait_for_load: bool = True,
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load a model into LM Studio with progress tracking.
        
        Args:
            model_id: The model key/ID to load
            config: Load config preset (maximum_quality, balanced, fast, memory_saver)
            identifier: Custom identifier for API calls
            wait_for_load: If True, wait and report loading progress
            ttl: Time-to-live in seconds (auto-unload after idle)
            
        Returns:
            Dict with load status and timing information
        """
        # Get load config
        load_config = LOAD_CONFIGS.get(config, LOAD_CONFIGS["balanced"])
        
        # Build lms load command
        cmd = ["lms", "load", model_id]
        
        # Add config options
        cmd.extend(["--context-length", str(load_config["context_length"])])
        
        if load_config["gpu_offload"] == "max":
            cmd.extend(["--gpu", "max"])
        else:
            cmd.extend(["--gpu", str(load_config["gpu_offload"])])
        
        if identifier:
            cmd.extend(["--identifier", identifier])
        
        if ttl:
            cmd.extend(["--ttl", str(ttl)])
        
        self.loading_model = model_id
        self._load_start_time = time.time()
        
        try:
            # Start loading
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            load_time = time.time() - self._load_start_time
            
            if result.returncode == 0:
                self.current_model = model_id
                self.loading_model = None
                
                return {
                    "success": True,
                    "model_id": model_id,
                    "status": "loaded",
                    "load_time_seconds": round(load_time, 2),
                    "config_used": config,
                    "config_details": load_config,
                    "message": f"✅ Model {model_id} loaded in {load_time:.1f}s"
                }
            else:
                self.loading_model = None
                return {
                    "success": False,
                    "model_id": model_id,
                    "status": "error",
                    "error": result.stderr or result.stdout,
                }
                
        except subprocess.TimeoutExpired:
            self.loading_model = None
            return {
                "success": False,
                "model_id": model_id,
                "status": "timeout",
                "error": "Model loading timed out after 5 minutes",
            }
        except FileNotFoundError:
            self.loading_model = None
            return {
                "success": False,
                "error": "lms CLI not found. Ensure LM Studio CLI is installed.",
            }
        except Exception as e:
            self.loading_model = None
            return {"success": False, "error": str(e)}
    
    def unload_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Unload a model from memory.
        
        Args:
            model_id: Specific model to unload, or None for all models
        """
        cmd = ["lms", "unload"]
        if model_id:
            cmd.append(model_id)
        else:
            cmd.append("--all")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                if model_id:
                    if self.current_model == model_id:
                        self.current_model = None
                else:
                    self.current_model = None
                
                return {
                    "success": True,
                    "message": f"Model{'s' if not model_id else ' ' + model_id} unloaded",
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or result.stdout,
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def switch_model(
        self,
        new_model_id: str,
        config: str = "balanced",
        unload_current: bool = True,
    ) -> Dict[str, Any]:
        """
        Switch to a different model with loading status.
        
        This is the main function for Claude/Windsurf to change models.
        
        Args:
            new_model_id: The model to switch to
            config: Load configuration preset
            unload_current: Whether to unload the current model first
            
        Returns:
            Dict with switch status and timing
        """
        result = {
            "action": "switch_model",
            "from_model": self.current_model,
            "to_model": new_model_id,
            "steps": [],
        }
        
        start_time = time.time()
        
        # Step 1: Unload current model if requested
        if unload_current and self.current_model:
            result["steps"].append({
                "step": "unload_current",
                "status": "⏳ Unloading current model..."
            })
            
            unload_result = self.unload_model(self.current_model)
            result["steps"][-1]["status"] = "✅ Unloaded" if unload_result.get("success") else "⚠️ Failed"
            result["steps"][-1]["details"] = unload_result
        
        # Step 2: Load new model
        result["steps"].append({
            "step": "load_new",
            "status": f"⏳ Loading {new_model_id}... (this may take 10-60 seconds)"
        })
        
        load_result = self.load_model(new_model_id, config=config)
        result["steps"][-1]["status"] = "✅ Loaded" if load_result.get("success") else "❌ Failed"
        result["steps"][-1]["details"] = load_result
        
        # Final status
        total_time = time.time() - start_time
        result["success"] = load_result.get("success", False)
        result["total_time_seconds"] = round(total_time, 2)
        result["current_model"] = self.current_model
        
        if result["success"]:
            result["message"] = f"✅ Switched to {new_model_id} in {total_time:.1f}s"
        else:
            result["message"] = f"❌ Failed to switch to {new_model_id}"
        
        return result
    
    # =========================================================================
    # MODEL RECOMMENDATIONS
    # =========================================================================
    
    def recommend_model(self, task: str) -> Dict[str, Any]:
        """
        Recommend the best Evony model for a specific task.
        
        Args:
            task: The task type (exploit_analysis, packet_decode, quick_help, etc.)
            
        Returns:
            Dict with recommended model and preset
        """
        task_lower = task.lower()
        
        # Map tasks to models
        task_mappings = {
            "exploit": ("evony-7b-3800-F16", "exploit_analysis"),
            "vulnerability": ("evony-7b-3800-F16", "exploit_analysis"),
            "security": ("evony-7b-3800-F16", "exploit_analysis"),
            "cve": ("evony-qwen3-8b-phase2pf16", "exploit_analysis"),
            "packet": ("evony-7b-3800-rtx30801", "packet_decode"),
            "decode": ("evony-7b-3800-rtx30801", "packet_decode"),
            "amf": ("evony-7b-3800-rtx30801", "packet_decode"),
            "protocol": ("evony-qwen3-8b-phase2pf16", "protocol_research"),
            "command": ("evony-7b-3800-rx7800xt", "quick_help"),
            "code": ("evony-7b-3800-F16", "code_review"),
            "source": ("evony-7b-3800-F16", "code_review"),
            "audit": ("evony-7b-3800-F16", "code_review"),
            "quick": ("evony-7b-3800-speed", "quick_help"),
            "fast": ("evony-7b-3800-speed", "quick_help"),
            "help": ("evony-7b-3800-rtx3080", "interactive"),
            "chat": ("evony-7b-3800-rtx3080", "interactive"),
            "document": ("evony-qwen3-8b-phase2k3_8", "creative"),
            "report": ("evony-qwen3-8b-phase2k3_8", "creative"),
        }
        
        # Find best match
        for keyword, (model, preset) in task_mappings.items():
            if keyword in task_lower:
                model_info = EVONY_MODELS.get(model)
                preset_info = INFERENCE_PRESETS.get(preset)
                
                return {
                    "success": True,
                    "task": task,
                    "recommended_model": model,
                    "recommended_preset": preset,
                    "model_info": {
                        "quantization": model_info.quantization if model_info else None,
                        "size_gb": model_info.size_gb if model_info else None,
                        "speed": model_info.speed if model_info else None,
                        "use_case": model_info.use_case if model_info else None,
                    },
                    "preset_info": preset_info,
                    "switch_command": f"switch_model('{model}', config='{preset_info.get('recommended_model', 'balanced')}')"
                }
        
        # Default recommendation
        return {
            "success": True,
            "task": task,
            "recommended_model": "evony-7b-3800-rtx30801",
            "recommended_preset": "interactive",
            "message": "Using balanced default. Specify task type for better recommendation.",
        }
    
    # =========================================================================
    # PRESETS AND SYSTEM PROMPTS
    # =========================================================================
    
    def get_system_prompt(self, prompt_type: str) -> str:
        """Get a predefined system prompt for Evony tasks."""
        return EVONY_SYSTEM_PROMPTS.get(prompt_type, EVONY_SYSTEM_PROMPTS["general_assistant"])
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get inference preset settings."""
        preset = INFERENCE_PRESETS.get(preset_name)
        if preset:
            # Add the full system prompt
            preset["full_system_prompt"] = self.get_system_prompt(preset.get("system_prompt", "general_assistant"))
        return preset
    
    def list_presets(self) -> List[Dict[str, Any]]:
        """List all available inference presets."""
        return [
            {
                "name": key,
                "display_name": preset["name"],
                "description": preset["description"],
                "recommended_model": preset["recommended_model"],
            }
            for key, preset in INFERENCE_PRESETS.items()
        ]
    
    def list_system_prompts(self) -> List[Dict[str, str]]:
        """List all available system prompts."""
        return [
            {"name": key, "preview": prompt[:100] + "..."}
            for key, prompt in EVONY_SYSTEM_PROMPTS.items()
        ]
    
    # =========================================================================
    # FILE SYSTEM ACCESS
    # =========================================================================
    
    def get_allowed_paths(self) -> Dict[str, Any]:
        """Get list of paths accessible to Evony models."""
        return {
            "evony_data_paths": {k: str(v) for k, v in EVONY_DATA_PATHS.items()},
            "allowed_drives": ALLOWED_DRIVES,
            "blocked_paths": BLOCKED_PATHS,
            "note": "Evony models can access all drives except Windows system folders",
        }
    
    def is_path_allowed(self, path: str) -> Tuple[bool, str]:
        """Check if a path is accessible to Evony models."""
        path_obj = Path(path)
        
        # Check if it's a blocked Windows path
        for blocked in BLOCKED_PATHS:
            if path.lower().startswith(blocked.lower()):
                return False, f"Access denied: {blocked} is a protected system path"
        
        # Check if it's in Evony data paths (always allowed)
        for name, evony_path in EVONY_DATA_PATHS.items():
            if str(path_obj).lower().startswith(str(evony_path).lower()):
                return True, f"Access granted: {name} data path"
        
        # Check drive letter
        if len(path) >= 2 and path[1] == ":":
            drive = path[:2].upper()
            if drive in [d.upper() for d in ALLOWED_DRIVES]:
                return True, f"Access granted: {drive} is an allowed drive"
            elif drive == "C:":
                # C: drive - check if outside system folders
                for blocked in BLOCKED_PATHS:
                    if path.lower().startswith(blocked.lower()):
                        return False, f"Access denied: System path"
                return True, "Access granted: C: drive (non-system path)"
        
        return True, "Access granted by default"


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_manager: Optional[LMStudioManager] = None

def get_lmstudio_manager() -> LMStudioManager:
    """Get the singleton LM Studio manager instance."""
    global _manager
    if _manager is None:
        _manager = LMStudioManager()
    return _manager


# =============================================================================
# CONVENIENCE FUNCTIONS FOR MCP TOOLS
# =============================================================================

def list_evony_models() -> Dict[str, Any]:
    """List all Evony models with their status."""
    return get_lmstudio_manager().list_models(filter_evony=True)

def switch_model(model_id: str, config: str = "balanced") -> Dict[str, Any]:
    """Switch to a different Evony model."""
    return get_lmstudio_manager().switch_model(model_id, config=config)

def recommend_model_for_task(task: str) -> Dict[str, Any]:
    """Get model recommendation for a task."""
    return get_lmstudio_manager().recommend_model(task)

def get_loading_status() -> Dict[str, Any]:
    """Get current model loading status."""
    manager = get_lmstudio_manager()
    current = manager.get_current_model()
    
    return {
        "current_model": current.get("id") if current else None,
        "current_state": current.get("state") if current else "none",
        "loading_model": manager.loading_model,
        "is_loading": manager.loading_model is not None,
    }

def get_preset_for_task(task: str) -> Dict[str, Any]:
    """Get the best preset and system prompt for a task."""
    manager = get_lmstudio_manager()
    recommendation = manager.recommend_model(task)
    
    if recommendation.get("recommended_preset"):
        preset = manager.get_preset(recommendation["recommended_preset"])
        return {
            "preset": preset,
            "system_prompt": preset.get("full_system_prompt") if preset else None,
            "recommendation": recommendation,
        }
    
    return recommendation


# =============================================================================
# CLI INTERFACE FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    manager = get_lmstudio_manager()
    
    if len(sys.argv) < 2:
        print("Evony LM Studio Manager")
        print("=" * 50)
        print("\nCommands:")
        print("  list              - List Evony models")
        print("  status            - Get current model status")
        print("  switch <model>    - Switch to a model")
        print("  recommend <task>  - Get model recommendation")
        print("  presets           - List inference presets")
        print("  prompts           - List system prompts")
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "list":
        result = manager.list_models()
        print(json.dumps(result, indent=2))
    
    elif cmd == "status":
        result = get_loading_status()
        print(json.dumps(result, indent=2))
    
    elif cmd == "switch" and len(sys.argv) > 2:
        model_id = sys.argv[2]
        print(f"Switching to {model_id}...")
        result = manager.switch_model(model_id)
        print(json.dumps(result, indent=2))
    
    elif cmd == "recommend" and len(sys.argv) > 2:
        task = " ".join(sys.argv[2:])
        result = manager.recommend_model(task)
        print(json.dumps(result, indent=2))
    
    elif cmd == "presets":
        presets = manager.list_presets()
        print(json.dumps(presets, indent=2))
    
    elif cmd == "prompts":
        prompts = manager.list_system_prompts()
        print(json.dumps(prompts, indent=2))
    
    else:
        print(f"Unknown command: {cmd}")
