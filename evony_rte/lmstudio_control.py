"""
LM Studio Control Module for Claude/Windsurf Integration
=========================================================
Provides comprehensive control over LM Studio from AI assistants.

Capabilities:
- Model loading/unloading/switching
- Preset management (list, apply, switch)
- Server status and management
- Chat completions with custom parameters
- Model information and statistics

Requirements:
- LM Studio running with server enabled (port 1234)
- lms CLI installed (npx lmstudio install-cli)
"""

import subprocess
import json
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

# Configuration
LMSTUDIO_HOST = "localhost"
LMSTUDIO_PORT = 1234
LMSTUDIO_BASE_URL = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}"
PRESETS_DIR = Path(os.environ.get("USERPROFILE", "")) / ".lmstudio" / "config-presets"
EVONY_PRESETS_DIR = Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\lmstudio_presets")


# =============================================================================
# CLI COMMANDS (via lms)
# =============================================================================

def run_lms_command(args: List[str], timeout: int = 60) -> Dict[str, Any]:
    """Execute an lms CLI command and return result."""
    try:
        result = subprocess.run(
            ["lms"] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except FileNotFoundError:
        return {"success": False, "error": "lms CLI not found. Run: npx lmstudio install-cli"}


def lms_server_status() -> Dict[str, Any]:
    """Get LM Studio server status."""
    return run_lms_command(["server", "status"])


def lms_server_start() -> Dict[str, Any]:
    """Start LM Studio server."""
    return run_lms_command(["server", "start"])


def lms_server_stop() -> Dict[str, Any]:
    """Stop LM Studio server."""
    return run_lms_command(["server", "stop"])


def lms_list_models() -> Dict[str, Any]:
    """List all downloaded models."""
    return run_lms_command(["ls"])


def lms_list_loaded() -> Dict[str, Any]:
    """List currently loaded models."""
    return run_lms_command(["ps"])


def lms_load_model(
    model_path: str,
    gpu_offload: str = "max",
    context_length: Optional[int] = None,
    ttl: Optional[int] = None,
    identifier: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a model into LM Studio.
    
    Args:
        model_path: Model identifier (e.g., "evony-7b-3800")
        gpu_offload: "max", "off", or 0.0-1.0
        context_length: Context window size
        ttl: Auto-unload after N seconds of idle
        identifier: Custom API identifier
    """
    args = ["load", model_path, "--yes", f"--gpu={gpu_offload}"]
    
    if context_length:
        args.append(f"--context-length={context_length}")
    if ttl:
        args.append(f"--ttl={ttl}")
    if identifier:
        args.append(f"--identifier={identifier}")
    
    return run_lms_command(args, timeout=120)


def lms_unload_model(identifier: Optional[str] = None, all_models: bool = False) -> Dict[str, Any]:
    """
    Unload a model from LM Studio.
    
    Args:
        identifier: Model identifier to unload
        all_models: If True, unload all models
    """
    if all_models:
        return run_lms_command(["unload", "--all"])
    elif identifier:
        return run_lms_command(["unload", identifier])
    else:
        return run_lms_command(["unload"])


def lms_estimate_resources(model_path: str, context_length: Optional[int] = None) -> Dict[str, Any]:
    """Estimate memory requirements for a model without loading it."""
    args = ["load", "--estimate-only", model_path]
    if context_length:
        args.append(f"--context-length={context_length}")
    return run_lms_command(args)


# =============================================================================
# REST API COMMANDS (via curl/requests)
# =============================================================================

def api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make an API request to LM Studio."""
    import urllib.request
    import urllib.error
    
    url = f"{LMSTUDIO_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode() if data else None,
                headers={"Content-Type": "application/json"},
                method=method
            )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            return {
                "success": True,
                "status": response.status,
                "data": json.loads(response.read().decode())
            }
    except urllib.error.URLError as e:
        return {"success": False, "error": str(e)}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON decode error: {e}"}


def api_list_models() -> Dict[str, Any]:
    """List all models via REST API (includes loaded state)."""
    return api_request("/api/v0/models")


def api_get_model(model_id: str) -> Dict[str, Any]:
    """Get specific model info via REST API."""
    return api_request(f"/api/v0/models/{model_id}")


def api_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    top_k: int = 40,
    stream: bool = False
) -> Dict[str, Any]:
    """
    Send a chat completion request.
    
    Args:
        messages: List of {"role": "user/assistant/system", "content": "..."}
        model: Model identifier (uses loaded model if None)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling
        top_k: Top-k sampling
        stream: Whether to stream response
    """
    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream
    }
    
    if model:
        data["model"] = model
    
    return api_request("/api/v0/chat/completions", method="POST", data=data)


def api_completion(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """Send a text completion request."""
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if model:
        data["model"] = model
    
    return api_request("/api/v0/completions", method="POST", data=data)


# =============================================================================
# PRESET MANAGEMENT
# =============================================================================

def list_presets() -> List[Dict[str, Any]]:
    """List all available presets."""
    presets = []
    
    # System presets
    if PRESETS_DIR.exists():
        for f in PRESETS_DIR.glob("*.preset.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    presets.append({
                        "name": data.get("name", f.stem),
                        "path": str(f),
                        "source": "system",
                        "identifier": data.get("identifier", "")
                    })
            except:
                pass
    
    # Evony presets
    if EVONY_PRESETS_DIR.exists():
        for f in EVONY_PRESETS_DIR.glob("*.preset.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    presets.append({
                        "name": data.get("name", f.stem),
                        "path": str(f),
                        "source": "evony",
                        "identifier": data.get("identifier", "")
                    })
            except:
                pass
    
    return presets


def get_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    """Get a preset by name."""
    # Check Evony presets first
    for f in EVONY_PRESETS_DIR.glob("*.preset.json"):
        if preset_name.lower() in f.stem.lower():
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except:
                pass
    
    # Check system presets
    for f in PRESETS_DIR.glob("*.preset.json"):
        if preset_name.lower() in f.stem.lower():
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except:
                pass
    
    return None


def install_preset(preset_name: str) -> Dict[str, Any]:
    """Copy an Evony preset to the LM Studio presets folder."""
    # Find the preset
    source_file = None
    for f in EVONY_PRESETS_DIR.glob("*.preset.json"):
        if preset_name.lower() in f.stem.lower():
            source_file = f
            break
    
    if not source_file:
        return {"success": False, "error": f"Preset '{preset_name}' not found"}
    
    # Ensure destination directory exists
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy the file
    import shutil
    dest_file = PRESETS_DIR / source_file.name
    
    try:
        shutil.copy2(source_file, dest_file)
        return {"success": True, "message": f"Installed {source_file.name} to {dest_file}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def install_all_evony_presets() -> Dict[str, Any]:
    """Install all Evony presets to LM Studio."""
    results = []
    for f in EVONY_PRESETS_DIR.glob("*.preset.json"):
        result = install_preset(f.stem)
        results.append({"preset": f.stem, **result})
    
    return {"success": True, "installed": results}


def extract_preset_params(preset: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inference parameters from a preset."""
    params = {}
    
    if "operation" in preset and "fields" in preset["operation"]:
        for field in preset["operation"]["fields"]:
            key = field.get("key", "")
            value = field.get("value")
            
            # Map LM Studio keys to API params
            if key == "llm.prediction.temperature":
                params["temperature"] = value
            elif key == "llm.prediction.topKSampling":
                params["top_k"] = value
            elif key == "llm.prediction.topPSampling":
                if isinstance(value, dict):
                    params["top_p"] = value.get("value", 0.9)
                else:
                    params["top_p"] = value
            elif key == "llm.prediction.repeatPenalty":
                if isinstance(value, dict):
                    params["repeat_penalty"] = value.get("value", 1.1)
                else:
                    params["repeat_penalty"] = value
            elif key == "llm.prediction.minPSampling":
                if isinstance(value, dict):
                    params["min_p"] = value.get("value", 0.05)
                else:
                    params["min_p"] = value
            elif key == "llm.prediction.maxPredictedTokens":
                if isinstance(value, dict):
                    params["max_tokens"] = value.get("value", 2048)
                else:
                    params["max_tokens"] = value
            elif key == "llm.prediction.systemPrompt":
                params["system_prompt"] = value
    
    return params


# =============================================================================
# HIGH-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def switch_model(model_name: str, context_length: int = 32768) -> Dict[str, Any]:
    """
    Switch to a different model (unload current, load new).
    
    Args:
        model_name: Model to load (e.g., "evony-7b-3800", "evony-7b-3800-rtx3060")
        context_length: Context window size
    """
    # First unload current model
    unload_result = lms_unload_model(all_models=True)
    
    # Then load new model
    load_result = lms_load_model(
        model_name,
        gpu_offload="max",
        context_length=context_length
    )
    
    return {
        "unload": unload_result,
        "load": load_result,
        "success": load_result.get("success", False)
    }


def chat_with_preset(
    messages: List[Dict[str, str]],
    preset_name: str,
    include_system_prompt: bool = True
) -> Dict[str, Any]:
    """
    Send a chat completion using parameters from a preset.
    
    Args:
        messages: Chat messages
        preset_name: Name of preset to use
        include_system_prompt: Whether to prepend the preset's system prompt
    """
    preset = get_preset(preset_name)
    if not preset:
        return {"success": False, "error": f"Preset '{preset_name}' not found"}
    
    params = extract_preset_params(preset)
    
    # Add system prompt if specified
    if include_system_prompt and "system_prompt" in params:
        system_msg = {"role": "system", "content": params["system_prompt"]}
        if messages[0].get("role") != "system":
            messages = [system_msg] + messages
    
    return api_chat_completion(
        messages=messages,
        temperature=params.get("temperature", 0.7),
        max_tokens=params.get("max_tokens", 2048),
        top_p=params.get("top_p", 0.9),
        top_k=params.get("top_k", 40)
    )


def get_status() -> Dict[str, Any]:
    """Get comprehensive LM Studio status."""
    server = lms_server_status()
    loaded = lms_list_loaded()
    models_api = api_list_models()
    
    loaded_models = []
    if models_api.get("success") and "data" in models_api.get("data", {}):
        for m in models_api["data"]["data"]:
            if m.get("state") == "loaded":
                loaded_models.append(m)
    
    return {
        "server": server,
        "loaded_models": loaded_models,
        "loaded_count": len(loaded_models),
        "cli_status": loaded
    }


# =============================================================================
# EVONY-SPECIFIC HELPERS
# =============================================================================

EVONY_MODELS = {
    "F16": "evony-7b-3800",
    "Q8": "evony-7b-3800-rtx3090ti",
    "Q6": "evony-7b-3800-rx7800xt",
    "Q4": "evony-7b-3800-rtx3060",
    "Q4_FAST": "evony-7b-3800-speed",
    "Q3_ULTRA": "evony-7b-3800-ultrafast"
}

EVONY_PRESETS = {
    "master": "evony-master-expert",
    "exploit": "evony-exploit-hunter",
    "protocol": "evony-protocol-decoder",
    "code": "evony-code-auditor",
    "quick": "evony-quick-reference",
    "forensic": "evony-forensic-analyst",
    "writer": "evony-creative-writer"
}


def load_evony_model(variant: str = "F16", context: int = 32768) -> Dict[str, Any]:
    """
    Load an Evony model variant.
    
    Args:
        variant: F16, Q8, Q6, Q4, Q4_FAST, Q3_ULTRA
        context: Context length
    """
    model = EVONY_MODELS.get(variant.upper())
    if not model:
        return {"success": False, "error": f"Unknown variant: {variant}. Options: {list(EVONY_MODELS.keys())}"}
    
    return switch_model(model, context)


def evony_chat(
    query: str,
    preset: str = "master",
    history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Chat with Evony model using a preset.
    
    Args:
        query: User's question
        preset: Preset name (master, exploit, protocol, code, quick, forensic, writer)
        history: Previous messages
    """
    preset_file = EVONY_PRESETS.get(preset.lower(), preset)
    
    messages = history or []
    messages.append({"role": "user", "content": query})
    
    return chat_with_preset(messages, preset_file)


# =============================================================================
# MAIN / CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
LM Studio Control - Claude/Windsurf Integration
================================================

Usage:
    python lmstudio_control.py status          - Get LM Studio status
    python lmstudio_control.py models          - List all models
    python lmstudio_control.py loaded          - List loaded models
    python lmstudio_control.py presets         - List available presets
    python lmstudio_control.py load <model>    - Load a model
    python lmstudio_control.py unload          - Unload current model
    python lmstudio_control.py switch <model>  - Switch to model
    python lmstudio_control.py install-presets - Install Evony presets
    
Evony Commands:
    python lmstudio_control.py evony-load <variant>  - Load Evony variant (F16/Q8/Q6/Q4)
    python lmstudio_control.py evony-chat <query>    - Quick chat with Evony
""")
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "status":
        print(json.dumps(get_status(), indent=2))
    
    elif cmd == "models":
        result = api_list_models()
        if result.get("success"):
            for m in result["data"]["data"]:
                state = "✓ LOADED" if m["state"] == "loaded" else ""
                print(f"{m['id']:50} {m.get('quantization', ''):10} {state}")
        else:
            print(f"Error: {result.get('error')}")
    
    elif cmd == "loaded":
        print(lms_list_loaded().get("stdout", "No models loaded"))
    
    elif cmd == "presets":
        for p in list_presets():
            print(f"[{p['source']:6}] {p['name']}")
    
    elif cmd == "load" and len(sys.argv) > 2:
        print(json.dumps(lms_load_model(sys.argv[2]), indent=2))
    
    elif cmd == "unload":
        print(json.dumps(lms_unload_model(), indent=2))
    
    elif cmd == "switch" and len(sys.argv) > 2:
        print(json.dumps(switch_model(sys.argv[2]), indent=2))
    
    elif cmd == "install-presets":
        print(json.dumps(install_all_evony_presets(), indent=2))
    
    elif cmd == "evony-load" and len(sys.argv) > 2:
        print(json.dumps(load_evony_model(sys.argv[2]), indent=2))
    
    elif cmd == "evony-chat" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        result = evony_chat(query)
        if result.get("success"):
            print(result["data"]["choices"][0]["message"]["content"])
        else:
            print(f"Error: {result.get('error')}")
    
    else:
        print(f"Unknown command: {cmd}")
