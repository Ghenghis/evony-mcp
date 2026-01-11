"""
Evony MCP - LM Studio Model Management Tools
=============================================
MCP tool handlers for real-time model switching, presets, and configuration.
These tools allow Claude/Windsurf to dynamically switch Evony models.
"""

import json
from typing import Dict, Any, List, Optional

from .lmstudio_manager import (
    get_lmstudio_manager,
    EVONY_MODELS,
    INFERENCE_PRESETS,
    EVONY_SYSTEM_PROMPTS,
    LOAD_CONFIGS,
)

# =============================================================================
# MCP TOOL DEFINITIONS
# =============================================================================

MCP_LMSTUDIO_TOOLS = [
    {
        "name": "evony_model_list",
        "description": "List all available Evony-trained models in LM Studio with their status (loaded/not-loaded), quantization, and recommended use cases.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "show_all": {
                    "type": "boolean",
                    "description": "If true, show all models including non-Evony models",
                    "default": False
                }
            }
        }
    },
    {
        "name": "evony_model_switch",
        "description": "Switch to a different Evony model. This will unload the current model and load the new one. Shows loading progress. Use this when you need a different model for a specific task (e.g., switch to F16 for deep analysis, Q4 for fast responses).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "The model ID to switch to (e.g., 'evony-7b-3800-F16', 'evony-7b-3800-speed')"
                },
                "config": {
                    "type": "string",
                    "enum": ["maximum_quality", "balanced", "fast", "memory_saver"],
                    "description": "Load configuration preset",
                    "default": "balanced"
                },
                "keep_current": {
                    "type": "boolean",
                    "description": "If true, don't unload the current model (load multiple)",
                    "default": False
                }
            },
            "required": ["model_id"]
        }
    },
    {
        "name": "evony_model_recommend",
        "description": "Get a recommendation for the best Evony model based on your current task. Recommends model, preset, and system prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Description of your task (e.g., 'exploit analysis', 'packet decoding', 'quick help', 'code review')"
                }
            },
            "required": ["task"]
        }
    },
    {
        "name": "evony_model_status",
        "description": "Get the current model loading status. Shows which model is loaded, if any model is currently loading, and load times.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "evony_model_unload",
        "description": "Unload a model from memory to free up VRAM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Specific model to unload, or omit to unload all"
                }
            }
        }
    },
    {
        "name": "evony_preset_list",
        "description": "List all available inference presets optimized for different Evony tasks.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "evony_preset_get",
        "description": "Get detailed preset configuration including system prompt, temperature, and sampling settings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset_name": {
                    "type": "string",
                    "enum": ["exploit_analysis", "packet_decode", "protocol_research", "code_review", "quick_help", "interactive", "creative"],
                    "description": "The preset to retrieve"
                }
            },
            "required": ["preset_name"]
        }
    },
    {
        "name": "evony_prompt_get",
        "description": "Get a specialized system prompt for Evony tasks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt_type": {
                    "type": "string",
                    "enum": ["exploit_analyst", "protocol_engineer", "source_auditor", "packet_decoder", "general_assistant", "fast_helper"],
                    "description": "The type of system prompt"
                }
            },
            "required": ["prompt_type"]
        }
    },
    {
        "name": "evony_config_optimal",
        "description": "Get optimal LM Studio load settings for a specific use case.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "use_case": {
                    "type": "string",
                    "enum": ["maximum_quality", "balanced", "fast", "memory_saver"],
                    "description": "The use case for optimization"
                }
            },
            "required": ["use_case"]
        }
    },
    {
        "name": "evony_access_check",
        "description": "Check if a file path is accessible to Evony models (all drives except Windows system).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to check"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "evony_data_paths",
        "description": "Get all Evony data paths that models can access (training data, exploits, protocol docs, source code).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
]

# =============================================================================
# TOOL HANDLERS
# =============================================================================

async def handle_evony_model_list(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_model_list tool call."""
    manager = get_lmstudio_manager()
    show_all = args.get("show_all", False)
    
    result = manager.list_models(filter_evony=not show_all)
    
    if result.get("success"):
        # Add model catalog info
        catalog_models = []
        for model_id, model_info in EVONY_MODELS.items():
            catalog_models.append({
                "id": model_id,
                "quantization": model_info.quantization,
                "size_gb": model_info.size_gb,
                "speed": model_info.speed,
                "use_case": model_info.use_case,
                "recommended_tasks": model_info.recommended_tasks,
            })
        
        result["model_catalog"] = catalog_models
    
    return result


async def handle_evony_model_switch(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_model_switch tool call."""
    manager = get_lmstudio_manager()
    
    model_id = args.get("model_id")
    config = args.get("config", "balanced")
    keep_current = args.get("keep_current", False)
    
    if not model_id:
        return {"success": False, "error": "model_id is required"}
    
    # Switch model
    result = manager.switch_model(
        new_model_id=model_id,
        config=config,
        unload_current=not keep_current
    )
    
    # Add helpful info about the loaded model
    if result.get("success"):
        model_info = EVONY_MODELS.get(model_id)
        if model_info:
            result["model_info"] = {
                "use_case": model_info.use_case,
                "speed": model_info.speed,
                "recommended_tasks": model_info.recommended_tasks,
            }
    
    return result


async def handle_evony_model_recommend(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_model_recommend tool call."""
    manager = get_lmstudio_manager()
    task = args.get("task", "")
    
    recommendation = manager.recommend_model(task)
    
    # Add the full preset info
    if recommendation.get("recommended_preset"):
        preset = manager.get_preset(recommendation["recommended_preset"])
        recommendation["preset_details"] = preset
    
    return recommendation


async def handle_evony_model_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_model_status tool call."""
    manager = get_lmstudio_manager()
    
    current = manager.get_current_model()
    
    return {
        "success": True,
        "current_model": current.get("id") if current else None,
        "model_state": current.get("state") if current else "none",
        "model_details": current if current else None,
        "loading_model": manager.loading_model,
        "is_loading": manager.loading_model is not None,
        "available_models": len(EVONY_MODELS),
    }


async def handle_evony_model_unload(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_model_unload tool call."""
    manager = get_lmstudio_manager()
    model_id = args.get("model_id")
    
    return manager.unload_model(model_id)


async def handle_evony_preset_list(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_preset_list tool call."""
    manager = get_lmstudio_manager()
    
    presets = manager.list_presets()
    
    return {
        "success": True,
        "presets": presets,
        "count": len(presets),
    }


async def handle_evony_preset_get(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_preset_get tool call."""
    manager = get_lmstudio_manager()
    preset_name = args.get("preset_name")
    
    if not preset_name:
        return {"success": False, "error": "preset_name is required"}
    
    preset = manager.get_preset(preset_name)
    
    if preset:
        return {"success": True, "preset": preset}
    else:
        return {"success": False, "error": f"Preset '{preset_name}' not found"}


async def handle_evony_prompt_get(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_prompt_get tool call."""
    manager = get_lmstudio_manager()
    prompt_type = args.get("prompt_type")
    
    if not prompt_type:
        return {"success": False, "error": "prompt_type is required"}
    
    prompt = manager.get_system_prompt(prompt_type)
    
    return {
        "success": True,
        "prompt_type": prompt_type,
        "system_prompt": prompt,
        "length": len(prompt),
    }


async def handle_evony_config_optimal(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_config_optimal tool call."""
    use_case = args.get("use_case", "balanced")
    
    config = LOAD_CONFIGS.get(use_case)
    
    if config:
        return {
            "success": True,
            "use_case": use_case,
            "config": config,
            "lm_studio_settings": {
                "context_length": config["context_length"],
                "gpu_offload": config["gpu_offload"],
                "flash_attention": config["flash_attention"],
                "kv_cache_quant": config["kv_cache_quant"],
                "eval_batch_size": config["eval_batch_size"],
                "keep_in_memory": config["keep_in_memory"],
                "mmap": config["mmap"],
            }
        }
    else:
        return {"success": False, "error": f"Unknown use case: {use_case}"}


async def handle_evony_access_check(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_access_check tool call."""
    manager = get_lmstudio_manager()
    path = args.get("path", "")
    
    allowed, message = manager.is_path_allowed(path)
    
    return {
        "success": True,
        "path": path,
        "allowed": allowed,
        "message": message,
    }


async def handle_evony_data_paths(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evony_data_paths tool call."""
    manager = get_lmstudio_manager()
    
    return {
        "success": True,
        **manager.get_allowed_paths()
    }


# =============================================================================
# TOOL DISPATCHER
# =============================================================================

TOOL_HANDLERS = {
    "evony_model_list": handle_evony_model_list,
    "evony_model_switch": handle_evony_model_switch,
    "evony_model_recommend": handle_evony_model_recommend,
    "evony_model_status": handle_evony_model_status,
    "evony_model_unload": handle_evony_model_unload,
    "evony_preset_list": handle_evony_preset_list,
    "evony_preset_get": handle_evony_preset_get,
    "evony_prompt_get": handle_evony_prompt_get,
    "evony_config_optimal": handle_evony_config_optimal,
    "evony_access_check": handle_evony_access_check,
    "evony_data_paths": handle_evony_data_paths,
}


async def dispatch_lmstudio_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to its handler."""
    handler = TOOL_HANDLERS.get(tool_name)
    
    if handler:
        return await handler(args)
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}


def get_lmstudio_tools() -> List[Dict[str, Any]]:
    """Get all LM Studio MCP tool definitions."""
    return MCP_LMSTUDIO_TOOLS
