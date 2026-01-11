"""
Evony RTE - Real-Time Engineering MCP Server
============================================
Complete reverse engineering toolkit for Evony game analysis.
Bot server auto-starts with MCP for live game interaction.
"""
import sys

__version__ = "2.0.0"

# Auto-start bot server when module is imported
_BOT_PORT = None

def get_bot_port():
    """Get the running bot server port."""
    global _BOT_PORT
    if _BOT_PORT is None:
        try:
            from .integrated_bot_server import ensure_bot_server_running
            _BOT_PORT = ensure_bot_server_running()
        except Exception as e:
            # Use stderr to avoid corrupting MCP JSON protocol on stdout
            print(f"[evony_rte] Bot server auto-start skipped: {e}", file=sys.stderr)
            _BOT_PORT = 9999  # Default port
    return _BOT_PORT

# Auto-start on import (lazy - only when first handler is called)
def _init_bot_server():
    """Initialize bot server on first use."""
    import atexit
    try:
        port = get_bot_port()
        if port:
            # Use stderr to avoid corrupting MCP JSON protocol on stdout
            print(f"[evony_rte] Bot server ready on port {port}", file=sys.stderr)
    except:
        pass

# Register initialization
try:
    _init_bot_server()
except:
    pass
