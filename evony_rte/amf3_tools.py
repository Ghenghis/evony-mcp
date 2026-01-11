"""
Evony AMF3 Packet Tools - Robust Analysis for Reverse Engineering
=================================================================
Comprehensive AMF3 encoding/decoding with Evony-specific analysis.
"""

import struct
import zlib
import json
import re
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# AMF3 TYPE CONSTANTS
# ============================================================================

class AMF3Types:
    UNDEFINED = 0x00
    NULL = 0x01
    FALSE = 0x02
    TRUE = 0x03
    INTEGER = 0x04
    DOUBLE = 0x05
    STRING = 0x06
    XML_DOC = 0x07
    DATE = 0x08
    ARRAY = 0x09
    OBJECT = 0x0A
    XML = 0x0B
    BYTEARRAY = 0x0C
    VECTOR_INT = 0x0D
    VECTOR_UINT = 0x0E
    VECTOR_DOUBLE = 0x0F
    VECTOR_OBJECT = 0x10
    DICTIONARY = 0x11
    
    TYPE_NAMES = {
        0x00: "undefined", 0x01: "null", 0x02: "false", 0x03: "true",
        0x04: "integer", 0x05: "double", 0x06: "string", 0x07: "xml_doc",
        0x08: "date", 0x09: "array", 0x0A: "object", 0x0B: "xml",
        0x0C: "bytearray", 0x0D: "vector_int", 0x0E: "vector_uint",
        0x0F: "vector_double", 0x10: "vector_object", 0x11: "dictionary"
    }

# ============================================================================
# EVONY COMMAND DATABASE - Known commands and their parameters
# ============================================================================

EVONY_COMMANDS = {
    # Troop Commands
    "troop.produceTroop": {
        "params": ["castleId", "troopType", "num", "barrackId", "heroId"],
        "param_types": {"castleId": "int", "troopType": "int", "num": "int", "barrackId": "int", "heroId": "int"},
        "vuln_fields": ["num"],  # Can overflow
        "description": "Produce troops - vulnerable to integer overflow"
    },
    "troop.disbandTroop": {
        "params": ["castleId", "troopType", "num"],
        "param_types": {"castleId": "int", "troopType": "int", "num": "int"},
        "vuln_fields": ["num"],  # Can be negative
        "description": "Disband troops - vulnerable to negative values"
    },
    "troop.getProduceQueue": {
        "params": ["castleId"],
        "param_types": {"castleId": "int"},
        "description": "Get troop production queue"
    },
    # Army Commands
    "army.newArmy": {
        "params": ["castleId", "heroId", "troops", "targetX", "targetY", "missionType"],
        "param_types": {"castleId": "int", "heroId": "int", "troops": "array", "targetX": "int", "targetY": "int", "missionType": "int"},
        "description": "Create new army march"
    },
    "army.callBackArmy": {
        "params": ["armyId"],
        "param_types": {"armyId": "int"},
        "description": "Recall army"
    },
    # Castle Commands
    "castle.UpdateCastleResources": {
        "params": ["castleId"],
        "param_types": {"castleId": "int"},
        "description": "Refresh resource counts"
    },
    # City Commands
    "city.transportResources": {
        "params": ["fromCastleId", "toCastleId", "wood", "food", "stone", "iron"],
        "param_types": {"fromCastleId": "int", "toCastleId": "int", "wood": "int", "food": "int", "stone": "int", "iron": "int"},
        "vuln_fields": ["wood", "food", "stone", "iron"],  # Can be negative
        "description": "Transport resources - vulnerable to negative values"
    },
    # Shop Commands
    "shop.useGoods": {
        "params": ["castleId", "itemId", "amount"],
        "param_types": {"castleId": "int", "itemId": "string", "amount": "int"},
        "vuln_fields": ["amount"],
        "description": "Use item from inventory"
    },
    "shop.buy": {
        "params": ["castleId", "itemId", "amount", "payType"],
        "param_types": {"castleId": "int", "itemId": "string", "amount": "int", "payType": "int"},
        "description": "Buy from shop"
    },
    # Hero Commands
    "hero.fireHero": {
        "params": ["castleId", "heroId"],
        "param_types": {"castleId": "int", "heroId": "int"},
        "description": "Fire/dismiss hero"
    },
    "hero.levelUpHero": {
        "params": ["castleId", "heroId"],
        "param_types": {"castleId": "int", "heroId": "int"},
        "description": "Level up hero"
    },
    # Common Commands
    "common.mapInfo": {
        "params": ["castleId", "x1", "y1", "x2", "y2"],
        "param_types": {"castleId": "int", "x1": "int", "y1": "int", "x2": "int", "y2": "int"},
        "description": "Get map tile information"
    },
    "common.getPlayerInfoByName": {
        "params": ["userName"],
        "param_types": {"userName": "string"},
        "description": "Lookup player by name"
    },
    # Login
    "login": {
        "params": ["user", "pwd"],
        "param_types": {"user": "string", "pwd": "string"},
        "description": "Login to game server"
    }
}

# Troop type mapping
TROOP_TYPES = {
    1: "Worker", 2: "Warrior", 3: "Scout", 4: "Pikeman", 5: "Swordsman",
    6: "Archer", 7: "Cavalry", 8: "Cataphract", 9: "Transporter", 10: "Ballista",
    11: "Battering Ram", 12: "Catapult"
}

# Resource cost per troop (for overflow calculation)
TROOP_COSTS = {
    1: {"food": 50},  # Worker
    6: {"food": 350, "gold": 500},  # Archer
    7: {"food": 500, "gold": 1000},  # Cavalry
    8: {"food": 800, "gold": 2000},  # Cataphract
}

# ============================================================================
# AMF3 DECODER - Full Featured
# ============================================================================

class AMF3Decoder:
    """Comprehensive AMF3 decoder with Evony analysis."""
    
    def __init__(self, data: bytes):
        self.stream = BytesIO(data)
        self.string_refs = []
        self.object_refs = []
        self.traits_refs = []
        self.raw_data = data
        self.decode_log = []  # Track what was decoded
        
    def read_byte(self) -> int:
        b = self.stream.read(1)
        return b[0] if b else 0
    
    def read_bytes(self, n: int) -> bytes:
        return self.stream.read(n)
    
    def read_u29(self) -> int:
        """Read variable-length unsigned 29-bit integer."""
        result = 0
        for i in range(4):
            b = self.read_byte()
            if i < 3:
                result = (result << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            else:
                result = (result << 8) | b
        return result
    
    def read_i29(self) -> int:
        """Read signed 29-bit integer."""
        n = self.read_u29()
        if n & 0x10000000:
            n -= 0x20000000
        return n
    
    def read_string(self) -> str:
        """Read AMF3 string with reference handling."""
        ref = self.read_u29()
        if (ref & 1) == 0:
            idx = ref >> 1
            if idx < len(self.string_refs):
                return self.string_refs[idx]
            return f"<invalid_ref:{idx}>"
        
        length = ref >> 1
        if length == 0:
            return ""
        
        s = self.stream.read(length).decode('utf-8', errors='replace')
        self.string_refs.append(s)
        return s
    
    def read_value(self) -> Any:
        """Read any AMF3 value with type tracking."""
        pos = self.stream.tell()
        type_marker = self.read_byte()
        type_name = AMF3Types.TYPE_NAMES.get(type_marker, f"unknown_{type_marker}")
        
        result = None
        
        if type_marker == AMF3Types.UNDEFINED:
            result = None
        elif type_marker == AMF3Types.NULL:
            result = None
        elif type_marker == AMF3Types.FALSE:
            result = False
        elif type_marker == AMF3Types.TRUE:
            result = True
        elif type_marker == AMF3Types.INTEGER:
            result = self.read_i29()
        elif type_marker == AMF3Types.DOUBLE:
            result = struct.unpack('>d', self.read_bytes(8))[0]
        elif type_marker == AMF3Types.STRING:
            result = self.read_string()
        elif type_marker == AMF3Types.DATE:
            ref = self.read_u29()
            if ref & 1:
                timestamp = struct.unpack('>d', self.read_bytes(8))[0]
                result = {"__type": "date", "timestamp": timestamp, "iso": datetime.fromtimestamp(timestamp/1000).isoformat()}
        elif type_marker == AMF3Types.ARRAY:
            result = self.read_array()
        elif type_marker == AMF3Types.OBJECT:
            result = self.read_object()
        elif type_marker == AMF3Types.BYTEARRAY:
            ref = self.read_u29()
            length = ref >> 1
            data = self.read_bytes(length)
            result = {"__type": "bytearray", "length": length, "hex": data.hex()[:200]}
        elif type_marker == AMF3Types.XML or type_marker == AMF3Types.XML_DOC:
            ref = self.read_u29()
            length = ref >> 1
            result = {"__type": "xml", "content": self.read_bytes(length).decode('utf-8', errors='replace')}
        else:
            result = {"__type": f"unknown_{type_marker}", "position": pos}
        
        self.decode_log.append({"pos": pos, "type": type_name, "value_preview": str(result)[:100]})
        return result
    
    def read_array(self) -> List:
        """Read AMF3 array with associative and dense portions."""
        ref = self.read_u29()
        if (ref & 1) == 0:
            idx = ref >> 1
            if idx < len(self.object_refs):
                return self.object_refs[idx]
            return []
        
        length = ref >> 1
        result = []
        self.object_refs.append(result)
        
        # Read associative portion (string keys)
        assoc = {}
        while True:
            key = self.read_string()
            if key == "":
                break
            assoc[key] = self.read_value()
        
        # Read dense portion
        for _ in range(length):
            result.append(self.read_value())
        
        # If we have associative values, return as object
        if assoc:
            return {"__type": "mixed_array", "dense": result, "assoc": assoc}
        
        return result
    
    def read_object(self) -> Dict:
        """Read AMF3 object with full traits support."""
        ref = self.read_u29()
        if (ref & 1) == 0:
            idx = ref >> 1
            if idx < len(self.object_refs):
                return self.object_refs[idx]
            return {}
        
        traits_ref = ref >> 1
        
        if (traits_ref & 1) == 0:
            # Traits reference
            traits_idx = traits_ref >> 1
            if traits_idx < len(self.traits_refs):
                traits = self.traits_refs[traits_idx]
            else:
                traits = {"class_name": "", "is_dynamic": True, "sealed_count": 0, "sealed_names": []}
        else:
            # Inline traits
            is_externalizable = (traits_ref >> 1) & 1
            is_dynamic = (traits_ref >> 2) & 1
            sealed_count = traits_ref >> 3
            
            class_name = self.read_string()
            sealed_names = [self.read_string() for _ in range(sealed_count)]
            
            traits = {
                "class_name": class_name,
                "is_externalizable": bool(is_externalizable),
                "is_dynamic": bool(is_dynamic),
                "sealed_count": sealed_count,
                "sealed_names": sealed_names
            }
            self.traits_refs.append(traits)
        
        result = {}
        if traits["class_name"]:
            result["__class"] = traits["class_name"]
        
        self.object_refs.append(result)
        
        # Read sealed properties
        for name in traits.get("sealed_names", []):
            result[name] = self.read_value()
        
        # Read dynamic properties
        if traits.get("is_dynamic", True):
            while True:
                key = self.read_string()
                if key == "":
                    break
                result[key] = self.read_value()
        
        return result
    
    def decode(self) -> Dict:
        """Decode complete packet and return analysis."""
        try:
            decoded = self.read_value()
            return {
                "success": True,
                "decoded": decoded,
                "stats": {
                    "bytes_total": len(self.raw_data),
                    "bytes_read": self.stream.tell(),
                    "string_refs": len(self.string_refs),
                    "object_refs": len(self.object_refs),
                    "decode_steps": len(self.decode_log)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "partial_decode": self.decode_log,
                "position": self.stream.tell()
            }

# ============================================================================
# AMF3 ENCODER - Full Featured
# ============================================================================

class AMF3Encoder:
    """Comprehensive AMF3 encoder for Evony packets."""
    
    def __init__(self):
        self.stream = BytesIO()
        self.string_refs = []
        self.object_refs = []
        
    def write_byte(self, b: int):
        self.stream.write(bytes([b & 0xFF]))
    
    def write_bytes(self, data: bytes):
        self.stream.write(data)
    
    def write_u29(self, n: int):
        """Write variable-length unsigned 29-bit integer."""
        n = n & 0x1FFFFFFF
        if n < 0x80:
            self.write_byte(n)
        elif n < 0x4000:
            self.write_byte((n >> 7) | 0x80)
            self.write_byte(n & 0x7F)
        elif n < 0x200000:
            self.write_byte((n >> 14) | 0x80)
            self.write_byte((n >> 7) | 0x80)
            self.write_byte(n & 0x7F)
        else:
            self.write_byte((n >> 22) | 0x80)
            self.write_byte((n >> 15) | 0x80)
            self.write_byte((n >> 8) | 0x80)
            self.write_byte(n & 0xFF)
    
    def write_string(self, s: str):
        """Write AMF3 string with reference optimization."""
        if s == "":
            self.write_u29(1)
            return
        
        if s in self.string_refs:
            ref = self.string_refs.index(s)
            self.write_u29(ref << 1)
            return
        
        self.string_refs.append(s)
        encoded = s.encode('utf-8')
        self.write_u29((len(encoded) << 1) | 1)
        self.write_bytes(encoded)
    
    def write_value(self, value: Any):
        """Write any Python value as AMF3."""
        if value is None:
            self.write_byte(AMF3Types.NULL)
        elif value is True:
            self.write_byte(AMF3Types.TRUE)
        elif value is False:
            self.write_byte(AMF3Types.FALSE)
        elif isinstance(value, int):
            if -0x10000000 <= value < 0x10000000:
                self.write_byte(AMF3Types.INTEGER)
                self.write_u29(value & 0x1FFFFFFF)
            else:
                # Large integer as double
                self.write_byte(AMF3Types.DOUBLE)
                self.write_bytes(struct.pack('>d', float(value)))
        elif isinstance(value, float):
            self.write_byte(AMF3Types.DOUBLE)
            self.write_bytes(struct.pack('>d', value))
        elif isinstance(value, str):
            self.write_byte(AMF3Types.STRING)
            self.write_string(value)
        elif isinstance(value, list):
            self.write_byte(AMF3Types.ARRAY)
            self.write_array(value)
        elif isinstance(value, dict):
            self.write_byte(AMF3Types.OBJECT)
            self.write_object(value)
        elif isinstance(value, bytes):
            self.write_byte(AMF3Types.BYTEARRAY)
            self.write_u29((len(value) << 1) | 1)
            self.write_bytes(value)
        else:
            self.write_byte(AMF3Types.STRING)
            self.write_string(str(value))
    
    def write_array(self, arr: List):
        """Write AMF3 array."""
        self.write_u29((len(arr) << 1) | 1)
        self.write_string("")  # No associative portion
        for item in arr:
            self.write_value(item)
    
    def write_object(self, obj: Dict):
        """Write AMF3 object (dynamic)."""
        # Dynamic anonymous object
        self.write_u29(0x0B)
        self.write_string("")  # Empty class name
        
        for key, value in obj.items():
            if not key.startswith("__"):  # Skip meta fields
                self.write_string(str(key))
                self.write_value(value)
        
        self.write_string("")  # End of dynamic properties
    
    def get_data(self) -> bytes:
        return self.stream.getvalue()

# ============================================================================
# EVONY PACKET ANALYSIS
# ============================================================================

@dataclass
class PacketAnalysis:
    """Detailed packet analysis results."""
    command: str
    data: Dict
    raw_hex: str
    raw_length: int
    is_compressed: bool
    known_command: bool
    command_info: Optional[Dict]
    vulnerabilities: List[Dict]
    field_analysis: List[Dict]
    warnings: List[str]

def analyze_evony_packet(data: bytes, try_decompress: bool = True) -> Dict:
    """
    Comprehensive Evony packet analysis.
    
    Handles:
    - Length prefix detection
    - Compression detection and handling
    - AMF3 decoding
    - Command identification
    - Vulnerability scanning
    - Field type analysis
    """
    result = {
        "raw_hex": data.hex(),
        "raw_length": len(data),
        "analysis": {},
        "vulnerabilities": [],
        "warnings": []
    }
    
    working_data = data
    
    # Check for 4-byte length prefix (Evony protocol)
    if len(data) >= 4:
        length_prefix = struct.unpack('>I', data[:4])[0]
        is_compressed = bool(length_prefix & 0x80000000)
        actual_length = length_prefix & 0x7FFFFFFF
        
        result["has_length_prefix"] = True
        result["is_compressed"] = is_compressed
        result["stated_length"] = actual_length
        
        if actual_length == len(data) - 4:
            working_data = data[4:]
            result["length_prefix_valid"] = True
        else:
            result["length_prefix_valid"] = False
            result["warnings"].append(f"Length mismatch: stated {actual_length}, actual {len(data)-4}")
    
    # Try decompression if flagged or data looks compressed
    if try_decompress and result.get("is_compressed", False):
        try:
            working_data = zlib.decompress(working_data)
            result["decompressed"] = True
            result["decompressed_length"] = len(working_data)
        except:
            result["decompressed"] = False
            result["warnings"].append("Decompression failed")
    
    # Decode AMF3
    decoder = AMF3Decoder(working_data)
    decode_result = decoder.decode()
    
    if decode_result["success"]:
        decoded = decode_result["decoded"]
        result["decoded"] = decoded
        result["decode_stats"] = decode_result["stats"]
        
        # Extract command info
        if isinstance(decoded, dict):
            cmd = decoded.get("cmd", "")
            data_obj = decoded.get("data", {})
            
            result["command"] = cmd
            result["command_data"] = data_obj
            
            # Check if known command
            if cmd in EVONY_COMMANDS:
                cmd_info = EVONY_COMMANDS[cmd]
                result["known_command"] = True
                result["command_info"] = cmd_info
                
                # Analyze fields
                field_analysis = []
                for param in cmd_info.get("params", []):
                    value = data_obj.get(param)
                    expected_type = cmd_info.get("param_types", {}).get(param, "unknown")
                    
                    analysis = {
                        "field": param,
                        "value": value,
                        "expected_type": expected_type,
                        "actual_type": type(value).__name__ if value is not None else "null"
                    }
                    
                    # Check for vulnerabilities
                    if param in cmd_info.get("vuln_fields", []):
                        analysis["vuln_check"] = True
                        
                        if isinstance(value, (int, float)):
                            # Integer overflow check
                            if value > 2147483647:
                                result["vulnerabilities"].append({
                                    "type": "integer_overflow",
                                    "field": param,
                                    "value": value,
                                    "severity": "HIGH",
                                    "description": f"Value {value} exceeds INT32_MAX (2147483647)"
                                })
                            # Negative value check
                            if value < 0:
                                result["vulnerabilities"].append({
                                    "type": "negative_value",
                                    "field": param,
                                    "value": value,
                                    "severity": "HIGH",
                                    "description": f"Negative value {value} may bypass validation"
                                })
                    
                    field_analysis.append(analysis)
                
                result["field_analysis"] = field_analysis
            else:
                result["known_command"] = False
                result["warnings"].append(f"Unknown command: {cmd}")
    else:
        result["decode_error"] = decode_result["error"]
        result["partial_decode"] = decode_result.get("partial_decode", [])
    
    return result

def encode_evony_packet(cmd: str, data: Dict, add_length_prefix: bool = True, compress: bool = False) -> Dict:
    """
    Encode an Evony packet with full AMF3 support.
    
    Returns:
    - Binary AMF3 data
    - Hex representation
    - Analysis of what was encoded
    """
    packet = {"cmd": cmd, "data": data}
    
    encoder = AMF3Encoder()
    encoder.write_value(packet)
    amf_data = encoder.get_data()
    
    result = {
        "command": cmd,
        "data": data,
        "amf3_binary": amf_data,
        "amf3_hex": amf_data.hex(),
        "amf3_length": len(amf_data),
    }
    
    if compress:
        compressed = zlib.compress(amf_data)
        result["compressed_binary"] = compressed
        result["compressed_hex"] = compressed.hex()
        result["compressed_length"] = len(compressed)
        result["compression_ratio"] = len(compressed) / len(amf_data)
        working_data = compressed
    else:
        working_data = amf_data
    
    if add_length_prefix:
        length = len(working_data)
        if compress:
            length |= 0x80000000  # Set compression flag
        prefix = struct.pack('>I', length)
        final_data = prefix + working_data
        result["with_prefix_binary"] = final_data
        result["with_prefix_hex"] = final_data.hex()
        result["with_prefix_length"] = len(final_data)
    
    # Add command analysis if known
    if cmd in EVONY_COMMANDS:
        cmd_info = EVONY_COMMANDS[cmd]
        result["command_info"] = cmd_info
        
        # Check for potential vulnerabilities in the data
        vulns = []
        for param, value in data.items():
            if param in cmd_info.get("vuln_fields", []):
                if isinstance(value, int):
                    if value > 2147483647:
                        vulns.append({
                            "field": param,
                            "issue": "overflow",
                            "value": value
                        })
                    if value < 0:
                        vulns.append({
                            "field": param,
                            "issue": "negative",
                            "value": value
                        })
        
        if vulns:
            result["potential_exploits"] = vulns
    
    return result

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_overflow_amount(troop_type: int, target_overflow: int = 2147483648) -> Dict:
    """
    Calculate the number of troops needed to cause integer overflow.
    
    This is useful for testing the troop production overflow exploit.
    """
    if troop_type not in TROOP_COSTS:
        return {"error": f"Unknown troop type: {troop_type}"}
    
    costs = TROOP_COSTS[troop_type]
    results = {}
    
    for resource, cost_per_unit in costs.items():
        # Calculate how many troops to overflow
        overflow_amount = (target_overflow // cost_per_unit) + 1
        total_cost = overflow_amount * cost_per_unit
        overflow_value = total_cost - 2147483648  # What it wraps to
        
        results[resource] = {
            "cost_per_troop": cost_per_unit,
            "troops_needed": overflow_amount,
            "total_cost": total_cost,
            "overflows_to": overflow_value if total_cost > 2147483647 else "no overflow"
        }
    
    troop_name = TROOP_TYPES.get(troop_type, f"Type {troop_type}")
    
    return {
        "troop_type": troop_type,
        "troop_name": troop_name,
        "overflow_analysis": results,
        "exploit_payload": {
            "cmd": "troop.produceTroop",
            "data": {
                "troopType": troop_type,
                "num": min(r["troops_needed"] for r in results.values())
            }
        }
    }

def hex_dump(data: bytes, bytes_per_line: int = 16) -> str:
    """Create a formatted hex dump for analysis."""
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f'{i:08x}  {hex_part:<{bytes_per_line*3}}  {ascii_part}')
    return '\n'.join(lines)

def find_patterns_in_packet(data: bytes) -> List[Dict]:
    """Find interesting patterns in packet data."""
    patterns = []
    hex_str = data.hex()
    
    # Look for command strings
    cmd_pattern = rb'[a-z]+\.[a-zA-Z]+'
    for match in re.finditer(cmd_pattern, data):
        patterns.append({
            "type": "command_string",
            "offset": match.start(),
            "value": match.group().decode('utf-8', errors='replace')
        })
    
    # Look for large integers (potential IDs)
    for i in range(len(data) - 4):
        val = struct.unpack('>I', data[i:i+4])[0]
        if 1000000 < val < 10000000000:  # Looks like an ID
            patterns.append({
                "type": "potential_id",
                "offset": i,
                "value": val
            })
    
    return patterns[:20]  # Limit results


# Export main functions
__all__ = [
    'AMF3Decoder', 'AMF3Encoder', 'AMF3Types',
    'analyze_evony_packet', 'encode_evony_packet',
    'calculate_overflow_amount', 'hex_dump', 'find_patterns_in_packet',
    'EVONY_COMMANDS', 'TROOP_TYPES', 'TROOP_COSTS'
]
