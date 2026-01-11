# Required Tools for Evony MCP Servers

The evony-rte MCP server integrates with these command-line tools for reverse engineering.

## Included Tools (in `tools/` folder)

| Tool             | Size       | Description                             |
| ---------------- | ---------- | --------------------------------------- |
| `Java21-JRE.zip` | 47 MB      | OpenJDK 21 JRE (required for FFDec)     |
| `RABCDAsm.zip`   | 0.13 MB    | ABC bytecode disassembly/assembly       |
| `FFDec.zip`      | 16.2 MB    | Flash SWF decompiler (JPEXS)            |
| `SWFTools.zip`   | 24.5 MB    | SWF manipulation utilities (swfextract) |
| `Graphviz.zip`   | 9.7 MB     | Flow diagram generation (dot)           |
| `Radare2.zip`    | 9.4 MB     | Advanced binary analysis (r2)           |
| **Total**        | **107 MB** | All CLI tools for evony-rte MCP         |

### Java 21 JRE (Required)
**Archive:** `tools/Java21-JRE.zip`

OpenJDK 21 runtime required for FFDec and other Java-based tools.

**Extract and set PATH:**
```powershell
Expand-Archive tools/Java21-JRE.zip -DestinationPath C:\Tools\Java21
$env:JAVA_HOME = "C:\Tools\Java21\jdk-21.0.5+11-jre"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
```

### RABCDAsm (ABC Bytecode Tools)
**Archive:** `tools/RABCDAsm.zip`

D source code for ABC bytecode manipulation:
- `rabcdasm` - Disassemble ABC bytecode
- `rabcasm` - Assemble ABC bytecode
- `abcexport` - Export ABC from SWF
- `abcreplace` - Replace ABC in SWF

**Build (requires D compiler):**
```bash
cd tools/RABCDAsm
dmd -of=rabcdasm rabcdasm.d
dmd -of=rabcasm rabcasm.d
dmd -of=abcexport abcexport.d
dmd -of=abcreplace abcreplace.d
```

Or download pre-built binaries from: https://github.com/lsalzman/RABCDAsm/releases

### FFDec / JPEXS Decompiler
**Archive:** `tools/FFDec.zip`

Flash SWF decompiler with full AS3 decompilation:
- Decompile ActionScript 3 code
- Export assets (images, sounds, shapes)
- Edit and recompile SWF files
- Deobfuscation support

**Extract and run:**
```bash
unzip tools/FFDec.zip -d C:\Tools\FFDec
# Run: C:\Tools\FFDec\ffdec.exe
```

---

## External Tools (Install Separately - Too Large for Repo)

### Ghidra (489 MB compressed)
**Purpose:** NSA reverse engineering framework for binary analysis

```powershell
# Windows (Chocolatey)
choco install ghidra

# Or download from:
# https://ghidra-sre.org/
```

Set `GHIDRA_HOME` environment variable after installation.

### Wireshark / TShark (100 MB compressed)
**Purpose:** Network packet capture and protocol analysis

```powershell
# Windows (Chocolatey)
choco install wireshark

# Or download from:
# https://www.wireshark.org/download.html
```

---

## Tool Status Check

Use the MCP tool to check installation status:
```
evony_rte: tools_status
```

This will show which tools are installed and their paths.

---

## Quick Install (Windows)

Install all tools with Chocolatey:
```powershell
# Install Chocolatey first if needed
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install tools
choco install ffdec ghidra wireshark swftools graphviz radare2 -y
```
