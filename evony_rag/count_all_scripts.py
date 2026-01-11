#!/usr/bin/env python3
"""
Count ALL scripts across ALL training data files.
"""
import json
import os
from pathlib import Path
from collections import Counter

# All training data locations
LOCATIONS = [
    Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\phase4-h200\dataset"),
    Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\Evony_Training_Data"),
    Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\Evony_Dataset"),
    Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\training"),
]

print("=" * 70)
print("COMPREHENSIVE SCRIPT COUNT ACROSS ALL TRAINING DATA")
print("=" * 70)

all_categories = Counter()
script_files = []
total_examples = 0
script_examples = 0

def count_in_file(filepath):
    global total_examples, script_examples, all_categories
    
    try:
        if filepath.suffix == '.jsonl':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        cat = d.get('category', 'unknown')
                        all_categories[cat] += 1
                        total_examples += 1
                        
                        # Check if it's script-related
                        instr = d.get('instruction', '').lower()
                        output = d.get('output', '').lower()
                        if (cat in ['scripts', 'scripting'] or 
                            'script' in cat.lower() or
                            'script' in instr[:100] or
                            any(cmd in output[:500] for cmd in ['setsilence', 'label ', 'goto ', 'loop ', 'endloop'])):
                            script_examples += 1
                    except:
                        pass
        elif filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict):
                            cat = d.get('category', 'unknown')
                            all_categories[cat] += 1
                            total_examples += 1
                            
                            instr = d.get('instruction', '').lower()
                            output = d.get('output', '').lower()
                            if (cat in ['scripts', 'scripting'] or 
                                'script' in cat.lower() or
                                'script' in instr[:100] or
                                any(cmd in output[:500] for cmd in ['setsilence', 'label ', 'goto ', 'loop ', 'endloop'])):
                                script_examples += 1
    except Exception as e:
        pass

# Search all locations
for loc in LOCATIONS:
    if loc.exists():
        print(f"\nSearching: {loc}")
        for root, dirs, files in os.walk(loc):
            for f in files:
                if f.endswith(('.json', '.jsonl')):
                    filepath = Path(root) / f
                    count_in_file(filepath)
                    
                # Count script files directly
                if f.endswith(('.txt', '.script')) and 'script' in root.lower():
                    script_files.append(filepath)

# Also check for script files in exploits and scripts folders
script_dirs = [
    Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\exploits"),
    Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\scripts"),
    Path(r"C:\Users\Admin\Downloads\Evony_Decrypted\keys"),
]

print("\n\nSearching script directories...")
for d in script_dirs:
    if d.exists():
        for f in d.rglob('*.txt'):
            script_files.append(f)
        for f in d.rglob('*.script'):
            script_files.append(f)
        for f in d.rglob('*.md'):
            # Check if it contains script content
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')[:500].lower()
                if any(cmd in content for cmd in ['setsilence', 'label ', 'goto ', 'loop ']):
                    script_files.append(f)
            except:
                pass

# Summary
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nTotal training examples found: {total_examples:,}")
print(f"Script-related examples: {script_examples:,}")
print(f"Script files found: {len(script_files):,}")

print("\n=== ALL CATEGORIES ===")
for cat, count in all_categories.most_common(30):
    marker = " <-- SCRIPT" if 'script' in cat.lower() else ""
    print(f"  {count:6,} | {cat}{marker}")

print(f"\n=== SAMPLE SCRIPT FILES ===")
unique_scripts = list(set(script_files))[:30]
for f in unique_scripts:
    print(f"  {f}")

print(f"\n=== TOTALS ===")
print(f"  Training examples: {total_examples:,}")
print(f"  Script examples:   {script_examples:,}")
print(f"  Script files:      {len(set(script_files)):,}")
