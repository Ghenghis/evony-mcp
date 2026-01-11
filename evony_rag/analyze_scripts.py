#!/usr/bin/env python3
"""Analyze script examples from training data."""
import json

TRAIN_FILE = r"C:\Users\Admin\Downloads\Evony_Decrypted\phase4-h200\dataset\training_jsonl\Evony_Training_Data_train.jsonl"

print("=" * 60)
print("SCRIPT EXAMPLES FROM TRAINING DATA")
print("=" * 60)

count = 0
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            d = json.loads(line)
            cat = d.get("category", "")
            instr = d.get("instruction", "")
            
            # Look for script-related examples
            if cat == "scripts" or "script" in instr.lower()[:80]:
                print(f"\n{'='*60}")
                print(f"CATEGORY: {cat}")
                print(f"QUESTION:\n{instr[:300]}")
                print(f"\nANSWER:\n{d.get('output', '')[:400]}...")
                count += 1
                if count >= 8:
                    break
        except:
            pass

print(f"\n\nFound {count} script examples")
