#!/usr/bin/env python3
"""
Test with properly formatted questions matching training data.
"""
import sys
sys.path.insert(0, ".")

from evony_rag.precision_rag import get_precision_rag
from evony_rag.question_formatter import get_question_formatter

print("=" * 60)
print("TEST WITH PROPER QUESTION FORMATS")
print("=" * 60)

formatter = get_question_formatter()
rag = get_precision_rag()

# Questions that MATCH training format
proper_questions = [
    # Class/function questions (most common in training)
    "What does the StratagemCommands class/function do in Evony?",
    "What parameters does ScriptManager accept?",
    "How does send_script work in the Evony client?",
    
    # Script command questions
    "How do I use the setsilence script command in Evony?",
    "How do I use the farmNPC script command in Evony?",
    
    # File purpose questions
    "What is the purpose of ScriptError.as in Evony?",
]

print("\n=== PROPER FORMAT QUESTIONS ===\n")
for q in proper_questions:
    print(f"Q: {q}")
    result = rag.query(q, use_cache=False, auto_format=False)  # Don't reformat
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Answer: {result.answer[:200]}...")
    print()

# Now test with vague questions that get reformatted
print("\n=== VAGUE QUESTIONS (AUTO-REFORMATTED) ===\n")

vague_questions = [
    "How do I farm NPCs?",
    "What scripts can I use for attacking?",
    "How does the script window work?",
]

for q in vague_questions:
    # First show how it gets formatted
    fmt = formatter.format_question(q)
    print(f"ORIGINAL: {q}")
    print(f"FORMATTED: {fmt.formatted}")
    print(f"ENTITIES: {fmt.entities_found[:3]}")
    
    # Then query with auto-format
    result = rag.query(q, use_cache=False, auto_format=True)
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Answer: {result.answer[:150]}...")
    print()

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
