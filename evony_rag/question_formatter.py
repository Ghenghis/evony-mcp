#!/usr/bin/env python3
"""
Question Formatter for Evony LLM
=================================
Transforms vague user questions into the format the model was trained on.

MODEL PURPOSE (CRITICAL):
- Reverse engineering AS3/Flash code (2009-2018 era)
- Script creation for bots (Autoevony, Roboevony)
- Exploits, CVEs, packet editing
- Protocol analysis
- NOT trained on gameplay - won't answer "how to play" well

TRAINING FORMAT PATTERNS:
1. "What does the [class/function] do?" - Code analysis
2. "Explain this code:\n```\n[code]\n```" - Reverse engineering
3. "Write a script to [action]" - Script creation
4. "How do I exploit [vulnerability]?" - Exploit questions
5. "What CVE affects [component]?" - CVE lookup
6. "How does the [protocol] packet work?" - Protocol analysis
7. "What parameters does [function] accept?" - API reference

The model expects SPECIFIC entity names, not vague topics.
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FormattedQuestion:
    """A question formatted for the Evony model."""
    original: str
    formatted: str
    pattern_used: str
    entities_found: List[str]
    needs_context: bool  # True if we need to retrieve context first


class EvonyQuestionFormatter:
    """
    Transforms user questions into training-compatible format.
    """
    
    # Patterns the model was trained on (reverse engineering focus)
    PATTERNS = {
        "file_purpose": "What is the purpose of {entity} in Evony?",
        "class_function": "What does the {entity} class/function do in Evony?",
        "how_works": "How does {entity} work in the Evony client?",
        "parameters": "What parameters does {entity} accept?",
        "example_usage": "Show me an example of using {entity} in Evony",
        "explain_code": "Explain this Evony code:\n```\n{code}\n```",
        "command_usage": "How do I use the {entity} command in Evony?",
        "script_explain": "What does this Evony script do?\n```\n{script}\n```",
        "script_command": "How do I use the {entity} script command in Evony?",
        "exploit": "How do I exploit {entity} in Evony?",
        "cve": "What CVE affects {entity}?",
        "protocol": "How does the {entity} protocol/packet work?",
        "write_script": "Write a script to {action}",
        "overflow": "How does the {entity} overflow exploit work?",
    }
    
    # Bot script commands (for script window)
    SCRIPT_COMMANDS = {
        'setsilence', 'echo', 'set', 'label', 'goto', 'if', 'endif', 'loop', 
        'endloop', 'wait', 'attack', 'farm', 'scout', 'sendmail', 'getmail',
        'deleteMail', 'reinforce', 'recall', 'build', 'research', 'train',
        'findNPC', 'farmNPC', 'scanMap', 'getArmyStatus', 'useItem',
    }
    
    def __init__(self):
        self._kg = None
        self._loaded = False
    
    def _lazy_load_kg(self):
        """Load KG for entity lookup."""
        if self._loaded:
            return
        try:
            from evony_rag.knowledge_graph import get_knowledge_graph
            self._kg = get_knowledge_graph()
            self._kg.load()
            self._loaded = True
        except:
            self._loaded = True
    
    def extract_entities(self, question: str) -> List[str]:
        """
        Extract potential entity names from a question.
        Looks for:
        - CamelCase names (class/function names)
        - snake_case names
        - Filenames (.as, .py, .txt, .md)
        - Quoted terms
        """
        entities = []
        
        # CamelCase (e.g., StratagemCommands, StoreListResponse)
        camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', question)
        entities.extend(camel)
        
        # snake_case (e.g., use_stratagem, get_army)
        snake = re.findall(r'\b([a-z]+_[a-z_]+)\b', question)
        entities.extend(snake)
        
        # Filenames
        files = re.findall(r'\b([\w]+\.(?:as|py|txt|md|json))\b', question, re.IGNORECASE)
        entities.extend(files)
        
        # Quoted terms
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        entities.extend(quoted)
        
        # Backticked terms
        backticked = re.findall(r'`([^`]+)`', question)
        entities.extend(backticked)
        
        # Command patterns (e.g., army.newArmy, stratagem.useStratagem)
        commands = re.findall(r'\b([a-z]+\.[a-zA-Z]+)\b', question)
        entities.extend(commands)
        
        return list(set(entities))
    
    def find_related_entities(self, topic: str, limit: int = 3) -> List[str]:
        """Find entity names related to a topic using KG."""
        self._lazy_load_kg()
        if not self._kg or len(self._kg.entities) == 0:
            return []
        
        related = []
        topic_lower = topic.lower()
        
        for name, entity_ids in self._kg.entity_by_name.items():
            if topic_lower in name or name in topic_lower:
                related.append(name)
                if len(related) >= limit:
                    break
        
        return related
    
    def format_question(self, question: str, context_entities: List[str] = None) -> FormattedQuestion:
        """
        Format a question to match training patterns.
        
        If the question already has specific entities, use them.
        If not, try to find related entities from KG.
        """
        original = question
        entities = self.extract_entities(question)
        
        # Add context entities if provided
        if context_entities:
            entities.extend(context_entities)
        
        entities = list(set(entities))
        
        # Determine the best pattern based on question type
        q_lower = question.lower()
        
        # If question contains code snippet
        if "```" in question or "explain this code" in q_lower:
            code_match = re.search(r'```(?:\w+)?\n?(.*?)```', question, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                return FormattedQuestion(
                    original=original,
                    formatted=self.PATTERNS["explain_code"].format(code=code),
                    pattern_used="explain_code",
                    entities_found=entities,
                    needs_context=False
                )
        
        # If asking about a script
        if "script" in q_lower and "```" in question:
            script_match = re.search(r'```(?:\w+)?\n?(.*?)```', question, re.DOTALL)
            if script_match:
                script = script_match.group(1).strip()
                return FormattedQuestion(
                    original=original,
                    formatted=self.PATTERNS["script_explain"].format(script=script),
                    pattern_used="script_explain",
                    entities_found=entities,
                    needs_context=False
                )
        
        # If we have specific entities, format with them
        if entities:
            entity = entities[0]  # Use first entity
            
            # Check if it's a script command
            if entity.lower() in self.SCRIPT_COMMANDS:
                pattern = "script_command"
            elif "parameter" in q_lower or "accept" in q_lower or "argument" in q_lower:
                pattern = "parameters"
            elif "purpose" in q_lower or "what is" in q_lower:
                if entity.endswith(('.as', '.py', '.txt', '.md')):
                    pattern = "file_purpose"
                else:
                    pattern = "class_function"
            elif "how" in q_lower and ("work" in q_lower or "use" in q_lower):
                if "." in entity:  # Command like army.newArmy
                    pattern = "command_usage"
                else:
                    pattern = "how_works"
            elif "example" in q_lower or "show" in q_lower:
                pattern = "example_usage"
            else:
                pattern = "class_function"
            
            formatted = self.PATTERNS[pattern].format(entity=entity)
            
            return FormattedQuestion(
                original=original,
                formatted=formatted,
                pattern_used=pattern,
                entities_found=entities,
                needs_context=False
            )
        
        # No specific entities - try to find related ones from KG
        # Extract key terms
        key_terms = [w for w in q_lower.split() if len(w) >= 4 and w not in 
                     {'what', 'does', 'how', 'show', 'tell', 'explain', 'evony', 'game', 'about'}]
        
        for term in key_terms[:3]:
            related = self.find_related_entities(term, limit=2)
            if related:
                entities.extend(related)
        
        if entities:
            entity = entities[0]
            formatted = self.PATTERNS["class_function"].format(entity=entity)
            return FormattedQuestion(
                original=original,
                formatted=formatted,
                pattern_used="class_function (inferred)",
                entities_found=entities,
                needs_context=True  # We inferred entities, may need context
            )
        
        # Can't find specific entities - return original but flag it
        return FormattedQuestion(
            original=original,
            formatted=question,
            pattern_used="none (too vague)",
            entities_found=[],
            needs_context=True
        )
    
    def generate_specific_questions(self, topic: str, limit: int = 5) -> List[str]:
        """
        Generate specific questions about a topic based on training format.
        Useful for exploring a topic the model knows about.
        """
        self._lazy_load_kg()
        questions = []
        
        # Find entities related to topic
        entities = self.find_related_entities(topic, limit=limit * 2)
        
        if not entities:
            # Try harder - search in KG
            if self._kg:
                for name, ids in list(self._kg.entity_by_name.items())[:1000]:
                    if topic.lower() in name.lower():
                        entities.append(name)
                    if len(entities) >= limit * 2:
                        break
        
        # Generate questions for each entity
        for entity in entities[:limit]:
            if entity.endswith(('.as', '.py', '.txt', '.md')):
                q = self.PATTERNS["file_purpose"].format(entity=entity)
            elif '.' in entity:
                q = self.PATTERNS["command_usage"].format(entity=entity)
            else:
                q = self.PATTERNS["class_function"].format(entity=entity)
            questions.append(q)
        
        return questions


# Singleton
_formatter = None

def get_question_formatter() -> EvonyQuestionFormatter:
    global _formatter
    if _formatter is None:
        _formatter = EvonyQuestionFormatter()
    return _formatter


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    formatter = get_question_formatter()
    
    print("=" * 60)
    print("EVONY QUESTION FORMATTER TEST")
    print("=" * 60)
    
    # Test vague questions
    vague_questions = [
        "How do I attack troops?",
        "What is the troop attack command?",
        "How does server authentication work?",
        "What commands are used for NPC farming?",
        "Tell me about exploits",
    ]
    
    print("\n=== VAGUE QUESTIONS (REFORMATTED) ===\n")
    for q in vague_questions:
        result = formatter.format_question(q)
        print(f"ORIGINAL:  {q}")
        print(f"FORMATTED: {result.formatted[:80]}...")
        print(f"PATTERN:   {result.pattern_used}")
        print(f"ENTITIES:  {result.entities_found}")
        print()
    
    # Test specific questions
    specific_questions = [
        "What does the StratagemCommands class do in Evony?",
        "What parameters does copyTo accept?",
        "How do I use the stratagem.useStratagem command in Evony?",
    ]
    
    print("\n=== SPECIFIC QUESTIONS (ALREADY CORRECT) ===\n")
    for q in specific_questions:
        result = formatter.format_question(q)
        print(f"ORIGINAL:  {q}")
        print(f"FORMATTED: {result.formatted[:80]}...")
        print(f"PATTERN:   {result.pattern_used}")
        print()
    
    # Generate topic-specific questions
    print("\n=== GENERATED QUESTIONS FOR TOPICS ===\n")
    for topic in ["army", "attack", "stratagem"]:
        print(f"Topic: {topic}")
        questions = formatter.generate_specific_questions(topic, limit=3)
        for q in questions:
            print(f"  - {q}")
        print()
