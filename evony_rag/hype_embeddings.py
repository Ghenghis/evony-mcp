"""
HyPE - Hypothetical Prompt Embeddings
=====================================
GAME-CHANGER: +42% context precision, +45% claim recall

Instead of embedding chunks directly, we precompute hypothetical QUESTIONS
that each chunk answers. At query time, we match question-to-question.

This is MORE ACCURATE than semantic similarity because:
- Questions have similar structure to queries
- Eliminates the query-document semantic gap
- Precomputed = no runtime LLM cost

Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .config import INDEX_PATH, DATASET_PATH


@dataclass
class HyPEChunk:
    """Chunk with precomputed hypothetical questions."""
    chunk_id: str
    file_path: str
    category: str
    content: str
    hypothetical_questions: List[str]
    question_embeddings: List[List[float]] = field(default_factory=list)


class HyPEGenerator:
    """
    Generates hypothetical questions for chunks.
    
    For Evony domain, we generate questions like:
    - "What is command ID X?"
    - "How does [function] work?"
    - "What parameters does [command] accept?"
    """
    
    # Domain-specific question templates
    TEMPLATES = {
        "source_code": [
            "What does the {symbol} function do?",
            "How is {symbol} implemented?",
            "What parameters does {symbol} accept?",
            "What class contains {symbol}?",
            "Show me the code for {symbol}",
        ],
        "protocol": [
            "What is command ID {cmd_id}?",
            "What does the {cmd_name} command do?",
            "What parameters does {cmd_name} accept?",
            "How do I send {cmd_name}?",
            "What is the response format for {cmd_name}?",
        ],
        "exploits": [
            "How does the {exploit_name} exploit work?",
            "What vulnerability does {exploit_name} target?",
            "How to execute {exploit_name}?",
            "What is the impact of {exploit_name}?",
            "How to detect {exploit_name}?",
        ],
        "keys": [
            "What is in the {filename} config file?",
            "What settings are in {filename}?",
            "What does {key_name} setting do?",
            "What is the value of {key_name}?",
            "Show config for {topic}",
        ],
        "scripts": [
            "How does the {script_name} script work?",
            "What does {script_name} do?",
            "Show me the {script_name} automation script",
            "What commands does {script_name} use?",
            "How to run {script_name}?",
        ],
        "documentation": [
            "What is {topic}?",
            "How does {topic} work?",
            "Explain {topic}",
            "What are the details about {topic}?",
        ],
    }
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize HyPE generator.
        
        Args:
            use_llm: If True, use LLM to generate questions (better but slower)
                     If False, use template-based generation (fast)
        """
        self.use_llm = use_llm
        self.lmstudio_url = "http://localhost:1234/v1"
    
    def extract_entities(self, content: str, category: str, file_path: str = "") -> Dict[str, List[str]]:
        """Extract entities from content for template filling."""
        import re
        import os
        
        entities = {
            "symbols": [],
            "cmd_ids": [],
            "cmd_names": [],
            "exploit_names": [],
            "topics": [],
            "filenames": [],
            "key_names": [],
            "script_names": [],
        }
        
        # Extract filename from path
        if file_path:
            basename = os.path.basename(file_path)
            name_only = os.path.splitext(basename)[0]
            entities["filenames"] = [name_only]
            entities["script_names"] = [name_only]
        
        # Extract function/class names
        functions = re.findall(r'(?:public|private|protected)\s+(?:static\s+)?function\s+(\w+)', content)
        classes = re.findall(r'class\s+(\w+)', content)
        variables = re.findall(r'(?:public|private)\s+(?:var|const)\s+(\w+)', content)
        entities["symbols"] = list(set(functions + classes + variables))
        
        # Extract command IDs
        cmd_ids = re.findall(r'(?:command|cmd|COMMAND)\s*(?:id|ID)?\s*[=:]\s*(\d+)', content, re.IGNORECASE)
        entities["cmd_ids"] = list(set(cmd_ids))
        
        # Extract command names from file path or content
        cmd_names = re.findall(r'(\w+Command)\b', content)
        entities["cmd_names"] = list(set(cmd_names))
        
        # Extract exploit names
        exploit_names = re.findall(r'(?:exploit|vuln|attack)[\s_:-]*(\w+)', content, re.IGNORECASE)
        entities["exploit_names"] = list(set(exploit_names))
        
        # Extract key names from JSON/config content
        key_names = re.findall(r'"(\w+)":\s*["\[\{]', content)
        key_names += re.findall(r'(\w+)\s*=\s*', content)
        entities["key_names"] = list(set(key_names))[:5]
        
        # Extract topics (capitalized phrases or words)
        topics = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
        # Also get words from filename
        if file_path:
            file_words = re.findall(r'[A-Za-z]+', os.path.basename(file_path))
            topics.extend(file_words)
        entities["topics"] = list(set(topics))[:5]
        
        return entities
    
    def generate_questions_template(self, chunk: Dict) -> List[str]:
        """Generate questions using templates (fast, no LLM)."""
        content = chunk.get("content", "")
        category = chunk.get("category", "documentation")
        file_path = chunk.get("file_path", "")
        
        questions = []
        entities = self.extract_entities(content, category, file_path)
        
        # Get templates for category
        templates = self.TEMPLATES.get(category, self.TEMPLATES["documentation"])
        
        # Fill templates with extracted entities
        for template in templates:
            if "{symbol}" in template and entities["symbols"]:
                for sym in entities["symbols"][:3]:
                    questions.append(template.format(symbol=sym))
            elif "{cmd_id}" in template and entities["cmd_ids"]:
                for cmd_id in entities["cmd_ids"][:3]:
                    questions.append(template.format(cmd_id=cmd_id))
            elif "{cmd_name}" in template and entities["cmd_names"]:
                for cmd in entities["cmd_names"][:3]:
                    questions.append(template.format(cmd_name=cmd))
            elif "{exploit_name}" in template and entities["exploit_names"]:
                for exp in entities["exploit_names"][:3]:
                    questions.append(template.format(exploit_name=exp))
            elif "{filename}" in template and entities["filenames"]:
                for fn in entities["filenames"][:2]:
                    questions.append(template.format(filename=fn))
            elif "{key_name}" in template and entities["key_names"]:
                for key in entities["key_names"][:3]:
                    questions.append(template.format(key_name=key))
            elif "{script_name}" in template and entities["script_names"]:
                for sn in entities["script_names"][:2]:
                    questions.append(template.format(script_name=sn))
            elif "{topic}" in template and entities["topics"]:
                for topic in entities["topics"][:3]:
                    questions.append(template.format(topic=topic))
        
        # Add generic questions based on content
        if "AMF" in content or "amf" in content.lower():
            questions.append("How does AMF3 encoding work?")
            questions.append("What is the AMF3 protocol format?")
        
        if "packet" in content.lower():
            questions.append("How are packets structured?")
            questions.append("How to decode network packets?")
        
        # If still no questions, generate fallback based on file path
        if not questions and file_path:
            import os
            basename = os.path.basename(file_path)
            questions.append(f"What is in {basename}?")
            questions.append(f"Show contents of {basename}")
            questions.append(f"What does {basename} contain?")
        
        # Deduplicate and limit
        return list(set(questions))[:10]
    
    def generate_questions_llm(self, chunk: Dict) -> List[str]:
        """Generate questions using LLM (better quality, slower)."""
        try:
            import requests
            
            content = chunk.get("content", "")[:1500]  # Limit context
            
            prompt = f"""Given this code/documentation chunk, generate 5 questions that this chunk would answer.
Focus on specific, searchable questions someone might ask.

Chunk:
```
{content}
```

Generate exactly 5 questions, one per line. Just the questions, no numbering or explanation."""

            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
                timeout=30,
            )
            
            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"]
                questions = [q.strip() for q in text.strip().split("\n") if q.strip()]
                return questions[:5]
            
        except Exception:
            pass
        
        # Fallback to template
        return self.generate_questions_template(chunk)
    
    def generate_questions(self, chunk: Dict) -> List[str]:
        """Generate hypothetical questions for a chunk."""
        if self.use_llm:
            return self.generate_questions_llm(chunk)
        return self.generate_questions_template(chunk)


class HyPEIndex:
    """
    HyPE Index - stores chunks with their hypothetical questions.
    
    At query time, we embed the query and find similar QUESTIONS
    (not documents), then return the chunks those questions belong to.
    """
    
    def __init__(self):
        self.generator = HyPEGenerator(use_llm=False)
        self.chunks: List[HyPEChunk] = []
        self.question_to_chunk: Dict[str, str] = {}  # question -> chunk_id
        self.question_embeddings: Dict[str, List[float]] = {}
        self.embedding_model = None
        self._loaded = False
    
    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is not None:
            return
        
        try:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
    
    def build_from_chunks(self, chunks: List[Dict], progress_callback=None):
        """
        Build HyPE index from existing chunks.
        
        Args:
            chunks: List of chunk dictionaries
            progress_callback: Optional callback(current, total)
        """
        self._load_embedding_model()
        
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            # Generate hypothetical questions
            questions = self.generator.generate_questions(chunk)
            
            if not questions:
                continue
            
            chunk_id = chunk.get("id", chunk.get("chunk_id", f"chunk_{i}"))
            
            # Store mapping
            for q in questions:
                self.question_to_chunk[q] = chunk_id
            
            # Create HyPE chunk
            hype_chunk = HyPEChunk(
                chunk_id=chunk_id,
                file_path=chunk.get("file_path", ""),
                category=chunk.get("category", ""),
                content=chunk.get("content", ""),
                hypothetical_questions=questions,
            )
            
            # Embed questions
            if self.embedding_model:
                embeddings = self.embedding_model.encode(questions)
                hype_chunk.question_embeddings = embeddings.tolist()
                for q, emb in zip(questions, embeddings):
                    self.question_embeddings[q] = emb.tolist()
            
            self.chunks.append(hype_chunk)
            
            if progress_callback and i % 100 == 0:
                progress_callback(i, total)
        
        self._loaded = True
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Search using question-to-question matching.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            List of (chunk_dict, score) tuples
        """
        if not self._loaded or not self.embedding_model:
            return []
        
        import numpy as np
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Find most similar questions
        scores = []
        for question, embedding in self.question_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            scores.append((question, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: -x[1])
        
        # Get unique chunks
        seen_chunks = set()
        results = []
        
        for question, score in scores:
            chunk_id = self.question_to_chunk.get(question)
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                
                # Find the chunk
                for chunk in self.chunks:
                    if chunk.chunk_id == chunk_id:
                        results.append(({
                            "chunk_id": chunk.chunk_id,
                            "file_path": chunk.file_path,
                            "category": chunk.category,
                            "content": chunk.content,
                            "matched_question": question,
                        }, float(score)))
                        break
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: Path):
        """Save HyPE index."""
        data = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "file_path": c.file_path,
                    "category": c.category,
                    "content": c.content[:500],  # Truncate for storage
                    "hypothetical_questions": c.hypothetical_questions,
                }
                for c in self.chunks
            ],
            "question_to_chunk": self.question_to_chunk,
        }
        
        with open(path / "hype_index.json", "w") as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> bool:
        """Load HyPE index."""
        try:
            with open(path / "hype_index.json") as f:
                data = json.load(f)
            
            self.question_to_chunk = data["question_to_chunk"]
            self._loaded = True
            return True
        except:
            return False


# Singleton
_hype_index = None


def get_hype_index() -> HyPEIndex:
    """Get singleton HyPE index."""
    global _hype_index
    if _hype_index is None:
        _hype_index = HyPEIndex()
    return _hype_index
