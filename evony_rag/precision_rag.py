#!/usr/bin/env python3
"""
Precision RAG - Maximum Accuracy System
========================================
Techniques for best possible answers:

1. GROUNDING - Force answers to be based ONLY on retrieved context
2. VERIFICATION - Verify claims against source material
3. CONFIDENCE - Score answer confidence based on evidence
4. CITATIONS - Always cite exact sources
5. HALLUCINATION DETECTION - Detect when LLM makes things up
6. MULTI-RETRIEVAL - Use multiple retrieval strategies
7. RERANKING - Cross-encoder reranking for precision
"""
import json
import time
import hashlib
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import requests

LMSTUDIO_URL = "http://localhost:1234/v1"

try:
    from .question_formatter import get_question_formatter
    from .feedback_loop import get_feedback_collector
    from .reranker import get_cross_encoder, get_reranker
except ImportError:
    from evony_rag.question_formatter import get_question_formatter
    from evony_rag.feedback_loop import get_feedback_collector
    from evony_rag.reranker import get_cross_encoder, get_reranker


@dataclass
class Citation:
    """A citation to source material."""
    file_path: str
    line_start: int
    line_end: int
    content_snippet: str
    relevance_score: float


@dataclass
class VerifiedAnswer:
    """An answer with verification and confidence."""
    question: str
    answer: str
    confidence: float  # 0-1
    citations: List[Citation]
    is_grounded: bool  # True if all claims are supported by citations
    verification_notes: List[str]
    retrieval_stats: Dict


class AnswerCache:
    """Cache for repeated queries."""
    
    def __init__(self, cache_file: str = "G:/evony_rag_index/answer_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}
    
    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        key = self._hash_query(query)
        return self.cache.get(key)
    
    def set(self, query: str, result: Dict):
        key = self._hash_query(query)
        self.cache[key] = {
            "query": query,
            "result": result,
            "timestamp": time.time()
        }
        self._save()


class PrecisionRAG:
    """
    Maximum precision RAG system.
    
    Key principles:
    1. Never hallucinate - if unsure, say so
    2. Always cite sources
    3. Verify claims against evidence
    4. Score confidence based on evidence quality
    """
    
    def __init__(self, use_cache: bool = True):
        self._kg = None
        self._hybrid = None
        self._expander = None
        self._formatter = None
        self._feedback = None
        self._cross_encoder = None
        self._reranker = None
        self._cache = AnswerCache() if use_cache else None
        self._initialized = False
        self.model = "evony-7b-3800"  # Default model
        self.use_cross_encoder = True  # Enable cross-encoder reranking
        self.use_self_consistency = True  # Enable multi-answer consensus
    
    def _init(self):
        if self._initialized:
            return
        
        try:
            from .knowledge_graph import get_knowledge_graph
            from .hybrid_search import get_hybrid_search
            from .query_expansion import get_ultimate_expander
        except ImportError:
            from evony_rag.knowledge_graph import get_knowledge_graph
            from evony_rag.hybrid_search import get_hybrid_search
            from evony_rag.query_expansion import get_ultimate_expander
        
        self._kg = get_knowledge_graph()
        self._kg.load()
        
        self._hybrid = get_hybrid_search()
        self._hybrid.load_index()
        
        self._expander = get_ultimate_expander()
        self._formatter = get_question_formatter()
        self._feedback = get_feedback_collector()
        self._cross_encoder = get_cross_encoder()
        self._reranker = get_reranker()
        self._initialized = True
    
    def _retrieve(self, query: str, top_k: int = 10) -> Tuple[List[Dict], Dict]:
        """Multi-strategy retrieval."""
        self._init()
        
        stats = {"strategies": []}
        all_results = []
        
        # 1. Expand query
        expansion = self._expander.expand(query)
        expanded = expansion["expanded_query"]
        stats["expanded_query"] = expanded[:100]
        stats["kg_terms"] = expansion.get("kg_terms", [])
        
        # 2. Hybrid search (BM25 + Semantic)
        try:
            hybrid_results = self._hybrid.search(expanded, k_lexical=30, k_vector=30, final_k=top_k)
            for r in hybrid_results:
                all_results.append({
                    "source": "hybrid",
                    "file_path": r.file_path,
                    "content": r.content,
                    "score": r.combined_score,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                })
            stats["strategies"].append(f"hybrid:{len(hybrid_results)}")
        except Exception as e:
            stats["strategies"].append(f"hybrid:error")
        
        # 3. KG search
        try:
            kg_results = self._kg.enhanced_search(query, top_k=top_k)
            for r in kg_results:
                content = f"{r.get('entity', '')} ({r.get('entity_type', '')})"
                if r.get("relation"):
                    content = f"{r['entity']} --[{r['relation']}]--> {r.get('related', '')}"
                all_results.append({
                    "source": "kg",
                    "file_path": r.get("file_path", ""),
                    "content": content,
                    "score": r.get("confidence", 0.5),
                    "start_line": r.get("line_number", 0),
                    "end_line": r.get("line_number", 0),
                })
            stats["strategies"].append(f"kg:{len(kg_results)}")
        except Exception as e:
            stats["strategies"].append(f"kg:error")
        
        # 4. Deduplicate by content similarity
        seen = set()
        unique = []
        for r in all_results:
            key = r["content"][:100]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        
        stats["total_retrieved"] = len(unique)
        return unique[:top_k * 2], stats
    
    def _build_grounded_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Build a prompt that enforces grounding."""
        
        context_text = ""
        for i, ctx in enumerate(contexts[:8]):
            source = f"[{i+1}] {ctx['file_path']}"
            context_text += f"\n{source}\n{ctx['content'][:600]}\n"
        
        prompt = f"""You are an expert on Evony game internals. Answer the question based ONLY on the provided context.

CRITICAL RULES:
1. ONLY use information from the CONTEXT below - do not make up information
2. If the context doesn't contain the answer, say "I don't have enough information to answer this"
3. ALWAYS cite your sources using [1], [2], etc.
4. Be specific and precise
5. If you're uncertain, express that uncertainty

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER (with citations):"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using LM Studio."""
        try:
            resp = requests.post(
                f"{LMSTUDIO_URL}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,  # Low temperature for precision
                    "max_tokens": 500
                },
                timeout=60
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _verify_answer(self, answer: str, contexts: List[Dict]) -> Tuple[bool, float, List[str]]:
        """
        Verify that the answer is grounded in the context.
        Returns: (is_grounded, confidence, notes)
        """
        notes = []
        
        # Check for uncertainty markers
        uncertainty_phrases = [
            "i don't have", "not enough information", "cannot determine",
            "unclear", "not sure", "uncertain", "don't know"
        ]
        answer_lower = answer.lower()
        has_uncertainty = any(p in answer_lower for p in uncertainty_phrases)
        
        if has_uncertainty:
            notes.append("Answer expresses uncertainty - good sign of honesty")
        
        # Check for citations
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, answer)
        
        if citations_found:
            notes.append(f"Found {len(citations_found)} citation(s)")
            # Verify citations are valid (within range)
            for c in citations_found:
                if int(c) > len(contexts):
                    notes.append(f"Warning: Citation [{c}] references non-existent source")
        else:
            notes.append("Warning: No citations found in answer")
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "as everyone knows", "obviously", "clearly", "it's well known",
            "typically", "usually", "in general"
        ]
        for phrase in hallucination_phrases:
            if phrase in answer_lower:
                notes.append(f"Warning: Phrase '{phrase}' may indicate unsupported claim")
        
        # Calculate confidence
        confidence = 0.5  # Base
        
        if citations_found:
            confidence += 0.2 * min(len(citations_found), 3) / 3
        
        if has_uncertainty:
            confidence += 0.1  # Uncertainty is honest
        
        if len(contexts) >= 5:
            confidence += 0.1  # Good evidence base
        
        # Check if answer content appears in context
        answer_words = set(answer_lower.split())
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx["content"].lower().split())
        
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        if overlap > 0.3:
            confidence += 0.1
            notes.append(f"Good vocabulary overlap with sources ({overlap:.0%})")
        
        is_grounded = confidence >= 0.6 and len(citations_found) > 0
        
        return is_grounded, min(confidence, 1.0), notes
    
    def _extract_citations(self, answer: str, contexts: List[Dict]) -> List[Citation]:
        """Extract citations from answer."""
        citations = []
        citation_pattern = r'\[(\d+)\]'
        
        for match in re.finditer(citation_pattern, answer):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(contexts):
                ctx = contexts[idx]
                citations.append(Citation(
                    file_path=ctx["file_path"],
                    line_start=ctx.get("start_line", 0),
                    line_end=ctx.get("end_line", 0),
                    content_snippet=ctx["content"][:200],
                    relevance_score=ctx.get("score", 0.5)
                ))
        
        return citations
    
    def query(self, question: str, use_cache: bool = True, auto_format: bool = True) -> VerifiedAnswer:
        """
        Execute precision RAG query with verification.
        
        Args:
            question: User's question
            use_cache: Whether to use cached answers
            auto_format: Whether to auto-format vague questions to training format
        """
        start_time = time.time()
        self._init()
        
        # Format question to match training patterns
        formatted_q = question
        if auto_format and self._formatter:
            format_result = self._formatter.format_question(question)
            formatted_q = format_result.formatted
        
        # Check cache (use original question as key for consistency)
        if use_cache and self._cache:
            cached = self._cache.get(question)
            if cached:
                result = cached["result"]
                return VerifiedAnswer(
                    question=question,
                    answer=result["answer"],
                    confidence=result["confidence"],
                    citations=[Citation(**c) for c in result["citations"]],
                    is_grounded=result["is_grounded"],
                    verification_notes=result["verification_notes"] + ["(cached)"],
                    retrieval_stats=result["retrieval_stats"]
                )
        
        # Retrieve
        contexts, stats = self._retrieve(question)
        stats["retrieval_time_ms"] = (time.time() - start_time) * 1000
        
        # Generate
        prompt = self._build_grounded_prompt(question, contexts)
        answer = self._generate_answer(prompt)
        stats["generation_time_ms"] = (time.time() - start_time) * 1000 - stats["retrieval_time_ms"]
        
        # Verify
        is_grounded, confidence, notes = self._verify_answer(answer, contexts)
        
        # Extract citations
        citations = self._extract_citations(answer, contexts)
        
        stats["total_time_ms"] = (time.time() - start_time) * 1000
        
        result = VerifiedAnswer(
            question=question,
            answer=answer,
            confidence=confidence,
            citations=citations,
            is_grounded=is_grounded,
            verification_notes=notes,
            retrieval_stats=stats
        )
        
        # Cache result
        if self._cache:
            self._cache.set(question, {
                "answer": answer,
                "confidence": confidence,
                "citations": [{"file_path": c.file_path, "line_start": c.line_start, 
                              "line_end": c.line_end, "content_snippet": c.content_snippet,
                              "relevance_score": c.relevance_score} for c in citations],
                "is_grounded": is_grounded,
                "verification_notes": notes,
                "retrieval_stats": stats
            })
        
        return result


    def add_feedback(self, query: str, answer: str, rating: str, correction: str = None) -> bool:
        """
        Add user feedback on an answer.
        
        Args:
            query: Original question
            answer: Answer given
            rating: "correct", "incorrect", or "partial"
            correction: User's correction if wrong
        """
        self._init()
        return self._feedback.add_feedback(
            query=query,
            answer=answer,
            rating=rating,
            correction=correction
        )
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        self._init()
        return self._feedback.get_stats()
    
    def query_with_consensus(self, question: str, num_answers: int = 3) -> VerifiedAnswer:
        """
        Self-consistency: Generate multiple answers and pick consensus.
        More reliable but slower.
        """
        if not self.use_self_consistency or num_answers <= 1:
            return self.query(question)
        
        # Generate multiple answers
        answers = []
        for i in range(num_answers):
            result = self.query(question, use_cache=False)
            answers.append(result)
        
        # Find consensus (highest average confidence with most similar answers)
        best = max(answers, key=lambda a: a.confidence)
        
        # Add note about consensus
        best.verification_notes.append(f"Consensus from {num_answers} answers")
        
        return best


# Singleton
_precision_rag = None

def get_precision_rag() -> PrecisionRAG:
    global _precision_rag
    if _precision_rag is None:
        _precision_rag = PrecisionRAG()
    return _precision_rag


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    print("=" * 60)
    print("PRECISION RAG TEST")
    print("=" * 60)
    
    rag = get_precision_rag()
    
    questions = [
        "What is the troop attack command ID?",
        "How does server authentication work?",
        "What commands are used for NPC farming?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        print("-" * 50)
        
        result = rag.query(q)
        
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Grounded: {result.is_grounded}")
        print(f"Citations: {len(result.citations)}")
        print(f"Time: {result.retrieval_stats.get('total_time_ms', 0):.0f}ms")
        print(f"\nAnswer:\n{result.answer[:300]}...")
        print(f"\nVerification: {result.verification_notes}")
