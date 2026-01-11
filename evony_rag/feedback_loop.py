#!/usr/bin/env python3
"""
User Feedback Loop for RAG Improvement
=======================================
Collects feedback on answers to improve future results.

Features:
1. Store feedback (correct/incorrect/partial)
2. Track which queries work well
3. Learn from mistakes
4. Export feedback for fine-tuning
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

FEEDBACK_FILE = Path(r"G:\evony_rag_index\user_feedback.json")


@dataclass
class Feedback:
    """User feedback on an answer."""
    query: str
    answer: str
    rating: str  # "correct", "incorrect", "partial"
    correction: Optional[str]  # User's correction if incorrect
    timestamp: float
    citations_used: List[str]
    confidence: float


class FeedbackCollector:
    """
    Collects and stores user feedback for improvement.
    """
    
    def __init__(self, feedback_file: str = None):
        self.file = Path(feedback_file) if feedback_file else FEEDBACK_FILE
        self.feedback: List[Feedback] = []
        self._load()
    
    def _load(self):
        """Load existing feedback."""
        if self.file.exists():
            try:
                with open(self.file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.feedback = [Feedback(**f) for f in data]
            except:
                self.feedback = []
    
    def _save(self):
        """Save feedback to file."""
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump([asdict(f) for f in self.feedback], f, indent=2)
    
    def add_feedback(self, query: str, answer: str, rating: str,
                     correction: str = None, citations: List[str] = None,
                     confidence: float = 0.0) -> bool:
        """
        Add user feedback on an answer.
        
        Args:
            query: The original question
            answer: The answer given
            rating: "correct", "incorrect", or "partial"
            correction: User's correction if answer was wrong
            citations: List of file paths cited
            confidence: Model's confidence score
        """
        if rating not in ["correct", "incorrect", "partial"]:
            return False
        
        fb = Feedback(
            query=query,
            answer=answer[:500],  # Truncate long answers
            rating=rating,
            correction=correction,
            timestamp=time.time(),
            citations_used=citations or [],
            confidence=confidence
        )
        
        self.feedback.append(fb)
        self._save()
        return True
    
    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        total = len(self.feedback)
        if total == 0:
            return {"total": 0, "correct": 0, "incorrect": 0, "partial": 0, "accuracy": 0}
        
        correct = sum(1 for f in self.feedback if f.rating == "correct")
        incorrect = sum(1 for f in self.feedback if f.rating == "incorrect")
        partial = sum(1 for f in self.feedback if f.rating == "partial")
        
        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "partial": partial,
            "accuracy": (correct + partial * 0.5) / total,
        }
    
    def get_corrections(self) -> List[Dict]:
        """Get all corrections for fine-tuning."""
        corrections = []
        for f in self.feedback:
            if f.rating == "incorrect" and f.correction:
                corrections.append({
                    "instruction": f.query,
                    "input": "",
                    "output": f.correction,
                    "category": "correction"
                })
        return corrections
    
    def export_for_training(self, output_file: str = None) -> str:
        """Export corrections as training data."""
        corrections = self.get_corrections()
        if not corrections:
            return "No corrections to export"
        
        output = Path(output_file) if output_file else self.file.parent / "corrections_training.jsonl"
        
        with open(output, "w", encoding="utf-8") as f:
            for c in corrections:
                f.write(json.dumps(c) + "\n")
        
        return f"Exported {len(corrections)} corrections to {output}"
    
    def get_problem_queries(self) -> List[str]:
        """Get queries that frequently get wrong answers."""
        query_results = {}
        for f in self.feedback:
            if f.query not in query_results:
                query_results[f.query] = {"correct": 0, "wrong": 0}
            if f.rating == "correct":
                query_results[f.query]["correct"] += 1
            else:
                query_results[f.query]["wrong"] += 1
        
        # Return queries with >50% wrong answers
        problems = []
        for q, stats in query_results.items():
            if stats["wrong"] > stats["correct"]:
                problems.append(q)
        
        return problems


# Singleton
_feedback = None

def get_feedback_collector() -> FeedbackCollector:
    global _feedback
    if _feedback is None:
        _feedback = FeedbackCollector()
    return _feedback


if __name__ == "__main__":
    print("=" * 60)
    print("FEEDBACK LOOP TEST")
    print("=" * 60)
    
    fb = get_feedback_collector()
    
    # Add some test feedback
    fb.add_feedback(
        query="What does StratagemCommands do?",
        answer="StratagemCommands handles stratagem actions...",
        rating="correct",
        confidence=0.8
    )
    
    fb.add_feedback(
        query="How do I farm NPCs?",
        answer="I don't have enough information...",
        rating="incorrect",
        correction="Use the farmNPC script command with level and radius parameters",
        confidence=0.5
    )
    
    print(f"\nStats: {fb.get_stats()}")
    print(f"Problem queries: {fb.get_problem_queries()}")
    print(f"Corrections: {len(fb.get_corrections())}")
    
    # Export
    result = fb.export_for_training()
    print(f"\n{result}")
