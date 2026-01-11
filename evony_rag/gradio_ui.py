#!/usr/bin/env python3
"""
Evony RAG Ultimate - Gradio UI
===============================
Interactive web interface for the precision RAG system.
"""
import sys
sys.path.insert(0, ".")

import gradio as gr
from evony_rag.precision_rag import get_precision_rag

# Initialize RAG
rag = None

def init_rag():
    global rag
    if rag is None:
        rag = get_precision_rag()
    return rag

def query_rag(question: str, use_cache: bool = True) -> tuple:
    """Execute RAG query and return formatted results."""
    global last_query
    if not question.strip():
        return "Please enter a question.", "", "", ""
    
    r = init_rag()
    result = r.query(question, use_cache=use_cache)
    
    # Store for feedback
    last_query["question"] = question
    last_query["answer"] = result.answer
    
    # Format answer with confidence indicator
    confidence_emoji = "🟢" if result.confidence >= 0.7 else "🟡" if result.confidence >= 0.5 else "🔴"
    grounded_emoji = "✅" if result.is_grounded else "⚠️"
    
    answer_text = f"{result.answer}"
    
    # Format stats
    stats_text = f"""**Confidence:** {confidence_emoji} {result.confidence:.0%}
**Grounded:** {grounded_emoji} {"Yes" if result.is_grounded else "No"}
**Citations:** {len(result.citations)}
**Time:** {result.retrieval_stats.get('total_time_ms', 0):.0f}ms
**Strategies:** {', '.join(result.retrieval_stats.get('strategies', []))}"""
    
    # Format citations
    citations_text = ""
    for i, c in enumerate(result.citations, 1):
        citations_text += f"**[{i}] {c.file_path}** (lines {c.line_start}-{c.line_end})\n"
        citations_text += f"```\n{c.content_snippet[:200]}...\n```\n\n"
    
    if not citations_text:
        citations_text = "*No citations found*"
    
    # Format verification
    verification_text = "\n".join([f"- {note}" for note in result.verification_notes])
    if not verification_text:
        verification_text = "*No verification notes*"
    
    return answer_text, stats_text, citations_text, verification_text

def clear_cache():
    """Clear the answer cache."""
    r = init_rag()
    if r._cache:
        r._cache.cache = {}
        r._cache._save()
        return "Cache cleared!"
    return "No cache available"

# Store last Q&A for feedback
last_query = {"question": "", "answer": ""}

def submit_feedback(rating: str, correction: str = ""):
    """Submit feedback on the last answer."""
    r = init_rag()
    if not last_query["question"]:
        return "No question to give feedback on"
    
    success = r.add_feedback(
        query=last_query["question"],
        answer=last_query["answer"],
        rating=rating,
        correction=correction if correction.strip() else None
    )
    
    stats = r.get_feedback_stats()
    return f"Feedback recorded! Total: {stats['total']}, Accuracy: {stats['accuracy']:.0%}"

def feedback_correct():
    return submit_feedback("correct")

def feedback_partial():
    return submit_feedback("partial")

def feedback_incorrect(correction: str):
    return submit_feedback("incorrect", correction)

# Build Gradio interface
with gr.Blocks(title="Evony RAG Ultimate", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎮 Evony RAG Ultimate
    ### Precision Knowledge Retrieval for Evony Game Internals
    
    Ask questions about Evony's protocols, commands, scripts, and exploits.
    The system uses multi-strategy retrieval with verification for accurate answers.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What commands are used for NPC farming?",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Ask", variant="primary")
                cache_toggle = gr.Checkbox(label="Use Cache", value=True)
                clear_btn = gr.Button("🗑️ Clear Cache")
        
        with gr.Column(scale=1):
            stats_output = gr.Markdown(label="Stats")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer")
        
        with gr.Column(scale=1):
            with gr.Accordion("📚 Citations", open=False):
                citations_output = gr.Markdown()
            
            with gr.Accordion("✅ Verification", open=False):
                verification_output = gr.Markdown()
    
    # Feedback section
    with gr.Row():
        gr.Markdown("**Was this answer helpful?**")
        correct_btn = gr.Button("✅ Correct", variant="secondary", size="sm")
        partial_btn = gr.Button("🟡 Partial", variant="secondary", size="sm")
        incorrect_btn = gr.Button("❌ Incorrect", variant="secondary", size="sm")
    
    with gr.Row():
        correction_input = gr.Textbox(
            label="Correction (if incorrect)",
            placeholder="Enter the correct answer...",
            lines=2,
            visible=True
        )
        submit_correction_btn = gr.Button("Submit Correction", size="sm")
    
    feedback_status = gr.Markdown("")
    
    # Wire feedback buttons
    correct_btn.click(fn=feedback_correct, outputs=feedback_status)
    partial_btn.click(fn=feedback_partial, outputs=feedback_status)
    submit_correction_btn.click(fn=feedback_incorrect, inputs=correction_input, outputs=feedback_status)
    
    # Example questions
    gr.Examples(
        examples=[
            "What is the troop attack command?",
            "How does server authentication work?",
            "What commands are used for NPC farming?",
            "Show the protocol encryption details",
            "What exploits are available?",
            "How do I send troops to attack?",
        ],
        inputs=question_input
    )
    
    # Wire up events
    submit_btn.click(
        fn=query_rag,
        inputs=[question_input, cache_toggle],
        outputs=[answer_output, stats_output, citations_output, verification_output]
    )
    
    question_input.submit(
        fn=query_rag,
        inputs=[question_input, cache_toggle],
        outputs=[answer_output, stats_output, citations_output, verification_output]
    )
    
    clear_btn.click(fn=clear_cache, outputs=stats_output)
    
    gr.Markdown("""
    ---
    **RAG Components:** Knowledge Graph (101K entities, 173K relationships) | 
    Hybrid Search (BM25 + Semantic) | Query Expansion | Answer Verification
    """)

if __name__ == "__main__":
    print("Starting Evony RAG Ultimate UI...")
    print("Open http://localhost:7860 in your browser")
    demo.launch(server_name="0.0.0.0", server_port=7860)
