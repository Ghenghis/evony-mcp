#!/usr/bin/env python3
"""
Generate embeddings for semantic search.
Uses sentence-transformers for fast local embedding generation.
"""
import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, ".")

print("=" * 60)
print("GENERATING EMBEDDINGS FOR SEMANTIC SEARCH")
print("=" * 60)

# Paths
CHUNKS_FILE = r"evony_rag\index\chunks.json"
EMBEDDINGS_FILE = r"G:\evony_rag_index\embeddings.npy"
METADATA_FILE = r"G:\evony_rag_index\embeddings_metadata.json"

# Load chunks
print("\n[1] Loading chunks...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"    Loaded {len(chunks):,} chunks")

# Use LM Studio embedding API
print("\n[2] Connecting to LM Studio embedding API...")
import requests

LMSTUDIO_URL = "http://localhost:1234/v1/embeddings"
model_name = "text-embedding-nomic-embed-text-v1.5"  # Available in LM Studio

# Test connection
try:
    test_resp = requests.post(LMSTUDIO_URL, json={"model": model_name, "input": ["test"]}, timeout=10)
    test_data = test_resp.json()
    embedding_dim = len(test_data["data"][0]["embedding"])
    print(f"    Model: {model_name}")
    print(f"    Embedding dimension: {embedding_dim}")
except Exception as e:
    print(f"    Error connecting to LM Studio: {e}")
    print("    Make sure LM Studio is running with an embedding model loaded")
    sys.exit(1)

def get_embeddings(texts):
    """Get embeddings from LM Studio."""
    resp = requests.post(LMSTUDIO_URL, json={"model": model_name, "input": texts}, timeout=120)
    data = resp.json()
    return [d["embedding"] for d in data["data"]]

# Generate embeddings in batches
print("\n[3] Generating embeddings...")
batch_size = 128
all_embeddings = []

texts = [c.get("content", "")[:512] for c in chunks]  # Limit to 512 chars
total_batches = (len(texts) + batch_size - 1) // batch_size

start_time = time.time()
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    try:
        batch_embeddings = get_embeddings(batch)
        all_embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"    Error at batch {i//batch_size}: {e}")
        # Add zero embeddings as fallback
        all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
    
    batch_num = i // batch_size + 1
    if batch_num % 10 == 0 or batch_num == total_batches:
        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        eta = (len(texts) - i - len(batch)) / rate if rate > 0 else 0
        print(f"    Batch {batch_num}/{total_batches} ({i + len(batch):,}/{len(texts):,}) - {rate:.0f} chunks/sec, ETA: {eta:.0f}s")

embeddings = np.array(all_embeddings)
elapsed = time.time() - start_time
print(f"    Generated {embeddings.shape[0]:,} embeddings in {elapsed:.1f}s")

# Save embeddings
print("\n[4] Saving embeddings...")
np.save(EMBEDDINGS_FILE, embeddings)
print(f"    Saved: {os.path.getsize(EMBEDDINGS_FILE):,} bytes")

# Save metadata
metadata = {
    "model": model_name,
    "dimension": model.get_sentence_embedding_dimension(),
    "num_embeddings": embeddings.shape[0],
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"    Metadata saved")

# Verify
print("\n[5] Verifying...")
loaded = np.load(EMBEDDINGS_FILE)
print(f"    Shape: {loaded.shape}")
print(f"    Dtype: {loaded.dtype}")

# Quick similarity test
print("\n[6] Quick similarity test...")
test_query = "troop attack command"
query_emb = np.array(get_embeddings([test_query])[0])
similarities = np.dot(loaded, query_emb) / (np.linalg.norm(loaded, axis=1) * np.linalg.norm(query_emb) + 1e-8)
top_indices = np.argsort(similarities)[-5:][::-1]

print(f"    Query: '{test_query}'")
print(f"    Top matches:")
for idx in top_indices:
    chunk = chunks[idx]
    print(f"      [{similarities[idx]:.3f}] {chunk.get('file_path', '')[-40:]} - {chunk.get('content', '')[:50]}...")

print("\n" + "=" * 60)
print("EMBEDDINGS GENERATION COMPLETE")
print("=" * 60)
