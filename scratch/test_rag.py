import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ridm_ultra.core import RIDM
from ridm_ultra.rag import HybridRetriever, SemanticChunker

def run_test():
    print("Testing Enterprise RAG System...")
    
    # 1. Setup mock RIDM
    vocab = ["<UNK>", "hello", "world", "this", "is", "a", "test", "document", "for", "rag", "system", "apple", "banana"]
    counts = {w: 1 for w in vocab}
    ridm = RIDM(vocab=vocab, counts=counts, dim=64, window=3)
    
    # Mock embeddings
    import numpy as np
    ridm.word_emb = np.random.randn(len(vocab), 64).astype(np.float32)
    ridm._Vt_k = ridm.word_emb.T
    ridm.context_vecs = ridm.word_emb
    
    # 2. Setup RAG
    docs = [
        "This is a test document for the RAG system.",
        "Another document about apple and banana.",
        "Hello world, this is just a test.",
        "RAG system is very important for context."
    ]
    
    retriever = HybridRetriever(ridm)
    chunker = SemanticChunker(window_size=5, overlap=2)
    
    chunks = []
    for i, doc in enumerate(docs):
        chunks.extend(chunker.chunk_document(doc, doc_id=f"doc_{i}"))
        
    print(f"Total chunks created: {len(chunks)}")
    retriever.add_chunks(chunks)
    
    # 3. Test Retrieval
    query = "test document RAG"
    print(f"\nQuerying: '{query}'")
    results = retriever.retrieve(query, top_k=2)
    
    for i, (content, score, meta) in enumerate(results):
        print(f"Rank {i+1} [Score: {score:.4f}] [DocID: {meta['doc_id']}]: {content}")
        
    print("\nTest completed successfully!")

if __name__ == "__main__":
    run_test()
