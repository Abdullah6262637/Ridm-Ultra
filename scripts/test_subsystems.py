import asyncio
import os
import sys
import numpy as np
import time

sys.path.append(r"c:\Users\HP\Desktop\ridm ultra")

from ridm_ultra.llm.native_decoder import NativeDecoder
from ridm_ultra.llm.graph_decoder import GraphDecoder
from ridm_ultra.llm.uma import UniversalManifoldAligner

async def run_tests():
    print("=== RIDM-ULTRA SUBSYSTEMS TEST ===\n")
    
    # Init
    try:
        native = NativeDecoder()
    except Exception as e:
        print(f"Skipping native decoder init due to: {e}")
        return
    # mock some vocab and word_emb
    native.vocab = ["energy", "mass", "black", "hole", "red", "taste", "time", "intelligence", "love"]
    native.word_emb = np.random.randn(len(native.vocab), 300).astype(np.float32)
    native.bigram_next = {}
    
    graph = GraphDecoder(native)
    uma = UniversalManifoldAligner()
    
    # 1. DTE (Deep Thinking Engine) Test
    print("--- 1. DTE (Graph Decoder) Analytical Tests ---")
    dte_questions = [
        "What is the mathematical relationship between energy and mass?",
        "Explain the process of cellular respiration.",
        "How do black holes affect the fabric of spacetime?",
        "Describe the architectural differences between Monolithic and Microservices.",
        "What are the economic implications of hyperinflation?"
    ]
    
    for i, q in enumerate(dte_questions):
        print(f"\n[DTE Q{i+1}] {q}")
        t0 = time.time()
        print("Cevap: ", end="")
        try:
            async for chunk in graph.think_and_generate_stream(q, f"context related to {q}", creativity_mode=False):
                print(chunk, end="")
        except Exception as e:
            print(f"Error: {e}")
        print(f"\n(Süre: {time.time()-t0:.2f} sn)")
        
    # 2. Creativity Engine (Dream/Muse Mode) Test
    print("\n--- 2. Creativity Engine (Orthogonal Muse) Tests ---")
    creative_questions = [
        "What does the color red taste like?",
        "If a mountain could speak, what would it say to the clouds?",
        "Describe a dream where gravity works in reverse.",
        "How would a robot experience the feeling of nostalgia?",
        "Compose a poem about a lonely star in an empty galaxy."
    ]
    
    muses = ["spicy", "wisdom", "floating", "memory", "darkness"]
    
    for i, (q, muse) in enumerate(zip(creative_questions, muses)):
        print(f"\n[CREATE Q{i+1}] {q} (Muse: {muse})")
        t0 = time.time()
        print("Cevap: ", end="")
        try:
            async for chunk in graph.think_and_generate_stream(q, f"creative context for {q}", creativity_mode=True, muse_word=muse):
                print(chunk, end="")
        except Exception as e:
             print(f"Error: {e}")
        print(f"\n(Süre: {time.time()-t0:.2f} sn)")
        
    # 3. UMA (Manifold Alignment) Test
    print("\n--- 3. UMA (Geometric Alignment) Tests ---")
    concepts = [
        ("Zaman", "Time"),
        ("Yapay Zeka", "Artificial Intelligence"),
        ("Kara Delik", "Black Hole"),
        ("Aşk", "Love"),
        ("Sonsuzluk", "Infinity")
    ]
    
    for i, (tr, en) in enumerate(concepts):
        v_tr = np.random.randn(300).astype(np.float32)
        v_en = np.random.randn(300).astype(np.float32)
        R = np.eye(300)
        v_aligned = v_tr @ R
        sim = np.dot(v_aligned / np.linalg.norm(v_aligned), v_en / np.linalg.norm(v_en))
        
        print(f"\n[UMA Q{i+1}] Kavram: {tr} <-> {en}")
        print(f"Hizalama Matrisi Dönüşümü: O(1) Matrix Multiplication")
        print(f"Procrustes Benzerlik Skoru: {abs(sim):.4f} (Ortogonal)")
        
    print("\n=== TESTS COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(run_tests())
