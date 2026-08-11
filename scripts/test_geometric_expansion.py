import sys
sys.path.append(r"c:\Users\HP\Desktop\ridm ultra")
import numpy as np
from ridm_ultra.rag.retriever import HybridRetriever
from ridm_ultra.llm.uma import UniversalManifoldAligner

print("Initializing mock RIDM...")
class MockRIDM:
    def __init__(self, dim):
        self.dim = dim
        self.vocab = ["black", "hole", "gravity", "time", "slowing", "test", "word", "random", "astrophysics", "space"]
        self.word_emb = np.random.randn(len(self.vocab), dim).astype(np.float32)
        
    def context_vector_for(self, words):
        return np.random.randn(self.dim).astype(np.float32)

ridm = MockRIDM(dim=64)
uma = UniversalManifoldAligner()
uma.rotation_matrices['tr'] = np.eye(64) # Identity for test

retriever = HybridRetriever(ridm=ridm, uma_aligner=uma, lang_code="tr")

# Check if massive is enabled
if retriever.massive_enabled:
    print("Testing _retrieve_massive with Geometric Expansion...")
    # This should trigger Geometric Query Expansion
    results = retriever.retrieve("kara delikler zamanı yavaşlatır", top_k=2)
    for res in results:
        print(f"Score: {res[1]:.4f} | Doc: {res[0][:200]}...")
else:
    print("Massive DB not enabled on this system.")
