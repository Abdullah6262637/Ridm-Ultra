import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from ridm_ultra.api.dependencies import get_chat_engine

async def run_tests():
    questions = [
        # Teknik / Sistem Soruları (RAG Odaklı)
        "RIDM Ultra hangi algoritmayı kullanarak kelime vektörlerini hesaplıyor?",
        "Zero-backprop nedir ve nasıl çalışır?",
        "Chat Engine hangi özellikleri destekliyor?",
        "Sliding context window ne işe yarar?",
        "SVD'nin makine öğrenmesindeki rolü nedir?",
        
        # Genel Bilgi / RAG Düşme İhtimali Olanlar
        "Yapay zeka modellerinde transformer mimarisi neden önemlidir?",
        "BM25 ve Dense arama arasındaki fark nedir?",
        "Reciprocal Rank Fusion (RRF) nasıl hesaplanır?",
        "Dökümanlar parçalara ayrılırken neden overlap (örtüşme) kullanılır?",
        "SQLite FTS5 ile vektör araması yapılabilir mi?",
        
        # Karmaşık / Matematik + Bilgi
        "Eğer 250 kelimelik 4 dökümanım varsa toplam kelime sayısı kaçtır?",
        "1024 * 8",
        
        # NLU - Gündelik ve Bağlamsal
        "Bana RAG sistemi hakkında bir masal kurgula.",
        "Nasılsın, sistemin düzgün çalışıyor mu?",
        
        # Ekstra testler
        "Cosine similarity formülü nedir?",
        "Bir dil modeli nasıl eğitilir?",
        "Cross-Encoder reranker neden düz vektör aramasına göre daha başarılıdır?",
        "FastAPI asenkron sorguları nasıl işler?",
        "Python'da bellek yönetimi nasıl yapılır?",
        "Yapay zekanın geleceği hakkında ne düşünüyorsun?"
    ]

    engine = get_chat_engine()
    
    # Pre-load RAG engine to show initialization time outside of question measuring
    if engine.rag_engine is not None:
        print("[!] Hybrid RAG Engine loaded successfully with", engine.rag_engine.dense.doc_count, "dense vectors.\n")
    
    print("# RIDM Ultra v8.3 - 20 Soru (Hybrid RAG + NLU) Test Raporu\n")
    print("Sistem; BM25+Dense Hybrid RAG, Cross-Encoder Reranker ve Semantic Routing üzerinden değerlendirilmiştir.\n")
    
    results_md = "# 20 Soru RAG & NLU Test Raporu\n\n"
    
    for i, q in enumerate(questions, 1):
        # We use chat() method directly for complete generation
        print(f"### Soru {i}: `{q}`")
        response = await engine.chat(
            user_message=q,
            session_id="test_session_rag_20",
            max_tokens=60
        )
        ans = response.content.strip()
        print(f"**RIDM Ultra:** {ans}\n")
        
        results_md += f"### Soru {i}: `{q}`\n"
        results_md += f"**RIDM Ultra:** {ans}\n\n"
        
    with open("artifacts/20_questions_rag_report.md", "w", encoding="utf-8") as f:
        f.write(results_md)

if __name__ == "__main__":
    asyncio.run(run_tests())
