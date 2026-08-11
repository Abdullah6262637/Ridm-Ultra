"""Ornek/demo korpuslar."""
import logging
import re

logger = logging.getLogger(__name__)

def build_default_corpus(lang="en", repeat=1500):
    if lang == "tr":
        text = (
            "kedi masaya cikti . kopek bahceye kacti . kedi kopekten korktu . "
            "kopek kediyi kovaladi . kediler masalara ciktilar . kopekler bahcelere "
            "kactilar . kedi pencereden atladi . kopek kapidan girdi . "
        ) * repeat
    else:
        text = "the cat sat on the mat . the dog sat on the log . the cat saw the dog . " * repeat
    return text.split()


def sanitize_rag_passage(text: str) -> str:
    """Strips dataset artifact noise (multiple-choice options, test questions, English noise)."""
    if not text:
        return ""

    parts = text.split(" | ")
    cleaned_parts = []

    for part in parts:
        p = part
        p = re.sub(r'Aşağıdaki\s+.*?\s+(yazın|soru|cevaplayın|yapın):?', '', p, flags=re.IGNORECASE)
        p = re.sub(r'Soru\s*\d*:?', '', p, flags=re.IGNORECASE)
        p = re.sub(r'Makale:\s*', '', p, flags=re.IGNORECASE)
        p = re.sub(r'SEÇENEK:?\s*\[\+\]\s*hayır;?\s*\[\+\]\s*evet;?', '', p, flags=re.IGNORECASE)
        p = re.sub(r'Cevap:\s*.*', '', p, flags=re.IGNORECASE)
        p = re.sub(r'^\s*[A-D]\)\s+.*', '', p, flags=re.MULTILINE)
        p = p.strip()

        if len(p) > 30 and not p.lower().startswith("soru") and not p.lower().startswith("cevap"):
            cleaned_parts.append(p)

    res = " ".join(cleaned_parts) if cleaned_parts else text.strip()
    res = re.sub(r'^yazın:\s*', '', res, flags=re.IGNORECASE)
    res = re.sub(r'Aşağıdaki\s+.*?\s+(yazın|soru|cevaplayın|yapın):?', '', res, flags=re.IGNORECASE)
    res = re.sub(r'Makale:\s*', '', res, flags=re.IGNORECASE)
    res = re.sub(r'Cevap:\s*.*', '', res, flags=re.IGNORECASE)
    return res.strip()


def load_rag_documents(max_docs=5000):
    from pathlib import Path

    import pandas as pd

    docs = []

    # English Pivot: Point directly to the 2 Billion token FineWeb parquet
    parquet_path = Path("data/raw/train-00003-of-00014.parquet")
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            for txt in df["text"].dropna():
                txt_str = txt.strip()
                if txt_str and len(txt_str.split()) > 10:
                    docs.append(txt_str)
                    if len(docs) >= max_docs:
                        break
        except Exception as e:
            logger.warning(f"Failed to load English RAG dataset {parquet_path}: {e}")

    return docs

