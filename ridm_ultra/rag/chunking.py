import re
from typing import Any, Dict, List


class SemanticChunker:
    """Intelligent text chunker using sliding windows over semantic sentence boundaries."""

    def __init__(self, window_size: int = 40, overlap: int = 15):
        """
        window_size: Approximate number of words per chunk.
        overlap: Number of words to overlap between chunks to preserve context.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sentence_boundaries = re.compile(r'([.?!])\s+')

    def split_into_sentences(self, text: str) -> List[str]:
        # Split by punctuation and keep the punctuation
        parts = self.sentence_boundaries.split(text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentences.append(parts[i].strip() + parts[i+1])
        if len(parts) % 2 != 0 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        return sentences

    def chunk_document(self, text: str, doc_id: str = "") -> List[Dict[str, Any]]:
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_words = []
        current_chunk_sentences = []

        for sentence in sentences:
            words = sentence.split()
            current_chunk_words.extend(words)
            current_chunk_sentences.append(sentence)

            if len(current_chunk_words) >= self.window_size:
                chunks.append({
                    "doc_id": doc_id,
                    "content": " ".join(current_chunk_sentences)
                })
                # Keep overlap sentences
                overlap_words_count = 0
                overlap_sentences = []
                for s in reversed(current_chunk_sentences):
                    s_words = s.split()
                    if overlap_words_count + len(s_words) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words_count += len(s_words)
                    else:
                        break

                # Minimum 1 sentence overlap if overlap is requested
                if self.overlap > 0 and not overlap_sentences and current_chunk_sentences:
                    overlap_sentences = [current_chunk_sentences[-1]]

                current_chunk_sentences = overlap_sentences
                current_chunk_words = []
                for s in current_chunk_sentences:
                    current_chunk_words.extend(s.split())

        # Add the last chunk if it has content and is not already exactly the last chunk
        if current_chunk_sentences:
            final_content = " ".join(current_chunk_sentences)
            if not chunks or chunks[-1]["content"] != final_content:
                chunks.append({
                    "doc_id": doc_id,
                    "content": final_content
                })

        return chunks
