import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class Template:
    """A rhetorical template extracted from a RAG sentence."""
    def __init__(self, skeleton: List[str], slot_indices: List[int], original_sentence: str, svd_score: float):
        self.skeleton = skeleton
        self.slot_indices = slot_indices
        self.original_sentence = original_sentence
        self.svd_score = svd_score

class RhetoricalTemplateEngine:
    """SVD-Guided Rhetorical Template Synthesizer.
    
    Instead of generating word-by-word, this engine:
    1. Extracts rhetorical templates from RAG context sentences
    2. Fills template slots using SVD cosine similarity + trigram constraints
    3. Scores candidates by trigram fluency
    4. Returns the most fluent and relevant candidate
    """
    
    CONTENT_WORD_MIN_LENGTH = 3
    SLOT_TOKEN = "[SLOT]"
    MAX_TEMPLATES = 5
    CANDIDATES_PER_TEMPLATE = 10
    
    def __init__(self, word_emb: np.ndarray, vocab: List[str], word2idx: Dict[str, int], 
                 bigram_next: Dict[Tuple[int, int], Dict[int, int]], stopwords: Optional[Set[str]] = None):
        self.word_emb = word_emb
        self.vocab = vocab
        self.word2idx = word2idx
        self.bigram_next = bigram_next
        # Pre-compute set of word IDs with trigram corpus presence (O(1) lookup)
        self._trigram_word_ids: set = set()
        for (k1, k2), nexts in self.bigram_next.items():
            self._trigram_word_ids.add(k1)
            self._trigram_word_ids.add(k2)
            self._trigram_word_ids.update(nexts.keys())
        if stopwords is None:
            self.stopwords = {
                "a", "an", "and", "are", "as", "at", "be", "but", "by",
                "for", "if", "in", "into", "is", "it", "no", "not", "of",
                "on", "or", "such", "that", "the", "their", "then", "there", "these",
                "they", "this", "to", "was", "will", "with", "from", "which", "were",
                "he", "she", "his", "her", "has", "have", "had"
            }
        else:
            self.stopwords = stopwords
            
    def _clean_word(self, word: str) -> str:
        return re.sub(r'[^a-z0-9]', '', word.lower())

    def _sentence_vector(self, sentence: str) -> np.ndarray:
        """Compute average SVD vector for a sentence (content words only)."""
        words = sentence.split()
        vecs = []
        for w in words:
            c = self._clean_word(w)
            if c and c not in self.stopwords and c in self.word2idx:
                vecs.append(self.word_emb[self.word2idx[c]])
        if vecs:
            avg_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(avg_vec)
            if norm > 0:
                return (avg_vec / norm).astype(np.float32)
        return np.zeros(self.word_emb.shape[1], dtype=np.float32)

    def extract_templates(self, sentences: List[str], query_vec: np.ndarray) -> List[Template]:
        """Extract rhetorical templates from RAG sentences."""
        query_norm = np.linalg.norm(query_vec)
        scores = []
        for sent in sentences:
            s_vec = self._sentence_vector(sent)
            s_norm = np.linalg.norm(s_vec)
            if s_norm > 0 and query_norm > 0:
                score = float(np.dot(s_vec, query_vec) / (s_norm * query_norm))
            else:
                score = 0.0
            scores.append((score, sent))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scores[:self.MAX_TEMPLATES]
        
        templates = []
        for score, sent in top_sentences:
            words = sent.split()
            skeleton = []
            slot_indices = []
            word_scores = []
            
            for i, w in enumerate(words):
                c = self._clean_word(w)
                if len(c) >= self.CONTENT_WORD_MIN_LENGTH and c in self.word2idx and c not in self.stopwords:
                    w_vec = self.word_emb[self.word2idx[c]]
                    w_norm = np.linalg.norm(w_vec)
                    if w_norm > 0 and query_norm > 0:
                        w_score = float(np.dot(w_vec, query_vec) / (w_norm * query_norm))
                    else:
                        w_score = 0.0
                    word_scores.append((i, w_score, w))
                else:
                    skeleton.append(w)
            
            # Keep at least 40% as skeleton
            min_skeleton = int(len(words) * 0.4)
            slots_allowed = len(words) - min_skeleton
            
            # Sort potential slots by furthest from query (lowest score first)
            word_scores.sort(key=lambda x: x[1])
            
            # Select which to make slots
            slot_word_indices = set([idx for idx, _, _ in word_scores[:slots_allowed]])
            
            final_skeleton = []
            final_slot_indices = []
            for i, w in enumerate(words):
                if i in slot_word_indices:
                    final_slot_indices.append(len(final_skeleton))
                    final_skeleton.append(self.SLOT_TOKEN)
                else:
                    final_skeleton.append(w)
                    
            templates.append(Template(final_skeleton, final_slot_indices, sent, score))
            
        return templates

    def fill_template(self, template: Template, query_vec: np.ndarray, 
                      rng: Optional[np.random.RandomState] = None) -> str:
        """Fill template slots with contextually appropriate words."""
        if rng is None:
            rng = np.random.RandomState()
            
        filled = list(template.skeleton)
        query_norm = np.linalg.norm(query_vec)
        
        for slot_idx in template.slot_indices:
            # Gather surrounding context vectors
            context_vecs = []
            prev_id = None
            prev2_id = None
            
            if slot_idx > 0:
                c = self._clean_word(filled[slot_idx - 1])
                if c in self.word2idx:
                    prev_id = self.word2idx[c]
                    context_vecs.append(self.word_emb[prev_id])
                if slot_idx > 1:
                    c2 = self._clean_word(filled[slot_idx - 2])
                    if c2 in self.word2idx:
                        prev2_id = self.word2idx[c2]
                        
            if slot_idx < len(filled) - 1:
                c = self._clean_word(filled[slot_idx + 1])
                if c in self.word2idx and c != self.SLOT_TOKEN:
                    context_vecs.append(self.word_emb[self.word2idx[c]])
                    
            ideal_vec = np.copy(query_vec)
            if context_vecs:
                avg_ctx = np.mean(context_vecs, axis=0)
                ideal_vec = 0.6 * query_vec + 0.4 * avg_ctx
                
            ideal_norm = np.linalg.norm(ideal_vec)
            if ideal_norm > 0:
                ideal_vec = ideal_vec / ideal_norm
                
            # Cosine similarity
            norms = np.linalg.norm(self.word_emb, axis=1)
            valid = norms > 0
            sims = np.zeros(len(self.vocab))
            sims[valid] = np.dot(self.word_emb[valid], ideal_vec) / norms[valid]
            
            top_k_indices = np.argsort(sims)[-50:]
            
            candidates = []
            # Collect IDs of neighboring skeleton words for bigram validation
            neighbor_ids = []
            if prev_id is not None:
                neighbor_ids.append(prev_id)
            if slot_idx < len(filled) - 1:
                next_w = self._clean_word(filled[slot_idx + 1])
                if next_w in self.word2idx and next_w != self.SLOT_TOKEN:
                    neighbor_ids.append(self.word2idx[next_w])
            
            for idx in top_k_indices:
                word = self.vocab[idx]
                if word in self.stopwords:
                    continue
                if not all(c.isascii() for c in word):
                    continue
                if len(word) < 2:
                    continue
                if idx not in self._trigram_word_ids:
                    continue
                # Must form at least ONE valid bigram with a neighbor
                has_neighbor_link = False
                for nid in neighbor_ids:
                    if self.bigram_next.get((nid, idx)) or self.bigram_next.get((idx, nid)):
                        has_neighbor_link = True
                        break
                    # Also check reverse: is this word a valid continuation?
                    for (k1, k2), nexts in []:  # Skip expensive reverse check
                        pass
                if not has_neighbor_link and neighbor_ids:
                    continue
                # Trigram sequence check with predecessors
                is_valid = True
                if prev_id is not None and prev2_id is not None:
                    next_dict = self.bigram_next.get((prev2_id, prev_id))
                    if next_dict and idx not in next_dict:
                        is_valid = False
                
                if is_valid:
                    candidates.append((sims[idx], word))
                    
            if not candidates:
                # Fallback: use original word from the template source sentence
                original_words = template.original_sentence.split()
                # Find the original word at this position
                original_pos = 0
                slot_count = 0
                for si, sw in enumerate(template.skeleton):
                    if sw == self.SLOT_TOKEN:
                        if slot_count == template.slot_indices.index(slot_idx):
                            original_pos = si
                            break
                        slot_count += 1
                if original_pos < len(original_words):
                    filled[slot_idx] = original_words[original_pos]
                else:
                    filled[slot_idx] = original_words[-1] if original_words else "the"
                continue
                        
            if not candidates:
                filled[slot_idx] = "something" # Extreme fallback
            else:
                # Temperature sampling
                scores = np.array([c[0] for c in candidates])
                words = [c[1] for c in candidates]
                
                temp = 0.7
                exp_scores = np.exp(scores / temp)
                probs = exp_scores / np.sum(exp_scores)
                
                chosen_idx = rng.choice(len(words), p=probs)
                filled[slot_idx] = words[chosen_idx]
                
        return " ".join(filled)

    def score_fluency(self, text: str) -> float:
        """Score text fluency using trigram log-probability average."""
        words = text.split()
        if len(words) < 3:
            return 0.0
            
        score = 0.0
        num_trigrams = 0
        
        for i in range(len(words) - 2):
            w1 = self._clean_word(words[i])
            w2 = self._clean_word(words[i+1])
            w3 = self._clean_word(words[i+2])
            
            w1_id = self.word2idx.get(w1, -1)
            w2_id = self.word2idx.get(w2, -1)
            w3_id = self.word2idx.get(w3, -1)
            
            if w1_id >= 0 and w2_id >= 0 and w3_id >= 0:
                next_dict = self.bigram_next.get((w1_id, w2_id))
                if next_dict and w3_id in next_dict:
                    count = next_dict[w3_id]
                    total = sum(next_dict.values())
                    prob = count / total
                    score += np.log(prob)
                else:
                    score += np.log(1e-6)
            else:
                score += np.log(1e-6)
                
            num_trigrams += 1
            
        if num_trigrams == 0:
            return 0.0
            
        return score / num_trigrams

    def generate_candidates(self, sentences: List[str], query_vec: np.ndarray,
                            num_candidates: int = 50) -> List[Tuple[float, str]]:
        """Generate and rank multiple candidate sentences."""
        templates = self.extract_templates(sentences, query_vec)
        rng = np.random.RandomState()
        
        candidates = []
        for t in templates:
            for _ in range(self.CANDIDATES_PER_TEMPLATE):
                filled_text = self.fill_template(t, query_vec, rng)
                fluency = self.score_fluency(filled_text)
                
                sent_vec = self._sentence_vector(filled_text)
                q_norm = np.linalg.norm(query_vec)
                s_norm = np.linalg.norm(sent_vec)
                coverage = 0.0
                if q_norm > 0 and s_norm > 0:
                    coverage = float(np.dot(sent_vec, query_vec) / (s_norm * q_norm))
                
                final_score = 0.4 * t.svd_score + 0.3 * fluency + 0.3 * coverage
                candidates.append((final_score, filled_text))
                
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:num_candidates]
