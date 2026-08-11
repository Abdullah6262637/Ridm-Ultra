import asyncio
import logging
import re
from typing import Dict, List, Set, Tuple

import numpy as np

from ridm_ultra.llm.template_engine import RhetoricalTemplateEngine

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "if", "in", "into", "is", "it", "no", "not", "of",
    "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "from", "which", "were",
    "he", "she", "his", "her", "has", "have", "had"
}

class Node:
    def __init__(self, id: int, word: str, word_id: int, score: float = 0.0, is_stopword: bool = False):
        self.id = id
        self.word = word
        self.word_id = word_id
        self.score = score
        self.is_stopword = is_stopword
        self.edges: Set[int] = set()
        self.is_terminal = False

class GraphDecoder:
    """Context-Aware N-Gram Graph Synthesizer (CANG-Gen)
    
    Generates high-quality fluent sentences by constructing a Word Graph from RAG Context, 
    but strictly enforces Bigram-overlap for sentence crossing, and scores paths using 
    SVD semantic similarity + Global 3-Gram probabilities.
    """

    def __init__(self, native_decoder):
        self.decoder = native_decoder
        self.vocab_set = set(self.decoder.vocab)
        self.template_engine = RhetoricalTemplateEngine(
            word_emb=self.decoder.word_emb,
            vocab=self.decoder.vocab,
            word2idx=self.decoder.word2idx,
            bigram_next=self.decoder.bigram_next,
        )

    def _clean_word(self, word: str) -> str:
        return re.sub(r'[^a-z0-9]', '', word.lower())

    async def generate_stream(self, query: str, context: str, beam_width: int = 15, max_len: int = 80):
        # We wrap the logic in a stream to match the interface, but it's computed instantly
        query_vec = self.decoder._get_context_vector(self.decoder._encode_prompt(query))

        # 1. Parse sentences
        sentences = re.split(r'(?<=[.!?]) +', context.replace('\n', ' '))
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]

        if not sentences:
            yield "Unable to generate abstractive summary."
            return

        # 2. Score entire sentences for fallback (Extractive Summarization)
        sentence_scores = []
        for i, sent in enumerate(sentences):
            words = sent.split()
            score = 0.0
            valid_words = 0
            for w in words:
                c = self._clean_word(w)
                if c and c not in STOPWORDS and c in self.vocab_set:
                    wid = self.decoder.word2idx[c]
                    w_vec = self.decoder.word_emb[wid]
                    n1 = np.linalg.norm(w_vec)
                    n2 = np.linalg.norm(query_vec)
                    if n1 > 0 and n2 > 0:
                        score += float(np.dot(w_vec, query_vec) / (n1 * n2))
                        valid_words += 1
            avg_score = score / max(1, valid_words)
            sentence_scores.append((avg_score, sent))

        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        best_extractive = sentence_scores[0][1]

        # 3. Build Word Graph with strict Bigram pivoting
        nodes: Dict[int, Node] = {}
        pivot_map: Dict[Tuple[str, str], int] = {}
        start_nodes: List[int] = []
        node_counter = 0

        query_words = set(self._clean_word(qw) for qw in query.split())

        for sent in sentences:
            words = sent.split()
            prev_node_id = None
            prev_clean = None

            for i, w in enumerate(words):
                w_clean = self._clean_word(w)
                if not w_clean:
                    continue

                is_stop = w_clean in STOPWORDS or len(w_clean) < 3
                word_id = self.decoder.word2idx.get(w_clean, 0)

                curr_id = None

                # Check bigram overlap to merge (this enforces grammatical continuity when jumping)
                if prev_clean is not None and not is_stop:
                    bigram = (prev_clean, w_clean)
                    if bigram in pivot_map:
                        curr_id = pivot_map[bigram]

                if curr_id is None:
                    curr_id = node_counter

                    # Compute SVD score
                    score = 0.0
                    if not is_stop and word_id > 0:
                        w_vec = self.decoder.word_emb[word_id]
                        n1 = np.linalg.norm(w_vec)
                        n2 = np.linalg.norm(query_vec)
                        if n1 > 0 and n2 > 0:
                            score = float(np.dot(w_vec, query_vec) / (n1 * n2))
                        if w_clean in query_words:
                            score += 2.0 # Higher boost for exact query word match

                    nodes[curr_id] = Node(curr_id, w, word_id, score, is_stop)

                    if prev_clean is not None and not is_stop:
                        pivot_map[(prev_clean, w_clean)] = curr_id

                    node_counter += 1

                if i == len(words) - 1:
                    nodes[curr_id].is_terminal = True

                if prev_node_id is not None:
                    if curr_id != prev_node_id:
                        nodes[prev_node_id].edges.add(curr_id)
                else:
                    start_nodes.append(curr_id)

                prev_node_id = curr_id
                prev_clean = w_clean

        # 3.5 Forward Score Propagation (Lexical Proximity QA)
        # Propagate high scores forward along edges to highlight the "answers" following query terms.
        decay_factor = 0.75
        max_prop_dist = 4

        high_score_nodes = [nid for nid, node in nodes.items() if node.score > 0.3]

        for start_nid in high_score_nodes:
            queue = [(start_nid, 0, nodes[start_nid].score)]
            visited = {start_nid}
            while queue:
                curr_id, dist, curr_score = queue.pop(0)
                if dist > 0:
                    nodes[curr_id].score += curr_score

                if dist < max_prop_dist:
                    next_score = curr_score * decay_factor
                    for child_id in nodes[curr_id].edges:
                        if child_id not in visited:
                            visited.add(child_id)
                            queue.append((child_id, dist + 1, next_score))

        # Normalize scores
        max_score = max([n.score for n in nodes.values()] + [1.0])
        for n in nodes.values():
            n.score = (n.score / max_score) * 2.0  # Scale up for beam search weight

        # 4. Beam Search with Trigram Scoring
        beam = []
        for sn in set(start_nodes):
            beam.append( (nodes[sn].score, [sn]) )

        completed_paths = []

        for step in range(max_len):
            new_beam = []
            for score, path in beam:
                last_node_id = path[-1]
                last_node = nodes[last_node_id]

                if last_node.is_terminal and len(path) >= 5:
                    length_penalty = (len(path) ** 0.6)
                    completed_paths.append((score / length_penalty, path))
                    continue

                for child_id in last_node.edges:
                    if child_id not in path:
                        child_node = nodes[child_id]
                        child_score = child_node.score

                        # Trigram Boost via NativeDecoder Language Model
                        trigram_boost = 0.0
                        if child_id == last_node_id + 1:
                            trigram_boost += 1.0 # Huge bonus for following original sentence structure

                        if len(path) >= 2:
                            w1_id = nodes[path[-2]].word_id
                            w2_id = nodes[path[-1]].word_id
                            w3_id = child_node.word_id

                            if w1_id > 0 and w2_id > 0 and w3_id > 0:
                                next_counts = self.decoder.bigram_next.get((w1_id, w2_id))
                                if next_counts and w3_id in next_counts:
                                    total = sum(next_counts.values())
                                    trigram_boost += 1.5 * (next_counts[w3_id] / total)
                                else:
                                    if child_id != last_node_id + 1:
                                        # Penalize unseen trigrams when crossing sentences
                                        trigram_boost -= 0.5

                        new_score = score + child_score + trigram_boost
                        new_beam.append( (new_score, path + [child_id]) )

            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]

            if not beam:
                break

        if not completed_paths and beam:
            # If no paths reached a terminal node within max_len, accept the best partial paths
            for score, path in beam:
                if len(path) >= 5:
                    length_penalty = (len(path) ** 0.6)
                    completed_paths.append((score / length_penalty, path))

        # 5. Adaptive Response Length — Query Complexity Analysis
        # Classify query type using interrogative word patterns
        q_lower = query.lower().strip()
        
        # Short answer queries: direct factual (who, when, where + short context)
        SHORT_PATTERNS = [
            r'^(who|when|where)\b',
            r'^(is|are|was|were|does|did|do|can|will|has|have)\b',
        ]
        # Long answer queries: explanatory, comparative, process
        LONG_PATTERNS = [
            r'^(how|why|explain|describe|compare|discuss|analyze|what are the)\b',
            r'(difference|advantages|disadvantages|process|mechanism|impact|relationship)\b',
        ]
        
        target_sentences = 2  # Default: medium
        for pat in LONG_PATTERNS:
            if re.search(pat, q_lower):
                target_sentences = 4
                break
        for pat in SHORT_PATTERNS:
            if re.search(pat, q_lower):
                target_sentences = 1
                break
        
        # "What is X" type → medium (2-3 sentences)
        if re.match(r'^what (is|are|was|were)\b', q_lower):
            target_sentences = 3
        
        # Scale target by available context richness
        target_sentences = min(target_sentences, len(sentences))
        
        print(f"\n[DEBUG] Found {len(completed_paths)} abstractive paths.")
        if completed_paths:
            completed_paths.sort(key=lambda x: x[0], reverse=True)
            for i, (score, path) in enumerate(completed_paths[:5]):
                words = " ".join([nodes[nid].word for nid in path])
                print(f"  Path {i+1} [score={score:.3f}]: {words}")

            print(f"[DEBUG] Best abstractive score: {completed_paths[0][0]}, Best extractive: {sentence_scores[0][0]}")
            print(f"[DEBUG] Target sentences: {target_sentences}")
            
            # 6. Multi-Sentence Composition — Select diverse, non-redundant paths
            selected_texts = []
            selected_word_sets = []
            
            for score, path in completed_paths:
                candidate_words = [nodes[nid].word for nid in path]
                candidate_text = " ".join(candidate_words)
                candidate_word_set = set(self._clean_word(w) for w in candidate_words if len(self._clean_word(w)) >= 3)
                
                # Redundancy check: reject if >70% word overlap with any already selected
                is_redundant = False
                for prev_set in selected_word_sets:
                    if not prev_set or not candidate_word_set:
                        continue
                    overlap = len(candidate_word_set & prev_set) / max(len(candidate_word_set), 1)
                    if overlap > 0.7:
                        is_redundant = True
                        break
                
                if not is_redundant:
                    # Capitalize and punctuate
                    if candidate_text:
                        candidate_text = candidate_text[0].upper() + candidate_text[1:]
                        if candidate_text[-1] not in ".!?":
                            candidate_text += "."
                    selected_texts.append(candidate_text)
                    selected_word_sets.append(candidate_word_set)
                
                if len(selected_texts) >= target_sentences:
                    break
            
            # If we still need more sentences, use extractive fallback
            if len(selected_texts) < target_sentences:
                for score, sent in sentence_scores:
                    sent_word_set = set(self._clean_word(w) for w in sent.split() if len(self._clean_word(w)) >= 3)
                    is_redundant = False
                    for prev_set in selected_word_sets:
                        if not prev_set or not sent_word_set:
                            continue
                        overlap = len(sent_word_set & prev_set) / max(len(sent_word_set), 1)
                        if overlap > 0.7:
                            is_redundant = True
                            break
                    if not is_redundant:
                        text = sent.strip()
                        if text:
                            text = text[0].upper() + text[1:]
                            if text[-1] not in ".!?":
                                text += "."
                        selected_texts.append(text)
                        selected_word_sets.append(sent_word_set)
                    if len(selected_texts) >= target_sentences:
                        break
            
            result = " ".join(selected_texts)
        else:
            # Fallback to Extractive
            print(f"[DEBUG] No abstractive paths found. Falling back to extractive (score: {sentence_scores[0][0]})")
            result = best_extractive
            if result:
                result = result[0].upper() + result[1:]
                if result[-1] not in ".!?":
                    result += "."

        # Stream output
        for word in result.split():
            yield word + " "

    def detect_contradiction(self, sentence1: str, sentence2: str) -> bool:
        """
        DTE Phase 2: Meta-Reflection Contradiction Detection
        Checks if two generated paths contradict each other using:
        1. Lexical antonym pairs
        2. SVD cosine direction analysis (negative cosine = opposite meaning)
        """
        words1 = set(sentence1.lower().split())
        words2 = set(sentence2.lower().split())

        antonym_pairs = [
            ("increase", "decrease"), ("good", "bad"), ("hot", "cold"),
            ("up", "down"), ("true", "false"), ("yes", "no"),
            ("high", "low"), ("large", "small"), ("more", "less"),
            ("before", "after"), ("positive", "negative"), ("success", "failure"),
            ("always", "never"), ("all", "none"), ("open", "closed"),
        ]
        for w1, w2 in antonym_pairs:
            if (w1 in words1 and w2 in words2) or (w2 in words1 and w1 in words2):
                return True

        # SVD cosine direction check
        vec1 = self.template_engine._sentence_vector(sentence1)
        vec2 = self.template_engine._sentence_vector(sentence2)
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1 > 0 and n2 > 0:
            cosine = float(np.dot(vec1, vec2) / (n1 * n2))
            if cosine < -0.3:
                logger.info(f"SVD contradiction detected: cosine={cosine:.3f}")
                return True

        return False

    async def think_and_generate_stream(self, query: str, context: str, sim_count: int = 50, creativity_mode: bool = False, muse_word: str = None):
        """
        Deep Thinking Engine (DTE) - Real Monte Carlo Template Simulation + Meta Reflection
        Generates multiple candidate hypotheses via template engine, scores them,
        performs contradiction detection, and selects the best path.
        """
        yield "[DÜŞÜNÜYOR] SVD Vektörleri Çarpıştırılıyor...\n"
        await asyncio.sleep(0.3)

        query_vec = self.decoder._get_context_vector(self.decoder._encode_prompt(query))

        if creativity_mode and muse_word:
            yield f"[DÜŞÜNÜYOR] 🎨 GEOMETRİK YARATICILIK AKTİF: '{muse_word}' vektörü ile Dik Açılı (Orthogonal) sapma hesaplanıyor...\n"
            await asyncio.sleep(0.3)
            muse_clean = re.sub(r'[^a-z0-9]', '', muse_word.lower())
            if muse_clean in self.decoder.word2idx:
                muse_vec = self.decoder.word_emb[self.decoder.word2idx[muse_clean]]
                query_vec = self.decoder.get_orthogonal_muse(query_vec, muse_vec, lambda_weight=0.3)

        # 1. Parse sentences from context
        sentences = re.split(r'(?<=[.!?]) +', context.replace('\n', ' '))
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]

        if not sentences:
            yield "Yeterli bağlam bulunamadı.\n"
            return

        yield f"[DÜŞÜNÜYOR] {len(sentences)} RAG cümlesinden şablon çıkarılıyor...\n"
        await asyncio.sleep(0.3)

        # 2. Generate candidates via Template Engine (Real Monte Carlo)
        yield f"[DÜŞÜNÜYOR] Monte Carlo: {sim_count} farklı hipotez üretiliyor...\n"
        await asyncio.sleep(0.3)

        candidates = self.template_engine.generate_candidates(
            sentences, query_vec, num_candidates=sim_count
        )
        logger.info(f"DTE Monte Carlo: generated {len(candidates)} candidates")

        if not candidates:
            # Fallback to extractive
            yield "[DÜŞÜNÜYOR] Şablon üretimi başarısız, extractive moda geçiliyor...\n"
            async for word in self.generate_stream(query, context, beam_width=20, max_len=100):
                yield word
            return

        yield f"[DÜŞÜNÜYOR] {len(candidates)} hipotez üretildi. En iyi skor: {candidates[0][0]:.3f}\n"
        await asyncio.sleep(0.3)

        # 3. Meta-Reflection: Contradiction detection between top candidates
        yield "[DÜŞÜNÜYOR] Meta-Reflection: Çelişki kontrolü yapılıyor...\n"
        await asyncio.sleep(0.3)

        contradiction_found = False
        if len(candidates) >= 2:
            top1_text = candidates[0][1]
            top2_text = candidates[1][1]
            if self.detect_contradiction(top1_text, top2_text):
                contradiction_found = True
                yield "[DÜŞÜNÜYOR] ⚠️ Hipotezler arasında çelişki tespit edildi! Alternatif yol seçiliyor...\n"
                await asyncio.sleep(0.3)
                # Remove contradicting candidate, use the next non-contradicting one
                filtered = [candidates[0]]  # Keep the best
                for score, text in candidates[2:]:
                    if not self.detect_contradiction(candidates[0][1], text):
                        filtered.append((score, text))
                candidates = filtered if filtered else candidates

        if not contradiction_found:
            yield "[DÜŞÜNÜYOR] ✓ Çelişki bulunamadı. Tutarlılık doğrulandı.\n"
            await asyncio.sleep(0.3)

        yield "[DÜŞÜNÜYOR] En yüksek SVD + Trigram + Coverage skoruna sahip rota bulundu.\n"
        yield "=" * 40 + "\n"

        # 4. Stream the best candidate
        best_text = candidates[0][1]
        if best_text:
            best_text = best_text[0].upper() + best_text[1:]
            if best_text[-1] not in '.!?':
                best_text += '.'

        for word in best_text.split():
            yield word + " "
            await asyncio.sleep(0.015)
