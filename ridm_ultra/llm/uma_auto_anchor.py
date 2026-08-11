"""Automatic Anchor Discovery for Universal Manifold Alignment.

Finds translation-equivalent word pairs between languages automatically
using cognate mining, frequency-rank alignment, and SVD neighborhood validation.
No human intervention needed.
"""
import logging
import numpy as np
import scipy.linalg as la
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Named constants
MIN_COGNATE_LENGTH = 4
MAX_EDIT_DISTANCE = 3
MIN_NEIGHBORHOOD_OVERLAP = 0.25
TOP_FREQ_WORDS = 2000
MIN_ANCHORS_REQUIRED = 50
NEIGHBORHOOD_K = 10

class AutoAnchorDiscovery:
    """Automatic cross-lingual anchor pair discovery using cognates and frequency alignment."""
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Wagner-Fischer dynamic programming edit distance."""
        len1, len2 = len(s1), len(s2)
        dp = np.zeros((len1 + 1, len2 + 1), dtype=np.int32)
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
            
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost # substitution
                )
        return int(dp[len1][len2])
        
    def discover_cognates(self, vocab_src: List[str], vocab_tgt: List[str],
                          max_edit_dist: int = MAX_EDIT_DISTANCE,
                          min_length: int = MIN_COGNATE_LENGTH) -> List[Tuple[str, str, float]]:
        """Find cognate pairs by edit distance.
        
        Only considers words of min_length or longer.
        Returns list of (src_word, tgt_word, similarity_score) sorted by score.
        Similarity = 1 - (edit_dist / max(len(s1), len(s2)))
        """
        logger.info(f"Discovering cognates between {len(vocab_src)} source and {len(vocab_tgt)} target words.")
        cognates = []
        try:
            # Pre-filter lists for length to save computation
            filtered_src = [w for w in vocab_src if len(w) >= min_length]
            filtered_tgt = [w for w in vocab_tgt if len(w) >= min_length]
            
            # Simple nested loop for cognates. O(N*M) can be large, but acceptable for filtered vocabs.
            for w_src in filtered_src:
                for w_tgt in filtered_tgt:
                    # Quick length difference check to prune
                    if abs(len(w_src) - len(w_tgt)) > max_edit_dist:
                        continue
                        
                    dist = self._edit_distance(w_src, w_tgt)
                    if dist <= max_edit_dist:
                        max_len = max(len(w_src), len(w_tgt))
                        sim = 1.0 - (dist / float(max_len))
                        cognates.append((w_src, w_tgt, sim))
                        
            cognates.sort(key=lambda x: x[2], reverse=True)
            logger.info(f"Discovered {len(cognates)} potential cognates.")
            return cognates
        except Exception as e:
            logger.error(f"Error during cognate discovery: {e}")
            raise

    def discover_by_frequency_rank(self, vocab_src: List[str], vocab_tgt: List[str],
                                    counts_src: Dict[str, int], counts_tgt: Dict[str, int],
                                    top_k: int = TOP_FREQ_WORDS) -> List[Tuple[str, str]]:
        """Align words by frequency rank position.
        
        Sort both vocabs by frequency, pair words at same rank position.
        Filter: only keep pairs where both words have length >= 3.
        """
        logger.info(f"Aligning top {top_k} frequency words.")
        try:
            # Sort words by frequency descending
            sorted_src = sorted(vocab_src, key=lambda w: counts_src.get(w, 0), reverse=True)
            sorted_tgt = sorted(vocab_tgt, key=lambda w: counts_tgt.get(w, 0), reverse=True)
            
            # Take top_k
            top_src = sorted_src[:top_k]
            top_tgt = sorted_tgt[:top_k]
            
            aligned_pairs = []
            max_align = min(len(top_src), len(top_tgt))
            
            for i in range(max_align):
                w_src = top_src[i]
                w_tgt = top_tgt[i]
                if len(w_src) >= 3 and len(w_tgt) >= 3:
                    aligned_pairs.append((w_src, w_tgt))
                    
            logger.info(f"Aligned {len(aligned_pairs)} pairs by frequency rank.")
            return aligned_pairs
        except Exception as e:
            logger.error(f"Error during frequency rank discovery: {e}")
            raise

    def validate_anchors(self, pairs: List[Tuple[str, str]],
                         emb_src: np.ndarray, emb_tgt: np.ndarray,
                         w2i_src: Dict[str, int], w2i_tgt: Dict[str, int],
                         k: int = NEIGHBORHOOD_K,
                         min_overlap: float = MIN_NEIGHBORHOOD_OVERLAP) -> List[Tuple[str, str]]:
        """Validate anchor pairs using SVD neighborhood overlap.
        
        For each pair (w_src, w_tgt):
          1. Find k nearest neighbors of w_src in source space
          2. Find k nearest neighbors of w_tgt in target space  
          3. Check if any neighbor pairs also appear in our anchor candidates
          4. If overlap ratio >= min_overlap, keep this anchor
        """
        logger.info(f"Validating {len(pairs)} candidate anchors using neighborhood overlap.")
        valid_pairs = []
        try:
            # Normalize embeddings for cosine similarity
            norm_src = np.linalg.norm(emb_src, axis=1, keepdims=True)
            norm_src[norm_src == 0] = 1.0
            norm_emb_src = emb_src / norm_src
            
            norm_tgt = np.linalg.norm(emb_tgt, axis=1, keepdims=True)
            norm_tgt[norm_tgt == 0] = 1.0
            norm_emb_tgt = emb_tgt / norm_tgt
            
            # Build quick lookup for candidate targets given a source
            candidate_map = {src: tgt for src, tgt in pairs}
            
            # Reverse mapping to look up words by index
            i2w_src = {i: w for w, i in w2i_src.items()}
            i2w_tgt = {i: w for w, i in w2i_tgt.items()}
            
            for w_src, w_tgt in pairs:
                if w_src not in w2i_src or w_tgt not in w2i_tgt:
                    continue
                    
                idx_src = w2i_src[w_src]
                idx_tgt = w2i_tgt[w_tgt]
                
                # Get k-NN in source
                vec_src = norm_emb_src[idx_src]
                sim_src = np.dot(norm_emb_src, vec_src)
                # k+1 because the word itself is the closest
                top_k_src_idx = np.argpartition(sim_src, -(k+1))[-(k+1):]
                
                # Get k-NN in target
                vec_tgt = norm_emb_tgt[idx_tgt]
                sim_tgt = np.dot(norm_emb_tgt, vec_tgt)
                top_k_tgt_idx = np.argpartition(sim_tgt, -(k+1))[-(k+1):]
                
                # Extract neighboring words
                neighbors_src = {i2w_src[i] for i in top_k_src_idx if i in i2w_src}
                neighbors_tgt = {i2w_tgt[i] for i in top_k_tgt_idx if i in i2w_tgt}
                
                # Check overlap: how many src neighbors have their candidate target in the tgt neighbors?
                overlap_count = 0
                for n_src in neighbors_src:
                    if n_src in candidate_map and candidate_map[n_src] in neighbors_tgt:
                        overlap_count += 1
                        
                # Overlap ratio relative to k
                overlap_ratio = overlap_count / float(k)
                if overlap_ratio >= min_overlap:
                    valid_pairs.append((w_src, w_tgt))
                    
            logger.info(f"Validation complete. {len(valid_pairs)} pairs passed min_overlap {min_overlap}.")
            return valid_pairs
        except Exception as e:
            logger.error(f"Error during anchor validation: {e}")
            raise

    def discover_and_compute_rotation(self, vocab_src: List[str], vocab_tgt: List[str],
                                       emb_src: np.ndarray, emb_tgt: np.ndarray,
                                       w2i_src: Dict[str, int], w2i_tgt: Dict[str, int],
                                       counts_src: Dict[str, int] = None, 
                                       counts_tgt: Dict[str, int] = None) -> Tuple[np.ndarray, int]:
        """Full pipeline: discover anchors → validate → compute Procrustes rotation.
        
        Returns (R_matrix, num_valid_anchors)
        """
        logger.info("Starting automatic anchor discovery and rotation computation.")
        
        try:
            # 1. Discover cognates
            cognate_triples = self.discover_cognates(vocab_src, vocab_tgt)
            candidate_pairs = [(src, tgt) for src, tgt, _ in cognate_triples]
            
            # 2. Discover by frequency rank (if counts are provided)
            if counts_src and counts_tgt:
                freq_pairs = self.discover_by_frequency_rank(vocab_src, vocab_tgt, counts_src, counts_tgt)
                # Merge and deduplicate
                seen_src = {src for src, _ in candidate_pairs}
                for src, tgt in freq_pairs:
                    if src not in seen_src:
                        candidate_pairs.append((src, tgt))
                        seen_src.add(src)
                        
            # 3. Validate anchors
            valid_anchors = self.validate_anchors(candidate_pairs, emb_src, emb_tgt, w2i_src, w2i_tgt)
            
            if len(valid_anchors) < MIN_ANCHORS_REQUIRED:
                raise ValueError(f"Not enough valid anchors found ({len(valid_anchors)} < {MIN_ANCHORS_REQUIRED}).")
                
            # 4. Prepare anchor matrices
            source_anchors_list = []
            target_anchors_list = []
            
            for src_word, tgt_word in valid_anchors:
                source_anchors_list.append(emb_src[w2i_src[src_word]])
                target_anchors_list.append(emb_tgt[w2i_tgt[tgt_word]])
                
            A = np.array(source_anchors_list)
            B = np.array(target_anchors_list)
            
            # 5. Compute Procrustes rotation directly (avoids circular import)
            C = np.dot(A.T, B)
            U, _S, Vt = la.svd(C)
            R_matrix = np.dot(U, Vt)
            
            logger.info(f"Successfully computed rotation matrix using {len(valid_anchors)} anchors.")
            return R_matrix, len(valid_anchors)
            
        except Exception as e:
            logger.error(f"Error computing rotation: {e}")
            raise
