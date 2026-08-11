from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

from ridm_ultra.llm.uma_auto_anchor import AutoAnchorDiscovery
from ridm_ultra.llm.uma_pivot_chain import UMAPivotChain

import logging

logger = logging.getLogger(__name__)


class UniversalManifoldAligner:
    """
    Phase 3: Universal Manifold Alignment (UMA)
    Mathematically rotates the SVD manifold of one language to perfectly align
    with the Anchor Space (e.g. English or a pure Latent Space) using Orthogonal Procrustes.
    
    Supports:
    - Manual anchor alignment (calculate_orthogonal_procrustes)
    - Automatic anchor discovery (auto_discover_and_align)
    - Pivot chain for low-resource languages (transform_with_pivot)
    """
    def __init__(self):
        self.rotation_matrices: Dict[str, np.ndarray] = {}
        self.anchor_discovery = AutoAnchorDiscovery()
        self.pivot_chain = UMAPivotChain()

    def calculate_orthogonal_procrustes(self, source_anchors: np.ndarray, target_anchors: np.ndarray) -> np.ndarray:
        """
        Given paired anchor vectors from two languages, computes the optimal rotation matrix R
        such that source_anchors * R is approximately equal to target_anchors.
        
        Args:
            source_anchors: (N, D) array of vectors in Language A
            target_anchors: (N, D) array of vectors in Language B (Anchor Space)
            
        Returns:
            R: (D, D) Orthogonal rotation matrix
        """
        assert source_anchors.shape == target_anchors.shape, "Anchor matrices must have the same shape."

        # SVD of A^T B
        C = np.dot(source_anchors.T, target_anchors)
        U, S, Vt = la.svd(C)

        # Optimal rotation matrix R = U * V^T
        R = np.dot(U, Vt)
        return R

    def align_language(self, lang_code: str, source_vectors: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Applies the rotation matrix R to the entire vocabulary space of a language.
        """
        aligned_vectors = np.dot(source_vectors, R)
        self.rotation_matrices[lang_code] = R
        self.pivot_chain._save_rotation(lang_code, R)
        logger.info(f"UMA: Language '{lang_code}' aligned and rotation cached.")
        return aligned_vectors

    def transform_vector(self, lang_code: str, vector: np.ndarray) -> np.ndarray:
        """
        Transforms a single vector (e.g., during query time) to the universal coordinate space.
        Falls back to pivot chain if direct rotation is unavailable.
        """
        if lang_code in self.rotation_matrices:
            return np.dot(vector, self.rotation_matrices[lang_code])

        # Try pivot chain
        pivot_R = self.pivot_chain._load_rotation(lang_code)
        if pivot_R is not None:
            self.rotation_matrices[lang_code] = pivot_R
            return np.dot(vector, pivot_R)

        # Try chained rotation
        if lang_code in self.pivot_chain.PIVOT_PATHS:
            chained_R = self.pivot_chain.compute_chained_rotation(lang_code, self.rotation_matrices)
            if chained_R is not None:
                self.rotation_matrices[lang_code] = chained_R
                return np.dot(vector, chained_R)

        return vector  # Assume already in universal space if no rotation found

    def auto_discover_and_align(
        self,
        lang_code: str,
        vocab_src: List[str],
        vocab_tgt: List[str],
        emb_src: np.ndarray,
        emb_tgt: np.ndarray,
        w2i_src: Dict[str, int],
        w2i_tgt: Dict[str, int],
        counts_src: Optional[Dict[str, int]] = None,
        counts_tgt: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Full automatic alignment pipeline:
        1. Discover anchor pairs automatically (cognates + frequency)
        2. Validate anchors with SVD neighborhood overlap
        3. Compute Procrustes rotation
        4. Cache and save the rotation matrix
        
        Returns:
            (R_matrix, num_valid_anchors)
        """
        R, num_anchors = self.anchor_discovery.discover_and_compute_rotation(
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            emb_src=emb_src,
            emb_tgt=emb_tgt,
            w2i_src=w2i_src,
            w2i_tgt=w2i_tgt,
            counts_src=counts_src,
            counts_tgt=counts_tgt,
        )

        self.rotation_matrices[lang_code] = R
        self.pivot_chain._save_rotation(lang_code, R)
        logger.info(f"UMA Auto-Align: '{lang_code}' aligned with {num_anchors} validated anchors.")
        return R, num_anchors

    def get_supported_languages(self) -> List[str]:
        """Return list of all languages with computed rotation matrices."""
        cached = set(self.rotation_matrices.keys())
        disk = set()
        for path in self.pivot_chain.rotation_dir.glob("rotation_*.npy"):
            lang = path.stem.replace("rotation_", "")
            disk.add(lang)
        return sorted(cached | disk)

