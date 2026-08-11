"""Pivot Chain for low-resource language support via UMA.

For languages without direct anchor pairs to English,
chain through intermediate pivot languages.
Example: Kazakh → Turkish → English
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class UMAPivotChain:
    """Chain UMA rotations through pivot languages for low-resource support."""
    
    # Known pivot paths (target is always English)
    PIVOT_PATHS: Dict[str, List[str]] = {
        "kk": ["tr", "en"],     # Kazakh → Turkish → English
        "az": ["tr", "en"],     # Azerbaijani → Turkish → English
        "uz": ["tr", "en"],     # Uzbek → Turkish → English
        "tk": ["tr", "en"],     # Turkmen → Turkish → English
        "ky": ["tr", "en"],     # Kyrgyz → Turkish → English
        "ar": ["en"],           # Arabic → English (direct)
        "fa": ["en"],           # Persian → English
        "de": ["en"],           # German → English
        "fr": ["en"],           # French → English
        "es": ["en"],           # Spanish → English
        "it": ["en"],           # Italian → English  
        "pt": ["en"],           # Portuguese → English
        "ru": ["en"],           # Russian → English
        "ja": ["en"],           # Japanese → English
        "ko": ["en"],           # Korean → English
        "zh": ["en"],           # Chinese → English
        "tr": ["en"],           # Turkish → English (direct)
    }
    
    def __init__(self, rotation_dir: str = "artifacts"):
        self.rotation_dir = Path(rotation_dir)
        self.rotation_dir.mkdir(parents=True, exist_ok=True)
        self._rotation_cache: Dict[str, np.ndarray] = {}
    
    def _load_rotation(self, lang_code: str) -> Optional[np.ndarray]:
        """Load a cached rotation matrix from disk."""
        if lang_code in self._rotation_cache:
            return self._rotation_cache[lang_code]
            
        file_path = self.rotation_dir / f"rotation_{lang_code}.npy"
        try:
            if file_path.exists():
                R = np.load(file_path)
                self._rotation_cache[lang_code] = R
                return R
            else:
                logger.warning(f"Rotation matrix for {lang_code} not found at {file_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load rotation for {lang_code}: {e}")
            return None
        
    def _save_rotation(self, lang_code: str, R: np.ndarray) -> None:
        """Save rotation matrix to disk."""
        file_path = self.rotation_dir / f"rotation_{lang_code}.npy"
        try:
            np.save(file_path, R)
            self._rotation_cache[lang_code] = R
            logger.info(f"Saved rotation matrix for {lang_code} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save rotation for {lang_code}: {e}")
            raise
    
    def get_pivot_path(self, lang_code: str) -> List[str]:
        """Get the pivot chain for a given language."""
        if lang_code not in self.PIVOT_PATHS:
            logger.warning(f"No pivot path defined for language: {lang_code}")
            return []
        return self.PIVOT_PATHS[lang_code]
        
    def compute_chained_rotation(self, lang_code: str,
                                  rotation_matrices: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the composite rotation R_total = R_1 · R_2 · ... · R_n"""
        path = self.get_pivot_path(lang_code)
        if not path:
            raise ValueError(f"Cannot compute chained rotation: no path for {lang_code}")
            
        logger.info(f"Computing chained rotation for {lang_code} via {path}")
        
        # Start with identity matrix. The dimension is inferred from the first available matrix
        # Let's find the first valid matrix to get the dimensions
        dim = None
        for p in path:
            if p in rotation_matrices:
                dim = rotation_matrices[p].shape[0]
                break
        
        if dim is None:
            # Check the source language itself if provided in the dict
            if lang_code in rotation_matrices:
                dim = rotation_matrices[lang_code].shape[0]
            else:
                raise ValueError("No rotation matrices provided to determine dimensions.")
                
        R_total = np.eye(dim)
        
        # Multiply through the chain
        # Note: If the path is ["tr", "en"], we need the rotation from 'lang_code' to 'tr',
        # and then 'tr' to 'en'.
        
        current_lang = lang_code
        for next_lang in path:
            # Look for the transition matrix from current_lang to next_lang
            # Assumes rotation_matrices contains these transition steps
            # or the direct matrix is named after the source language.
            # Here we assume rotation_matrices[lang] gives the matrix from 'lang' to its next pivot.
            if current_lang in rotation_matrices:
                R_step = rotation_matrices[current_lang]
                R_total = np.dot(R_total, R_step)
            else:
                # Attempt to load from disk if not in memory
                R_step = self._load_rotation(current_lang)
                if R_step is not None:
                    R_total = np.dot(R_total, R_step)
                else:
                    raise KeyError(f"Missing rotation matrix for step {current_lang} -> {next_lang}")
                    
            current_lang = next_lang
            
        return R_total
        
    def transform_vector(self, lang_code: str, vector: np.ndarray,
                         rotation_matrices: Dict[str, np.ndarray]) -> np.ndarray:
        """Transform a vector from source language to English via pivot chain."""
        try:
            R_total = self.compute_chained_rotation(lang_code, rotation_matrices)
            return np.dot(vector, R_total)
        except Exception as e:
            logger.error(f"Transformation failed for {lang_code}: {e}")
            raise
    
    def get_supported_languages(self) -> List[str]:
        """Return list of all supported language codes."""
        return list(self.PIVOT_PATHS.keys())
