"""Vocabulary Sharding for large-scale RIDM Ultra deployments.

Splits large embedding matrices into semantic clusters (shards) for 
memory-efficient retrieval. Each shard is stored as a separate .npy file
and loaded on-demand via memory mapping.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

# Constants
DEFAULT_NUM_SHARDS = 8
SHARD_PREFIX = "emb_shard"
METADATA_FILE = "shard_metadata.json"

class ShardManager:
    """Manages vocabulary embedding shards for memory-efficient retrieval."""
    
    def __init__(self, shard_dir: str = "artifacts/shards"):
        self.shard_dir = Path(shard_dir)
        self.shard_metadata: Dict[int, Dict] = {}  # shard_id -> {vocab_indices, centroid}
        self.centroids: Optional[np.ndarray] = None
        self._loaded_shards: Dict[int, np.ndarray] = {}  # Cache
        self.vocab: Optional[List[str]] = None
    
    def create_shards(self, word_emb: np.ndarray, vocab: List[str],
                      num_shards: int = DEFAULT_NUM_SHARDS) -> None:
        """Split embeddings into semantic clusters using mini-batch k-means.
        
        Algorithm:
        1. Run MiniBatchKMeans on word_emb to get cluster assignments
        2. For each cluster, save the sub-matrix as shard_{id}.npy
        3. Save metadata (which vocab indices belong to which shard, centroids)
        """
        logger.info(f"Creating {num_shards} shards for vocabulary of size {len(vocab)}...")
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = vocab
        
        kmeans = MiniBatchKMeans(n_clusters=num_shards, random_state=42, n_init=3)
        labels = kmeans.fit_predict(word_emb)
        self.centroids = kmeans.cluster_centers_
        
        metadata = {
            "num_shards": num_shards,
            "vocab": vocab,
            "shards": {}
        }
        
        for i in range(num_shards):
            # Find indices for this cluster
            indices = np.where(labels == i)[0]
            shard_data = word_emb[indices]
            
            # Save shard
            shard_path = self.shard_dir / f"{SHARD_PREFIX}_{i}.npy"
            np.save(shard_path, shard_data)
            
            # Save metadata for this shard
            metadata["shards"][str(i)] = {
                "indices": indices.tolist(),
                "centroid": self.centroids[i].tolist()
            }
            logger.info(f"Saved shard {i} with {len(indices)} items.")
            
        # Save main metadata
        meta_path = self.shard_dir / METADATA_FILE
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
            
        self._sync_metadata(metadata)
        logger.info("Sharding complete.")

    def _sync_metadata(self, metadata: Dict) -> None:
        self.shard_metadata = {}
        self.vocab = metadata.get("vocab", [])
        num_shards = metadata.get("num_shards", 0)
        
        centroids_list = []
        for i in range(num_shards):
            s_id = str(i)
            if s_id in metadata.get("shards", {}):
                self.shard_metadata[i] = {
                    "indices": metadata["shards"][s_id]["indices"],
                    "centroid": np.array(metadata["shards"][s_id]["centroid"], dtype=np.float32)
                }
                centroids_list.append(self.shard_metadata[i]["centroid"])
        
        if centroids_list:
            self.centroids = np.vstack(centroids_list)

    def load_metadata(self) -> bool:
        """Load shard metadata from disk."""
        meta_path = self.shard_dir / METADATA_FILE
        if not meta_path.exists():
            logger.warning(f"Shard metadata not found at {meta_path}")
            return False
            
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self._sync_metadata(metadata)
            logger.info(f"Loaded metadata for {len(self.shard_metadata)} shards.")
            return True
        except Exception as e:
            logger.error(f"Error loading shard metadata: {e}")
            return False

    def _find_relevant_shards(self, query_vec: np.ndarray, top_k_shards: int = 3) -> List[int]:
        """Find most relevant shards by cosine similarity to centroids."""
        if self.centroids is None or len(self.shard_metadata) == 0:
            return []
            
        # Normalize centroids and query vector
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_centroids = self.centroids / norms
        
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        norm_query = query_vec / q_norm
        
        sims = np.dot(norm_centroids, norm_query)
        top_indices = np.argsort(sims)[::-1][:top_k_shards]
        
        return top_indices.tolist()

    def _load_shard(self, shard_id: int) -> np.ndarray:
        """Load a shard with memory mapping."""
        if shard_id in self._loaded_shards:
            return self._loaded_shards[shard_id]
            
        shard_path = self.shard_dir / f"{SHARD_PREFIX}_{shard_id}.npy"
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
            
        try:
            # Load with mmap to save RAM
            shard_data = np.load(shard_path, mmap_mode='r')
            self._loaded_shards[shard_id] = shard_data
            return shard_data
        except Exception as e:
            logger.error(f"Failed to load shard {shard_id}: {e}")
            raise

    def query(self, query_vec: np.ndarray, top_k: int = 10,
              vocab: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Query across relevant shards for nearest neighbors.
        
        1. Find top 3 relevant shards by centroid similarity
        2. Load those shards (mmap)
        3. Compute cosine similarity within each shard
        4. Merge results and return top_k
        """
        active_vocab = vocab if vocab is not None else self.vocab
        if not active_vocab:
            logger.warning("No vocabulary available for shard querying.")
            return []
            
        relevant_shards = self._find_relevant_shards(query_vec, top_k_shards=3)
        if not relevant_shards:
            return []
            
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        norm_query = query_vec / q_norm
            
        all_results = []
        
        for shard_id in relevant_shards:
            shard_data = self._load_shard(shard_id)
            indices = self.shard_metadata[shard_id]["indices"]
            
            # Compute similarities
            norms = np.linalg.norm(shard_data, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_shard = shard_data / norms
            
            sims = np.dot(norm_shard, norm_query)
            
            # Get top k from this shard
            local_k = min(top_k, len(sims))
            top_local_idx = np.argsort(sims)[::-1][:local_k]
            
            for idx in top_local_idx:
                global_idx = indices[idx]
                if global_idx < len(active_vocab):
                    word = active_vocab[global_idx]
                    score = float(sims[idx])
                    all_results.append((word, score))
                    
        # Sort combined results
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def get_stats(self) -> Dict:
        """Return shard statistics."""
        return {
            "num_shards": len(self.shard_metadata),
            "vocab_size": len(self.vocab) if self.vocab else 0,
            "loaded_shards": len(self._loaded_shards),
            "shard_sizes": {k: len(v["indices"]) for k, v in self.shard_metadata.items()}
        }
