"""Yerleşik kaynak modülleri için çalıştırılabilir paket köprüsü.

Kaynakların geriye dönük konumunu korurken ``python -m ridm_ultra.cli``
çalışmasını sağlar.
"""
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
# Önce gerçek alt paketler (ör. ``ridm_ultra.llm``), sonra geriye dönük düz
# kaynak modülleri aranır.
__path__ = [str(_PACKAGE_DIR), str(_PACKAGE_DIR.parent)]

# Source tree is intentionally flat for backwards compatibility; expose the
# same public API from the installable package name as well.
# ruff: noqa: E402
from .attention import ClosedFormAttention, CooccurrenceRelationBasis, LearnedRelationAttention, TransformerBlockStack
from .backend import BackendInfo, ComputeBackend
from .benchmark import BenchmarkRunner
from .calibration import AutoregressiveCalibrator, NeuralReranker
from .constants import SENTENCE_BOUNDARY_TOKENS, UNK_TOKEN
from .core import RIDM
from .corpus import build_default_corpus
from .datasets import DatasetSplit, TextDataset
from .deep_understanding import DeepGenerator, WordTypeModel, CoherenceScorer, build_deep_understanding
from .graph_retrieval import LSHIndex, SemanticGraph, SimpleRAG
from .hierarchical import HierarchicalContextMemory, adaptive_context
from .hybrid import HybridLM
from .memory import ProductKeyMemory, SparseDistributedMemory
from .moe import MixtureOfExperts
from .multisense import MultiSenseEmbedding
from .ngram import NgramBaseline
from .reasoning import ReasoningChain, ReasoningController
from .reservoir import DeepReservoirScorer, DeepReservoirStack
from .subword import SubwordHasher
from .tokenizer import BPETokenizer
from .training import Trainer, TrainingConfig, TrainingResult
from .vocab import build_vocab, encode, train_test_split

__all__ = [
    "UNK_TOKEN", "SENTENCE_BOUNDARY_TOKENS", "build_vocab", "encode", "train_test_split",
    "SubwordHasher", "BPETokenizer", "RIDM", "SparseDistributedMemory", "ProductKeyMemory",
    "NgramBaseline", "HierarchicalContextMemory", "adaptive_context", "SemanticGraph", "LSHIndex",
    "SimpleRAG", "ClosedFormAttention", "CooccurrenceRelationBasis", "LearnedRelationAttention",
    "TransformerBlockStack", "DeepReservoirStack", "DeepReservoirScorer", "MultiSenseEmbedding",
    "ReasoningChain", "ReasoningController", "AutoregressiveCalibrator", "NeuralReranker",
    "MixtureOfExperts", "HybridLM", "build_default_corpus", "BenchmarkRunner", "ComputeBackend",
    "BackendInfo", "TextDataset", "DatasetSplit", "Trainer", "TrainingConfig", "TrainingResult",
    "DeepGenerator", "WordTypeModel", "CoherenceScorer", "build_deep_understanding",
]
