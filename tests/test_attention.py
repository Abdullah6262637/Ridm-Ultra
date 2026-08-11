"""Unit tests for ClosedFormAttention & Relation Basis modules."""
from __future__ import annotations

import numpy as np
from ridm_ultra.attention import (
    ClosedFormAttention,
    CooccurrenceRelationBasis,
    LearnedRelationAttention,
    TransformerBlockStack,
)
from ridm_ultra.reservoir import DeepReservoirScorer


def test_closed_form_attention():
    attn = ClosedFormAttention(dim=32, n_heads=4, seed=42)
    context_vecs = np.random.randn(5, 32).astype(np.float32)

    out, weights = attn.forward(context_vecs)
    assert out.shape == (32,)
    assert weights.shape == (5,)
    assert np.isclose(weights.sum(), 1.0)


def test_cooccurrence_relation_basis(sample_ridm, sample_token_ids):
    basis = CooccurrenceRelationBasis(sample_ridm, sample_token_ids, window=3, k=16, seed=42)
    assert basis.query_basis.shape[0] == sample_ridm.V
    assert basis.key_basis.shape[0] == sample_ridm.V

    rel_attn = LearnedRelationAttention(sample_ridm, basis)
    scores = rel_attn.forward([1, 2])
    assert scores.shape == (sample_ridm.V,)


def test_transformer_block_stack(sample_ridm, sample_token_ids):
    attentions = [ClosedFormAttention(dim=32, n_heads=4, seed=i) for i in range(2)]
    reservoir_scorer = DeepReservoirScorer(sample_ridm, hidden_dims=(32, 32))
    reservoir_scorer.fit(sample_token_ids)

    stack = TransformerBlockStack(dim=32, attention_layers=attentions, ffn_stack=reservoir_scorer.stack, n_blocks=2)
    cvecs = np.random.randn(4, 32).astype(np.float32)
    out = stack.forward(cvecs)
    assert out.shape == (32,)
