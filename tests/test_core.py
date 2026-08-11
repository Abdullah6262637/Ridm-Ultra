"""Unit tests for RIDM SVD Core & ComputeBackend."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from ridm_ultra.backend import ComputeBackend
from ridm_ultra.core import RIDM


def test_compute_backend_info():
    backend = ComputeBackend(backend="auto", device="cpu")
    info = backend.info
    assert info.requested == "auto"
    assert info.device == "cpu"
    assert info.threads >= 1


def test_ridm_fit_and_finalize(sample_vocab, sample_token_ids):
    model = RIDM(vocab=sample_vocab, dim=32, window=3, seed=42)
    model.partial_fit(sample_token_ids)
    assert model.total_contexts > 0

    model.finalize(k=16)
    assert model.word_emb is not None
    assert model.word_emb.shape == (len(sample_vocab), min(16, len(sample_vocab)))
    assert model.k == min(16, len(sample_vocab))



def test_ridm_prediction_and_evaluation(sample_ridm, sample_token_ids):
    topk = sample_ridm.predict_topk(["the", "cat"], k=3)
    assert len(topk) == 3
    assert isinstance(topk[0][0], str)

    probs = sample_ridm.softmax_probs([1, 2])
    assert len(probs) == sample_ridm.V
    assert np.isclose(probs.sum(), 1.0)

    eval_res = sample_ridm.evaluate(sample_token_ids)
    assert "accuracy" in eval_res
    assert "perplexity" in eval_res


def test_ridm_incremental_update_and_drift(sample_ridm):
    drift_before = sample_ridm.drift_estimate([1, 2, 3])
    assert not np.isnan(drift_before)

    # Perform incremental update with new token sequence
    sample_ridm.incremental_update([1, 3, 5, 7])
    drift_after = sample_ridm.drift_estimate([1, 2, 3])
    assert not np.isnan(drift_after)


def test_ridm_serialization(sample_ridm):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.npz"
        sample_ridm.save(save_path)
        assert save_path.exists()

        loaded_model = RIDM.load(save_path)
        assert loaded_model.V == sample_ridm.V
        assert loaded_model.dim == sample_ridm.dim
        assert loaded_model.k == sample_ridm.k
