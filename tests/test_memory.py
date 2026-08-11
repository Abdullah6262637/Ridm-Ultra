"""Unit tests for SparseDistributedMemory & ProductKeyMemory."""
from __future__ import annotations

import numpy as np
from ridm_ultra.memory import ProductKeyMemory, SparseDistributedMemory


def test_sparse_distributed_memory():
    sdm = SparseDistributedMemory(addr_dim=32, content_dim=16, n_locations=128, seed=42)
    addr = np.random.randn(32).astype(np.float32)
    content = np.random.randn(16).astype(np.float32)

    # Initial read (empty memory)
    empty_read = sdm.read(addr)
    assert empty_read.shape == (16,)

    # Write and read back
    sdm.write(addr, content, weight=1.5)
    assert sdm.coverage() > 0.0

    read_back = sdm.read(addr)
    assert read_back.shape == (16,)

    # Test Ebbinghaus decay
    sdm.decay(rate=0.5)
    decayed_read = sdm.read(addr)
    assert np.all(np.abs(decayed_read) <= np.abs(read_back))


def test_product_key_memory():
    pkm = ProductKeyMemory(dim=32, value_dim=32, n_sub=16, top_k=4, seed=42)
    q = np.random.randn(32).astype(np.float32)
    val = np.random.randn(32).astype(np.float32)

    # Write value
    pkm.write(q, val, weight=1.0)
    assert pkm.utilization() > 0.0

    # Read value
    recalled = pkm.read(q)
    assert recalled.shape == (32,)

    # Decay & consolidation
    pkm.decay(rate=0.1)
    pruned = pkm.consolidate(min_count=0.5)
    assert pruned >= 0
