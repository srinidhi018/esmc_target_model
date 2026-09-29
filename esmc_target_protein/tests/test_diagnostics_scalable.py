"""Unit tests for scalable diagnostics, deterministic sampling, and failure output (Tests AU - AZ)."""

import pytest
import torch
from esmc_target.diagnostics import (
    biological_sanity_report,
    norm_distribution,
    pairwise_cosine_stats,
    run_diagnostics,
)


def test_diagnostics_exact_for_small_n():
    # Test AV: Exact pairwise diagnostics for N <= 1000
    embeddings = {
        f"seq_{i}": torch.randn(1152) for i in range(10)
    }
    res = pairwise_cosine_stats(embeddings, seed=0)
    assert res["pairwise_method"] == "exact"
    assert res["count"] == 10 * 9 // 2
    assert res["N"] == 10


def test_diagnostics_sampled_and_blockwise_for_large_n():
    # Test AU, AW, AX: For N > 1000, sampled pairs used, no N x N matrix constructed (blockwise)
    N = 1050
    torch.manual_seed(42)
    embeddings = {
        f"seq_{i}": torch.randn(128) for i in range(N)
    }
    
    # 1. Pairwise cosine stats
    res_cos = pairwise_cosine_stats(embeddings, max_pairs=500, seed=123)
    assert res_cos["pairwise_method"] == "sampled"
    assert res_cos["N"] == N
    assert res_cos["sample_size"] == 500
    assert res_cos["seed"] == 123

    # 2. Biological sanity report for large N
    labels = {f"seq_{i}": {"gene_symbol": f"GENE_{i}"} for i in range(N)}
    res_bio = biological_sanity_report(embeddings, labels, top_k=5)
    assert "nearest_pairs" in res_bio
    assert len(res_bio["nearest_pairs"]) == 5
    assert len(res_bio["farthest_pairs"]) == 5


def test_diagnostics_never_modify_embeddings():
    # Test AZ: Diagnostics never modify original tensors
    original_vec = torch.tensor([1.0, 2.0, 3.0])
    vec_clone = original_vec.clone()
    embeddings = {"seq_1": original_vec}

    norm_distribution(embeddings)
    assert torch.allclose(original_vec, vec_clone)


def test_diagnostic_failure_produces_artifact():
    # Test AY: Diagnostic failure writes JSON payload with status="failed" and embeddings_modified=False
    bad_embeddings = "not_a_mapping"  # Will trigger AttributeError/TypeError in run_diagnostics
    res = run_diagnostics(bad_embeddings, encoder=None, cache_entries={},
                          sequences_by_hash={}, labels={})
    assert res["status"] == "failed"
    assert res["embeddings_modified"] is False
    assert "error_type" in res
    assert "error" in res


def test_diagnostics_streaming_memory_bound_n3000():
    # Test streaming memory bound for N = 3000
    import tracemalloc

    N = 3000
    torch.manual_seed(42)
    embeddings = {f"seq_{i}": torch.randn(128) for i in range(N)}
    labels = {f"seq_{i}": {"gene_symbol": f"GENE_{i}"} for i in range(N)}

    tracemalloc.start()
    res = biological_sanity_report(embeddings, labels, top_k=10)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(res["nearest_pairs"]) == 10
    assert len(res["farthest_pairs"]) == 10
    # Peak memory allocated by Python structures during sanity report must stay under 45 MB
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 45.0, f"Peak memory was {peak_mb:.2f} MB, expected < 45 MB for streaming top-k"

