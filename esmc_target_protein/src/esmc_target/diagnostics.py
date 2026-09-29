"""Scientific diagnostics for a completed protein-embedding run (Section 21A).

Diagnostics produce **warnings, never failures**, and their thresholds are
configurable and uncalibrated. Nothing here modifies a scientific artifact.

a. Distribution of embedding L2 norms across unique proteins.
b. Pairwise cosine similarity across unique proteins (collapse detection).
c. Determinism: re-encode three proteins in-process and compare to the cache.
d. Chunking consistency: normal encode vs a forced small ``chunk_size``.
e. Biological sanity report: nearest/farthest protein pairs, labelled only with
   columns present in the input (no external annotation lookups).
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .utils import get_logger, sha256_text

LOGGER = get_logger("esmc_target.diagnostics")


def _percentiles(values: Sequence[float], points: Sequence[float] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {f"p{p}": float(torch.quantile(tensor, p / 100.0).item()) for p in points}


def norm_distribution(embeddings: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """(a) L2-norm distribution + per-protein scalar diagnostics."""
    norms: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    for vector in embeddings.values():
        flat = vector.detach().to(torch.float32).flatten()
        norms.append(float(flat.norm().item()))
        means.append(float(flat.mean().item()))
        stds.append(float(flat.std(unbiased=False).item()))
    if not norms:
        return {"count": 0}
    return {
        "count": len(norms),
        "l2_norm": {
            "min": min(norms),
            "median": float(torch.tensor(norms, dtype=torch.float64).median().item()),
            "max": max(norms),
            "mean": sum(norms) / len(norms),
            "std": float(torch.tensor(norms, dtype=torch.float64).std(unbiased=False).item()),
            **_percentiles(norms),
        },
        "embedding_mean": {"min": min(means), "max": max(means), "mean": sum(means) / len(means)},
        "embedding_std": {"min": min(stds), "max": max(stds), "mean": sum(stds) / len(stds)},
    }


def pairwise_cosine_stats(embeddings: Mapping[str, torch.Tensor], max_pairs: int = 20000,
                          collapse_threshold: float = 0.995, seed: int = 0) -> Dict[str, Any]:
    """(b) Scalable Pairwise Cosine Similarity (collapse detection).

    If N <= 1000: exact pairwise evaluation across all pairs.
    If N > 1000: NEVER construct full N×N matrix. Deterministically sample pairs with fixed seed.
    """
    keys = sorted(embeddings)
    N = len(keys)
    if N < 2:
        return {"count": 0, "N": N, "note": "fewer than two unique proteins; no pairs to compare"}

    matrix = torch.stack([embeddings[k].detach().to(torch.float32).flatten() for k in keys], dim=0)
    norms = matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = matrix / norms
    total_possible = N * (N - 1) // 2

    values: List[float] = []
    method = "exact"

    if N <= 1000:
        method = "exact"
        for i in range(N):
            for j in range(i + 1, N):
                values.append(float(torch.dot(unit[i], unit[j]).item()))
    else:
        method = "sampled"
        g = torch.Generator()
        g.manual_seed(seed)
        num_samples = min(max_pairs, total_possible)
        sampled_i = torch.randint(0, N, (num_samples * 3,), generator=g)
        sampled_j = torch.randint(0, N, (num_samples * 3,), generator=g)
        
        seen_pairs = set()
        for idx in range(len(sampled_i)):
            i_idx = int(sampled_i[idx].item())
            j_idx = int(sampled_j[idx].item())
            if i_idx == j_idx:
                continue
            pair = (min(i_idx, j_idx), max(i_idx, j_idx))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            val = float(torch.dot(unit[pair[0]], unit[pair[1]]).item())
            values.append(val)
            if len(values) >= num_samples:
                break

    if not values:
        return {"count": 0, "N": N, "total_possible_pairs": total_possible}

    median = float(torch.tensor(values, dtype=torch.float64).median().item())
    high = sum(1 for v in values if v >= collapse_threshold) / len(values)
    warnings: List[str] = []
    if high > 0.99:
        warnings.append(
            f"POSSIBLE EMBEDDING COLLAPSE: {high * 100:.2f}% of protein pairs have cosine "
            f">= {collapse_threshold}. This pattern usually indicates a pooling or alignment bug "
            f"rather than biology. Threshold is configurable and uncalibrated.")
    return {
        "count": len(values),
        "pairwise_method": method,
        "seed": seed,
        "sample_size": len(values),
        "pair_count": len(values),
        "N": N,
        "total_possible_pairs": total_possible,
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": median,
        "fraction_above_threshold": high,
        "collapse_threshold": collapse_threshold,
        **_percentiles(values),
        "warnings": warnings,
    }


def biological_sanity_report(embeddings: Mapping[str, torch.Tensor], labels: Mapping[str, Dict[str, Any]],
                             top_k: int = 10) -> Dict[str, Any]:
    """(e) Nearest/farthest pairs, labelled from input columns only. Scalable for N > 1000."""
    keys = sorted(embeddings)
    N = len(keys)
    if N < 2:
        return {"note": "fewer than two unique proteins"}

    matrix = torch.stack([embeddings[k].detach().to(torch.float32).flatten() for k in keys], dim=0)
    unit = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)

    if N <= 1000:
        sims = unit @ unit.T
        pairs: List[Tuple[float, str, str]] = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((float(sims[i, j].item()), keys[i], keys[j]))
        pairs.sort(reverse=True)
        nearest = pairs[:top_k]
        farthest = [(s, a, b) for s, a, b in reversed(pairs[-top_k:])]
    else:
        # Bounded streaming top-k using min/max heaps for N > 1000
        nearest_heap: List[Tuple[float, str, str]] = []  # min-heap storing (sim, a, b)
        farthest_heap: List[Tuple[float, str, str]] = [] # min-heap storing (-sim, a, b)
        block_size = 250
        for start_i in range(0, N, block_size):
            end_i = min(start_i + block_size, N)
            block = unit[start_i:end_i]  # [block_size, hidden]
            sims_block = block @ unit.T   # [block_size, N]
            for i_local, i_global in enumerate(range(start_i, end_i)):
                for j_global in range(i_global + 1, N):
                    sim = float(sims_block[i_local, j_global].item())
                    ki, kj = keys[i_global], keys[j_global]

                    # Push to nearest heap
                    if len(nearest_heap) < top_k:
                        heapq.heappush(nearest_heap, (sim, ki, kj))
                    elif sim > nearest_heap[0][0]:
                        heapq.heappushpop(nearest_heap, (sim, ki, kj))

                    # Push to farthest heap
                    neg_sim = -sim
                    if len(farthest_heap) < top_k:
                        heapq.heappush(farthest_heap, (neg_sim, ki, kj))
                    elif neg_sim > farthest_heap[0][0]:
                        heapq.heappushpop(farthest_heap, (neg_sim, ki, kj))

        nearest = sorted(nearest_heap, reverse=True)
        farthest = [(-neg_sim, a, b) for neg_sim, a, b in sorted(farthest_heap, reverse=True)]

    def describe(a: str, b: str, similarity: float) -> Dict[str, Any]:
        return {
            "cosine_similarity": similarity,
            "a": {"sequence_hash": a, **labels.get(a, {})},
            "b": {"sequence_hash": b, **labels.get(b, {})},
            "same_gene_symbol": bool(labels.get(a, {}).get("gene_symbol")
                                     and labels.get(a, {}).get("gene_symbol") == labels.get(b, {}).get("gene_symbol")),
        }

    return {
        "nearest_pairs": [describe(a, b, s) for s, a, b in nearest],
        "farthest_pairs": [describe(a, b, s) for s, a, b in farthest],
        "k": top_k,
        "N": N,
        "interpretation": "informational only; an expert should check that paralogs/family members "
                          "in the data trend closer than unrelated proteins. No external "
                          "annotations were fetched; labels come from input columns.",
    }


def determinism_check(encoder: Any, cache_entries: Mapping[str, Mapping[str, Any]],
                      sequences_by_hash: Mapping[str, str],
                      pick_for_determinism: Sequence[Any] = ()) -> Dict[str, Any]:
    """(c) In-process determinism check: re-encode picked sequences and compare to cache."""
    if encoder is None or not cache_entries or not sequences_by_hash:
        return {"status": "skipped", "reason": "encoder or cache or sequence mapping unavailable"}
    
    tested = 0
    passed = 0
    details = []
    for item in pick_for_determinism:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            label, seq = item
        else:
            label, seq = str(item), str(item)
        
        seq_hash = sha256_text(seq)
        cached_entry = cache_entries.get(seq_hash)
        if cached_entry is None or "embedding" not in cached_entry:
            continue
        
        try:
            encoding = encoder.encode(seq)
            cached_emb = cached_entry["embedding"]
            diff = float((encoding.embedding - cached_emb).abs().max().item())
            is_identical = bool(torch.allclose(encoding.embedding, cached_emb, atol=1e-5))
            tested += 1
            if is_identical:
                passed += 1
            details.append({
                "label": label,
                "sequence_hash": seq_hash,
                "max_abs_diff": diff,
                "deterministic": is_identical,
            })
        except Exception as exc:
            details.append({
                "label": label,
                "sequence_hash": seq_hash,
                "error": str(exc),
                "deterministic": False,
            })
    return {
        "status": "success" if (tested == 0 or passed == tested) else "warning",
        "tested_count": tested,
        "passed_count": passed,
        "details": details,
    }


def chunking_sensitivity_diagnostic(encoder: Any, probe_sequence: str,
                                    forced_chunk_size: int = 512) -> Dict[str, Any]:
    """(d) Chunking sensitivity diagnostic: encode with normal vs forced small chunk_size.
    Different chunk sizes legitimately change contextual representations, so a low cosine is
    informational, not a correctness failure.
    """
    if encoder is None or not probe_sequence:
        return {"status": "skipped", "reason": "encoder or probe sequence unavailable"}
    
    try:
        normal = encoder.encode(probe_sequence)
        forced = encoder.encode(probe_sequence, force_chunk_size=forced_chunk_size)
        diff = float((normal.embedding - forced.embedding).abs().max().item())
        cos_sim = float(torch.dot(normal.embedding / normal.embedding.norm(),
                                  forced.embedding / forced.embedding.norm()).item())
        return {
            "status": "success",
            "normal_chunks": normal.num_chunks,
            "forced_chunks": forced.num_chunks,
            "max_abs_diff": diff,
            "cosine_similarity": cos_sim,
            "note": "chunking sensitivity is informational only; different context windows legitimately alter embeddings",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
        }


def chunking_consistency_check(encoder: Any, probe_sequence: str,
                                forced_chunk_size: int = 512) -> Dict[str, Any]:
    """Alias for backwards compatibility."""
    return chunking_sensitivity_diagnostic(encoder, probe_sequence, forced_chunk_size=forced_chunk_size)


def run_diagnostics(embeddings: Mapping[str, torch.Tensor], encoder: Any,
                    cache_entries: Mapping[str, Mapping[str, Any]],
                    sequences_by_hash: Mapping[str, str],
                    labels: Mapping[str, Dict[str, Any]],
                    chunk_probe_sequence: Optional[str] = None,
                    forced_chunk_size: int = 512,
                    max_pairs: int = 20000,
                    collapse_threshold: float = 0.995,
                    pick_for_determinism: Sequence[str] = (),
                    seed: int = 0) -> Dict[str, Any]:
    """Assemble the full ``embedding_diagnostics.json`` payload, capturing failures explicitly."""
    from .utils import local_now_iso

    try:
        norms = norm_distribution(embeddings)
        cosines = pairwise_cosine_stats(embeddings, max_pairs=max_pairs,
                                        collapse_threshold=collapse_threshold, seed=seed)
        warnings: List[str] = list(cosines.get("warnings", []))
        if norms.get("count"):
            values = norms["l2_norm"]
            if values["std"] == 0.0 and values["min"] > 0:
                warnings.append("All protein embeddings share an identical L2 norm; check pooling.")
            if values["max"] > 0 and values["min"] / values["max"] < 1e-3:
                warnings.append("Embedding L2 norms span more than three orders of magnitude; check "
                                "for a truncated or mis-aligned chunk.")
        determinism = determinism_check(encoder, cache_entries, sequences_by_hash, pick_for_determinism)
        chunking = chunking_sensitivity_diagnostic(encoder, chunk_probe_sequence or "",
                                                    forced_chunk_size=forced_chunk_size) if chunk_probe_sequence else {}
        biology = biological_sanity_report(embeddings, labels)
        return {
            "status": "success",
            "embeddings_modified": False,
            "timestamp": local_now_iso(),
            "diagnostic_version": "1.0",
            "norm_distribution": norms,
            "pairwise_cosine_similarity": cosines,
            "determinism_check": determinism,
            "chunking_sensitivity_diagnostic": chunking,
            "chunking_consistency_check": chunking,
            "biological_sanity_report": biology,
            "warnings": warnings,
            "thresholds": {
                "cosine_collapse_threshold": collapse_threshold,
                "max_pairs": max_pairs,
                "forced_chunk_size": forced_chunk_size,
                "seed": seed,
            },
            "note": "diagnostics only; warnings never modify or invalidate a scientific artifact",
        }
    except Exception as exc:
        LOGGER.warning("Diagnostics evaluation encountered an error: %s: %s", type(exc).__name__, exc)
        return {
            "status": "failed",
            "embeddings_modified": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timestamp": local_now_iso(),
            "diagnostic_version": "1.0",
            "warnings": [f"Diagnostic evaluation failed: {type(exc).__name__}: {exc}"],
        }
