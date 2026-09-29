#!/usr/bin/env python
"""Validate every primary artifact of a run and print VALIDATION PASSED/FAILED.

Checks (Section 18):
  * all primary artifacts exist and load;
  * drug_target_embeddings_*.pt is a plain dict of Tensor[dim] float32;
  * the cache is readable and its fingerprint matches the manifest;
  * provenance row counts match; successful + failed == input rows;
  * every successful row's hash exists in the cache;
  * no NaN/Inf, norm > 0;
  * each drug embedding equals the recomputed mean of that drug's unique
    successful protein embeddings;
  * drugs_without_successful_targets are absent from the dict;
  * the schema json exists and matches the .pt file and the manifest;
  * row_embeddings / row_embedding_row_indices / row_valid_mask are mutually
    consistent (compact successful-only layout).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from esmc_target.identifiers import canonicalize_nsc_id  # noqa: E402
from esmc_target.utils import setup_logging  # noqa: E402


class Validator:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.checks: List[str] = []

    def ok(self, message: str) -> None:
        self.checks.append(f"PASS  {message}")

    def fail(self, message: str) -> None:
        self.errors.append(f"FAIL  {message}")

    def check(self, condition: bool, message: str) -> bool:
        (self.ok if condition else self.fail)(message)
        return bool(condition)

    def report(self) -> int:
        for line in self.checks:
            print(line)
        for line in self.errors:
            print(line)
        print()
        if self.errors:
            print(f"VALIDATION FAILED ({len(self.errors)} problem(s), {len(self.checks)} check(s) passed)")
            return 1
        print(f"VALIDATION PASSED ({len(self.checks)} checks)")
        return 0


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate(output_dir: Path) -> Validator:
    v = Validator()
    manifest_path = output_dir / "manifest.json"
    if not v.check(manifest_path.exists(), "manifest.json exists"):
        return v
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backbone_dim = int(manifest.get("backbone_embedding_dim", manifest.get("hidden_size", 1152)))
    v.check(backbone_dim == 1152, f"backbone_embedding_dim is 1152 (found {backbone_dim})")
    
    projection = bool(manifest.get("projection", {}).get("enabled", False))
    final_dim = int(manifest.get("final_output_dim", 256 if projection else backbone_dim))
    expected_name = f"drug_target_embeddings_{final_dim}.pt"

    drug_pt = output_dir / expected_name
    cache_pt = output_dir / "protein_cache.pt"
    audit_pt = output_dir / "target_embeddings.pt"
    schema_json = output_dir / "drug_target_embeddings_schema.json"
    row_csv = output_dir / "target_provenance.csv"
    drug_csv = output_dir / "drug_provenance.csv"
    failed_csv = output_dir / "failed_rows.csv"

    for label, path in (("drug embeddings", drug_pt), ("protein cache", cache_pt),
                        ("audit package", audit_pt), ("schema", schema_json),
                        ("row provenance", row_csv), ("drug provenance", drug_csv),
                        ("failed rows", failed_csv)):
        v.check(path.exists(), f"{label} artifact exists ({path.name})")
    if v.errors:
        return v

    drugs = torch.load(drug_pt, map_location="cpu", weights_only=False)
    v.check(isinstance(drugs, dict), f"{expected_name} is a plain dict")
    v.check(all(isinstance(k, str) for k in drugs), f"{expected_name} is keyed by string nsc_id")
    
    non_canonical_nsc = [k for k in drugs if canonicalize_nsc_id(k) != k]
    v.check(not non_canonical_nsc, f"all drug keys are canonical NSC IDs ({len(non_canonical_nsc)} non-canonical)")

    bad_shape = [k for k, t in drugs.items() if not (isinstance(t, torch.Tensor) and t.dim() == 1
                                                    and int(t.shape[0]) == final_dim)]
    v.check(not bad_shape, f"every drug vector is a 1-D Tensor[{final_dim}] ({len(drugs)} vectors)")
    bad_dtype = [k for k, t in drugs.items() if isinstance(t, torch.Tensor) and t.dtype != torch.float32]
    v.check(not bad_dtype, "every drug vector is float32")
    nonfinite = [k for k, t in drugs.items()
                 if isinstance(t, torch.Tensor) and not torch.isfinite(t).all()]
    v.check(not nonfinite, f"no NaN/Inf in drug embeddings ({len(nonfinite)} bad)")
    zero_norm = [k for k, t in drugs.items()
                 if isinstance(t, torch.Tensor) and float(t.norm().item()) <= 0]
    v.check(not zero_norm, f"every drug vector has norm > 0 ({len(zero_norm)} bad)")

    cache = torch.load(cache_pt, map_location="cpu", weights_only=False)
    v.check("fingerprint" in cache and "entries" in cache, "cache has fingerprint and entries")
    v.check(cache["fingerprint"].get("fingerprint_hash")
            == manifest["embedding_fingerprint"].get("fingerprint_hash"),
            "cache fingerprint matches the manifest")
    entries = cache["entries"]
    v.check(all(e["embedding"].shape == (backbone_dim,) for e in entries.values()),
            f"every cached protein embedding is Tensor[{backbone_dim}]")
    v.check(all(bool(torch.isfinite(e["embedding"]).all()) and float(e["embedding"].norm()) > 0
                for e in entries.values()), "cached embeddings are finite with norm > 0")

    rows = _read_csv_rows(row_csv)
    drug_rows = _read_csv_rows(drug_csv)
    failed = _read_csv_rows(failed_csv)
    n_success = sum(1 for r in rows if r["status"] == "success")
    n_failed = sum(1 for r in rows if r["status"] == "failed")
    v.check(n_success + n_failed == len(rows), "successful_rows + failed_rows == input rows")
    v.check(n_failed == len(failed), f"failed_rows.csv holds every failed row ({n_failed})")
    v.check(len(rows) == int(manifest["row_counts"]["input"]),
            "row provenance count matches the manifest input count")
    v.check(len(drug_rows) == int(manifest["dataset_statistics"]["unique_nsc_id"]),
            "drug provenance count matches the number of unique nsc_id")
    v.check(len(drugs) + len(manifest["drugs_without_successful_targets"]) == len(drug_rows),
            "drugs with embeddings + drugs without successful targets == all drugs")

    missing_hashes = [r["source_row_index"] for r in rows
                      if r["status"] == "success" and r["sequence_hash"] not in entries]
    v.check(not missing_hashes,
            f"every successful row's sequence_hash exists in the cache ({len(missing_hashes)} missing)")
    v.check(all(r.get("sequence_length_mismatch") in ("True", "False", "") for r in rows),
            "sequence_length_mismatch is recorded for every row")
    v.check(not [r for r in rows if r["status"] == "failed" and not r["error"]],
            "every failed row carries an error message")

    absent = [nsc for nsc in manifest["drugs_without_successful_targets"] if nsc in drugs]
    v.check(not absent,
            f"drugs without successful targets are absent from the dict ({len(absent)} present)")

    # recompute drug means from the cache, independently of the pipeline
    per_drug: Dict[str, Dict[str, torch.Tensor]] = {}
    for row in rows:
        if row["status"] != "success" or row["sequence_hash"] not in entries:
            continue
        per_drug.setdefault(row["nsc_id"], {})
        per_drug[row["nsc_id"]].setdefault(row["sequence_hash"], entries[row["sequence_hash"]]["embedding"])
    mismatched = []
    for nsc, unique in per_drug.items():
        if nsc not in drugs:
            mismatched.append(f"{nsc}: absent from dict")
            continue
        expected = torch.stack(list(unique.values()), dim=0).mean(dim=0)
        if not torch.allclose(expected, drugs[nsc], atol=1e-5, rtol=1e-4):
            mismatched.append(f"{nsc}: max|diff|={float((expected - drugs[nsc]).abs().max()):.3e}")
    v.check(not mismatched, f"drug embeddings equal the recomputed mean of unique proteins "
                            f"({len(mismatched)} mismatched) {mismatched[:3]}")
    v.check(len(per_drug) == len(drugs), "every drug with successful rows has an embedding")

    audit = torch.load(audit_pt, map_location="cpu", weights_only=False)
    for key in ("row_embeddings", "row_embedding_row_indices", "row_valid_mask", "row_metadata",
                "drug_embeddings", "drug_metadata", "config", "model_name", "model_revision",
                "embedding_dim", "projection_enabled", "aggregation_method"):
        v.check(key in audit, f"audit package contains '{key}'")
    v.check(int(audit["embedding_dim"]) == final_dim,
            "audit embedding_dim matches the artifact")
    first_key = next(iter(audit["drug_embeddings"]), None)
    if first_key is not None:
        v.check(int(audit["drug_embeddings"][first_key].shape[0]) == final_dim, "audit drug_embeddings vector dimension matches final_dim")
    row_emb = audit["row_embeddings"]
    idx = audit["row_embedding_row_indices"]
    mask = audit["row_valid_mask"]
    v.check(row_emb.dim() == 2 and int(row_emb.shape[1]) == backbone_dim, f"row_embeddings is [N_success, {backbone_dim}]")
    v.check(idx.dtype == torch.long and int(idx.shape[0]) == int(row_emb.shape[0]),
            "row_embedding_row_indices is a LongTensor aligned to row_embeddings")
    v.check(mask.dtype == torch.bool and int(mask.shape[0]) == len(audit["row_metadata"]),
            "row_valid_mask is a bool tensor over all input rows")
    v.check(int(idx.shape[0]) == int(mask.sum()),
            "len(row_embedding_row_indices) equals row_valid_mask.sum()")
    metadata_source_indices = {int(m["source_row_index"]) for m in audit["row_metadata"] if "source_row_index" in m}
    if metadata_source_indices:
        v.check(all(int(i) in metadata_source_indices for i in idx.tolist()),
                "every stored index in row_embedding_row_indices exists in source_row_index metadata")
    else:
        v.check(all(0 <= int(i) < len(audit["row_metadata"]) for i in idx.tolist()),
                "every row index is in range")
    v.check(bool(torch.isfinite(row_emb).all()) and float(row_emb.norm(dim=1).min()) > 0 if row_emb.shape[0] > 0 else True,
            "row embeddings are finite with norm > 0")

    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    expected_schema = {
        "embedding_dim": final_dim,
        "key_type": "nsc_id",
        "dtype": "float32",
        "aggregation": "mean",
        "sequence_identity": "SHA256(cleaned_sequence)",
        "layernorm_applied": projection,
        "projection_applied": projection,
    }
    for key, expected in expected_schema.items():
        v.check(schema.get(key) == expected,
                f"schema['{key}'] == {expected!r} (found {schema.get(key)!r})")
    v.check(schema.get("model_id") == manifest["model_id_resolved"],
            "schema model_id matches the manifest")
    v.check(schema.get("model_revision") == manifest["model_revision"],
            "schema model_revision matches the manifest")
    v.check(len(drugs) == len(audit["drug_embeddings"]), "audit and primary drug dicts agree")
    return v


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate run artifacts.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    setup_logging(None)
    return validate(Path(args.output_dir)).report()


if __name__ == "__main__":
    raise SystemExit(main())
