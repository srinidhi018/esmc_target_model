"""Versioned, atomic, resumable protein embedding cache.

Cache key is ``sequence_hash = SHA256(cleaned sequence)`` - never ``target_id``
or ``uniprot_id`` (the current file has 245 unique sequences but only 244 unique
target_ids, so ``target_id`` is provably not a safe identity key).

The cache is scoped to an embedding **fingerprint** (model id/revision, config
hashes, cleaning version, chunking, pooling, dtype, implementation version).
A fingerprint mismatch is refused unless ``--rebuild-cache`` is passed: an
embedding produced under different settings is never served.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from filelock import FileLock

from .errors import CacheCorruptError, FatalError, FingerprintMismatchError
from .utils import atomic_path, get_logger, sha256_json

LOGGER = get_logger("esmc_target.cache")

#: Bump when the cleaning/pooling implementation changes in a way that would
#: invalidate previously cached vectors.
SEQUENCE_CLEANING_VERSION = "clean-v1"
POOLING_IMPLEMENTATION_VERSION = "pool-v1"
ENCODER_IMPLEMENTATION_VERSION = "encoder-v1"
CACHE_SCHEMA_VERSION = 1

FINGERPRINT_FIELDS = (
    "cache_schema_version",
    "model_id",
    "model_revision",
    "tokenizer_id",
    "tokenizer_revision",
    "model_class",
    "tokenizer_class",
    "model_config_hash",
    "tokenizer_config_hash",
    "transformers_version",
    "torch_version",
    "effective_attention_implementation",
    "sequence_cleaning_version",
    "special_residue_policy",
    "special_residue_tokenizer_support",
    "model_max_positions",
    "residue_capacity",
    "chunk_size",
    "overlap",
    "pooling_method",
    "inference_dtype",
    "pooling_implementation_version",
    "encoder_implementation_version",
)


def build_fingerprint(**kwargs: Any) -> Dict[str, Any]:
    """Build the cache fingerprint from the resolved scientific settings."""
    kwargs.setdefault("encoder_implementation_version", ENCODER_IMPLEMENTATION_VERSION)
    if "tokenizer_id" not in kwargs or kwargs["tokenizer_id"] is None:
        kwargs["tokenizer_id"] = kwargs.get("model_id")
    if "tokenizer_revision" not in kwargs or kwargs["tokenizer_revision"] is None:
        kwargs["tokenizer_revision"] = kwargs.get("model_revision")

    fingerprint = {name: kwargs.get(name) for name in FINGERPRINT_FIELDS}
    missing = [k for k, v in fingerprint.items()
               if v is None and k not in ("transformers_version", "torch_version", "model_max_positions", "residue_capacity", "encoder_implementation_version")]
    if missing:
        raise FatalError(f"cache fingerprint is missing required field(s): {missing}")

    # If revisions are unresolved/unknown, inject a unique runtime salt so the fingerprint
    # is NEVER equal across different runs with unresolved provenance.
    rev_resolved = bool(kwargs.get("revision_resolved", True))
    mod_rev = str(kwargs.get("model_revision", ""))
    tok_rev = str(kwargs.get("tokenizer_revision", ""))
    if not rev_resolved or mod_rev in ("", "unknown", "unresolved") or tok_rev in ("", "unknown", "unresolved"):
        import uuid
        fingerprint["_unresolved_revision_salt"] = uuid.uuid4().hex

    fingerprint["fingerprint_hash"] = sha256_json(fingerprint)
    return fingerprint


def fingerprint_diff(stored: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that differ between a stored and the current fingerprint."""
    diff: Dict[str, Any] = {}
    for key in FINGERPRINT_FIELDS:
        old, new = stored.get(key), current.get(key)
        if old != new:
            diff[key] = {"cached": old, "current": new}
    return diff


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    saves: int = 0
    same_branch_cache_hits: int = 0
    cross_branch_cache_hits: int = 0
    total_protein_cache_hits: int = 0
    new_protein_encodes: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "saves": self.saves,
            "same_branch_cache_hits": self.same_branch_cache_hits,
            "cross_branch_cache_hits": self.cross_branch_cache_hits,
            "total_protein_cache_hits": self.total_protein_cache_hits,
            "new_protein_encodes": self.new_protein_encodes,
        }


class ProteinCache:
    """Fingerprinted ``{sequence_hash: entry}`` store with atomic persistence and inter-process locking."""

    def __init__(self, path: str | Path, fingerprint: Dict[str, Any],
                 rebuild: bool = False) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock = FileLock(str(self.lock_path))
        self.fingerprint = fingerprint
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.stats = CacheStats()
        self._since_save = 0
        self.rebuild = rebuild
        self.is_rebuild = rebuild
        if rebuild and self.path.exists():
            LOGGER.warning("--rebuild-cache: ignoring existing cache at %s", self.path)
        else:
            self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> None:
        with self.lock:
            if not self.path.exists():
                LOGGER.info("No existing cache at %s; starting a new one", self.path)
                return
            try:
                payload = torch.load(self.path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise CacheCorruptError(
                    f"Cache at {self.path} could not be read ({type(exc).__name__}: {exc}). "
                    f"Delete it or re-run with --rebuild-cache."
                )
            if not isinstance(payload, dict) or "fingerprint" not in payload or "entries" not in payload:
                raise CacheCorruptError(
                    f"Cache at {self.path} is missing the 'fingerprint'/'entries' keys. "
                    f"Re-run with --rebuild-cache."
                )
            stored = payload["fingerprint"]
            if stored.get("fingerprint_hash") != self.fingerprint.get("fingerprint_hash"):
                diff = fingerprint_diff(stored, self.fingerprint)
                raise FingerprintMismatchError(
                    f"Cache at {self.path} was produced under a different embedding fingerprint and "
                    f"will NOT be reused. Differing fields: {diff}\n"
                    f"Re-run with --rebuild-cache to discard the old cache, or point --cache-path at a "
                    f"cache that matches the current settings."
                )
            self.entries = dict(payload["entries"])
            LOGGER.info("Loaded %d cached protein embeddings from %s", len(self.entries), self.path)

    def _merge_on_disk(self) -> None:
        """Merge latest on-disk entries inside lock to prevent lost updates."""
        if self.is_rebuild or not self.path.exists():
            return
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise CacheCorruptError(f"On-disk cache at {self.path} is corrupt: {exc}") from exc

        if isinstance(payload, dict):
            disk_fp = payload.get("fingerprint")
            if disk_fp:
                diff = fingerprint_diff(disk_fp, self.fingerprint)
                if diff:
                    diff_keys = list(diff.keys())
                    raise FingerprintMismatchError(
                        f"Cannot save cache to {self.path}: on-disk cache has a different fingerprint. "
                        f"Differing field(s): {diff_keys}. Refusing to merge or overwrite incompatible cache."
                    )
            disk_entries = payload.get("entries", {})
            for k, v in disk_entries.items():
                if k not in self.entries:
                    self.entries[k] = v
                else:
                    # Merge branch provenance & IDs
                    existing = self.entries[k]
                    branches = sorted(set(existing.get("branches", [])) | set(v.get("branches", [])))
                    existing["branches"] = branches
                    for id_key in ("uniprot_ids", "target_ids"):
                        merged = sorted(set(existing.get(id_key, [])) | set(v.get(id_key, [])))
                        existing[id_key] = merged

    def save(self, force: bool = False, save_every: int = 100) -> bool:
        """Atomically persist with inter-process lock & Read-Modify-Write merge. A crash can never leave a corrupt cache."""
        with self.lock:
            if not force and self._since_save < save_every:
                return False
            self._merge_on_disk()
            payload = {
                "fingerprint": self.fingerprint,
                "entries": self.entries,
                "schema_version": CACHE_SCHEMA_VERSION,
            }
            with atomic_path(self.path) as tmp:
                torch.save(payload, tmp)
            self._since_save = 0
            self.stats.saves += 1
            self.rebuild = False
            return True

    # -- access --------------------------------------------------------
    def __contains__(self, sequence_hash: str) -> bool:
        with self.lock:
            return sequence_hash in self.entries

    def __len__(self) -> int:
        with self.lock:
            return len(self.entries)

    def get(self, sequence_hash: str, branch: str = "target") -> Optional[torch.Tensor]:
        res = self.get_entry(sequence_hash, branch=branch)
        return res[0] if res is not None else None

    def get_entry(self, sequence_hash: str, branch: str = "target") -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Atomic retrieval of embedding + metadata under a single lock acquisition."""
        with self.lock:
            entry = self.entries.get(sequence_hash)
            if entry is None:
                self.stats.misses += 1
                return None
            
            branches = set(entry.get("branches", ["target"]))
            if branch in branches:
                self.stats.same_branch_cache_hits += 1
            else:
                self.stats.cross_branch_cache_hits += 1
                branches.add(branch)
                entry["branches"] = sorted(branches)
                self._since_save += 1

            self.stats.hits += 1
            self.stats.total_protein_cache_hits += 1
            return entry["embedding"], entry

    def entry(self, sequence_hash: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.entries.get(sequence_hash)

    def put(self, sequence_hash: str, embedding: torch.Tensor, uniprot_ids: Sequence[Any],
            target_ids: Sequence[Any], sequence_length: int, num_chunks: int,
            diagnostics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None,
            branch: str = "target") -> None:
        """Insert/refresh an entry. Provenance lists are deduplicated and sorted;
        they never influence cache identity or the embedding computation."""
        with self.lock:
            existing = self.entries.get(sequence_hash, {})
            existing_branches = set(existing.get("branches", []))
            existing_branches.add(branch)

            entry = {
                "embedding": embedding.detach().to(dtype=torch.float32, device="cpu").clone(),
                "uniprot_ids": sorted({str(u) for u in uniprot_ids if u}),
                "target_ids": sorted({str(t) for t in target_ids if t}),
                "branches": sorted(existing_branches),
                "sequence_length": int(sequence_length),
                "num_chunks": int(num_chunks),
                "embedding_mean": float(diagnostics.get("embedding_mean", 0.0)),
                "embedding_std": float(diagnostics.get("embedding_std", 0.0)),
                "embedding_l2_norm": float(diagnostics.get("embedding_l2_norm", 0.0)),
            }
            if metadata:
                entry["metadata"] = dict(metadata)
            # Provenance from an earlier row for the same sequence is preserved.
            for key in ("uniprot_ids", "target_ids"):
                merged = set(entry[key]) | set(existing.get(key, []))
                entry[key] = sorted(merged)
            self.entries[sequence_hash] = entry
            self.stats.stores += 1
            self.stats.new_protein_encodes += 1
            self._since_save += 1
