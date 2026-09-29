"""End-to-end orchestration: CSV -> canonicalize -> clean -> ESMC -> cache -> drugs.

This is pooling points **A** (residue -> protein, parameter-free) and **B**
(protein -> drug target, parameter-free mean). Nothing here trains anything, and
nothing here projects to 256-D by default.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from . import __version__ as PACKAGE_VERSION
from .aggregation import AggregationResult, aggregate_drug_embeddings
from .cache import (
    CACHE_SCHEMA_VERSION,
    POOLING_IMPLEMENTATION_VERSION,
    SEQUENCE_CLEANING_VERSION,
    ProteinCache,
    build_fingerprint,
)
from .config import REQUIRED_HIDDEN_SIZE, AppConfig
from .data import RowRecord, dataset_statistics, filter_rows, prepare_rows, read_input
from .diagnostics import run_diagnostics
from .errors import FatalError, IsolationError, PipelineInterruptedError, RowError
from .esmc_encoder import (
    DTYPE_MAP,
    TargetProteinEncoder,
    is_oom,
    probe_special_residue_support,
    resolve_model,
    support_map,
)
from .outputs import (
    build_schema,
    embedding_filename,
    provenance_columns,
    write_csv,
    write_diagnostics,
    write_drug_embeddings,
    write_embedding_csv,
    write_failed_rows,
    write_manifest,
    write_schema,
    write_target_embeddings,
)
from .projection import load_projection, project_embeddings
from .sequence import clean_sequence, sequence_hash
from .signals import SignalHandler
from .utils import (
    atomic_write_json,
    get_logger,
    gpu_info,
    human_bytes,
    local_now_iso,
    peak_gpu_memory_bytes,
    run_id as make_run_id,
    seed_everything,
    setup_logging,
    software_versions,
)

LOGGER = get_logger("esmc_target.pipeline")
PRODUCTION_DIR_MARKERS = ("full",)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise FatalError("runtime.device='cuda' requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enforce_run_isolation(output_dir: Path, is_subset_run: bool, full_run_dir: Optional[Path]) -> None:
    """``--max-rows`` / ``--nsc-ids`` must never write into the production dir."""
    if not is_subset_run:
        return
    if full_run_dir is not None and output_dir.resolve() == Path(full_run_dir).resolve():
        raise IsolationError(
            f"--max-rows/--nsc-ids is a subset run, so an explicit --output-dir other than the "
            f"full-run directory ({full_run_dir}) is required. Refusing to write subset results "
            f"into the production directory (cache contamination)."
        )
    if output_dir.name.lower() in PRODUCTION_DIR_MARKERS:
        raise IsolationError(
            f"--max-rows/--nsc-ids requires a separate output directory; {output_dir} looks like the "
            f"full-run directory. Use e.g. --output-dir outputs/smoke_test or outputs/edge_cases."
        )


@dataclass
class RunResult:
    output_dir: Path
    manifest: Dict[str, Any]
    drug_embeddings_path: Path
    protein_cache_path: Path
    target_embeddings_path: Path
    summary: Dict[str, Any]


class PreprocessingPipeline:
    def __init__(self, config: AppConfig, output_dir: str | Path,
                 input_path: str | Path | None = None,
                 max_rows: Optional[int] = None, nsc_ids: Optional[Sequence[str]] = None,
                 resume: bool = True, rebuild_cache: bool = False,
                 export_csv: bool = False, enable_projection: bool = False,
                 cache_path: Optional[str] = None,
                 resolved_model: Any = None,
                 run_id: Optional[str] = None) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.input_path = Path(input_path) if input_path else None
        self.max_rows = max_rows
        if isinstance(nsc_ids, str):
            self.nsc_ids = [s.strip() for s in nsc_ids.split(",") if s.strip()]
        elif nsc_ids:
            self.nsc_ids = list(nsc_ids)
        else:
            self.nsc_ids = None
        self.resume = resume
        self.rebuild_cache = rebuild_cache
        self.export_csv = export_csv or config.output.export_csv
        self.enable_projection = enable_projection or config.projection.enabled
        self.cache_path_override = Path(cache_path) if cache_path else None
        self._resolved_model = resolved_model
        self.run_id = run_id or make_run_id()
        self.is_subset_run = bool(max_rows or nsc_ids)
        self.seeds = seed_everything(config.runtime.seed)
        self.logger = setup_logging(self.output_dir / "run.log")
        self.cache: Optional[ProteinCache] = None
        self.encoder: Optional[TargetProteinEncoder] = None
        self.timings: Dict[str, float] = {}
        self.started_at = local_now_iso()
        self._t0 = time.perf_counter()
        self._rows_by_hash: Dict[str, List[RowRecord]] = {}
        self._chunk_log: Dict[str, Dict[str, Any]] = {}

    # -- helpers -------------------------------------------------------
    @property
    def cache_path(self) -> Path:
        if self.cache_path_override:
            return self.cache_path_override
        if self.config.cache.path:
            return Path(self.config.cache.path)
        return self.output_dir / "protein_cache.pt"

    def _log_device(self, device: torch.device) -> None:
        if device.type == "cuda":
            info = gpu_info()
            LOGGER.info("Device: CUDA | %s | total memory %s", info.get("gpu_name"),
                        human_bytes(info.get("gpu_memory_total_bytes")))
        else:
            LOGGER.info("Device: CPU (no CUDA device available) - expect slow inference")

    def _resolve_model(self):
        if self._resolved_model is not None:
            LOGGER.info("Using a pre-resolved model (injected by tests/preflight)")
            return self._resolved_model
        return resolve_model(
            self.config.model.candidates,
            trust_remote_code=self.config.model.trust_remote_code,
            cache_dir=self.config.model.cache_dir,
            local_files_only=self.config.model.local_files_only,
            allow_masked_lm_fallback=self.config.model.allow_masked_lm_fallback,
            allow_unresolved_revision=self.config.model.allow_unresolved_revision,
        )

    def _encode_record(self, record: RowRecord):
        t0 = time.perf_counter()
        try:
            encoding = self.encoder.encode(record.sequence)
        except FatalError:
            raise
        except RowError as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.error_kind = getattr(exc, "kind", "inference_error")
            LOGGER.error("Row %d (%s) failed during inference: %s", record.source_row_index,
                         record.nsc_id, record.error)
            return None
        self.timings[f"protein:{record.sequence_hash[:12]}"] = time.perf_counter() - t0
        return encoding

    def _store(self, record: RowRecord, encoding) -> None:
        uniprot, targets = [], []
        for other in self._rows_by_hash.get(record.sequence_hash, []):
            if other.uniprot_id:
                uniprot.append(other.uniprot_id)
            if other.target_id:
                targets.append(other.target_id)
        self.cache.put(record.sequence_hash, encoding.embedding, uniprot, targets,
                       sequence_length=len(record.sequence),
                       num_chunks=encoding.num_chunks, diagnostics=encoding.diagnostics,
                       metadata={"chunk_boundaries": [list(b) for b in encoding.chunk_boundaries],
                                 "coverage": encoding.coverage},
                       branch="target")
        self._chunk_log[record.sequence_hash] = {
            "sequence_length": len(record.sequence),
            "num_chunks": encoding.num_chunks,
            "chunk_boundaries": [list(b) for b in encoding.chunk_boundaries],
            "coverage": encoding.coverage,
        }

    # -- main ----------------------------------------------------------
    def run(self) -> RunResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Run id: %s", self.run_id)
        LOGGER.info("Output directory: %s", self.output_dir)

        signal_handler = SignalHandler()
        signal_handler.register()

        newly_computed = 0
        interrupted = False

        try:
            # 1. input ------------------------------------------------------------
            if self.input_path is None:
                raise FatalError("No --input file was provided")
            source = read_input(self.input_path)
            frame, subset_applied = filter_rows(source.frame, self.max_rows, self.nsc_ids)
            if subset_applied["max_rows"] or subset_applied["nsc_ids"]:
                LOGGER.info("Subset run: max_rows=%s nsc_ids=%s (rows kept: %d of %d)",
                            subset_applied["max_rows"], subset_applied["nsc_ids"],
                            frame.shape[0], source.n_rows)
            extra_columns = [c for c in frame.columns if c != "sequence"]

            # 2. model ------------------------------------------------------------
            device = resolve_device(self.config.runtime.device)
            self._log_device(device)
            resolved = self._resolve_model()
            if resolved.hidden_size != REQUIRED_HIDDEN_SIZE:
                raise FatalError(f"resolved model hidden_size={resolved.hidden_size} != {REQUIRED_HIDDEN_SIZE}")

            # 3. encoder + startup inspection -------------------------------------
            encoder = TargetProteinEncoder(
                resolved=resolved, device=device, dtype=self.config.runtime.dtype,
                chunk_size=self.config.sequence.chunk_size, overlap=self.config.sequence.overlap,
                model_max_positions_override=(self.config.sequence.max_model_positions
                                              if self.config.sequence.max_model_positions not in (None, "auto")
                                              else None))
            self.encoder = encoder

            from .esmc_encoder import inspect_runtime
            inspection = inspect_runtime(resolved.model, resolved.tokenizer, device=device,
                                         dtype=DTYPE_MAP[self.config.runtime.dtype])
            LOGGER.info("Model output: type=%s field=%s shape=%s", inspection.output_type,
                        inspection.residue_field, inspection.hidden_state_shape)
            LOGGER.info("Derived capacity: max_positions=%d special_tokens=%d residue_capacity=%d "
                        "chunk_size=%d overlap=%d", inspection.model_max_positions,
                        inspection.special_tokens, inspection.residue_capacity, encoder.chunk_size,
                        encoder.overlap)

            # 4. special-residue support (full tokenizer->model->extraction path) ---
            support = probe_special_residue_support(resolved.model, resolved.tokenizer, device=device,
                                                    dtype=DTYPE_MAP[self.config.runtime.dtype])
            support_flags = support_map(support)
            unsupported = sorted([s for s, ok in support_flags.items() if not ok])
            if unsupported and self.config.sequence.special_residue_policy == "error":
                LOGGER.warning("Policy 'error' is set but these symbols are unsupported by the "
                               "tokenizer: %s (rows containing them will fail)", unsupported)

            # 5. rows --------------------------------------------------------------
            records = prepare_rows(frame, self.config.sequence.special_residue_policy, support_flags)
            self._rows_by_hash = {}
            for record in records:
                if record.sequence_hash:
                    self._rows_by_hash.setdefault(record.sequence_hash, []).append(record)
            stats = dataset_statistics(records, residue_capacity=encoder.residue_capacity)
            LOGGER.info("Dataset: %d rows, %d unique sequences, length %s..%s, "
                        "%d rows > 2048 residues, %d unique sequences need chunking (capacity=%d)",
                        stats["input_rows"], stats["unique_sequences"], stats["sequence_length_min"],
                        stats["sequence_length_max"], stats["rows_over_2048_residues"],
                        stats.get("unique_sequences_requiring_chunking", 0), encoder.residue_capacity)
            for record in records:
                if record.status == "failed":
                    LOGGER.warning("Row %d (%s) failed cleaning: %s", record.source_row_index,
                                   record.nsc_id, record.error)

            # 6. fingerprint + cache ------------------------------------------------
            fingerprint = build_fingerprint(
                cache_schema_version=CACHE_SCHEMA_VERSION,
                model_id=resolved.model_id,
                model_revision=resolved.model_revision,
                tokenizer_id=resolved.tokenizer_id,
                tokenizer_revision=resolved.tokenizer_revision,
                model_class=resolved.model_class,
                tokenizer_class=resolved.tokenizer_class,
                model_config_hash=resolved.model_config_hash,
                tokenizer_config_hash=resolved.tokenizer_config_hash,
                transformers_version=resolved.transformers_version or "unknown",
                torch_version=torch.__version__,
                effective_attention_implementation=getattr(resolved, "effective_attention_implementation", "eager"),
                sequence_cleaning_version=SEQUENCE_CLEANING_VERSION,
                special_residue_policy=self.config.sequence.special_residue_policy,
                special_residue_tokenizer_support=support_flags,
                model_max_positions=encoder.model_max_positions,
                residue_capacity=encoder.residue_capacity,
                chunk_size=encoder.chunk_size,
                overlap=encoder.overlap,
                pooling_method="residue_mean",
                inference_dtype=self.config.runtime.dtype,
                pooling_implementation_version=POOLING_IMPLEMENTATION_VERSION,
            )
            if self.config.cache.enabled:
                self.cache = ProteinCache(self.cache_path, fingerprint, rebuild=self.rebuild_cache)
            else:
                LOGGER.warning("cache.enabled is false: using temporary cache")
                self.cache = ProteinCache(self.output_dir / "_disabled_cache.pt", fingerprint, rebuild=True)
                self.cache.path = self.output_dir / "protein_cache.pt"

            # 7. encode unique sequences -------------------------------------------
            unique: Dict[str, RowRecord] = {}
            for record in records:
                if record.status == "pending" and record.sequence_hash and record.sequence_hash not in unique:
                    unique[record.sequence_hash] = record
            LOGGER.info("Unique sequence-embedding jobs: %d", len(unique))

            from tqdm import tqdm
            sorted_unique = sorted(unique.values(), key=lambda r: (-len(r.sequence or ""), r.sequence_hash or ""))
            for record in tqdm(sorted_unique, desc="unique proteins", unit="protein", leave=False):
                seq_hash = record.sequence_hash
                if signal_handler.shutdown_requested:
                    interrupted = True
                    LOGGER.warning("Shutdown requested (%s). Stopping encoding loop safely.", signal_handler.shutdown_signal)
                    break
                if record.status != "pending":
                    continue
                cached_res = self.cache.get_entry(seq_hash, branch="target")
                if cached_res is not None:
                    cached_emb, cached_entry = cached_res
                    record.status = "cache_hit_pending"
                    record.cache_hit = True
                    record.num_chunks = int(cached_entry.get("num_chunks", 0))
                    record.embedding_l2_norm = float(cached_entry.get("embedding_l2_norm", 0.0))
                    continue
                encoding = self._encode_record(record)
                if encoding is None:
                    continue
                newly_computed += 1
                record.status = "success"
                record.cache_hit = False
                record.num_chunks = encoding.num_chunks
                record.embedding_l2_norm = encoding.diagnostics["embedding_l2_norm"]
                self._store(record, encoding)
                if self.config.cache.enabled:
                    self.cache.save(save_every=self.config.cache.save_every)

            if self.config.cache.enabled:
                self.cache.save(force=True)

            if interrupted:
                manifest = self._build_manifest(
                    source=source, subset_applied=subset_applied,
                    extra_columns=extra_columns, resolved=resolved,
                    inspection=inspection, support=support, support_flags=support_flags,
                    records=records, stats=stats, fingerprint=fingerprint,
                    result=AggregationResult({}, {}, [], 0), encoder=encoder,
                    final_dim=resolved.hidden_size, projection_applied=False
                )
                manifest["status"] = "interrupted"
                manifest["shutdown_requested"] = True
                manifest["shutdown_signal"] = signal_handler.shutdown_signal
                manifest["run_summary"] = {
                    "total_rows": len(records),
                    "unique_sequences": stats["unique_sequences"],
                    "newly_computed_proteins": newly_computed,
                    "remaining_proteins": len(unique) - newly_computed - self.cache.stats.hits,
                }
                write_manifest(self.output_dir / "manifest.json", manifest)
                raise PipelineInterruptedError(
                    f"Run interrupted by signal {signal_handler.shutdown_signal}. Saved cache and manifest.",
                    shutdown_signal=signal_handler.shutdown_signal
                )

            LOGGER.info("Newly computed protein embeddings: %d (cache hits: %d, misses: %d)",
                        newly_computed, self.cache.stats.hits, self.cache.stats.misses)

            # 8. map back to rows ---------------------------------------------------
            embeddings_by_hash: Dict[str, torch.Tensor] = {}
            for seq_hash in self.cache.entries:
                embeddings_by_hash[seq_hash] = self.cache.entries[seq_hash]["embedding"]
            for record in records:
                if record.status == "cache_hit_pending":
                    record.status = "success" if record.sequence_hash in embeddings_by_hash else "failed"
                    if record.status == "failed":
                        record.error = f"no embedding in cache for sequence_hash={record.sequence_hash}"
                        record.error_kind = "missing_embedding"
                    continue
                if record.status == "pending":
                    if record.sequence_hash and record.sequence_hash in embeddings_by_hash:
                        record.status = "success"
                        entry = self.cache.entry(record.sequence_hash) or {}
                        record.num_chunks = int(entry.get("num_chunks", 0))
                        record.embedding_l2_norm = entry.get("embedding_l2_norm")
                        record.cache_hit = True
                    else:
                        record.status = "failed"
                        record.error = record.error or "sequence was not encoded"
                        record.error_kind = record.error_kind or "not_encoded"
                    continue
                if record.status == "success" and record.sequence_hash not in embeddings_by_hash:
                    record.status = "failed"
                    record.error = f"no embedding in cache for sequence_hash={record.sequence_hash}"
                    record.error_kind = "missing_embedding"

            # 9. aggregate over drugs ------------------------------------------------
            result = aggregate_drug_embeddings(records, embeddings_by_hash,
                                               method=self.config.aggregation.method,
                                               embedding_dim=resolved.hidden_size)
            LOGGER.info("Drug aggregation: %d drugs with embeddings, %d without successful targets, "
                        "%d within-drug duplicates removed", len(result.drug_embeddings),
                        len(result.drugs_without_successful_targets), result.within_drug_duplicates_removed)

            # 10. optional projection (aggregate FIRST, then Linear + LayerNorm) ----
            projection_applied = False
            final_drug_embeddings = result.drug_embeddings
            if self.enable_projection:
                projection = load_projection(self.config.projection.checkpoint,
                                             input_dim=resolved.hidden_size,
                                             output_dim=self.config.projection.output_dim)
                final_drug_embeddings = project_embeddings(result.drug_embeddings, projection)
                projection_applied = True
                LOGGER.info("Applied trained projection %s -> %s",
                            self.config.projection.checkpoint, self.config.projection.output_dim)
            final_dim = self.config.projection.output_dim if projection_applied else resolved.hidden_size

            # 11. artifacts ------------------------------------------------------------
            manifest = self._build_manifest(source=source, subset_applied=subset_applied,
                                            extra_columns=extra_columns, resolved=resolved,
                                            inspection=inspection, support=support, support_flags=support_flags,
                                            records=records, stats=stats, fingerprint=fingerprint,
                                            result=result, encoder=encoder,
                                            final_dim=final_dim, projection_applied=projection_applied)
            paths = self._write_artifacts(records=records, result=result, extra_columns=extra_columns,
                                         resolved=resolved, config=self.config,
                                         final_drug_embeddings=final_drug_embeddings,
                                         final_dim=final_dim, projection_applied=projection_applied,
                                         source=source)
            self._write_diagnostics(records, embeddings_by_hash, encoder, resolved)

            failed_unique_jobs = sum(1 for r in unique.values() if r.status == "failed")
            accounting_total = newly_computed + self.cache.stats.hits + failed_unique_jobs
            accounting_ok = bool(accounting_total == len(unique))

            summary = {
                "total_rows": len(records),
                "successful_rows": sum(1 for r in records if r.status == "success"),
                "failed_rows": sum(1 for r in records if r.status == "failed"),
                "unique_sequences": stats["unique_sequences"],
                "unique_sequence_embedding_jobs": len(unique),
                "newly_computed_proteins": newly_computed,
                "failed_unique_jobs": failed_unique_jobs,
                "cache_hits": self.cache.stats.hits,
                "cache_misses": self.cache.stats.misses,
                "same_branch_cache_hits": self.cache.stats.same_branch_cache_hits,
                "cross_branch_cache_hits": self.cache.stats.cross_branch_cache_hits,
                "total_protein_cache_hits": self.cache.stats.total_protein_cache_hits,
                "accounting_invariant_satisfied": accounting_ok,
                "long_sequences_rows": stats["rows_over_2048_residues"],
                "unique_sequences_requiring_chunking": stats.get("unique_sequences_requiring_chunking", 0),
                "esmc_chunk_forward_passes": encoder.chunk_forward_passes,
                "embedding_dim": final_dim,
                "drugs_with_embeddings": len(result.drug_embeddings),
            }
            manifest["run_summary"] = summary
            manifest["timings"]["total_seconds"] = round(time.perf_counter() - self._t0, 3)
            write_manifest(self.output_dir / "manifest.json", manifest)

            LOGGER.info("=" * 78)
            LOGGER.info("RUN SUMMARY")
            for key, value in summary.items():
                LOGGER.info("  %-38s %s", key, value)
            LOGGER.info("  %-38s %s", "peak GPU memory", human_bytes(peak_gpu_memory_bytes()))
            LOGGER.info("  %-38s %.1f", "total seconds", manifest["timings"]["total_seconds"])
            LOGGER.info("=" * 78)
            return RunResult(output_dir=self.output_dir, manifest=manifest,
                             drug_embeddings_path=paths["drug_embeddings"],
                             protein_cache_path=paths["cache"],
                             target_embeddings_path=paths["target_embeddings"],
                             summary=summary)
        finally:
            signal_handler.restore()

    def _build_manifest(self, *, source, subset_applied, extra_columns, resolved, inspection,
                        support, support_flags, records, stats, fingerprint, result, encoder,
                        final_dim, projection_applied) -> Dict[str, Any]:
        successful = [r for r in records if r.status == "success"]
        long_detail = []
        for seq_hash, info in self._chunk_log.items():
            if info["num_chunks"] > 1:
                long_detail.append({"sequence_hash": seq_hash, **info})
        special_events = []
        for record in records:
            for event in record.special_residue_events:
                special_events.append({"nsc_id": record.nsc_id, "gene_symbol": record.gene_symbol,
                                       "uniprot_id": record.uniprot_id,
                                       "sequence_hash": record.sequence_hash, **event})
        return {
            "run_id": self.run_id,
            "status": "completed",
            "package_version": PACKAGE_VERSION,
            "timestamp": self.started_at,
            "completed_at": local_now_iso(),
            "input_filename": source.path.name,
            "input_path": str(source.path),
            "input_sha256": source.sha256,
            "input_size_bytes": source.size_bytes,
            "subset_run": self.is_subset_run,
            "subset_filter_applied": subset_applied,
            "input_columns_as_found": list(source.frame.columns),
            "input_columns_canonical": extra_columns,
            "model_id_requested": list(self.config.model.candidates),
            "model_id_resolved": resolved.model_id,
            "model": resolved.to_dict(),
            "model_id": resolved.model_id,
            "model_revision": resolved.model_revision,
            "tokenizer_revision": resolved.tokenizer_revision,
            "model_config_hash": resolved.model_config_hash,
            "tokenizer_config_hash": resolved.tokenizer_config_hash,
            "hidden_size": resolved.hidden_size,
            "backbone_embedding_dim": resolved.hidden_size,
            "final_output_dim": final_dim,
            "model_max_positions": encoder.model_max_positions,
            "special_tokens": encoder.special_tokens,
            "residue_capacity": encoder.residue_capacity,
            "chunk_size": encoder.chunk_size,
            "overlap": encoder.overlap,
            "pooling_method": "residue_mean",
            "esmc_output_field_used_for_residues": inspection.residue_field,
            "esmc_applies_final_layer_norm": resolved.applies_final_layer_norm,
            "projection": {"enabled": projection_applied,
                           "checkpoint": self.config.projection.checkpoint if projection_applied else None,
                           "output_dim": self.config.projection.output_dim if projection_applied else None,
                           "applied_to_aggregate": True if projection_applied else False},
            "aggregation": {"method": result.method, "pooling_point": "B_protein_to_drug"},
            "dtype": {"inference": self.config.runtime.dtype, "outputs": "float32"},
            "device": str(encoder.device),
            "gpu": gpu_info(),
            "config_hash": self.config.config_hash(),
            "config": self.config.to_dict(),
            "embedding_fingerprint": fingerprint,
            "sequence_cleaning_version": SEQUENCE_CLEANING_VERSION,
            "pooling_implementation_version": POOLING_IMPLEMENTATION_VERSION,
            "dataset_statistics": stats,
            "row_counts": {"input": len(records), "successful": len(successful),
                           "failed": len(records) - len(successful)},
            "cache": {
                "path": str(self.cache_path),
                "hits": self.cache.stats.hits,
                "misses": self.cache.stats.misses,
                "same_branch_cache_hits": self.cache.stats.same_branch_cache_hits,
                "cross_branch_cache_hits": self.cache.stats.cross_branch_cache_hits,
                "total_protein_cache_hits": self.cache.stats.total_protein_cache_hits,
                "new_protein_encodes": self.cache.stats.new_protein_encodes,
                "stores": self.cache.stats.stores,
                "saves": self.cache.stats.saves,
                "entries": len(self.cache.entries),
                "save_every": self.config.cache.save_every,
                "rebuild": self.rebuild_cache,
                "resume": self.resume,
            },
            "chunk_forward_passes": encoder.chunk_forward_passes,
            "max_sequence_length": stats.get("sequence_length_max"),
            "long_sequences": {"rows_over_2048_residues": stats.get("rows_over_2048_residues"),
                               "unique_sequences_over_2048_residues": stats.get("unique_sequences_over_2048_residues"),
                               "unique_sequences_requiring_chunking": stats.get("unique_sequences_requiring_chunking"),
                               "per_sequence_chunk_counts_and_boundaries": long_detail},
            "tokenizer_support": {s: r.to_dict() for s, r in support.items()},
            "special_residue_tokenizer_support": support_flags,
            "special_residue_events": special_events,
            "preflight": {"startup_inspection": inspection.to_dict()},
            "drugs_without_successful_targets": result.drugs_without_successful_targets,
            "within_drug_duplicates_removed": result.within_drug_duplicates_removed,
            "aggregation_summary": result.summary(),
            "software_versions": software_versions().to_dict(),
            "random_seed": self.seeds,
            "timings": {"per_protein_seconds": dict(self.timings)},
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes(),
            "shutdown_requested": False,
            "shutdown_signal": None,
            "run_log": str(self.output_dir / "run.log"),
        }

    def _write_artifacts(self, *, records, result, extra_columns, resolved, config,
                         final_drug_embeddings, final_dim, projection_applied, source):
        paths: Dict[str, Path] = {}
        dim = resolved.hidden_size

        # drug embeddings (PRIMARY ARTIFACT 2)
        drug_name = embedding_filename(dim, projection_applied)
        paths["drug_embeddings"] = write_drug_embeddings(self.output_dir / drug_name,
                                                         final_drug_embeddings)
        paths["schema"] = write_schema(self.output_dir / "drug_target_embeddings_schema.json",
                                       build_schema(model_id=resolved.model_id,
                                                    model_revision=resolved.model_revision,
                                                    hidden_size=final_dim,
                                                    projection_applied=projection_applied,
                                                    layernorm_applied=projection_applied))

        # compact successful-only row tensors
        successful = [r for r in records if r.status == "success"]
        if successful:
            row_embeddings = torch.stack(
                [self.cache.entries[r.sequence_hash]["embedding"].to(torch.float32) for r in successful])
            row_indices = torch.tensor([r.source_row_index for r in successful], dtype=torch.long)
        else:
            row_embeddings = torch.zeros((0, dim), dtype=torch.float32)
            row_indices = torch.zeros((0,), dtype=torch.long)
        n_mask = len(records)
        row_valid_mask = torch.zeros(n_mask, dtype=torch.bool)
        for idx, r in enumerate(records):
            if r.status == "success":
                row_valid_mask[idx] = True
        row_metadata = [self._row_metadata(r) for r in records]

        paths["target_embeddings"] = write_target_embeddings(
            path=self.output_dir / "target_embeddings.pt", row_embeddings=row_embeddings,
            row_embedding_row_indices=row_indices, row_valid_mask=row_valid_mask,
            row_metadata=row_metadata,
            drug_embeddings=final_drug_embeddings,
            drug_metadata={k: v for k, v in result.drug_metadata.items()},
            config=config.to_dict(), model_name="ESMC-600M", model_revision=resolved.model_revision,
            embedding_dim=final_dim, projection_enabled=projection_applied,
            aggregation_method=result.method)

        # provenance CSVs
        rows_payload = [self._row_metadata(r, with_hash=True) for r in records]
        columns = provenance_columns(extra_columns)
        paths["target_provenance"] = write_csv(self.output_dir / "target_provenance.csv",
                                               columns, rows_payload)
        drug_rows = [result.drug_metadata[k] for k in sorted(result.drug_metadata)]
        paths["drug_provenance"] = write_csv(self.output_dir / "drug_provenance.csv",
                                             ["nsc_id", "drug_name", "num_target_proteins",
                                              "num_unique_target_proteins", "num_successful_targets",
                                              "num_failed_targets", "num_duplicate_targets_removed",
                                              "aggregation_method", "status", "error", "embedding_dim"],
                                             drug_rows)
        paths["failed_rows"] = write_failed_rows(
            self.output_dir / "failed_rows.csv",
            [self._row_metadata(r, with_hash=True) for r in records if r.status == "failed"])

        if self.export_csv:
            stage = "projected_256" if projection_applied else f"esmc_{dim}"
            drug_rows_with_vectors = [result.drug_metadata[k] for k in sorted(result.drug_metadata)
                                      if k in final_drug_embeddings]
            paths["drug_csv"] = write_embedding_csv(
                self.output_dir / "drug_target_embeddings.csv", drug_rows_with_vectors,
                [final_drug_embeddings[r["nsc_id"]].tolist() for r in drug_rows_with_vectors],
                final_dim, stage)
            successful_rows_payload = [self._row_metadata(r, with_hash=True) for r in successful]
            paths["target_csv"] = write_embedding_csv(
                self.output_dir / "target_embeddings.csv", successful_rows_payload,
                [self.cache.entries[r.sequence_hash]["embedding"].tolist() for r in successful],
                dim, stage)

        paths["cache"] = self.cache.path
        if self.config.cache.enabled:
            self.cache.save(force=True)
        for key, path in paths.items():
            if path is not None:
                LOGGER.info("Wrote %s", path.name)
        return paths

    def _row_metadata(self, record: RowRecord, with_hash: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source_row_index": record.source_row_index,
            "nsc_id": record.nsc_id,
            "drug_name": record.drug_name,
            "chembl_id": record.metadata.get("chembl_id"),
            "target_id": record.target_id,
            "gene_symbol": record.gene_symbol,
            "uniprot_id": record.uniprot_id,
            "sequence_length": record.declared_length,
            "actual_sequence_length": record.actual_length,
            "sequence_length_mismatch": record.length_mismatch,
            "sequence_hash": record.sequence_hash,
            "embedding_cache_hit": bool(record.cache_hit),
            "num_chunks": int(record.num_chunks or self._chunk_log.get(record.sequence_hash or "", {}).get("num_chunks", 0)),
            "special_residue_events": (json_dumps(record.special_residue_events)
                                       if record.special_residue_events else ""),
            "embedding_l2_norm": record.embedding_l2_norm,
            "status": record.status,
            "error": record.error or record.warning or "",
        }
        for column, value in record.metadata.items():
            if column not in payload and column != "sequence":
                payload[column] = value
        return payload

    def _write_diagnostics(self, records, embeddings_by_hash, encoder, resolved) -> None:
        if not self.config.output.diagnostics or not embeddings_by_hash:
            return
        sequences_by_hash: Dict[str, str] = {}
        labels: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not record.sequence_hash or record.sequence_hash not in embeddings_by_hash:
                continue
            if record.sequence_hash not in sequences_by_hash and record.sequence:
                sequences_by_hash[record.sequence_hash] = record.sequence
            labels.setdefault(record.sequence_hash, {
                "gene_symbol": record.gene_symbol, "uniprot_id": record.uniprot_id,
                "target_id": record.target_id, "nsc_id": record.nsc_id,
                "sequence_length": record.actual_length,
            })
        # (c) one short, one with a special residue if present, one chunked
        pick: List[Tuple[str, str]] = []
        for seq_hash, seq in sorted(sequences_by_hash.items()):
            if len(pick) < 1 and len(seq) < 300:
                pick.append((f"short:{labels[seq_hash].get('gene_symbol') or seq_hash[:10]}", seq))
        for record in records:
            if record.special_residue_symbols and record.sequence and record.sequence_hash in sequences_by_hash:
                pick.append((f"special:{record.special_residue_symbols[0]}", record.sequence))
                break
        chunk_probe = None
        for record in records:
            info = self._chunk_log.get(record.sequence_hash or "")
            if record.sequence and info and info.get("num_chunks", 0) > 1:
                chunk_probe = record.sequence
                pick.append((f"chunked:{record.gene_symbol or record.nsc_id}", record.sequence))
                break
        probe_for_chunking = None
        for record in records:
            if record.sequence and 100 < len(record.sequence) <= encoder.residue_capacity:
                probe_for_chunking = record.sequence
                break
        try:
            diagnostics = run_diagnostics(
                embeddings=embeddings_by_hash, encoder=encoder,
                cache_entries=self.cache.entries, sequences_by_hash=sequences_by_hash,
                labels=labels, chunk_probe_sequence=probe_for_chunking,
                forced_chunk_size=min(512, encoder.chunk_size),
                max_pairs=self.config.output.diagnostics_max_pairs,
                collapse_threshold=self.config.output.cosine_collapse_warn_threshold,
                pick_for_determinism=pick)
        except Exception as exc:  # diagnostics never break a run
            LOGGER.warning("Diagnostics could not be completed (%s: %s); the run is unaffected",
                           type(exc).__name__, exc)
            return
        write_diagnostics(self.output_dir / "embedding_diagnostics.json", diagnostics)
        for warning in diagnostics.get("warnings", []):
            LOGGER.warning("DIAGNOSTIC: %s", warning)
        LOGGER.info("Wrote embedding_diagnostics.json")


def json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, sort_keys=True)
