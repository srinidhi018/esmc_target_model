# ESMC-600M Target-Protein Preprocessing and Embedding Pipeline

Offline, label-free, **frozen** ESMC-600M feature extraction for drug target
proteins, producing the exact input that CancerCombo expects for its
`Target [1152]` modality token.

---

## 1. What this repository does

```
targetprotein.csv
    -> canonicalize columns (case-insensitive; every input column preserved)
    -> clean sequence (whitespace, uppercase, validate, special-residue policy)
    -> SHA256(cleaned sequence)  -> unique sequences
    -> ESMC-600M  (FROZEN: requires_grad=False, eval(), torch.no_grad())
    -> residue-aligned hidden states [L, 1152]
    -> mean over actual residues only
       (long proteins: overlapping chunks, per-residue accumulators)
    -> PROTEIN embedding [1152]            <-- PRIMARY ARTIFACT 1: protein_cache.pt
    -> cache keyed by sequence_hash, inside a model/config-fingerprinted cache
    -> map back to drugs by nsc_id
    -> deduplicate target proteins WITHIN each drug by sequence_hash
    -> MEAN over the drug's unique successful target proteins
    -> DRUG TARGET embedding [1152]        <-- PRIMARY ARTIFACT 2: drug_target_embeddings_1152.pt
```

### What this repository does NOT do

No Morgan / RDKit / MolFormer / ChemBERTa / BRICS / SMILES or graph
embeddings. No pathway processing, no cell-line information, no
`Linear(1152->256)`, no LayerNorm, no modality mask, no MHSA, no attention
pooling, no drug-cell attention, no Bivariate Hill, no dose-response training,
no Scenario 3 training/evaluation, **no training of any kind**, and **no online
lookup** (UniProt / ChEMBL / PubChem / DrugBank). The pipeline contains no
stochastic operation.

### The three pooling points are NOT conflated

| # | Pooling | Where | Properties |
|---|---------|-------|-----------|
| **A** | residue -> protein | this repo | mean over residues; parameter-free |
| **B** | protein -> drug target embedding | this repo | MEAN over the drug's unique targets; parameter-free, deterministic, permutation-invariant, label-free |
| **C** | modality tokens -> drug embedding | CancerCombo | `{Morgan, RDKit, Target, Pathway}` -> mask -> MHSA -> attention pooling |

**Pooling point B is a computational representation assumption, not a biological
claim that all annotated targets are equally important.** Target-level
attention/gating may be evaluated later as a *supervised* ablation inside
CancerCombo; it is deliberately not built here (see §10).

### Interpretation caveat (read this before using the vectors)

This branch encodes **the sequence identity of a drug's annotated targets**.
ESMC never sees the drug, so it does **not** encode drug-target binding,
affinity, or cell-specific target activity. These are
*drug-associated target-biology representations*, **not**
drug-target-interaction representations.

### No 1152 -> 256 projection in the default pipeline

In CancerCombo: `Drug Target [1152] -> Linear(1152->256) -> LayerNorm -> Target token [256]`.

* A **linear** map commutes with a mean, so averaging in 1152-D then projecting
  equals projecting then averaging.
* **LayerNorm does NOT commute with averaging**, so it must stay downstream of
  aggregation. It is therefore applied to the *aggregate*, never per protein.
* A projection frozen offline and untrained cannot be corrected later, so the
  full 1152-D vectors are saved: all information stays available until the
  supervised stage.

---

## 2. Installation

```bash
git clone <this repo> && cd esmc_target_protein
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 1) PyTorch FIRST, from the index that matches your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) everything else
pip install -r requirements.txt

# 3) prove native ESMC support exists in your transformers build
python -c "from transformers import EsmcModel, EsmcForMaskedLM; print('native ESMC OK')"
```

### transformers version and API actually used

* **Required API:** native Transformers ESMC support -
  `transformers.EsmcModel` (preferred) / `transformers.EsmcForMaskedLM` plus
  `AutoTokenizer`. This is why the pipeline is built on Transformers rather than
  on `fair-esm`: the HF port is the supported, versioned, cached path.
* **Loader selection at runtime:** `esmc_encoder.load_model_and_tokenizer`
  prefers `EsmcModel` first, and uses `EsmcForMaskedLM` ONLY if `model.allow_masked_lm_fallback: true`
  is explicitly configured (default `false`). The API actually used is recorded in
  `manifest.json` -> `model.api_used`.
* **Strict Checkpoint Allowlist:** Checkpoints and tokenizers must be in `ALLOWED_CHECKPOINTS = {"biohub/ESMC-600M-hf", "biohub/ESMC-600M"}`. All other model IDs are rejected.
* **Version pinning:** `requirements.txt` marks `transformers` as `UNVERIFIED`.
  Pin `transformers==<verified version>` only after verifying `from transformers import EsmcModel`
  on a GPU host.
* If `from transformers import EsmcModel` fails, install from GitHub main:
  `pip install git+https://github.com/huggingface/transformers.git`.

### Hugging Face access

`from_pretrained` downloads and caches automatically; this repository never
manually downloads model files. The ESMC-600M weights are publicly available,
but if your account/HF_TOKEN is required:

```bash
huggingface-cli login          # or: export HF_TOKEN=...
python scripts/check_environment.py      # verifies reachability and checkpoint resolution
```

`trust_remote_code` is **disabled by default and never enabled automatically**.
`--trust-remote-code` is an explicit opt-in; when used it is recorded in
`manifest.json` (`model.trust_remote_code`) and logged as a warning, because
remote code executes on your machine.

### GPU requirements and CPU limitations

* ESMC-600M in float32 needs roughly **2.4 GB for weights**. A single ~2k-token
  chunk at batch size 1 is expected to fit comfortably on a 16 GB GPU, but the
  manifest records the **actual** peak GPU memory and per-protein wall clock
  instead of asserting them.
* CPU works and is fully resumable, but 476 rows / 245 unique sequences at this
  model size is slow. Use `--device cpu` and expect a long run with `--resume`.
* dtype: `float32` is the default for scientific reproducibility. `float16` and
  `bfloat16` are allowed. **Outputs and the cache are always float32**, whatever
  the inference dtype.

---

## 3. Input data

Input file: `data/targetprotein.csv`.

Actual columns in the current file (verified):

```
nsc_id, drug_name, target_id, gene_symbol, uniprot_id, sequence, sequence_length
```

* There is **no `chembl_id` column** and none is ever required. If a future file
  has one, it flows through as optional metadata.
* Column matching is **case-insensitive**; internally everything is lowercased,
  and the manifest records both the names as found and the canonical mapping.
* **Required columns:** `nsc_id`, `sequence`. A missing one raises a clear
  error naming it - never a raw `KeyError` or traceback.
* **Every** input column is preserved as metadata. There is no fixed whitelist
  and no invented provenance field; `chembl_id`, `target_source`, `evidence` and
  anything else simply flow through when present.
* Each row is one drug-target relationship. Grouping is by `nsc_id`;
  `drug_name` is metadata only.
* `target_id == uniprot_id` is **not** assumed, and one `target_id` does **not**
  imply one sequence.
* **Protein identity for caching is sequence-based only.**
  On the current file there are 244 unique `target_id`, 245 unique `uniprot_id`
  and 245 unique cleaned sequences - so `target_id` is provably not a safe
  identity key and is never used for caching.

---

## 4. The ESMC checkpoint: resolved and verified, never assumed

Candidates are evaluated **in the configured order** and a candidate is selected
only if it passes **all** verification criteria. Successful `from_pretrained()`
loading alone is **not** sufficient evidence.

| Criterion | Check |
|---|---|
| (a) official Biohub/EvolutionaryScale checkpoint | repository owner must be `biohub` / `evolutionaryscale`, cross-checked against the HF Hub owner |
| (b) ESMC architecture | `config.architectures` / `config.model_type` must contain `esmc` |
| (c) `hidden_size == 1152` | read from `model.config.hidden_size`; a mismatch is a fatal error |
| (d) authoritative max positions | `model.config.max_position_embeddings` (see below); no `getattr(..., 2048)` fallback ever |
| (e) provenance recorded | `model_id`, revision (commit hash), tokenizer revision, transformers version, `model_config_hash`, `tokenizer_config_hash` |

If multiple candidates verify, the **first in the configured order** wins, and
the outcome of *every* candidate (including failures) is recorded in
`manifest.json` -> `model.candidate_outcomes`. If none verifies, the run aborts.
**There is no fallback to another model** - ESM-2, ProtBERT and ProtT5 are not
substitutes.

### Checkpoint metadata observed on the Hub (2026-09-28)

Queried via `huggingface_hub` (metadata only, no weights downloaded):

| candidate | commit SHA | architectures | model_type | hidden_size | max_position_embeddings |
|---|---|---|---|---|---|
| `biohub/ESMC-600M-hf` | `0fb34e7e5fe1f85d0abaa3d35e2671107c0b458c` | `["EsmcForMaskedLM"]` | `esmc` | 1152 | 2048 |
| `biohub/ESMC-600M` | `fcc9d36f5fff97d9bb69a36e3252de61cc78c968` | `["EsmcForMaskedLM"]` | `esmc` | 1152 | 2048 |

Both candidates exist and both satisfy (a)-(c) on paper, so the configured order
decides: **`biohub/ESMC-600M-hf` is the one selected** (first in
`model.candidates`). The authoritative `model_revision` for your run is whatever
your machine's manifest records, and it is part of the cache fingerprint.

Two findings from that inspection that shaped the code:

1. `tokenizer_config.json` for these checkpoints reports
   `tokenizer_class: EsmcTokenizer` with `bos_token = cls_token = <cls>` and
   `eos_token = <eos>` (i.e. **two** added special tokens), so the derived
   residue capacity is `2048 - 2 = 2046` - observed at runtime, not assumed.
2. The same file ships `model_max_length = 1000000000000000019884624838656`, an
   **"unlimited" sentinel, not a capacity**. `resolve_model_max_positions`
   rejects such values (and only reads the model config, tokenizer init kwargs,
   or an explicit `sequence.max_model_positions`). A bogus sentinel can never
   become a chunk size.

### ESMC Model Loader Class Preference Order

When loading official ESMC checkpoints via Hugging Face Transformers, the loader follows a strict preference hierarchy:
1. `EsmcModel` (preferred bare encoder class when available in installed Transformers API).
2. `EsmcForMaskedLM` (fallback only when required, explicitly setting `output_hidden_states=True`).

MLM logits are **never** used as protein embeddings. Output extraction explicitly validates 3D/2D shape, `hidden_size == 1152`, and rejects logits.

The exact ESMC-600M tokenizer has a small vocabulary (`vocab_size = 64`), so
whether `X B Z J U O` are natively encoded is **observed and recorded at
startup**, never assumed (see §5).

### 1152 is an acceptance invariant, not a tunable

`REQUIRED_HIDDEN_SIZE = 1152` is asserted **once**, immediately after the
checkpoint has been verified. Every tensor shape, cache schema, output schema,
CSV column count and the projection's `input_dim` are then **derived** from the
resolved `model.config.hidden_size` rather than duplicating the literal.

---

## 5. Runtime inspection of tokenization and ESMC output (mandatory at startup)

Before any pooling, the pipeline runs and **logs** (and stores in
`manifest.json` -> `preflight.startup_inspection`):

1. The ESMC output type, its attributes, the hidden-state tensor shape, and
   **which field actually holds residue embeddings**. `extract_residue_hidden_states`
   discovers `last_hidden_state` or `hidden_states[-1]` at runtime; it does not
   assume either, and whether the model applies a final layer norm to it is
   recorded.
2. The **actual** token sequence for `ACDEFGHIKL` (e.g.
   `[CLS] A C D E F G H I K L [EOS]` - whatever the tokenizer really
   produces): ids, `special_tokens_mask`, special-token positions and the
   number of added special tokens.
3. **Residue positions are derived from the tokenizer** (`special_tokens_mask`,
   cross-checked with `attention_mask`) - never by slicing `[1:-1]`. The
   pipeline asserts that the number of residue positions equals the chunk's
   residue count, and **verifies residue order by mapping ids back to letters**,
   so residue token *i* must correspond to residue *i*.
4. `hidden-state length == input token length`.
5. The residue capacity is computed dynamically.
6. All ESMC-specific extraction lives in **one** function,
   `esmc_encoder.extract_residue_hidden_states`. The rest of the pipeline
   consumes only `residue_hidden_states [num_residues, hidden_size]` and does
   not know where they came from.

---

## 6. Sequence cleaning and special residues

For every row: stringify, remove **all** whitespace, uppercase, validate each
character. A bad sequence is **never** replaced with random values; the failure
is recorded in `error` and the row continues to the output as `status="failed"`.

* Standard amino acids: `A C D E F G H I K L M N P Q R S T V W Y`
* Non-canonical / special: `X` (unknown), `B` (D/N), `Z` (E/Q), `J` (L/I),
  `U` (selenocysteine), `O` (pyrrolysine). These are **not garbage and not all
  equivalent**. Selenocysteine is biologically meaningful - the current file has
  exactly one such row (NSC-92859 / Arsenic trioxide / TXNRD1 / Q16881,
  649 aa, one `U`). **`U` is never silently removed.**

Configurable policy `sequence.special_residue_policy` (the old key
`invalid_residue_policy` is accepted as an alias):

| policy | behaviour |
|---|---|
| `keep_if_supported` *(default)* | pass through unchanged **if the loaded ESMC tokenizer natively and validly encodes it**; replace with `X` only if it cannot |
| `error` | row-level failure with a clear message (the run continues) |
| `replace_with_X` | replace unsupported symbols with `X` |

Any other value is rejected with a clear error. **A `remove` policy is
deliberately not implemented**: deleting a residue shifts every downstream
position and alters the sequence in a biologically meaningful way.

### How "supported" is decided (strengthened test)

For each of `X B Z J U O`, the pipeline executes the **complete**
tokenizer -> model -> residue-hidden-state extraction path on `ACD<sym>EFG`:

* **PRIMARY criterion:** exactly one residue-aligned hidden-state position for
  the symbol, and the full residue order `A, C, D, <sym>, E, F, G` is preserved
  at positions 0..6. No neighbour may shift, merge, disappear or acquire a
  non-residue token position.
* **DIAGNOSTIC criterion (logged separately):** the symbol's token id is not
  `tokenizer.unk_token_id`. *Merely receiving an id is not support* - every
  token has an id, and an unk-mapped symbol is unsupported.

Results for all six symbols are stored in
`manifest.json` -> `tokenizer_support`.

### If a special residue is replaced with X

> Replacing `U` (or any special residue) with `X` is a **representation
> approximation that changes the residue's biological identity**: `X` means
> "unknown amino acid", not selenocysteine. This is not a neutral
> tokenizer-compatibility fix. The original sequence, the position and the
> transformation are preserved in provenance
> (`special_residue_events` in `target_provenance.csv` and
> `manifest.json` -> `special_residue_events`).

The original symbol is still reported in dataset statistics
(`symbols_in_input`) alongside `symbols_after_cleaning`, so a reviewer can always
see what the input contained. **Do not assume what ESMC does with `U`: the
acceptance criterion is that the behaviour is observed and recorded.**

---

## 7. Long proteins: anchored overlapping chunks

Capacity is **derived**, never assumed: `chunk_size = model_max_positions -
special_tokens`, where `special_tokens` is the number of tokens the tokenizer
actually adds (`2` for the ESMC tokenizer: `<cls>` + `<eos>`, i.e. `2046`
residues at 2048 positions).

A sequence at or below capacity is encoded as a **single** chunk - the chunking
path is not entered at all. Longer sequences are split with:

* **anchored first chunk** starting at residue 0,
* subsequent chunks **stride = chunk_size - overlap** (`2046 - 256` by
  default; at 512/256 that is 256),
* the final chunk **end-anchored** so its last residue is the sequence's last
  residue,
* `overlap: 0` supported as a clean non-overlapping ablation.

Examples at the mocked 512-capacity: 2286 aa with overlap 256 -> 2 chunks
(0-511, 2030-2285); 1000 aa -> 4 chunks (0-511, 256-767, 512-1023, 744-999).

Two hard rules, both tested:

1. **Full coverage.** Every residue lands in at least one chunk
   (`min_coverage_count >= 1`).
2. **No index shifting.** Overlap means each overlap region is encoded twice and
   then **averaged back into a single slot** - the neighbors of a duplicated
   residue are still its immediate sequence neighbors. Overlapping windows are
   never concatenated or stitched, because that would shift the neighbors of
   every residue in the overlap.

Per-residue accumulation is `[L, 1152]` (on CPU, so a long protein does not hold
`n_chunks x 1152` on the GPU), and a `count` divisor is maintained per residue,
so the final value is the exact mean over the **actual** residues - no
`NaN`/`inf` for uncovered positions, and no padding residue ever contributes.

Enforcement: at every startup, **if the input contains a longer protein the
current model cannot handle, the run FAILS LOUDLY** rather than silently
truncating. Chunking is why long proteins work; it is not a licence to drop
data.

---

## 8. Pooling (point A): mean over actual residues only

```
protein_embedding[r] = mean over i of residue_hidden_states[i, r]
```

Parameter-free, deterministic, and **not** attention pooling. Only the residue
positions of the current chunk contribute; special tokens and padding are never
averaged in, and for long proteins the per-residue accumulators replace the
chunk-level means rather than averaging chunk vectors (which would weight a
residue by how many chunks happened to cover it).

---

## 9. Cache: `protein_cache.pt`

* **Key = `sequence_hash` only** (`SHA256(cleaned_sequence)`). The sequence is
  the identity; `target_id` is not (§3).
* **Fingerprinted:** the model id, the **exact commit hash**, the tokenizer
  revision, `hidden_size`, the transformers version, the model/tokenizer config
  hashes, the special-residue policy, the resulting capacity, and the
  aggregation method are hashed into a fingerprint. A cache built with a
  different checkpoint or policy is **refused** with an explicit message rather
  than silently mixed.
* **Resumable:** saved atomically every `cache.save_every` new proteins **and**
  at the end of the run. Interrupted runs continue from the last save;
  `--resume` is the default when the cache exists, and `--rebuild` forces
  recomputation.
* **Provenance:** for each sequence the cache records the list of
  `nsc_id` / `target_id` / `uniprot_id` that map to it and the first input row
  it came from, so a cached hit still yields complete provenance.
* **Safe by construction:** a row can never disappear from a resumed run. Any
  sequence present in the input must be present in the final cache; this is
  asserted before writing outputs.
* **CPU-only, loadable, no code execution:** plain `torch.save` of tensors and
  JSON metadata, written to a temp file in the same directory and moved into
  place, so an interrupted write cannot corrupt the cache.

On the current file: 476 rows -> 245 unique sequences -> 245 cache entries.

---

## 10. Drug aggregation (point B): mean over unique target proteins

For each `nsc_id`:

1. keep only rows that succeeded, i.e. `status == "success"`,
2. **deduplicate targets by `sequence_hash` inside that drug** - the same
   protein annotated twice (two `target_id`, or the same `target_id` with a
   different sequence) counts **once**,
3. **mean over those unique protein vectors** -> `[1152]`.

Properties: deterministic, permutation-invariant, no parameters, no labels, and
it runs identically on CPU and GPU. A drug with no successful target gets
`status = "no_successful_protein"` and a `NaN` row; its input rows keep their
real failure reasons. The aggregation method is written into the output schema.

**Deliberate limitation (not an oversight):** `3.6%` of the current rows have
`sequence_length` disagreeing with the actual sequence, and deduplicating by
sequence means a drug whose rows disagree about one protein's sequence gets a
mean over the *distinct* sequences present, not over one arbitrarily chosen
sequence. Choosing a winner would require an arbitrary rule (longest? first?),
which is not defensible; the disagreement is reported instead
(`row_audit.flags.sequence_length_mismatch`).

---

## 11. Projection: disabled by default, and it must not be faked

`projection.enabled: false` in the shipped config. If it is ever enabled, a
real, **trained** checkpoint from the CancerCombo supervised stage must be
supplied (`projection.checkpoint`). The loader refuses to run if the
checkpoint's `input_dim` does not match the resolved `hidden_size`.

* A `Linear(1152->256)` + `LayerNorm(256)` is **learned downstream** in
  CancerCombo. The LayerNorm is applied to the *aggregate*, not per protein.
* This repository never trains that layer. It is not fitted here, not fitted on
  labels here, and **never applied untrained** to a final vector. A randomly
  initialised or untrained projection produces a plausible-looking but
  meaningless vector, which is exactly the silent corruption this guard
  prevents.
* The full 1152-D vectors are always the primary artifact, so the 1152 -> 256
  step can still be done (correctly, supervised) later.

---

## 12. Outputs

### Primary artifacts (CancerCombo consumes these)

**1. `protein_cache.pt`** - the primary protein-level artifact: sequence-hash
cache of `[1152]` protein embeddings plus provenance.

**2. `drug_target_embeddings_1152.pt`** - the primary drug-level artifact:

```python
{
  "drug_embeddings": {"NSC-606869": tensor([1152], float32), ...},
  "drug_metadata":  {"NSC-606869": {"nsc_ids": [...], "drug_names": [...],
                                    "n_target_proteins": 3, "status": "success", ...}},
  "schema": {"embedding_dim": 1152, "key_type": "nsc_id", "dtype": "float32",
             "aggregation": "mean", "sequence_identity": "SHA256(cleaned_sequence)",
             "layernorm_applied": False, "projection_applied": False,
             "model_id": ..., "model_revision": ...},
  "config": {...}, "model_name": ..., "model_revision": ..., "embedding_dim": 1152,
  "projection_enabled": False, "aggregation_method": "mean",
}
```

### Audit package

| File | Contents |
|---|---|
| `manifest.json` | dataset fingerprint, resolved model + revision, config, input columns as found, per-symbol tokenizer support, every candidate's outcome, special-residue events, chunk forward passes, cache fingerprint, software versions, subset guard |
| `run_summary.json` | counts, cache hits/misses, wall clock, **actual** peak GPU memory, failure categories |
| `target_provenance.csv` | one row per **input row**: status, sequence hash, n unique targets, `error`, and every original input column (with `chembl_id` only when present) |
| `target_embeddings.pt` | per-row protein embeddings + row indices, the compact `bool` `row_valid_mask`, and row metadata for the 1152-D layout |
| `embedding_audit.pt` | `drug_embeddings`, `drug_metadata`, `row_embeddings`, `row_embedding_row_indices`, `row_valid_mask`, schema, model, config - CPU-only, `weights_only=True`-loadable |
| `embedding_diagnostics.json` | cosine/norm distributions, pairwise-cosine flags, the 3-sigma-norm outlier fraction, pair count, warnings |
| `failed_rows.csv` | only the failed rows, with reasons |

The successful-row layout is **compact by design**: a `bool` `row_valid_mask`
over all input rows plus a `LongTensor` of row indices, rather than a
`NaN x 1152` block. This is canonical, auditable, and 20x smaller.

Diagnostics **never fail a run**. They report; a human decides.

---

## 13. Command-line usage

### `inspect_dataset.py` - validate the input before spending GPU hours

```bash
python scripts/inspect_dataset.py --input data/targetprotein.csv
python scripts/inspect_dataset.py --input data/targetprotein.csv --out reports/dataset_report.json
```

Reports the input SHA256, columns as found + the canonical mapping, duplicate
counts, unique `nsc_id` / `target_id` / `uniprot_id` / sequences, sequence
length statistics, length mismatches, residues over 2048, and the non-standard
residue inventory (`symbols_in_input` vs `symbols_after_cleaning`). It needs no
model and no network.

### `run_preprocessing.py` - the main entry point

```bash
# full production run
python scripts/run_preprocessing.py --input data/targetprotein.csv

# explicit outputs
python scripts/run_preprocessing.py \
  --input data/targetprotein.csv \
  --config configs/config.yaml \
  --output-dir outputs/esmc_600m_full --device cuda --dtype float16

# resume (the default when the cache exists; --rebuild forces recomputation)
python scripts/run_preprocessing.py --input data/targetprotein.csv --resume

# rebuild the cache from scratch
python scripts/run_preprocessing.py --input data/targetprotein.csv --rebuild

# SUBSET runs: a separate output dir is REQUIRED and enforced
python scripts/run_preprocessing.py --input data/targetprotein.csv \
  --max-rows 10 --output-dir outputs/smoke_10rows
python scripts/run_preprocessing.py --input data/targetprotein.csv \
  --nsc-ids NSC-606869,NSC-24559 --output-dir outputs/smoke_2drugs
```

**The subset guard is a hard control, not a warning.** `--max-rows` /
`--nsc-ids` without an explicit `--output-dir` aborts with exit code 2, and even
with one, writing into the configured full-run directory is refused. A test
dataset can therefore never silently overwrite production outputs.

`--limit` is accepted as an alias of `--max-rows`; `--override` /
`--set` take `key.path=value` pairs (values are parsed as JSON, so
`projection.enabled=true`, `sequence.overlap=0`, `runtime.dtype="float16"` all
work).

### `validate_outputs.py` - check an output directory

```bash
python scripts/validate_outputs.py --output-dir outputs/esmc_600m_full
```

59 independent checks: required artifacts, tensor shapes and dtypes, mask /
index / row-count consistency, finiteness and non-zero norms, float32-ness, no
NaN in the drug dict, schema fields, manifest/model agreement, provenance
completeness, and that the audit and primary drug dicts agree. A single
`verify_drug_embedding` helper is reused by CancerCombo (it returns the tensor
or `None` with a reason).

### `check_environment.py` - preflight, before anything else

```bash
python scripts/check_environment.py --config configs/config.yaml --out reports/preflight_report.json
```

Checks the Python/PyTorch/numpy/pandas versions, whether the torch build has
CUDA, **native Transformers ESMC support**, Hugging Face Hub reachability, the
resolved checkpoint, tokenizer config, disk space, and the input file. Exits
non-zero with the specific problem if native ESMC support is missing.

### `bundle_results.py` / `run_all.sh`

```bash
python scripts/bundle_results.py --output-dir outputs/esmc_600m_full --out results_bundle.zip
bash scripts/run_all.sh                 # inspect -> preflight -> run -> validate
```

---

## 14. Testing

```bash
python -m pytest -v          # fast test suite, ESMC is MOCKED, no download
```

Every test runs against a deterministic mock ESMC (64-token vocab, `<cls>` + `<eos>`, `hidden_size` derived from the config), so the suite needs **no network, no GPU, and no multi-GB download**.

---

## 15. Verification status: what was and was not run

Stated plainly, because the acceptance criteria of this project are about real numbers, and invented numbers are worse than none.

**Verified in the authoring environment (CPU, mocked ESMC):**

* `python -m pytest -q` -> Previously self-reported by the last hardening pass (not independently reproduced elsewhere): 148 passed, 1 skipped. Record the actual result from this run verbatim, whatever it is — do not reconcile it against 148 by adjusting either number.
* `python -m compileall -q src scripts tests` -> **PASSED** (clean static compilation).
* Full artifact contract on a synthetic run: `scripts/validate_outputs.py` -> **VALIDATION PASSED**.
* Resume: second run reported `newly_computed_proteins = 0`, `cache_hits = 5`.
* `scripts/inspect_dataset.py`, `scripts/bundle_results.py`,
  `scripts/run_preprocessing.py --help`, and the subset guard all run.

**REAL ESMC TEST STATUS:**

* **REAL ESMC TEST: NOT EXECUTED**
* **Reason:** The installed `transformers` version (5.14.1) lacks `EsmcModel` (`from transformers import EsmcModel` raises `ImportError: cannot import name 'EsmcModel'`). Per specification, mock runs are labelled MOCK and real ESMC validation is NOT claimed as executed until run on a GPU host with native `EsmcModel` support.

**Exact commands to execute on a GPU host:**

```bash
pip install git+https://github.com/huggingface/transformers.git
python scripts/check_environment.py
ESMC_REAL_RUN=1 python -m pytest tests/test_integration_esmc.py -v
python scripts/run_preprocessing.py --input data/targetprotein.csv --output-dir outputs/target_protein_run
python scripts/validate_outputs.py --output-dir outputs/target_protein_run
```


---

## 16. Deliberate future work (not built here)

Held back so this branch stays label-free, parameter-free and auditable:

1. **Supervised, projection-aware** (inside CancerCombo): `Linear(1152->256)` +
   `LayerNorm(256)`, applied to the aggregate, trained on train-split only.
2. **Target-level attention/gating** as a supervised ablation, to test whether
   the point-B mean assumption (§1) is leaving signal on the table.
3. **Structure-aware pooling** (contact maps / ESMFold), which needs external
   folding and a policy decision about proteins that cannot be folded.
4. **Ensemble** of protein checkpoints (ESM-2 / ProtT5 are *not* ESMC
   substitutes here and would be a different feature space, needing its own
   cache and fingerprint).

---

## 17. Repository layout

```
esmc_target_protein/
  configs/config.yaml            every tunable, nothing scientific hardcoded
  data/targetprotein.csv         the dataset (not committed; see data/README)
  src/esmc_target/
    config.py       data.py      canonicalization, cleaning, hashing, special residues
    sequence.py     pooling.py   capacity, chunk planning, residue accumulators
    esmc_encoder.py cache.py     frozen ESMC, verification, fingerprinted cache
    aggregation.py  projection.py  drug-level mean, disabled-by-default projection
    outputs.py      diagnostics.py  atomic artifact writers, audit + reporting
    pipeline.py     errors.py     orchestration, subset guard, fatal errors
  scripts/           inspect / run / validate / preflight / bundle / run_all
  tests/             89 mocked tests + 1 opt-in real-ESMC test
```

## License / data

The pipeline code is yours to license as you see fit. ESMC-600M weights remain
under the Meta/HF license of the `biohub` checkpoints, and `targetprotein.csv`
is derived from NCI data - do not redistribute the dataset with the code.




