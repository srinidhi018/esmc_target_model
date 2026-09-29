# Input data

`targetprotein.csv` is **not committed** (see `../.gitignore`). Place the real
file here before running the pipeline.

## Expected file

```
data/targetprotein.csv
```

One row per drug-target relationship.

### Columns (current file, verified)

| column | required | meaning |
|---|---|---|
| `nsc_id` | **yes** | drug key; grouping key for aggregation |
| `sequence` | **yes** | protein amino-acid sequence, single-letter |
| `drug_name` | no | metadata only, carried through to provenance |
| `target_id` | no | metadata; **never** used as a cache identity |
| `gene_symbol` | no | metadata |
| `uniprot_id` | no | metadata |
| `sequence_length` | no | declared length; a disagreement is a **warning** |

* Column names are matched **case-insensitively**; internal names are lowercase.
* Missing required column -> a clear error naming it, not a `KeyError`.
* **Every** input column is preserved as metadata. Optional columns such as
  `chembl_id` are carried through automatically when present, and are never
  required.
* `target_id` and `uniprot_id` are treated as independent fields. Do not assume
  `target_id == uniprot_id`, or that one `target_id` implies one sequence - on
  the current file there are 244 unique `target_id` but 245 unique sequences.
* **Cache identity is `SHA256(cleaned_sequence)` only.**

## Accepted residue symbols

* standard: `A C D E F G H I K L M N P Q R S T V W Y`
* non-standard / special (policy-driven, never silently deleted):
  `X B Z J U O`
* anything else (for example `*` when it stands alone) -> the row fails with a
  reason in `error` and lands in `failed_rows.csv`

See `../README.md` §6 for the special-residue policy and §4 for the identity
problem that makes `sequence_length` optional.

## Before you run

```bash
python scripts/inspect_dataset.py --input data/targetprotein.csv
python scripts/check_environment.py
```

`inspect_dataset.py` reports the input SHA256, the column mapping, duplicates,
length statistics and mismatches, and the non-standard residue inventory - all
without a model, a GPU, or a network.
