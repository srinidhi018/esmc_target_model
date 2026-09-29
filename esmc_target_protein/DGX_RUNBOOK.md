# DGX Execution Runbook
**Target-Protein ESMC-600M Preprocessing Pipeline**

> [!IMPORTANT]
> Steps 1–11 in this runbook have NEVER been executed end-to-end anywhere yet, as authoring and local verification occurred on a CPU machine lacking CUDA and native `EsmcModel` support in the installed transformers version. Steps 1–11 are staged for execution on a fresh DGX / GPU host session.

---

### Step 1: Environment Gate 1 (PyTorch & CUDA Status)
Run to verify PyTorch installation and GPU availability:
```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

---

### Step 2: Environment Gate 2 (Transformers Native ESMC Status)
Verify native `EsmcModel` imports from transformers:
```bash
python -c "import transformers; print('transformers:', transformers.__version__); from transformers import EsmcModel; print('native EsmcModel: OK')"
```

---

### Step 3: Install Native ESMC Dependencies (If Needed)
If Transformers is missing or native `EsmcModel` is unavailable, install the verified Transformers version required for the ESMC-600M checkpoint rather than blindly upgrading to latest:
```bash
pip install "transformers==<VERIFIED_TRANSFORMERS_VERSION>"
```
*Note: Do NOT blindly upgrade with `pip install -U transformers`. The exact version is frozen only after successful real ESMC execution on DGX.*

---

### Step 4: Local Test Suite Verification
Verify local unit tests pass:
> Previously self-reported by the last hardening pass (not independently reproduced elsewhere): 148 passed, 1 skipped. Record the actual result from this run verbatim, whatever it is — do not reconcile it against 148 by adjusting either number.

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

---

### Step 5: Preflight Check (Environment & Dataset Facts)
Run preflight check to inspect model resolution, CUDA capability, and special residue (U/TXNRD1) support prior to any inference/download:
```bash
python scripts/check_environment.py --config configs/config.yaml
```

---

### Step 6: Real ESMC-600M Integration Test
Execute the opt-in real pretrained model forward pass test on GPU (this is the 1 skipped test in Step 4; on DGX it must execute):
```bash
ESMC_REAL_RUN=1 python -m pytest tests/test_integration_esmc.py -v
```

> **Frozen Environment Lock Generation**:
> Immediately after Step 6 passes cleanly, generate the reproducible environment lockfile:
> ```bash
> pip freeze > requirements.lock
> ```
> *Note: `requirements.lock` serves as the frozen, reproducible environment record once real ESMC execution succeeds.*

---

### Step 7: Smoke Test
Run isolated smoke test on the first 5 rows:
```bash
python scripts/run_preprocessing.py --input data/targetprotein.csv --config configs/config.yaml --max-rows 5 --output-dir outputs/smoke_test
```

---

### Step 8: Edge Cases Verification
Test known stress rows:
- `NSC-92859` → TXNRD1 → Q16881 → 649 aa: tests the known U-containing sequence case.
- `NSC-24559` → MUC6 → Q6W4X9 → 2439 aa: tests a 2439-aa long protein.
- `NSC-756645` → ROS1 → P08922 → 2347 aa: tests a 2347-aa long protein.
- `NSC-606869` → POLE → Q07864 → 2286 aa: tests a 2286-aa long protein.

```bash
python scripts/run_preprocessing.py --input data/targetprotein.csv --config configs/config.yaml --nsc-ids NSC-92859,NSC-24559,NSC-756645,NSC-606869 --output-dir outputs/edge_cases
```
*Note: These are edge-case validation runs, not changes to the production pipeline.*

---

### Step 9: Full Production Run
Execute full preprocessing pipeline on all 476 target dataset rows:
```bash
python scripts/run_preprocessing.py --input data/targetprotein.csv --config configs/config.yaml --output-dir outputs/target_protein_run --resume
```

---

### Step 10: Cache Resume Verification
Re-run full pipeline command against the output directory to verify atomic cache hit and 0 newly computed proteins:
```bash
python scripts/run_preprocessing.py --input data/targetprotein.csv --config configs/config.yaml --output-dir outputs/target_protein_run --resume
```

---

### Step 11: Output Validation
Validate generated embeddings, provenance files, drug aggregation, and manifests:
```bash
python scripts/validate_outputs.py --output-dir outputs/target_protein_run
```
