#!/usr/bin/env bash
# Full acceptance run: preflight -> smoke -> edge -> full -> resume -> validate.
# Every step tees to a log; the script stops at the first failure.
# Usage:  bash scripts/run_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
INPUT="${INPUT:-data/targetprotein.csv}"
CONFIG="${CONFIG:-configs/config.yaml}"
LOGS="$REPO_ROOT/outputs/logs"
mkdir -p "$LOGS"

step() {
  echo ""
  echo "=============================================================================="
  echo ">>> $* "
  echo "=============================================================================="
}

step "0. Environment preflight (needs no dataset)"
"$PY" scripts/check_environment.py --config "$CONFIG" --out outputs/preflight_report.json 2>&1 \
  | tee "$LOGS/00_preflight.log"

step "1. Smoke test: first five rows only (isolated output dir)"
"$PY" scripts/run_preprocessing.py --input "$INPUT" --config "$CONFIG" \
  --max-rows 5 --output-dir outputs/smoke_test 2>&1 | tee "$LOGS/01_smoke_test.log"

step "2. Edge cases: selenocysteine row, three long proteins, POLE across two drugs"
"$PY" scripts/run_preprocessing.py --input "$INPUT" --config "$CONFIG" \
  --nsc-ids NSC-92859,NSC-24559,NSC-756645,NSC-606869,NSC-613327 \
  --output-dir outputs/edge_cases 2>&1 | tee "$LOGS/02_edge_cases.log"

step "3. Full run"
"$PY" scripts/run_preprocessing.py --input "$INPUT" --config "$CONFIG" \
  --output-dir outputs/full --resume 2>&1 | tee "$LOGS/03_full_run.log"

step "4. Resume re-run (must compute 0 new proteins)"
"$PY" scripts/run_preprocessing.py --input "$INPUT" --config "$CONFIG" \
  --output-dir outputs/full --resume 2>&1 | tee "$LOGS/04_resume.log"

step "5. Validate the full run"
"$PY" scripts/validate_outputs.py --output-dir outputs/full 2>&1 | tee "$LOGS/05_validate.log"

step "6. Bundle logs, manifests and provenance for review"
"$PY" scripts/bundle_results.py --output-dir outputs/full --out outputs/results_bundle.zip 2>&1 \
  | tee "$LOGS/06_bundle.log"

echo ""
echo "ALL STEPS COMPLETED. Review outputs/logs/*.log, outputs/full/manifest.json and"
echo "outputs/results_bundle.zip. The projection-enabled path is deliberately NOT run:"
echo "it requires a genuinely trained CancerCombo projection checkpoint."
