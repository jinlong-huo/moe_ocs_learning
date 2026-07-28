#!/bin/bash
# End-to-end OCS preset pipeline
# Train → capture affinity → compute plan → preset inference → compare
#
# Usage:
#   bash scripts/run_preset_pipeline.sh
#   bash scripts/run_preset_pipeline.sh --trace data/routing_traces/routing.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Config ---
TRACE="${1:-data/routing_traces/routing.json}"
MAX_CIRCUITS="${2:-16}"
EXPERTS_PER_RANK="${3:-4}"
WORLD_SIZE="${4:-4}"
OUTPUT_DIR="outputs/preset_pipeline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "OCS PRESET PIPELINE"
echo "============================================================"
echo "Trace:         $TRACE"
echo "Max circuits:  $MAX_CIRCUITS"
echo "World size:    $WORLD_SIZE"
echo "Experts/rank:  $EXPERTS_PER_RANK"
echo "Output dir:    $OUTPUT_DIR"
echo ""

# ---- Step 1: Validate affinity consistency ----
echo "[Step 1/4] Validating affinity consistency..."
if python3 scripts/validate_affinity.py \
    --train-trace "$TRACE" \
    --infer-trace "$TRACE" \
    --num-experts $((WORLD_SIZE * EXPERTS_PER_RANK)) \
    --top-k 2 \
    --max-circuits "$MAX_CIRCUITS" \
    --output "$OUTPUT_DIR/affinity_report.json" 2>&1; then
    echo "  Affinity validation OK"
else
    echo "  WARNING: affinity validation may be incomplete (single trace used for both)"
fi
echo ""

# ---- Step 2: Compute placement plan ----
echo "[Step 2/4] Computing OCS circuit placement plan..."
PLAN_PATH="$OUTPUT_DIR/preset_plan.json"
python3 scripts/compute_preset_plan.py \
    --trace "$TRACE" \
    --output "$PLAN_PATH" \
    --max-circuits "$MAX_CIRCUITS" \
    --experts-per-rank "$EXPERTS_PER_RANK" \
    --world-size "$WORLD_SIZE"
echo ""

# ---- Step 3A: Run EPS baseline ----
echo "[Step 3/4] Running EPS baseline (serial)..."
python3 -m src.launcher \
    --config configs/synthetic_moe.yaml \
    --runtime.mode serial \
    --profiling.trace_dir "$OUTPUT_DIR/traces_eps_baseline" \
    2>&1 | tail -2
echo ""

# ---- Step 3B: Run OCS pipeline (runtime reconfig) ----
echo "[Step 3/4] Running OCS pipeline (runtime reconfig)..."
python3 -m src.launcher \
    --config configs/ocs_demo.yaml \
    --profiling.trace_dir "$OUTPUT_DIR/traces_ocs_runtime" \
    2>&1 | tail -2
echo ""

# ---- Step 3C: Run OCS preset ----
echo "[Step 3/4] Running OCS preset (pre-configured circuits)..."
python3 -m src.launcher \
    --config configs/ocs_preset.yaml \
    --ocs.preset.source plan \
    --ocs.preset.plan_path "$PLAN_PATH" \
    --profiling.trace_dir "$OUTPUT_DIR/traces_ocs_preset" \
    2>&1 | tail -2
echo ""

# ---- Step 4: Compare results ----
echo "[Step 4/4] Comparing results..."
python3 -c "
import json, os, glob

def load_trace_meta(trace_dir):
    files = sorted(glob.glob(os.path.join(trace_dir, 'rank_*_trace.json')))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)

for label, path in [
    ('EPS baseline', '$OUTPUT_DIR/traces_eps_baseline'),
    ('OCS runtime', '$OUTPUT_DIR/traces_ocs_runtime'),
    ('OCS preset',  '$OUTPUT_DIR/traces_ocs_preset'),
]:
    meta = load_trace_meta(path)
    if meta:
        ocs = meta.get('metadata', {}).get('ocs', {})
        metrics = ocs.get('metrics', {})
        reuse = metrics.get('reuse_ratio', 0)
        reconfig = metrics.get('total_reconfig_time_us', 0)
        print(f'{label}: reuse_ratio={reuse:.2f} reconfig_total={reconfig:.0f}us')
    else:
        print(f'{label}: no trace data found')
"

echo ""
echo "============================================================"
echo "Pipeline complete. Results in: $OUTPUT_DIR"
echo "============================================================"
