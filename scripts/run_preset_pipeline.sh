#!/bin/bash
# End-to-end OCS preset pipeline on real Qwen weights
# Compute plan → EPS baseline → OCS runtime → OCS preset → compare
#
# Usage:
#   bash scripts/run_preset_pipeline.sh
#   bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json
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

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "OCS PRESET PIPELINE (real Qwen weights)"
echo "============================================================"
echo "Trace:         $TRACE"
echo "Max circuits:  $MAX_CIRCUITS"
echo "World size:    $WORLD_SIZE"
echo "Experts/rank:  $EXPERTS_PER_RANK"
echo "Output dir:    $OUTPUT_DIR"
echo ""

# ---- Step 1: Compute placement plan ----
echo "[Step 1/3] Computing OCS circuit placement plan..."
PLAN_PATH="$OUTPUT_DIR/preset_plan.json"
python3 scripts/compute_preset_plan.py \
    --trace "$TRACE" \
    --output "$PLAN_PATH" \
    --max-circuits "$MAX_CIRCUITS" \
    --experts-per-rank "$EXPERTS_PER_RANK" \
    --world-size "$WORLD_SIZE"
echo ""

# ---- Step 2A: Run EPS baseline (overlap mode, no OCS) ----
echo "[Step 2A/3] Running EPS baseline (overlap, real Qwen experts)..."
EPS_TRACE_DIR="$OUTPUT_DIR/traces_eps_baseline"
python3 -m src.launcher \
    --config configs/qwen_replay.yaml \
    --trace-dir "$EPS_TRACE_DIR" \
    2>&1 | tail -5
echo ""

# ---- Step 2B: Run OCS pipeline (runtime reconfig) ----
echo "[Step 2B/3] Running OCS pipeline (runtime reconfig)..."
OCS_TRACE_DIR="$OUTPUT_DIR/traces_ocs_runtime"
python3 -m src.launcher \
    --config configs/qwen_ocs_pipeline.yaml \
    --trace-dir "$OCS_TRACE_DIR" \
    2>&1 | tail -5
echo ""

# ---- Step 2C: Run OCS preset ----
# Create a temporary config that points to the computed plan
echo "[Step 2C/3] Running OCS preset (pre-configured circuits)..."
PRESET_TRACE_DIR="$OUTPUT_DIR/traces_ocs_preset"
PRESET_CONFIG="$OUTPUT_DIR/ocs_preset_temp.yaml"

python3 -c "
import yaml, os, sys, json
sys.path.insert(0, '.')

# Fully resolve the qwen OCS base config (so extends: is inlined)
from src.launcher import load_config
cfg = load_config('configs/qwen_ocs_base.yaml')

# Point to the computed plan
cfg.setdefault('ocs', {})['preset'] = {
    'source': 'plan',
    'plan_path': '$PLAN_PATH',
    'strategy': 'coactivation',
}
cfg['runtime'] = {'mode': 'ocs_preset', 'num_steps': 3}

with open('$PRESET_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
print(f'Temp config written -> $PRESET_CONFIG')
"

python3 -m src.launcher \
    --config "$PRESET_CONFIG" \
    --trace-dir "$PRESET_TRACE_DIR" \
    2>&1 | tail -5
rm -f "$PRESET_CONFIG"
echo ""

# ---- Step 3: Compare results ----
echo "[Step 3/3] Comparing results..."
python3 -c "
import json, os, glob

def load_trace_meta(trace_dir):
    files = sorted(glob.glob(os.path.join(trace_dir, 'rank_*_trace.json')))
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)

for label, path in [
    ('EPS baseline', '$EPS_TRACE_DIR'),
    ('OCS runtime', '$OCS_TRACE_DIR'),
    ('OCS preset',  '$PRESET_TRACE_DIR'),
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
