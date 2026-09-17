#!/usr/bin/env bash
# Run ASTRA-sim against this repo's network configs. Works from any directory.
#
# ── Prerequisites (macOS) ────────────────────────────────────────────────────
#   xcode-select --install                     # clang / make
#   brew install cmake protobuf                # verified with protoc 35.1
#   cd ~/astra-sim && git submodule update --init --recursive   # 7 submodules
#
#   The binary is built by configs/astra_sim/build_astrasim.sh, which this script
#   calls automatically if it is missing (or if ASTRA_BUILD=1 is set).
#
# ── Usage ────────────────────────────────────────────────────────────────────
#   bash configs/astra_sim/run_astrasim.sh            # OCS-promoted config
#   bash configs/astra_sim/run_astrasim.sh eps        # EPS with sigma=4
#   bash configs/astra_sim/run_astrasim.sh compare    # both, with the ratio
#   bash configs/astra_sim/run_astrasim.sh /path/to/other.yml
#
#   Overrides: ASTRA_SIM_DIR, WORKLOAD, SYSTEM, REMOTE_MEMORY, BINARY
#
# Both configs are a 16-NPU all-to-all over a single ring tier; they differ only
# in that tier's bandwidth (12.5 vs 50 GB/s), which is this repo's model of what
# an optical circuit buys (a tier promotion, not extra bandwidth).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASTRA="${ASTRA_SIM_DIR:-$HOME/astra-sim}"
BIN_DIR="$ASTRA/build/astra_analytical/build/bin"
BIN="${BINARY:-$BIN_DIR/AstraSim_Analytical_Congestion_Unaware}"

WORKLOAD="${WORKLOAD:-$ASTRA/examples/workload/microbenchmarks/all_to_all/16npus_1MB/all_to_all}"
SYSTEM="${SYSTEM:-$ASTRA/examples/system/native_collectives/HGX-H100-validated.json}"
REMOTE_MEMORY="${REMOTE_MEMORY:-$ASTRA/examples/remote_memory/analytical/no_memory_expansion.json}"

die() { echo "error: $*" >&2; exit 1; }

# ── dependency checks, each with the command that fixes it ───────────────────
command -v cmake  >/dev/null || die "cmake not found        -> brew install cmake"
command -v protoc >/dev/null || die "protoc not found       -> brew install protobuf"
[ -d "$ASTRA" ] || die "no ASTRA-sim checkout at $ASTRA   -> git clone https://github.com/astra-sim/astra-sim.git, or set ASTRA_SIM_DIR"
[ -f "$ASTRA/build/astra_analytical/build.sh" ] || die "$ASTRA is not an ASTRA-sim checkout"
[ -n "$(ls -A "$ASTRA/extern/graph_frontend/chakra" 2>/dev/null)" ] || \
  die "submodules are empty   -> cd $ASTRA && git submodule update --init --recursive"
[ -f "$WORKLOAD".0.et ] || die "workload not found: $WORKLOAD.et (set WORKLOAD=...)"

if [ ! -x "$BIN" ] || [ "${ASTRA_BUILD:-0}" = "1" ]; then
  echo "[run] binary missing or ASTRA_BUILD=1 -> building first"
  bash "$HERE/build_astrasim.sh"
fi
[ -x "$BIN" ] || die "build did not produce $BIN"

# ── presets ──────────────────────────────────────────────────────────────────
run_one() {  # $1 = label, $2 = network yml
  local label="$1" net="$2" cycles
  [ -f "$net" ] || die "network config not found: $net"
  cycles="$("$BIN" \
      --workload-configuration="$WORKLOAD" \
      --system-configuration="$SYSTEM" \
      --remote-memory-configuration="$REMOTE_MEMORY" \
      --network-configuration="$net" 2>&1 \
    | grep -oE 'finished, [0-9]+ cycles' | head -1 | grep -oE '[0-9]+')"
  [ -n "$cycles" ] || die "no cycle count in output (run the binary by hand to see why)"
  echo "$cycles"
}

case "${1:-ocs}" in
  eps|sigma4)  NET="$HERE/ring16_eps_sigma4.yml";   LABEL="EPS sigma=4" ;;
  ocs|promoted) NET="$HERE/ring16_ocs_promoted.yml"; LABEL="OCS promoted" ;;
  compare)     NET=""; LABEL="compare" ;;
  *)           NET="$1"; LABEL="custom" ;;
esac

echo "[run] binary   : $BIN"
echo "[run] workload : $(basename "$WORKLOAD") ($(ls "$WORKLOAD".*.et | wc -l | tr -d ' ') ranks)"
echo

if [ "$LABEL" = "compare" ]; then
  EPS=$(run_one "eps" "$HERE/ring16_eps_sigma4.yml")
  OCS=$(run_one "ocs" "$HERE/ring16_ocs_promoted.yml")
  printf '[run] EPS sigma=4 (12.5 GB/s) : %s cycles\n' "$EPS"
  printf '[run] OCS promoted (50 GB/s)  : %s cycles\n' "$OCS"
  python3 - "$EPS" "$OCS" <<'PY'
import sys
e, o = (float(x) for x in sys.argv[1:3])
print(f"[run] speedup                 : {e/o:.2f}x  (bandwidth ratio 4.00x)")
PY
else
  [ -f "$NET" ] || die "network config not found: $NET"
  echo "[run] network  : $NET"
  printf '[run] %s: %s cycles\n' "$LABEL" "$(run_one "$LABEL" "$NET")"
fi
