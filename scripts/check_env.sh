#!/usr/bin/env bash
# check_env.sh — one command that proves this repo is runnable, and names why not.
#
# Why this exists: the test suite passes only under an interpreter that has BOTH
# numpy and scipy.  Measured on 2026-09-17:
#
#   python3        3.14.6 (Homebrew)   no numpy        -> 2 errors
#   .venv/bin/python 3.12 (project)    no scipy        -> 1 error
#   python3.12     3.12.4              numpy 2.4.4 + scipy 1.16.3 -> 33 pass
#
# so "33 tests passing" is not reproducible from the README as written.  This
# script finds the interpreter that actually works and reports it.  It changes
# nothing outside itself.
#
#   bash scripts/check_env.sh          # probe + run the suite
#   bash scripts/check_env.sh --quiet  # print the interpreter only
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

CANDIDATES=()
for c in python3.12 python3.13 python3.11 python3; do
  command -v "$c" >/dev/null 2>&1 && CANDIDATES+=("$(command -v "$c")")
done
[ -x "$REPO/.venv/bin/python" ] && CANDIDATES+=("$REPO/.venv/bin/python")

PY=""
for p in "${CANDIDATES[@]}"; do
  if "$p" -c 'import numpy, scipy, sys; sys.exit(0)' >/dev/null 2>&1; then
    PY="$p"; break
  fi
done

echo "repo: $REPO"
echo "build: configs/astra_sim/build_astrasim.sh (ASTRA-sim cross-check, optional)"
if [ -z "$PY" ]; then
  echo "FAIL: no interpreter with numpy+scipy found. Tried:"
  for p in "${CANDIDATES[@]}"; do
    v="$("$p" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null || echo '?')"
    miss="$("$p" -c 'import importlib.util as u;print(",".join(m for m in ("numpy","scipy") if u.find_spec(m) is None))' 2>/dev/null || echo 'n/a')"
    printf '  %-52s %-8s missing: %s\\n' "$p" "$v" "$miss"
  done
  echo "  fix: python3.12 -m pip install numpy scipy"
  exit 1
fi

"$PY" -c 'import sys, numpy, scipy; print("python : %s" % sys.version.split()[0]); print("numpy  : %s" % numpy.__version__); print("scipy  : %s" % scipy.__version__)'
echo "interp : $PY"
echo "tests  : $PY -m unittest discover -s tests"

if [ "${1:-}" = "--quiet" ]; then exit 0; fi

"$PY" -m unittest discover -s tests 2>&1 | tail -3
