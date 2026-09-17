#!/usr/bin/env bash
# Build ASTRA-sim's analytical backend on macOS. Idempotent, env-independent.
#
#   bash configs/astra_sim/build_astrasim.sh
#   ASTRA_SIM_DIR=/path/to/astra-sim bash configs/astra_sim/build_astrasim.sh
#
# Three macOS-specific things this handles, each of which fails confusingly if
# done by hand:
#
#   1. `nproc` is a Linux-ism -> shim it.
#   2. Homebrew protobuf 35.x links against abseil. The default
#      `find_package(Protobuf)` (module mode) does NOT pull it in transitively,
#      so the link fails with unresolved `absl::log_internal::*` symbols.
#      Setting PROTOBUF_FROM_SOURCE=True makes CMakeLists use CONFIG mode
#      (`protobuf::libprotobuf`), which carries abseil.
#   3. A build directory configured in one protobuf mode and re-run in the OTHER
#      keeps a half-populated cache: `Protobuf_INCLUDE_DIR` ends up empty while
#      `Protobuf_LIBRARIES` is still set, and the compile dies with
#      "google/protobuf/runtime_version.h file not found" — even though the
#      header exists. Hence the unconditional `rm -rf` of the build directory.

set -euo pipefail

ASTRA="${ASTRA_SIM_DIR:-$HOME/astra-sim}"
[ -d "$ASTRA" ] || { echo "error: no ASTRA-sim checkout at $ASTRA (set ASTRA_SIM_DIR)" >&2; exit 1; }
[ -f "$ASTRA/build/astra_analytical/build.sh" ] || { echo "error: $ASTRA is not an ASTRA-sim checkout" >&2; exit 1; }

# --- 1. nproc shim -----------------------------------------------------------
SHIM="$(mktemp -d)"
printf '#!/bin/sh\nsysctl -n hw.ncpu\n' > "$SHIM/nproc"
chmod +x "$SHIM/nproc"
trap 'rm -rf "$SHIM"' EXIT

# --- sanity: protoc must exist and must be the one whose headers we compile against
command -v protoc >/dev/null || { echo "error: protoc not on PATH (brew install protobuf)" >&2; exit 1; }
echo "[build] protoc: $(command -v protoc) ($(protoc --version))"
PBINC="$(dirname "$(dirname "$(command -v protoc)")")/include/google/protobuf"
if [ -e "$PBINC/runtime_version.h" ]; then
  echo "[build] protobuf headers: $PBINC (runtime_version.h present)"
else
  echo "[build] WARNING: $PBINC/runtime_version.h missing — generated code may not compile" >&2
fi

# --- 2/3. clean build dir, then build with CONFIG-mode protobuf --------------
# Also drop the *generated* chakra protobuf files. If they survive from a build
# that used a different protoc, the compile mixes a new .pb.h with old headers and
# fails with the same "runtime_version.h file not found". This is what the
# upstream `build.sh -c` does; we do it unconditionally.
rm -f "$ASTRA"/extern/graph_frontend/chakra/schema/protobuf/et_def.pb.h \
      "$ASTRA"/extern/graph_frontend/chakra/schema/protobuf/et_def.pb.cc \
      "$ASTRA"/extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py
rm -rf "$ASTRA/build/astra_analytical/build"
cd "$ASTRA"
echo "[build] building (CONFIG-mode protobuf, clean build dir) ..."
PROTOBUF_FROM_SOURCE=True PATH="$SHIM:$PATH" bash build/astra_analytical/build.sh

# --- verify ------------------------------------------------------------------
BIN="$ASTRA/build/astra_analytical/build/bin"
echo
for b in AstraSim_Analytical_Congestion_Aware AstraSim_Analytical_Congestion_Unaware; do
  if [ -x "$BIN/$b" ]; then echo "[build] OK  $b"; else echo "[build] FAIL missing $b" >&2; exit 1; fi
done
echo "[build] both binaries at $BIN"
