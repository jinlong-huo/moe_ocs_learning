# ASTRA-sim bridge for this repo

Two scripts and four network configs that let this repo's fabric model be
cross-checked in ASTRA-sim. Nothing here patches ASTRA-sim; it drives it.

```bash
bash configs/astra_sim/run_astrasim.sh compare      # the whole experiment, from any cwd
```

## Prerequisites — install these first

Both scripts check for these and print the exact fix if one is missing.

```bash
xcode-select --install                                   # clang / make
brew install cmake protobuf                              # verified with protoc 35.1
git clone https://github.com/astra-sim/astra-sim.git ~/astra-sim
cd ~/astra-sim && git submodule update --init --recursive # 7 submodules, required
```

The submodules are **not optional** — the analytical network backend, the Chakra
workload reader and `fmt`/`spdlog` all live in them; without them CMake has
nothing to compile.

## Build

```bash
bash configs/astra_sim/build_astrasim.sh            # ~2.5 min from clean, idempotent
```

Do **not** call `build/astra_analytical/build.sh` by hand on macOS — three things
fail, two of them with misleading errors:

| what | without it |
| --- | --- |
| shim `nproc` (a Linux-ism) | `NUM_THREADS` empty → `cmake --build -j` fails |
| `PROTOBUF_FROM_SOURCE=True` (protobuf **CONFIG** mode) | Homebrew protobuf 35.x needs abseil; module mode does not link it → unresolved `absl::log_internal::*` at **link** time |
| wipe the build dir **and** the generated `et_def.pb.{h,cc}` | a tree configured in one protobuf mode then re-run in the other leaves `Protobuf_INCLUDE_DIR` empty while `Protobuf_LIBRARIES` is set → `google/protobuf/runtime_version.h file not found` at **compile** time, although the header exists |

Upstream's own clean step is `bash build/astra_analytical/build.sh -c`, which
deletes the same four artifacts. Docs:
[build](https://astra-sim.github.io/astra-sim-docs/getting-started/build.html),
[running](https://astra-sim.github.io/astra-sim-docs/2.2/getting-started/running-astra-sim.html),
[ns-3 backend](https://astra-sim.github.io/astra-sim-docs/network-backend/ns3-network-backend.html).

## Run

```bash
bash configs/astra_sim/run_astrasim.sh              # OCS-promoted config
bash configs/astra_sim/run_astrasim.sh eps          # EPS with sigma=4
bash configs/astra_sim/run_astrasim.sh compare      # both, with the ratio
bash configs/astra_sim/run_astrasim.sh /path/other.yml
```

Overrides: `ASTRA_SIM_DIR`, `WORKLOAD`, `SYSTEM`, `REMOTE_MEMORY`, `BINARY`,
`ASTRA_BUILD=1` (force a rebuild). It builds automatically if the binary is
missing, and exits non-zero with an actionable message if a dependency is absent.

## Measured: 16-NPU all-to-all, one ring tier, 4x bandwidth

`ring16_eps_sigma4.yml` (12.5 GB/s = NIC/4) vs `ring16_ocs_promoted.yml`
(50 GB/s = full NIC rate). The promotion is exactly this repo's tier-promotion
model of a circuit.

| backend | EPS sigma=4 | OCS promoted | speedup |
| --- | --- | --- | --- |
| congestion **unaware** | 415 220 cycles | 195 620 | 2.12x |
| congestion **aware** | 586 120 cycles | 195 925 | 2.99x |

Three findings:

1. **Contention only exists when oversubscribed.** Promoted, the congestion-aware
   and congestion-unaware models agree to 0.2 % (195 925 vs 195 620); at sigma=4
   they differ by **41 %**. That is the mechanism this repo's cost model assumes,
   confirmed independently: a circuit removes contention rather than adding
   bandwidth.
2. **The gain is sublinear in sigma.** 4x bandwidth buys 2.1–3.0x time. Fitting
   `T = F + k/bw` gives a fixed component `F ≈ 122 000` cycles — about **29 % of
   the EPS time** — so only ~70 % of it scales with bandwidth.
3. **This suggests our cost model overstates the OCS gain.** `cost_model.py`
   credits the full `1/sigma` on the byte term with alpha as a small additive
   latency; ASTRA-sim says a third of the time is not bandwidth-proportional.

## Dead end, recorded so you don't repeat it

`eps_sigma4_16npu.yml` / `ocs_promoted_16npu.yml` are the same experiment on the
HGX-shaped 2-tier fabric (`npus_count: [8, 2]`). The 16-NPU all-to-all **did not
traverse the cross tier** — both bandwidths gave identical `57252` cycles — so the
single-tier ring is the config that actually exercises the bandwidth axis.

## Not yet done (the modification plan)

* **Affinity → system layer.** ASTRA-sim's workload states a collective and its
  size; the per-pair split comes from the *system layer*. Our placement-dependent,
  uneven dispatch matrix therefore needs a **custom collective** —
  `examples/run_scripts/analytical/congestion_aware/run_analytical_with_custom_collective.sh`
  plus `examples/system/custom_collectives/` is the template.
* **OCS at pair granularity.** The analytical backend's config is **tier-level**
  (`bandwidth: [...]` per tier), so it cannot express "this one rank pair is
  promoted" and has no port budget. Per-pair promotion needs the **ns-3 backend**,
  whose topology is a per-link JSON (`examples/network/ns3/sample_16nodes_1D.json`).
* **Reconfiguration cost** is not modelled by either config; it would have to be
  added per plan change.
