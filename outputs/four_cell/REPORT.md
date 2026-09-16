
## Qwen3.6-35B-A3B-4bit (E=256, K=8, 40 MoE layers, world=32)

`outputs/four_cell/qwen36.ws32.json` — fit 81920 cells / eval 40960 cells, leave-categories-out

Regime **multi_pod_s4** (2 pods, 4 nodes, σ_core=4.0, σ_pod=1.0); 16 circuits, 2 ports/rank

| placement | EPS bottleneck (us) | static OCS gain | oracle OCS gain | circuits / promotable covered | port-saturated ranks |
| --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 2944.1 | +2.53 | +4.55 | 16 / 6.6% | 11 |
| `load_balanced_layer` | 5411.9 | +8.44 | +9.60 | 16 / 6.7% | 10 |
| `load_balanced` | 3703.4 | +2.60 | +2.60 | 16 / 6.6% | 9 |
| `linear` | 3956.2 | +4.76 | +8.37 | 16 / 6.9% | 14 |
| `random` | 4406.1 | +8.45 | +8.48 | 16 / 6.9% | 14 |
| `affinity_layer` | 8082.0 | +8.32 | +9.31 | 16 / 8.7% | 16 |
| `promote_aware_layer` | 3118.4 | +5.53 | +5.98 | 16 / 6.8% | 13 |
| `bottleneck_search_layer` | 3172.5 | +4.22 | +4.30 | 16 / 6.6% | 12 |

**The 2x2** (A = `linear` EPS, B = affinity-coordinated EPS, C = `linear` + OCS, D = affinity-coordinated + OCS)

| cell | bottleneck (us) | vs A |
| --- | --- | --- |
| A  eps, no affinity | 3956.2 | +0.00% |
| B  eps, affinity | 2944.1 | +25.58% |
| C  ocs, no affinity | 3768.0 | +4.76% |
| D  ocs + affinity | 2869.7 | +27.46% |

- affinity alone: **+25.58%**
- OCS alone: **+4.76%**
- both: **+27.46%**
- synergy `delta_both - (delta_aff + delta_ocs)` = **-2.87%** -> **substitutes**
- D is the minimum cell: **True** (best = D)

**Co-design (the fourth cell), vs the best electrical placement**

| search | on the OCS substrate | on the EPS substrate | its own circuit gain |
| --- | --- | --- | --- |
| `promote_aware_layer` | -2.66% | -5.92% | +5.53% |
| `bottleneck_search_layer` | -5.89% | -7.76% | +4.22% |

(`bottleneck_search_layer` is the control: the same search with no circuits at all. If it loses on the EPS substrate too, a co-design loss is the search's fault rather than OCS's.)

**Static OCS gain by regime** (per placement)

| regime | `affinity_coordinated_layer` | `affinity_layer` | `bottleneck_search_layer` | `linear` | `load_balanced` | `load_balanced_layer` | `random` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi_pod_s4 | +2.53% | +8.32% | +4.22% | +4.76% | +2.60% | +8.44% | +8.45% |
| multi_pod_s2 | +2.89% | +5.15% | +3.23% | +4.80% | +2.34% | +5.29% | +5.37% |
| multi_pod_s8 | +2.31% | +10.20% | +4.01% | +4.73% | +2.83% | +10.32% | +10.27% |
| multi_pod_pod_s2 | +2.54% | +2.80% | +1.56% | +4.56% | +2.07% | +7.57% | +7.60% |
| single_pod | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| realistic | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Port / circuit budget envelope** (focus regime)

| placement | circuits | ports/rank | static OCS gain | oracle gain | value of prediction | covered |
| --- | --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 16 | 1 | +4.55% | +4.55% | +0.00% | 6.5% |
| `affinity_coordinated_layer` | 16 | 2 | +2.53% | +4.55% | +2.02% | 6.6% |
| `affinity_coordinated_layer` | 16 | 4 | +2.53% | +4.76% | +2.24% | 6.6% |
| `affinity_coordinated_layer` | 32 | 1 | +4.55% | +4.55% | +0.00% | 6.5% |
| `affinity_coordinated_layer` | 32 | 2 | +8.57% | +8.74% | +0.17% | 13.0% |
| `affinity_coordinated_layer` | 32 | 4 | +5.83% | +6.15% | +0.32% | 13.2% |
| `affinity_coordinated_layer` | 8 | 1 | +2.53% | +4.55% | +2.02% | 3.3% |
| `affinity_coordinated_layer` | 8 | 2 | +0.41% | +4.55% | +4.14% | 3.4% |
| `affinity_coordinated_layer` | 8 | 4 | +0.41% | +4.55% | +4.14% | 3.4% |
| `linear` | 16 | 1 | +4.69% | +4.85% | +0.16% | 6.4% |
| `linear` | 16 | 2 | +4.76% | +8.37% | +3.61% | 6.9% |
| `linear` | 16 | 4 | +4.76% | +7.68% | +2.92% | 7.0% |
| `linear` | 32 | 1 | +4.69% | +4.85% | +0.16% | 6.4% |
| `linear` | 32 | 2 | +9.24% | +9.14% | -0.10% | 12.6% |
| `linear` | 32 | 4 | +4.76% | +11.43% | +6.68% | 13.7% |
| `linear` | 8 | 1 | +4.69% | +4.37% | -0.32% | 3.5% |
| `linear` | 8 | 2 | +3.89% | +4.76% | +0.87% | 3.6% |
| `linear` | 8 | 4 | +4.76% | +4.76% | +0.00% | 3.6% |

**Reconfiguration class envelope** (affinity-coordinated placement)

| class | reconfig | static OCS gain | breakeven token passes |
| --- | --- | --- | --- |
| mems_10ms | 10000 us | +2.53% | 3.36 |
| mems_1ms | 1000 us | +2.53% | 0.34 |
| fast_10us | 10 us | +2.53% | 0.0 |
| ideal_0 | 0 us | +2.53% | 0.0 |

**Dispatch semantics** (the verdict must be reported per mode; see F10)

| mode | total-byte spread across placements | affinity | OCS | both | synergy | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| REPLICATED | 0.0% | +5.01% | +7.62% | +9.96% | -2.67% | substitutes |
| DEDUP_RANK | 33.793456% | +25.58% | +4.76% | +27.46% | -2.87% | substitutes |
| DEDUP_NODE | 26.334779% | +18.05% | +6.84% | +23.42% | -1.47% | substitutes |

(`total-byte spread` is the placement-to-placement spread in dispatch volume: exactly 0 under REPLICATED, non-zero under the DEDUP modes — a numeric check of the documented dispatch semantics.)

**Completion time, communication component only** (prefill 26040 cells, decode step 640 cells)

| placement | EPS TTFT (us) | OCS TTFT (us) | TTFT gain | EPS ITL (us) | OCS ITL (us) | ITL gain |
| --- | --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 2278.9 | 2211.1 | +2.98% | 190.3 | 168.0 | +11.73% |
| `load_balanced_layer` | 2926.3 | 2914.3 | +0.41% | 505.5 | 465.0 | +8.01% |
| `load_balanced` | 2844.0 | 2622.3 | +7.80% | 364.0 | 352.0 | +3.30% |
| `linear` | 2785.7 | 2576.1 | +7.52% | 347.9 | 335.9 | +3.45% |
| `random` | 2699.7 | 2583.6 | +4.30% | 350.0 | 322.8 | +7.78% |
| `affinity_layer` | 5267.7 | 4707.7 | +10.63% | 276.3 | 264.3 | +4.34% |
| `bottleneck_search_layer` | 2346.6 | 2309.2 | +1.59% | 239.1 | 227.1 | +5.02% |

**Optimality bounds**

layer 0: 2048 cells, 1035 distinct expert sets

| placement | linearised max | exact ingress max | vs linear bound | vs union bound |
| --- | --- | --- | --- | --- |
| `linear` | 919 | 886 | +79.5% | +91.4% |
| `random` | 945 | 784 | +84.6% | +69.3% |
| `load_balanced` | 908 | 714 | +77.3% | +54.2% |
| `load_balanced_layer` | 521 | 515 | +1.8% | +11.2% |
| `affinity_layer` | 2132 | 609 | +316.4% | +31.5% |
| `affinity_coordinated_layer` | 1485 | 500 | +190.0% | +8.0% |

- linear-objective bound: **512.0** (LP relaxation, 321 constraints)
- linear MIP: bound 512.0, incumbent 538.0, gap 0.048327 (30.038s)
- union bound: **463.0** = max(perfectly-spread ideal 64.0, largest expert reach 463)
layer 1: 2048 cells, 1284 distinct expert sets

| placement | linearised max | exact ingress max | vs linear bound | vs union bound |
| --- | --- | --- | --- | --- |
| `linear` | 920 | 794 | +79.7% | +59.4% |
| `random` | 1133 | 927 | +121.3% | +86.1% |
| `load_balanced` | 1127 | 938 | +120.1% | +88.4% |
| `load_balanced_layer` | 521 | 521 | +1.8% | +4.6% |
| `affinity_layer` | 2928 | 1105 | +471.9% | +121.9% |
| `affinity_coordinated_layer` | 1458 | 716 | +184.8% | +43.8% |

- linear-objective bound: **512.0** (LP relaxation, 321 constraints)
- linear MIP: bound 512.0, incumbent 601.0, gap 0.148087 (33.545s)
- union bound: **498.0** = max(perfectly-spread ideal 64.0, largest expert reach 498)
layer 2: 2048 cells, 1421 distinct expert sets

| placement | linearised max | exact ingress max | vs linear bound | vs union bound |
| --- | --- | --- | --- | --- |
| `linear` | 863 | 779 | +68.6% | +63.0% |
| `random` | 909 | 826 | +77.5% | +72.8% |
| `load_balanced` | 1192 | 936 | +132.8% | +95.8% |
| `load_balanced_layer` | 545 | 540 | +6.4% | +13.0% |
| `affinity_layer` | 2966 | 1107 | +479.3% | +131.6% |
| `affinity_coordinated_layer` | 1931 | 863 | +277.1% | +80.5% |

- linear-objective bound: **512.0** (LP relaxation, 321 constraints)
- linear MIP: bound 512.0, incumbent 565.0, gap 0.093805 (30.055s)
- union bound: **478.0** = max(perfectly-spread ideal 64.0, largest expert reach 478)

## Qwen3.8-Whittle-MoE-27B-A17.8B-4bit (E=64, K=16, 64 MoE layers, world=32)

`outputs/four_cell/whittle.ws32.json` — fit 127360 cells / eval 65536 cells, leave-categories-out

Regime **multi_pod_s4** (2 pods, 4 nodes, σ_core=4.0, σ_pod=1.0); 16 circuits, 2 ports/rank

| placement | EPS bottleneck (us) | static OCS gain | oracle OCS gain | circuits / promotable covered | port-saturated ranks |
| --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 10120.3 | +2.91 | +3.75 | 16 / 6.4% | 11 |
| `load_balanced_layer` | 19134.0 | +8.47 | +8.47 | 16 / 6.9% | 15 |
| `load_balanced` | 11211.2 | +2.47 | +2.68 | 16 / 6.4% | 13 |
| `linear` | 12639.2 | +8.46 | +8.46 | 16 / 6.7% | 14 |
| `random` | 11966.8 | +8.43 | +8.43 | 16 / 6.7% | 14 |
| `affinity_layer` | 22057.9 | +8.39 | +8.38 | 16 / 8.0% | 16 |
| `promote_aware_layer` | 10833.8 | +8.51 | +7.97 | 16 / 6.6% | 13 |
| `bottleneck_search_layer` | 10908.9 | +0.81 | +4.37 | 16 / 6.4% | 12 |

**The 2x2** (A = `linear` EPS, B = affinity-coordinated EPS, C = `linear` + OCS, D = affinity-coordinated + OCS)

| cell | bottleneck (us) | vs A |
| --- | --- | --- |
| A  eps, no affinity | 12639.2 | +0.00% |
| B  eps, affinity | 10120.3 | +19.93% |
| C  ocs, no affinity | 11569.4 | +8.46% |
| D  ocs + affinity | 9825.5 | +22.26% |

- affinity alone: **+19.93%**
- OCS alone: **+8.46%**
- both: **+22.26%**
- synergy `delta_both - (delta_aff + delta_ocs)` = **-6.13%** -> **substitutes**
- D is the minimum cell: **True** (best = D)

**Co-design (the fourth cell), vs the best electrical placement**

| search | on the OCS substrate | on the EPS substrate | its own circuit gain |
| --- | --- | --- | --- |
| `promote_aware_layer` | -0.88% | -7.05% | +8.51% |
| `bottleneck_search_layer` | -10.13% | -7.79% | +0.81% |

(`bottleneck_search_layer` is the control: the same search with no circuits at all. If it loses on the EPS substrate too, a co-design loss is the search's fault rather than OCS's.)

**Static OCS gain by regime** (per placement)

| regime | `affinity_coordinated_layer` | `affinity_layer` | `bottleneck_search_layer` | `linear` | `load_balanced` | `load_balanced_layer` | `random` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi_pod_s4 | +2.91% | +8.39% | +0.81% | +8.46% | +2.47% | +8.47% | +8.43% |
| multi_pod_s2 | +2.90% | +5.10% | +0.90% | +5.19% | +2.46% | +5.16% | +5.16% |
| multi_pod_s8 | +2.92% | +10.33% | +0.75% | +10.39% | +2.39% | +10.43% | +9.61% |
| multi_pod_pod_s2 | +2.80% | +2.54% | +2.06% | +7.62% | +2.64% | +2.54% | +7.20% |
| single_pod | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| realistic | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Port / circuit budget envelope** (focus regime)

| placement | circuits | ports/rank | static OCS gain | oracle gain | value of prediction | covered |
| --- | --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 16 | 1 | +4.25% | +4.44% | +0.19% | 6.3% |
| `affinity_coordinated_layer` | 16 | 2 | +2.91% | +3.75% | +0.84% | 6.4% |
| `affinity_coordinated_layer` | 16 | 4 | +2.19% | +4.08% | +1.89% | 6.4% |
| `affinity_coordinated_layer` | 32 | 1 | +4.25% | +4.44% | +0.19% | 6.3% |
| `affinity_coordinated_layer` | 32 | 2 | +8.34% | +8.72% | +0.37% | 12.6% |
| `affinity_coordinated_layer` | 32 | 4 | +3.20% | +4.17% | +0.97% | 12.7% |
| `affinity_coordinated_layer` | 8 | 1 | +2.19% | +3.75% | +1.57% | 3.2% |
| `affinity_coordinated_layer` | 8 | 2 | +0.48% | +3.49% | +3.01% | 3.2% |
| `affinity_coordinated_layer` | 8 | 4 | +2.19% | +2.19% | +0.00% | 3.2% |
| `linear` | 16 | 1 | +4.37% | +4.37% | +0.00% | 6.3% |
| `linear` | 16 | 2 | +8.46% | +8.46% | +0.00% | 6.7% |
| `linear` | 16 | 4 | +10.52% | +11.56% | +1.04% | 6.9% |
| `linear` | 32 | 1 | +4.37% | +4.37% | +0.00% | 6.3% |
| `linear` | 32 | 2 | +8.46% | +8.46% | +0.00% | 12.6% |
| `linear` | 32 | 4 | +14.76% | +14.76% | +0.00% | 13.3% |
| `linear` | 8 | 1 | +4.37% | +4.37% | +0.00% | 3.3% |
| `linear` | 8 | 2 | +8.46% | +8.46% | +0.00% | 3.4% |
| `linear` | 8 | 4 | +10.52% | +10.52% | +0.00% | 3.5% |

**Reconfiguration class envelope** (affinity-coordinated placement)

| class | reconfig | static OCS gain | breakeven token passes |
| --- | --- | --- | --- |
| mems_10ms | 10000 us | +2.91% | 0.53 |
| mems_1ms | 1000 us | +2.91% | 0.05 |
| fast_10us | 10 us | +2.91% | 0.0 |
| ideal_0 | 0 us | +2.91% | 0.0 |

**Dispatch semantics** (the verdict must be reported per mode; see F10)

| mode | total-byte spread across placements | affinity | OCS | both | synergy | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| REPLICATED | 0.0% | +9.59% | +8.47% | +12.01% | -6.05% | substitutes |
| DEDUP_RANK | 17.775395% | +19.93% | +8.46% | +22.26% | -6.13% | substitutes |
| DEDUP_NODE | 12.663037% | +9.91% | +8.63% | +16.75% | -1.79% | substitutes |

(`total-byte spread` is the placement-to-placement spread in dispatch volume: exactly 0 under REPLICATED, non-zero under the DEDUP modes — a numeric check of the documented dispatch semantics.)

**Completion time, communication component only** (prefill 80576 cells, decode step 1024 cells)

| placement | EPS TTFT (us) | OCS TTFT (us) | TTFT gain | EPS ITL (us) | OCS ITL (us) | ITL gain |
| --- | --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 13792.6 | 13239.1 | +4.01% | 898.4 | 886.4 | +1.34% |
| `load_balanced_layer` | 21445.8 | 19537.5 | +8.90% | 1271.3 | 1259.3 | +0.94% |
| `load_balanced` | 15014.5 | 14041.3 | +6.48% | 1049.5 | 1037.5 | +1.14% |
| `linear` | 14835.8 | 13904.3 | +6.28% | 1033.3 | 930.8 | +9.91% |
| `random` | 15404.8 | 14095.7 | +8.50% | 1027.2 | 1015.2 | +1.17% |
| `affinity_layer` | 25560.3 | 23301.5 | +8.84% | 1256.2 | 1244.2 | +0.96% |
| `bottleneck_search_layer` | 14104.9 | 13905.3 | +1.42% | 967.2 | 955.2 | +1.24% |

**Optimality bounds**

layer 0: 1990 cells, 1830 distinct expert sets

| placement | linearised max | exact ingress max | vs linear bound | vs union bound |
| --- | --- | --- | --- | --- |
| `linear` | 3292 | 1914 | +97.2% | +15.7% |
| `random` | 2512 | 1767 | +50.5% | +6.8% |
| `load_balanced` | 2157 | 1744 | +29.2% | +5.4% |
| `load_balanced_layer` | 1669 | 1661 | +0.0% | +0.4% |
| `affinity_layer` | 3292 | 1914 | +97.2% | +15.7% |
| `affinity_coordinated_layer` | 2586 | 1660 | +54.9% | +0.4% |

- linear-objective bound: **1669.0** (LP relaxation, 129 constraints)
- linear MIP: bound 1669.0, incumbent 1669.0, gap 0.0 (1.656s)
- union bound: **1654.0** = max(perfectly-spread ideal 497.5, largest expert reach 1654)

## Qwen1.5-MoE-A2.7B-Chat-4bit (E=60, K=4, 24 MoE layers, world=30)

`outputs/four_cell/qwen15.ws30.json` — fit 79488 cells / eval 45072 cells, leave-categories-out

Regime **multi_pod_s4** (2 pods, 4 nodes, σ_core=4.0, σ_pod=1.0); 15 circuits, 2 ports/rank

| placement | EPS bottleneck (us) | static OCS gain | oracle OCS gain | circuits / promotable covered | port-saturated ranks |
| --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 2222.7 | +3.83 | +7.76 | 15 / 7.2% | 10 |
| `load_balanced_layer` | 2466.4 | +0.81 | +2.33 | 15 / 7.2% | 12 |
| `load_balanced` | 2424.9 | +2.68 | +2.68 | 15 / 7.1% | 11 |
| `linear` | 2412.8 | +1.90 | +2.72 | 15 / 7.2% | 10 |
| `random` | 2393.6 | +0.95 | +2.54 | 15 / 7.2% | 11 |
| `affinity_layer` | 2285.5 | +0.53 | +4.22 | 15 / 7.5% | 12 |
| `promote_aware_layer` | 2309.9 | +4.56 | +7.58 | 15 / 7.3% | 10 |
| `bottleneck_search_layer` | 2378.7 | +4.68 | +9.16 | 15 / 7.2% | 11 |

**The 2x2** (A = `linear` EPS, B = affinity-coordinated EPS, C = `linear` + OCS, D = affinity-coordinated + OCS)

| cell | bottleneck (us) | vs A |
| --- | --- | --- |
| A  eps, no affinity | 2412.8 | +0.00% |
| B  eps, affinity | 2222.7 | +7.88% |
| C  ocs, no affinity | 2366.9 | +1.90% |
| D  ocs + affinity | 2137.7 | +11.40% |

- affinity alone: **+7.88%**
- OCS alone: **+1.90%**
- both: **+11.40%**
- synergy `delta_both - (delta_aff + delta_ocs)` = **+1.62%** -> **complements**
- D is the minimum cell: **True** (best = D)

**Co-design (the fourth cell), vs the best electrical placement**

| search | on the OCS substrate | on the EPS substrate | its own circuit gain |
| --- | --- | --- | --- |
| `promote_aware_layer` | -3.13% | -3.92% | +4.56% |
| `bottleneck_search_layer` | -6.07% | -7.02% | +4.68% |

(`bottleneck_search_layer` is the control: the same search with no circuits at all. If it loses on the EPS substrate too, a co-design loss is the search's fault rather than OCS's.)

**Static OCS gain by regime** (per placement)

| regime | `affinity_coordinated_layer` | `affinity_layer` | `bottleneck_search_layer` | `linear` | `load_balanced` | `load_balanced_layer` | `random` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi_pod_s4 | +3.83% | +0.53% | +4.68% | +1.90% | +2.68% | +0.81% | +0.95% |
| multi_pod_s2 | +3.54% | +3.04% | +3.39% | +2.27% | +2.58% | +1.78% | +1.51% |
| multi_pod_s8 | +3.66% | +0.28% | +5.45% | +1.79% | +2.75% | +0.26% | +0.61% |
| multi_pod_pod_s2 | +0.49% | +2.60% | +0.45% | +1.84% | +2.16% | +1.35% | +1.08% |
| single_pod | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| realistic | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Port / circuit budget envelope** (focus regime)

| placement | circuits | ports/rank | static OCS gain | oracle gain | value of prediction | covered |
| --- | --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 16 | 1 | +4.79% | +5.25% | +0.46% | 6.6% |
| `affinity_coordinated_layer` | 16 | 2 | +3.83% | +7.76% | +3.94% | 7.7% |
| `affinity_coordinated_layer` | 16 | 4 | +0.54% | +5.83% | +5.29% | 7.7% |
| `affinity_coordinated_layer` | 32 | 1 | +4.79% | +5.25% | +0.46% | 6.6% |
| `affinity_coordinated_layer` | 32 | 2 | +9.10% | +9.41% | +0.31% | 13.2% |
| `affinity_coordinated_layer` | 32 | 4 | +7.98% | +7.76% | -0.22% | 15.2% |
| `affinity_coordinated_layer` | 8 | 1 | +0.54% | +5.25% | +4.71% | 3.9% |
| `affinity_coordinated_layer` | 8 | 2 | +0.54% | +3.83% | +3.29% | 3.9% |
| `affinity_coordinated_layer` | 8 | 4 | +0.54% | +3.83% | +3.29% | 3.9% |
| `linear` | 16 | 1 | +4.29% | +4.70% | +0.41% | 6.6% |
| `linear` | 16 | 2 | +1.90% | +2.72% | +0.81% | 7.7% |
| `linear` | 16 | 4 | +1.90% | +2.72% | +0.81% | 7.7% |
| `linear` | 32 | 1 | +4.29% | +4.70% | +0.41% | 6.6% |
| `linear` | 32 | 2 | +8.58% | +9.06% | +0.48% | 13.1% |
| `linear` | 32 | 4 | +2.37% | +6.82% | +4.45% | 15.2% |
| `linear` | 8 | 1 | +1.90% | +0.57% | -1.34% | 3.9% |
| `linear` | 8 | 2 | +1.90% | +0.57% | -1.34% | 3.9% |
| `linear` | 8 | 4 | +1.90% | +0.57% | -1.34% | 3.9% |

**Reconfiguration class envelope** (affinity-coordinated placement)

| class | reconfig | static OCS gain | breakeven token passes |
| --- | --- | --- | --- |
| mems_10ms | 10000 us | +3.83% | 4.9 |
| mems_1ms | 1000 us | +3.83% | 0.49 |
| fast_10us | 10 us | +3.83% | 0.0 |
| ideal_0 | 0 us | +3.83% | 0.0 |

**Dispatch semantics** (the verdict must be reported per mode; see F10)

| mode | total-byte spread across placements | affinity | OCS | both | synergy | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| REPLICATED | 0.0% | -1.31% | +2.58% | +1.48% | +0.21% | additive |
| DEDUP_RANK | 10.462241% | +7.88% | +1.90% | +11.40% | +1.62% | complements |
| DEDUP_NODE | 6.090805% | +6.25% | +5.55% | +11.22% | -0.58% | substitutes |

(`total-byte spread` is the placement-to-placement spread in dispatch volume: exactly 0 under REPLICATED, non-zero under the DEDUP modes — a numeric check of the documented dispatch semantics.)

**Completion time, communication component only** (prefill 21864 cells, decode step 576 cells)

| placement | EPS TTFT (us) | OCS TTFT (us) | TTFT gain | EPS ITL (us) | OCS ITL (us) | ITL gain |
| --- | --- | --- | --- | --- | --- | --- |
| `affinity_coordinated_layer` | 1172.2 | 1059.4 | +9.62% | 91.5 | 79.5 | +13.11% |
| `load_balanced_layer` | 1445.0 | 1314.5 | +9.03% | 115.3 | 101.3 | +12.12% |
| `load_balanced` | 1352.1 | 1237.4 | +8.49% | 96.6 | 82.6 | +14.46% |
| `linear` | 1398.1 | 1268.2 | +9.30% | 95.9 | 83.4 | +13.02% |
| `random` | 1363.4 | 1239.8 | +9.06% | 103.1 | 87.9 | +14.81% |
| `affinity_layer` | 1329.3 | 1317.3 | +0.90% | 97.2 | 81.8 | +15.88% |
| `bottleneck_search_layer` | 1230.2 | 1120.9 | +8.89% | 96.9 | 84.9 | +12.38% |

**Optimality bounds**

layer 0: 3312 cells, 2463 distinct expert sets

| placement | linearised max | exact ingress max | vs linear bound | vs union bound |
| --- | --- | --- | --- | --- |
| `linear` | 619 | 619 | +36.3% | +46.0% |
| `random` | 617 | 612 | +35.9% | +44.3% |
| `load_balanced` | 639 | 607 | +40.7% | +43.2% |
| `load_balanced_layer` | 454 | 452 | +0.0% | +6.6% |
| `affinity_layer` | 707 | 634 | +55.7% | +49.5% |
| `affinity_coordinated_layer` | 580 | 452 | +27.8% | +6.6% |

- linear-objective bound: **454.0** (LP relaxation, 121 constraints)
- linear MIP: bound 454.0, incumbent 454.0, gap 0.0 (5.591s)
- union bound: **424.0** = max(perfectly-spread ideal 220.8, largest expert reach 424)
