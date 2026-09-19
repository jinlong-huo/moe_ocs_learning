# Phase 5 — both affinities in one objective: the answer is no

`scripts/affinity_ab.py`, results in `outputs/affinity/ab_compare.*.json`.
Out-of-sample by construction: fit on window A, scored on a **disjoint** window
(leave-categories-out), with the inter-layer transitions built from the fit window
only.

## The arms

| arm | what it optimises | how |
| --- | --- | --- |
| `linear` | nothing (reference) | the deployed default |
| **A — intra only** | co-activation + balance | `affinity_coordinated_layer` |
| **B — inter only** | ExFlow's objective (min cross-layer re-routes) | swap-based local search on their exact objective — a stand-in for their ILP, not their solver |
| **C(λ) — both** | `eps_proxy + λ · reroute_fraction` | NEW combined swap search, started from A so the intra term is already good and the search must *trade* |

Two different metrics are reported, because they disagree: the real one is
`evaluate()`'s bottleneck in **µs** (tier-weighted, max over ranks); the other is
ExFlow's **re-route fraction** (hop counts, placement-only).

## Result, three models, out-of-sample

EPS bottleneck in µs (lower is better):

| model | `linear` | **A intra** | B inter | best C(λ) | A vs linear |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6 (E=256, L=40) | 3956.2 | **2944.1** | 4162.9 | 3451.7 (λ=0) | **−25.6 %** |
| Qwen1.5 (E=60, L=24) | 2412.8 | **2222.7** | 2419.0 | 2265.7 (λ=0) | **−7.9 %** |
| Whittle (E=64, L=64) | 12639.2 | **10120.3** | 13028.1 | 11208.4 (λ=0) | **−19.9 %** |

Re-route fraction (ExFlow's metric, lower is better):

| model | A intra | B inter | C best |
| --- | --- | --- | --- |
| Qwen3.6 | 0.9765 | **0.8269** | 0.8615 |
| Qwen1.5 | 0.9655 | **0.8572** | 0.8876 |
| Whittle | 0.9843 | **0.8800** | 0.9061 |

## Three findings

1. **The conflict is confirmed out-of-sample, in both directions.** A wins the µs
   metric in all three models; B wins its own metric in all three. Neither is a
   good proxy for the other. This is the measured version of the
   rho = −0.49 / −0.32 / −0.32 correlation from `docs/exflow_vs_ours.md`.
2. **The inter-layer term is not merely unhelpful on this metric — it is
   actively harmful.** B is *worse than the deployed default* `linear` on µs in
   all three models (+5.2 %, +0.3 %, +3.1 %). It buys its hop reduction by moving
   bytes onto hotter rank pairs, and the metro metric is a max over ranks.
3. **"Carry both terms" does not pay.** C never beats A on µs in any model, at any
   λ. It does land between A and B, so the combination is coherent — but the intra
   term alone is the better plan, and the honest recommendation is **A alone**.

## The most useful failure inside the experiment

At λ=0 the combined search ignores the inter-layer term entirely and optimises only
`eps_proxy` (average per-cell max-ingress, a count-space stand-in for the µs
metric). Starting from A, it improves that proxy substantially — Qwen3.6:
**0.4585 → 0.3657** — and yet the real out-of-sample bottleneck gets **worse**:
2944 µs → 3452 µs.

That is a clean demonstration that **message counts are not bytes**: the proxy
ignores tier weighting entirely, so "balanced" in count space can be worse in the
tier that actually binds. It is the same lesson as `docs/ownership_measurement.md`
arriving from a different direction.

## Limits

* The combined search optimises a **cheap proxy**, not the µs metric itself. So the
  claim is "a search on this proxy does not beat A", not "no joint objective can".
  A search that evaluated the real metric per candidate would be ~1000× slower per
  swap and is the obvious next experiment — and is the only version of this test
  that could overturn the finding.
* λ is swept, not tuned; λ=0 already wins, which is the strongest form of the
  negative result.
* B is a local search on ExFlow's objective, not their ILP. A better inter-layer
  optimiser would move B's own metric, but finding 2 makes the µs outcome
  unlikely to flip: the harm is structural (hotter pairs), not overhead.

Two definitions, matching the code (`src/eval/affinity_graph.py`, `scripts/exflow_compare.py`). Ours is the **intra**-layer one.

## Intra-layer — ours

Indicator of token $t$'s top-$K$ set at layer $l$, and the co-selection matrix:

$$
x_{t,l}\in\{0,1\}^{E},\qquad
x_{t,l}[e]=\mathbb{1}\!\left[e\in S_{t,l}\right],
\qquad
A^{(l)}_{ef}=\sum_{t} x_{t,l}[e]\,x_{t,l}[f],\quad e\neq f
$$

In matrix form (what the code computes), with $X_l=[x_{1,l},\dots,x_{N_l,l}]^{\top}$:

$$
A^{(l)}=X_l^{\top}X_l-\operatorname{diag}\!\left(X_l^{\top}X_l\right)
$$

Normalised variant (Jaccard) and the gate-weighted variant actually used:

$$
\tilde A^{(l)}_{ef}=\frac{A^{(l)}_{ef}}{A^{(l)}_{ee}+A^{(l)}_{ff}-A^{(l)}_{ef}},
\qquad
A^{(l,\text{w})}_{ef}=\sum_{t}\min\!\left(g_{t,l}[e],\,g_{t,l}[f]\right)
$$

## Inter-layer — ExFlow's conditional

With $r_{t,l}$ the **top-1** expert of token $t$ at layer $l$ and $n^{(l)}_{ef}$ the transition count:

$$
n^{(l)}_{ef}=\sum_{t}\mathbb{1}\!\left[r_{t,l}=e\ \wedge\ r_{t,l+1}=f\right],
\qquad
P^{(l)}_{ef}=\Pr\!\left(E_{f,l+1}\mid E_{e,l}\right)=\frac{n^{(l)}_{ef}}{\sum_{f'} n^{(l)}_{ef'}}
$$

Their objective (min cross-layer re-routes) and the placement variable $P:[L,E]\to[W]$:

$$
\rho(P)=\frac{1}{|T|}\sum_{t}\sum_{l=1}^{L-1}
\mathbb{1}\!\left[P_l\!\left(r_{t,l}\right)\neq P_{l+1}\!\left(r_{t,l+1}\right)\right]
$$

## How they enter our objective

$$
\min_{P}\ \underbrace{\max_{r}\sum_{s}\frac{b_{rs}(P)}{\beta_{\tau(r,s)}}+\alpha_{\tau_{\max}}}_{\text{EPS bottleneck (ours)}}
\qquad\text{s.t.}\qquad
\underbrace{A^{(l)}\ \text{drives the initialiser}}_{\text{co-location}}
$$

Measured: the two objectives **conflict** — $\rho\!\left(\text{inter-layer objective},\ \text{bottleneck}\right)=-0.49,\,-0.32,\,-0.32$ over the three models, so neither is a proxy for the other (details in `docs/affinity_ab.md`).

Notation table for the equations above. One fix first: I reused $P$ for both the placement and the conditional — rename the placement to $\Pi$ so the two don't collide.

| symbol         | meaning                                                  | domain                                     |
| -------------- | -------------------------------------------------------- | ------------------------------------------ |
| $E$            | number of experts                                        | $\mathbb{N}$, e.g. 256                     |
| $K$            | experts per token (top-$K$)                              | $\mathbb{N}$, e.g. 8                       |
| $L$            | number of MoE layers                                     | $\mathbb{N}$, e.g. 40                      |
| $W$            | world size (ranks)                                       | $\mathbb{N}$, e.g. 32                      |
| $l$            | layer index                                              | $1..L$                                     |
| $t$            | token index                                              | $1..N_l$                                   |
| $N_l$          | tokens routed at layer $l$                               | $\mathbb{N}$                               |
| $e,f$          | expert indices                                           | $1..E$, $e\neq f$                          |
| $S_{t,l}$      | top-$K$ expert set of token $t$ at layer $l$             | $|S_{t,l}|=K$                              |
| $g_{t,l}[e]$   | gate weight of expert $e$ for token $t$ at layer $l$     | $\mathbb{R}_{\ge0}$, $\sum_e g = 1$        |
| $x_{t,l}$      | indicator of $S_{t,l}$                                   | $\{0,1\}^{E}$                              |
| $X_l$          | indicator matrix, rows $x_{t,l}^\top$                    | $\{0,1\}^{N_l\times E}$                    |
| $A^{(l)}$      | **intra-layer** affinity (ours)                          | $\mathbb{R}_{\ge0}^{E\times E}$, symmetric |
| $A^{(l)}_{ef}$ | co-selection count of $e,f$ in layer $l$                 | $\mathbb{R}_{\ge0}$                        |
| $r_{t,l}$      | top-1 expert of token $t$ at layer $l$                   | $1..E$                                     |
| $n^{(l)}_{ef}$ | transition count $e\to f$, layers $l\to l{+}1$           | $\mathbb{N}$                               |
| $P^{(l)}_{ef}$ | **inter-layer** conditional $\Pr(E_{f,l+1}\mid E_{e,l})$ | $[0,1]$, $\sum_f P^{(l)}_{ef}=1$           |
| $T$            | set of scored token-transitions                          | —                                          |
| $\Pi$          | expert$\to$rank **placement**                            | $[L,E]\to[W]$                              |
| $\Pi_l(e)$     | rank hosting expert $e$ at layer $l$                     | $1..W$                                     |
| $b_{rs}(\Pi)$  | bytes on rank pair $(r,s)$ under $\Pi$                   | $\mathbb{R}_{\ge0}$, $[W,W]$               |
| $\tau(r,s)$    | tier of pair $(r,s)$                                     | $\{\text{node, pod, cross, optical}\}$     |
| $\beta_\tau$   | per-pair bandwidth of tier $\tau$                        | GB/s: 450 / 50 / 12.5 / 50                 |
| $\alpha_\tau$  | latency of tier $\tau$                                   | µs: 2 / 5 / 12 / 6                         |
| $\rho(\Pi)$    | ExFlow re-route fraction                                 | $[0,1]$, lower better                      |

Indices vs. sets: $A^{(l)}$ and $P^{(l)}$ are ; $\Pi$ is the only object being *optimised*.

With the rename, the two definitions read:

$$
A^{(l)}_{ef}=\sum_{t} x_{t,l}[e]\,x_{t,l}[f]
\qquad\qquad
P^{(l)}_{ef}=\frac{n^{(l)}_{ef}}{\sum_{f'} n^{(l)}_{ef'}}
$$

and the objective over $\Pi$ (not $P$):

$$
\min_{\Pi}\ \max_{r}\sum_{s}\frac{b_{rs}(\Pi)}{\beta_{\tau(r,s)}}+\alpha_{\tau_{\max}}
$$

Note the repo has this same collision in `docs/ocs_moe_program.md` (where $P$ is the placement); worth renaming there too if you lift these into the paper.





Good instinct to check — but the phase assumption is backwards, and the repo says so explicitly. Let me give you the formulation first, then the correction.

## The intra-layer formulation

**The data.** For each MoE layer $l\in\{1..L\}$ and each token $t$ in the trace, the router gives a top-$K$ set $S_{t,l}\subseteq\{1..E\}$, $|S_{t,l}|=K$, plus gate weights $g_{t,l}[e]$, $\sum_e g=1$. Every cell also carries two labels: a **phase** $\varphi(t)\in\{\text{prefill},\text{decode}\}$ and a **request id** $\rho(t)$.

**The statistic** — per layer, co-selection within that layer:
$$
\boxed{\;A^{(l)}_{ef}=\sum_{t} x_{t,l}[e]\,x_{t,l}[f],\qquad e\neq f\;}
$$

with $x_{t,l}\in\{0,1\}^E$ the indicator of $S_{t,l}$. In matrix form $A^{(l)}=X_l^{\top}X_l-\operatorname{diag}(\cdot)$, symmetric, $E\times E$.

**The three restrictions that matter for your question** — the same statistic, over different slices of $t$:

$$
A^{(l)}_{\text{pooled}}=\sum_{r}A^{(l)}_{r},
\qquad
A^{(l)}_{r}=\sum_{t:\,\rho(t)=r}(\cdot),
\qquad
A^{(l)}_{\varphi}=\sum_{t:\,\varphi(t)=\varphi}(\cdot)
$$

So "per request" and "per phase" are just index-set restrictions on the same sum — the code implements the pooled one (`pooled_affinity`, `layer_affinities`), and the request-level one exists separately for cross-tenant analysis.

**Normalisations** (all off-diagonal): Jaccard $\tilde A_{ef}=A_{ef}/(A_{ee}+A_{ff}-A_{ef})$, and the gate-weighted variant $A^{\mathrm{w}}_{ef}=\sum_t\min(g_{t,l}[e],g_{t,l}[f])$.

**How it's consumed** — placement is the decision, affinity is the guide:

$$
\max_{\Pi}\ \sum_{l}\sum_{e<f}A^{(l)}_{ef}\,\mathbb{1}\!\left[\Pi_l(e)=\Pi_l(f)\right]
\;+\;\text{balance term}
$$

The balance term is not optional: co-location alone lowers the *sum* of bytes and raises the *max* over ranks (measured: −32 % volume, **+83 % bottleneck vs random**).

## The correction: the pipeline runs on **decode**, not prefill

`src/eval/trace_ir.py` keeps the phase — line 57: `phase uint8 0 = prefill, 1 = decode` — and `decode_only()` selects `phase == 1`. Then look at who calls it:

- `ownership_ocs_envelope.py`, `affinity_ab.py`, `bmatching_vs_greedy.py`, `astra_placement_order.py`, `exflow_compare.py`, `source_model_sensitivity.py` — **`decode_only=True`**;
- `ocs_four_cell.py` — loads the full table, then `fit.decode_only(), ev.decode_only()`.

So **every number we've discussed in this session is a decode number**, and the prefill tokens are present in the trace but filtered out. Your "the decode can also use such affinity?" — it already does; decode *is* the phase in use.

## "This time, not next time" — yes, and that's the right way to say it

The affinity is an **offline statistic**, computed once from a calibration window, frozen, and then applied. The repo implements the freeze literally: the probe suite is split by category into a **fit** window and a disjoint **eval** window (`split_by_category` → `by_runs`), the plan is built from fit, and scored on eval.

And we measured exactly whether it survives — `scripts/plan_window_stability.py`, plan(fit) vs plan(eval) Jaccard:

| source model      | Jaccard     | consequence                       |
| ----------------- | ----------- | --------------------------------- |
| `per_sequence`    | **0.0000**  | plan totally stale → 0.088 % gain |
| `hash`            | 0.03 – 0.14 | mostly stale                      |
| measured (spread) | **1.0000**  | identical plan → 8.4 % gain       |

So the honest statement for the paper is not "we compute affinity from this request" but **"we compute it from a calibration window and require the plan to survive to the serving window"** — and that survival is the thing that decides whether the affinity is worth anything.

## What the phase correction implies

Since decode is what's in use, the open question flips: **does a decode-estimated affinity describe prefill?** It's measurable in one command — build $A^{(l)}_{\text{prefill}}$ and $A^{(l)}_{\text{decode}}$ on the same runs and compare them with the repo's `graph_similarity`. If they agree, the choice of phase is a detail; if they don't, any "affinity drives placement" claim has to name the phase it was fitted on, exactly as it now names the ownership model.

Say the word and I'll run that comparison — it's cheap, and it's the last unmeasured assumption in the affinity half.



Slide-ready, LaTeX:

```latex
\begin{align}
S_{t,l} &\subseteq \{1,\dots,E\},\quad |S_{t,l}| = K
   && \text{router output: top-}K\text{ experts for token } t,\ \text{layer } l\\[3pt]
x_{t,l}[e] &= \mathbb{1}\!\left[\,e \in S_{t,l}\,\right] \in \{0,1\}^{E}
   && \text{indicator of the set}\\[3pt]
A^{(l)}_{ef} &= \sum_{t} x_{t,l}[e]\,x_{t,l}[f]
   \;=\; \#\left\{\,t \;:\; e \in S_{t,l} \;\wedge\; f \in S_{t,l}\,\right\},
   \quad e \neq f
   && \textbf{intra-layer affinity}\\[3pt]
A^{(l)} &= X_{l}^{\top} X_{l} - \operatorname{diag}\!\left(X_{l}^{\top} X_{l}\right)
   && \text{matrix form, } X_l = [x_{1,l},\dots,x_{N_l,l}]^{\top}
\end{align}
```

Objective line (if the slide shows where it goes):

```latex
\max_{\Pi}\ \sum_{l}\sum_{e<f} A^{(l)}_{ef}\,
\mathbb{1}\!\left[\Pi_l(e) = \Pi_l(f)\right]
\;+\; \lambda\cdot\text{balance}
```

Plain-text slide bullets, if you prefer:

- $S_{t,l}$ — the router's top-$K$ expert set for token $t$ at layer $l$, $|S_{t,l}|=K$
- $x_{t,l}$ — its 0/1 indicator, so set membership becomes algebra
- $A^{(l)}_{ef}$ — how many tokens in layer $l$ picked **both** $e$ and $f$; symmetric, off-diagonal
- diagonal $A_{ee}$ = marginal (popularity), **removed** — popularity is not partnership
- computed **offline** from a calibration window, then frozen; measured on decode
- alone it gives −21.8 % volume but **+83 % bottleneck** vs random → the balance term is what makes it win
