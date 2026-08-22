#!/usr/bin/env python3
"""Generate system architecture diagrams for the MoE + OCS Communication Research Testbed.

Pure CSS rendering — no matplotlib, no external fonts. Open the output .html in a browser.

Usage:
  python scripts/draw_system_diagram.py                    # detailed multi-section overview
  python scripts/draw_system_diagram.py --mode slide       # single-slide presentation diagram
  python scripts/draw_system_diagram.py -m slide -o custom.html
"""

import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ═══════════════════════════════════════════════════════════════
#  Slide-mode HTML — single cohesive diagram for presentations
# ═══════════════════════════════════════════════════════════════

SLIDE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoE + OCS — System Architecture</title>
<style>
  :root {
    --bg: #0b0e14;
    --surface: #13171f;
    --border: #252b36;
    --text: #c9d1d9;
    --dim: #6e7681;
    --blue: #58a6ff;
    --green: #3fb950;
    --orange: #d29922;
    --purple: #bc8cff;
    --pink: #f778ba;
    --red: #f85149;
    --cyan: #39d2c0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; padding: 20px;
  }

  /* ── Slide container (16:9-ish) ── */
  .slide {
    width: 100%; max-width: 1200px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 32px 24px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto auto auto;
    gap: 18px 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
  }

  /* ── Title bar (full width) ── */
  .title-bar {
    grid-column: 1 / -1;
    display: flex; align-items: center; gap: 14px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
  }
  .title-bar .icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
  }
  .title-bar h1 { font-size: 20px; font-weight: 700; color: #f0f6fc; letter-spacing: -0.3px; }
  .title-bar .sub { font-size: 11px; color: var(--dim); margin-top: 1px; }
  .title-bar .badge {
    margin-left: auto; font-size: 10px; font-weight: 600; padding: 4px 10px;
    border-radius: 20px; border: 1px solid var(--border);
    color: var(--dim); white-space: nowrap;
  }

  /* ── Section headers ── */
  .sec-title {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; color: var(--dim); margin-bottom: 10px;
    display: flex; align-items: center; gap: 7px;
  }
  .sec-title::before {
    content: ''; width: 8px; height: 8px; border-radius: 2px;
  }

  /* ── 1. Pipeline (left column, row 2) ── */
  .pipeline-panel {
    grid-column: 1; grid-row: 2;
  }
  .pipeline-panel .sec-title::before { background: var(--blue); }

  .pipe-flow {
    display: flex; align-items: center; gap: 0;
    flex-wrap: wrap; padding: 6px 0;
  }
  .pnode {
    display: flex; flex-direction: column; align-items: center;
    padding: 10px 13px; border-radius: 8px;
    font-size: 12px; font-weight: 600; text-align: center;
    min-width: 70px;
  }
  .pnode .picon { font-size: 18px; margin-bottom: 3px; }
  .pnode .plbl { font-size: 12px; }
  .pnode .pdetail { font-size: 10px; color: var(--dim); font-weight: 400; margin-top: 2px; }

  .pn-input { background: #1c2128; border: 1.5px solid #30363d; color: #c9d1d9; }
  .pn-router { background: #12291a; border: 1.5px solid #1f4d2a; color: #7ee787; }
  .pn-scatter { background: #0d2137; border: 1.5px solid #1a4478; color: #79c0ff; }
  .pn-compute { background: #271c0a; border: 1.5px solid #5c4311; color: #e3b341; }
  .pn-gather { background: #1a0e2a; border: 1.5px solid #482f6b; color: #d2a8ff; }
  .pn-combine { background: #2a0e1c; border: 1.5px solid #662c45; color: #fda4d4; }
  .pn-output { background: #1c2128; border: 1.5px solid #30363d; color: #c9d1d9; }

  .parrow {
    color: var(--dim); font-size: 18px; font-weight: 300;
    padding: 0 5px; flex-shrink: 0;
  }

  .pipe-note {
    font-size: 10px; color: var(--dim); margin-top: 8px;
    padding: 6px 10px; background: #1c212830; border-radius: 5px;
    border-left: 2px solid var(--orange);
  }
  .pipe-note b { color: var(--orange); }

  /* ── 2. Transport comparison (right column, row 2) ── */
  .transport-panel {
    grid-column: 2; grid-row: 2;
  }
  .transport-panel .sec-title::before { background: var(--orange); }

  .tp-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
  }
  .tp-card {
    background: #1c2128; border: 1px solid var(--border);
    border-radius: 9px; padding: 14px;
  }
  .tp-card h4 { font-size: 13px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
  .tp-eps h4 { color: var(--orange); }
  .tp-ocs h4 { color: var(--blue); }

  .tp-row { display: flex; justify-content: space-between; padding: 4px 0;
             font-size: 11px; border-bottom: 1px solid #1c2128; }
  .tp-row .k { color: var(--dim); }
  .tp-row .v { font-weight: 600; color: #f0f6fc; font-variant-numeric: tabular-nums; }
  .v-good { color: var(--green) !important; }
  .v-warn { color: var(--orange) !important; }

  /* ── 3. Circuit pool (left column, row 3) ── */
  .circuit-panel {
    grid-column: 1; grid-row: 3;
  }
  .circuit-panel .sec-title::before { background: var(--cyan); }

  .circuit-flow {
    display: flex; align-items: center; gap: 8px;
  }
  .cstage {
    background: #1c2128; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px; min-width: 130px;
  }
  .cstage h5 { font-size: 10px; color: var(--dim); text-align: center; margin-bottom: 6px;
               text-transform: uppercase; letter-spacing: 0.5px; }
  .cslot {
    display: flex; align-items: center; gap: 6px; padding: 4px 7px;
    border-radius: 4px; margin: 3px 0; font-size: 10px; font-family: 'SF Mono', monospace;
  }
  .cs-hot { background: #12291a; border: 1px solid #1f4d2a44; color: #7ee787; }
  .cs-cold { background: #271c0a; border: 1px solid #5c431144; color: #e3b341; }
  .cs-evict { background: #2d1111; border: 1px solid #f8514944; color: #f85149; }
  .cs-empty { background: #0b0e14; border: 1px dashed #1c2128; color: #30363d; }

  .cdot { width: 16px; height: 16px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: 8px;
          font-weight: 700; flex-shrink: 0; }
  .cdot-src { background: #1a447833; color: var(--blue); }
  .cdot-dst { background: #482f6b33; color: var(--purple); }

  .carrow { color: var(--dim); font-size: 16px; }

  .circuit-meta { font-size: 10px; color: var(--dim); margin-top: 8px; }
  .circuit-meta span { margin-right: 14px; }
  .circuit-meta .k { color: var(--dim); }

  /* ── 4. Overlap timeline (right column, row 3) ── */
  .timeline-panel {
    grid-column: 2; grid-row: 3;
  }
  .timeline-panel .sec-title::before { background: var(--green); }

  .tl-chart {
    background: #1c2128; border: 1px solid var(--border);
    border-radius: 9px; padding: 14px;
  }
  .tl-chart h5 { font-size: 11px; color: var(--orange); margin-bottom: 8px; }
  .tl-row { display: flex; align-items: center; margin: 6px 0; gap: 0; }
  .tl-label { width: 50px; font-size: 10px; color: var(--dim); text-align: right;
              padding-right: 8px; flex-shrink: 0; }
  .tl-bar {
    height: 20px; border-radius: 4px; margin: 0 1px;
    display: flex; align-items: center; justify-content: center;
    font-size: 9px; font-weight: 700;
  }
  .tb-scatter { background: var(--blue); color: #fff; }
  .tb-compute { background: var(--orange); color: #000; }
  .tb-gather  { background: var(--purple); color: #fff; }
  .tb-pre-hot { background: var(--green); color: #000; }
  .tb-pre-cold{ background: var(--red); color: #fff; }
  .tl-gap { width: 4px; flex-shrink: 0; }

  .tl-overlap-marker {
    border: 1.5px dashed var(--orange); border-radius: 4px;
    padding: 2px 7px; font-size: 9px; color: var(--orange);
    margin: 0 3px; white-space: nowrap;
  }
  .tl-legend {
    display: flex; gap: 12px; margin-top: 8px; font-size: 10px; color: var(--dim);
    flex-wrap: wrap;
  }
  .tl-legend span { display: flex; align-items: center; gap: 4px; }
  .tl-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }

  /* ── 5. Results bar (full width, row 4) ── */
  .results-bar {
    grid-column: 1 / -1; grid-row: 4;
    display: flex; align-items: center; gap: 20px;
    padding: 14px 18px;
    background: linear-gradient(135deg, #1c2128, #13171f);
    border: 1px solid var(--border); border-radius: 10px;
    flex-wrap: wrap;
  }
  .results-bar .rb-title {
    font-size: 12px; font-weight: 700; color: var(--dim);
    text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap;
  }
  .rb-metric { text-align: center; }
  .rb-metric .val { font-size: 20px; font-weight: 800; }
  .rb-metric .lbl { font-size: 10px; color: var(--dim); }
  .rb-delta { font-size: 18px; font-weight: 800; color: var(--green); }
  .rb-bar-wrap { flex: 1; min-width: 200px; }
  .rb-bar-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .rb-bar-lbl { width: 70px; text-align: right; font-size: 10px; color: var(--dim); flex-shrink: 0; }
  .rb-bar-track { flex: 1; background: #0b0e14; border-radius: 3px; height: 18px; overflow: hidden; }
  .rb-bar-fill { height: 100%; border-radius: 3px; display: flex; align-items: center;
                  padding-left: 7px; font-size: 10px; font-weight: 700; }
  .rb-bar-val { width: 65px; font-size: 10px; font-weight: 600; flex-shrink: 0; }

  /* ── Footer ── */
  .slide-footer {
    grid-column: 1 / -1;
    text-align: center; font-size: 9px; color: var(--dim);
    padding-top: 4px;
  }

  /* ── Responsive: stack on narrow screens ── */
  @media (max-width: 900px) {
    .slide { grid-template-columns: 1fr; }
    .pipeline-panel { grid-column: 1; grid-row: auto; }
    .transport-panel { grid-column: 1; grid-row: auto; }
    .circuit-panel { grid-column: 1; grid-row: auto; }
    .timeline-panel { grid-column: 1; grid-row: auto; }
    .results-bar { grid-column: 1; grid-row: auto; flex-direction: column; align-items: flex-start; }
  }
</style>
</head>
<body>

<div class="slide">

  <!-- ═══ Title ═══ -->
  <div class="title-bar">
    <div class="icon">⚡</div>
    <div>
      <h1>MoE + OCS Communication Research Testbed</h1>
      <div class="sub">CPU-based Expert-Parallel MoE with Optical Circuit Switching — mechanism verification testbed</div>
    </div>
    <div class="badge">Slide Diagram</div>
  </div>

  <!-- ═══ 1. MoE EP Pipeline ═══ -->
  <div class="pipeline-panel">
    <div class="sec-title">MoE Expert-Parallel Pipeline</div>
    <div class="pipe-flow">
      <div class="pnode pn-input">
        <div class="picon">📥</div>
        <div class="plbl">Tokens</div>
        <div class="pdetail">B×S × H</div>
      </div>
      <div class="parrow">→</div>
      <div class="pnode pn-router">
        <div class="picon">🎯</div>
        <div class="plbl">Router</div>
        <div class="pdetail">top-K gating</div>
      </div>
      <div class="parrow">→</div>
      <div class="pnode pn-scatter">
        <div class="picon">📤</div>
        <div class="plbl">Scatter</div>
        <div class="pdetail">all_to_all</div>
      </div>
      <div class="parrow">→</div>
      <div class="pnode pn-compute">
        <div class="picon">⚙️</div>
        <div class="plbl">Compute</div>
        <div class="pdetail">local experts</div>
      </div>
      <div class="parrow">→</div>
      <div class="pnode pn-gather">
        <div class="picon">📥</div>
        <div class="plbl">Gather</div>
        <div class="pdetail">all_to_all</div>
      </div>
      <div class="parrow">→</div>
      <div class="pnode pn-combine">
        <div class="picon">🔗</div>
        <div class="plbl">Combine</div>
        <div class="pdetail">softmax &Sigma;</div>
      </div>
      <div class="parrow">→</div>
      <div class="pnode pn-output">
        <div class="picon">✅</div>
        <div class="plbl">Output</div>
        <div class="pdetail">next layer</div>
      </div>
    </div>
    <div class="pipe-note">
      <b>Key bottleneck:</b> Scatter &amp; Gather each require one <b>all_to_all_single</b> — N ranks → N×(N−1) data streams. <b>Expert mapping:</b> expert_id → rank = expert_id // experts_per_rank
    </div>
  </div>

  <!-- ═══ 2. Transport Layer ═══ -->
  <div class="transport-panel">
    <div class="sec-title">Transport Layer: EPS vs OCS</div>
    <div class="tp-grid">
      <div class="tp-card tp-eps">
        <h4>⚡ EPS (Electrical Packet Switching)</h4>
        <div class="tp-row"><span class="k">Connection</span><span class="v">Always-on, per-packet</span></div>
        <div class="tp-row"><span class="k">Setup</span><span class="v v-good">0 µs (statistical mux)</span></div>
        <div class="tp-row"><span class="k">Per-hop latency</span><span class="v v-warn">100–500 µs</span></div>
        <div class="tp-row"><span class="k">Concurrency</span><span class="v">Unlimited</span></div>
        <div class="tp-row"><span class="k">Best for</span><span class="v">Unpredictable traffic</span></div>
      </div>
      <div class="tp-card tp-ocs">
        <h4>🔬 OCS (Optical Circuit Switching)</h4>
        <div class="tp-row"><span class="k">Connection</span><span class="v">Finite circuit pool</span></div>
        <div class="tp-row"><span class="k">Setup</span><span class="v v-warn">reconfig_time (cold)</span></div>
        <div class="tp-row"><span class="k">Hot-path latency</span><span class="v v-good">1–2 µs (optical)</span></div>
        <div class="tp-row"><span class="k">Concurrency</span><span class="v">max_circuits (ports)</span></div>
        <div class="tp-row"><span class="k">Best for</span><span class="v">Stable routing patterns</span></div>
      </div>
    </div>
  </div>

  <!-- ═══ 3. OCS Circuit Pool ═══ -->
  <div class="circuit-panel">
    <div class="sec-title">OCS Circuit Pool — Port Reassignment</div>
    <div class="circuit-flow">
      <div class="cstage">
        <h5>Cold Start</h5>
        <div class="cslot cs-cold">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">1</div>
          &nbsp;cold · 50µs
        </div>
        <div class="cslot cs-cold">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">2</div>
          &nbsp;cold · 50µs
        </div>
        <div class="cslot cs-empty">— empty —</div>
      </div>
      <div class="carrow">→</div>
      <div class="cstage">
        <h5>Hot Reuse</h5>
        <div class="cslot cs-hot">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">1</div>
          &nbsp;hot · 1µs ✓
        </div>
        <div class="cslot cs-hot">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">2</div>
          &nbsp;hot · 1µs ✓
        </div>
        <div class="cslot cs-hot">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">3</div>
          &nbsp;hot · 1µs ✓
        </div>
      </div>
      <div class="carrow">→</div>
      <div class="cstage">
        <h5>Port reassignment</h5>
        <div class="cslot cs-evict">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">4</div>
          &nbsp;evicted 0→2
        </div>
        <div class="cslot cs-hot">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">1</div>
          &nbsp;hot ✓
        </div>
        <div class="cslot cs-hot">
          <div class="cdot cdot-src">0</div>→<div class="cdot cdot-dst">3</div>
          &nbsp;hot ✓
        </div>
      </div>
    </div>
    <div class="circuit-meta">
      <span><b>Data structure:</b> OrderedDict[(src,dst)]</span>
      <span><b>Eviction:</b> popitem(last=False) — O(1)</span>
      <span><b>Promotion:</b> move_to_end() — O(1)</span>
    </div>
  </div>

  <!-- ═══ 4. Overlap Timeline ═══ -->
  <div class="timeline-panel">
    <div class="sec-title">Communication / Compute Overlap</div>
    <div class="tl-chart">
      <h5>🔬 OCS Pipeline Mode — pre-establish circuits, then overlap</h5>
      <div class="tl-row">
        <span class="tl-label">Rank 0</span>
        <span class="tl-bar tb-pre-cold" style="width:36px;">pre₀</span>
        <span class="tl-bar tb-scatter" style="width:70px;">scatter₀</span>
        <span class="tl-gap"></span>
        <span class="tl-overlap-marker">overlap zone ↓</span>
        <span class="tl-bar tb-pre-hot" style="width:22px;">pre₁</span>
        <span class="tl-bar tb-scatter" style="width:70px;">scatter₁</span>
      </div>
      <div class="tl-row" style="margin-bottom: 2px;">
        <span class="tl-label"></span>
        <span style="font-size:8px;color:var(--red);width:36px;text-align:center;">50µs</span>
        <span style="width:70px;"></span>
        <span style="width:4px;"></span>
        <span style="font-size:8px;color:var(--green);width:100px;text-align:center;">hidden behind<br>compute₀</span>
      </div>
      <div class="tl-row">
        <span class="tl-label">Rank 1</span>
        <span class="tl-bar tb-scatter" style="width:70px;opacity:0.35;"></span>
        <span class="tl-gap"></span>
        <span class="tl-bar tb-scatter" style="width:70px;">scatter₀ rx</span>
        <span class="tl-bar tb-compute" style="width:55px;">comp₀</span>
        <span class="tl-bar tb-gather" style="width:70px;">gather₀</span>
      </div>
      <div class="tl-legend">
        <span><span class="tl-swatch" style="background:var(--blue);"></span> Scatter</span>
        <span><span class="tl-swatch" style="background:var(--orange);"></span> Compute</span>
        <span><span class="tl-swatch" style="background:var(--purple);"></span> Gather</span>
        <span><span class="tl-swatch" style="background:var(--green);"></span> Pre-est. (hot)</span>
        <span><span class="tl-swatch" style="background:var(--red);"></span> Pre-est. (cold)</span>
      </div>
    </div>
  </div>

  <!-- ═══ 5. Results ═══ -->
  <div class="results-bar">
    <span class="rb-title">Sample Results</span>
    <div class="rb-metric">
      <div class="val" style="color:var(--orange);">23,269 µs</div>
      <div class="lbl">EPS Baseline</div>
    </div>
    <div class="rb-metric">
      <div class="val" style="color:var(--green);">17,424 µs</div>
      <div class="lbl">OCS Pipeline</div>
    </div>
    <div class="rb-delta">−25.1%</div>
    <div class="rb-metric">
      <div class="val" style="color:var(--green);">96.7%</div>
      <div class="lbl">Circuit Reuse</div>
    </div>
    <div class="rb-metric">
      <div class="val" style="color:var(--green);">0</div>
      <div class="lbl">Port reassignments</div>
    </div>
    <div class="rb-bar-wrap">
      <div class="rb-bar-row">
        <span class="rb-bar-lbl">EPS</span>
        <div class="rb-bar-track"><div class="rb-bar-fill" style="width:100%;background:var(--orange);">23,269 µs</div></div>
        <span class="rb-bar-val">100%</span>
      </div>
      <div class="rb-bar-row">
        <span class="rb-bar-lbl">OCS</span>
        <div class="rb-bar-track"><div class="rb-bar-fill" style="width:74.9%;background:var(--green);">17,424 µs</div></div>
        <span class="rb-bar-val" style="color:var(--green);">74.9%</span>
      </div>
    </div>
  </div>

  <!-- ═══ Footer ═══ -->
  <div class="slide-footer">
    MoE Communication Research Testbed · OCS Integration · 2026 &nbsp;|&nbsp;
    <code>python scripts/draw_system_diagram.py --mode slide</code>
  </div>

</div>

</body>
</html>"""

# ═══════════════════════════════════════════════════════════════
#  Detailed multi-section HTML (original mode)
# ═══════════════════════════════════════════════════════════════

DETAIL_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoE + OCS 通信实验平台 — 系统架构图</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, 'PingFang SC', 'Noto Sans SC', 'Microsoft YaHei', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; line-height: 1.6; }
  h1 { text-align: center; color: #58a6ff; font-size: 24px; margin-bottom: 4px; }
  .subtitle { text-align: center; color: #8b949e; font-size: 13px; margin-bottom: 32px; }

  .section { max-width: 960px; margin: 0 auto 32px; }
  .section-title { font-size: 18px; color: #f0f6fc; border-left: 3px solid #58a6ff;
                   padding-left: 10px; margin-bottom: 14px; }

  /* === 图 1: MoE EP 流水线 === */
  .pipeline { display: flex; flex-direction: column; gap: 0; background: #161b22;
              border: 1px solid #30363d; border-radius: 10px; padding: 20px; overflow-x: auto; }
  .pipeline .row { display: flex; align-items: center; gap: 0; min-width: 820px; }
  .pipeline .label { width: 90px; font-size: 12px; color: #8b949e; text-align: right;
                     padding-right: 10px; flex-shrink: 0; }
  .box { border-radius: 6px; padding: 10px 14px; font-size: 13px; font-weight: 600;
         text-align: center; white-space: nowrap; margin: 3px 2px; }
  .box-router { background: #1a3a2a; border: 1.5px solid #3fb950; color: #7ee787; }
  .box-scatter { background: #0d2b3f; border: 1.5px solid #58a6ff; color: #79c0ff; }
  .box-compute { background: #2d1a0a; border: 1.5px solid #d29922; color: #e3b341; }
  .box-gather { background: #1e0a2d; border: 1.5px solid #bc8cff; color: #d2a8ff; }
  .box-combine { background: #2d0a1e; border: 1.5px solid #f778ba; color: #fda4d4; }
  .box-token { background: #161b22; border: 1.5px solid #484f58; color: #c9d1d9; }
  .arrow { color: #484f58; font-size: 16px; padding: 0 4px; flex-shrink: 0; }
  .note { font-size: 11px; color: #8b949e; padding-left: 6px; }

  /* === 图 2: EPS vs OCS === */
  .cmp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .cmp-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
              padding: 18px; }
  .cmp-card h3 { font-size: 15px; margin-bottom: 12px; }
  .eps-title { color: #d29922; } .ocs-title { color: #58a6ff; }
  .cmp-row { display: flex; justify-content: space-between; padding: 7px 0;
             border-bottom: 1px solid #21262d; font-size: 13px; }
  .cmp-row .lbl { color: #8b949e; }
  .cmp-row .val { font-weight: 600; color: #f0f6fc; }
  .highlight-good { color: #3fb950; }
  .highlight-bad { color: #f85149; }

  /* === 图 3: 电路池 === */
  .pool-viz { display: flex; gap: 10px; align-items: stretch; flex-wrap: wrap; }
  .pool-col { flex: 1; min-width: 220px; background: #161b22; border: 1px solid #30363d;
              border-radius: 10px; padding: 14px; }
  .pool-col h4 { font-size: 13px; margin-bottom: 8px; color: #f0f6fc; text-align: center; }
  .circuit-slot { display: flex; align-items: center; gap: 8px; padding: 7px 10px;
                  border-radius: 5px; margin: 4px 0; font-size: 12px; font-family: monospace; }
  .slot-hot { background: #1a3a2a; border: 1px solid #3fb95033; color: #7ee787; }
  .slot-cold { background: #2d1a0a; border: 1px solid #d2992233; color: #e3b341; }
  .slot-empty { background: #0d1117; border: 1px dashed #30363d; color: #484f58; }
  .slot-label { width: 24px; height: 24px; border-radius: 50%; display: flex;
                align-items: center; justify-content: center; font-size: 10px;
                font-weight: 700; }
  .label-src { background: #58a6ff33; color: #58a6ff; }
  .label-dst { background: #bc8cff33; color: #bc8cff; }

  /* === 图 4: overlap 流水线 === */
  .timeline { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
              padding: 20px; overflow-x: auto; }
  .timeline-title { font-size: 13px; color: #f0f6fc; margin-bottom: 12px; }
  .tl-row { display: flex; align-items: center; margin: 8px 0; min-width: 800px; }
  .tl-rank { width: 56px; font-size: 11px; color: #8b949e; text-align: right;
             padding-right: 8px; flex-shrink: 0; }
  .tl-bar { height: 22px; border-radius: 4px; margin: 0 1px; display: flex;
            align-items: center; justify-content: center; font-size: 10px;
            font-weight: 600; min-width: 18px; }
  .tl-scatter { background: #58a6ff; color: #fff; }
  .tl-compute { background: #d29922; color: #000; }
  .tl-gather { background: #bc8cff; color: #fff; }
  .tl-pre { background: #3fb950; color: #000; }
  .tl-pre-exposed { background: #f85149; color: #fff; }
  .tl-gap { width: 6px; flex-shrink: 0; }
  .tl-overlap-bracket { border: 1.5px dashed #d29922; border-radius: 6px;
                        padding: 4px 8px; margin: 0 4px; font-size: 10px;
                        color: #d29922; display: flex; align-items: center;
                        flex-shrink: 0; }
  .legend { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 5px; color: #8b949e; }
  .legend-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }

  /* === 图 5: 实验设计 === */
  .exp-flow { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
              background: #161b22; border: 1px solid #30363d; border-radius: 10px;
              padding: 20px; justify-content: center; }
  .exp-node { background: #0d1117; border: 1.5px solid #30363d; border-radius: 8px;
              padding: 12px 16px; text-align: center; font-size: 13px; }
  .exp-node .icon { font-size: 22px; }
  .exp-node .desc { font-size: 11px; color: #8b949e; margin-top: 4px; }
  .exp-arrow-char { font-size: 20px; color: #484f58; }
  .exp-vs { background: #2d1a0a; border-color: #d29922; font-weight: 600; }

  /* === 图 6: 结果摘要 === */
  .result-bar { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
  .result-label { width: 100px; text-align: right; font-size: 12px; color: #8b949e; flex-shrink: 0; }
  .result-track { flex: 1; background: #21262d; border-radius: 4px; height: 22px; overflow: hidden; position: relative; }
  .result-fill { height: 100%; border-radius: 4px; display: flex; align-items: center;
                 padding-left: 8px; font-size: 11px; font-weight: 600; }
  .result-value { width: 80px; font-size: 12px; font-weight: 600; flex-shrink: 0; }

  /* === 响应式 === */
  @media (max-width: 700px) {
    .cmp-grid { grid-template-columns: 1fr; }
    .pool-viz { flex-direction: column; }
  }
</style>
</head>
<body>

<h1>MoE + OCS 通信实验平台</h1>
<div class="subtitle">系统架构 · 传输层对比 · 电路池模型 · 实验设计</div>

<!-- ====== 图 1: MoE + EP 流水线 ====== -->
<div class="section">
  <div class="section-title">1. MoE 专家并行 (EP) 通信流水线</div>
  <div class="pipeline">

    <div class="row">
      <span class="label">输入</span>
      <span class="box box-token">B×S 个 Token<br><span class="note">[batch × seq, hidden]</span></span>
      <span class="arrow">→</span>
      <span class="box box-router">Router 路由<br><span class="note">top-K gating</span></span>
      <span class="arrow">→</span>
      <span class="box box-token">expert_ids<br>gate_weights</span>
    </div>

    <div style="height:8px"></div>

    <div class="row">
      <span class="label"></span>
      <span class="box box-scatter">🔵 Scatter<br><span class="note">all_to_all → target rank<br>(expert_id // E_per_rank)</span></span>
      <span class="arrow">→</span>
      <span class="box box-compute">🟠 Compute<br><span class="note">每个 rank 计算<br>本地 expert 前向</span></span>
      <span class="arrow">→</span>
      <span class="box box-gather">🟣 Gather<br><span class="note">all_to_all 回传<br>按 orig_idx 重排</span></span>
      <span class="arrow">→</span>
      <span class="box box-combine">🩷 Combine<br><span class="note">softmax 加权求和<br>(top-K ≥ 2 时)</span></span>
    </div>

    <div style="height:8px"></div>

    <div class="row">
      <span class="label"></span>
      <span style="font-size:12px;color:#8b949e;">
        ⬆ 关键通信瓶颈: Scatter/Gather 各需要一次 <b>all_to_all_single</b>（N 个 rank → N×(N−1) 条数据流）
      </span>
    </div>
  </div>
</div>

<!-- ====== 图 2: EPS vs OCS ====== -->
<div class="section">
  <div class="section-title">2. 传输层对比: EPS 电交换 vs OCS 光交换</div>
  <div class="cmp-grid">
    <div class="cmp-card">
      <h3 class="eps-title">⚡ EPS (传统电分组交换)</h3>
      <div class="cmp-row"><span class="lbl">连接模型</span><span class="val">始终在线，逐包路由</span></div>
      <div class="cmp-row"><span class="lbl">建立开销</span><span class="val highlight-good">0（统计复用）</span></div>
      <div class="cmp-row"><span class="lbl">每跳延迟</span><span class="val highlight-bad">高（交换芯片排队+转发）</span></div>
      <div class="cmp-row"><span class="lbl">并发容量</span><span class="val">理论上无上限</span></div>
      <div class="cmp-row"><span class="lbl">典型延迟</span><span class="val">100–500 µs / 跳</span></div>
      <div class="cmp-row"><span class="lbl">适合场景</span><span class="val">流量不可预测、连接数多变</span></div>
      <div style="margin-top:10px; padding:8px; background:#2d1a0a22; border-radius:5px; font-size:12px; color:#d29922;">
        ⚠ 每次 all-to-all 都付出完整的逐跳延迟<br>
        delay = latency + bytes / BW
      </div>
    </div>

    <div class="cmp-card">
      <h3 class="ocs-title">🔬 OCS (光电路交换)</h3>
      <div class="cmp-row"><span class="lbl">连接模型</span><span class="val highlight-good">有限电路池，按需建立</span></div>
      <div class="cmp-row"><span class="lbl">建立开销</span><span class="val highlight-bad">冷启动: reconfig_time µs</span></div>
      <div class="cmp-row"><span class="lbl">热路径延迟</span><span class="val highlight-good">极低（纯光学直通）</span></div>
      <div class="cmp-row"><span class="lbl">并发容量</span><span class="val">max_circuits 条电路</span></div>
      <div class="cmp-row"><span class="lbl">典型延迟</span><span class="val">1–2 µs / 跳（热路径）</span></div>
      <div class="cmp-row"><span class="lbl">适合场景</span><span class="val">流量模式稳定、可预测</span></div>
      <div style="margin-top:10px; padding:8px; background:#1a3a2a22; border-radius:5px; font-size:12px; color:#3fb950;">
        ✅ 热路径: delay = circuit_latency + bytes / BW<br>
        ❌ 冷路径: 外加 reconfig_time（MEMS 微镜切换）
      </div>
    </div>
  </div>
</div>

<!-- ====== 图 3: OCS 电路池 ====== -->
<div class="section">
  <div class="section-title">3. OCS 电路池 端口重分配机制</div>
  <div class="pool-viz">

    <div class="pool-col">
      <h4>初始状态（空池）</h4>
      <div class="circuit-slot slot-empty">槽 0 · 空闲</div>
      <div class="circuit-slot slot-empty">槽 1 · 空闲</div>
      <div class="circuit-slot slot-empty">槽 2 · 空闲</div>
      <div class="circuit-slot slot-empty">槽 3 · 空闲</div>
      <div style="font-size:11px;color:#8b949e;margin-top:6px;">max_circuits = 4</div>
    </div>

    <div style="color:#484f58;font-size:22px;display:flex;align-items:center;">→</div>

    <div class="pool-col">
      <h4>Step 1: 冷启动</h4>
      <div class="circuit-slot slot-cold">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">1</span>
        cold · 50µs
      </div>
      <div class="circuit-slot slot-cold">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">2</span>
        cold · 50µs
      </div>
      <div class="circuit-slot slot-cold">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">3</span>
        cold · 50µs
      </div>
      <div class="circuit-slot slot-empty">槽 3 · 空闲</div>
      <div style="font-size:11px;color:#d29922;margin-top:6px;">3 次建立，每次 50µs</div>
    </div>

    <div style="color:#484f58;font-size:22px;display:flex;align-items:center;">→</div>

    <div class="pool-col">
      <h4>Step 2: 热路径复用</h4>
      <div class="circuit-slot slot-hot">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">1</span>
        hot ✓ · 1µs
      </div>
      <div class="circuit-slot slot-hot">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">2</span>
        hot ✓ · 1µs
      </div>
      <div class="circuit-slot slot-hot">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">3</span>
        hot ✓ · 1µs
      </div>
      <div class="circuit-slot slot-empty">槽 3 · 空闲</div>
      <div style="font-size:11px;color:#3fb950;margin-top:6px;">3 次复用，每次 1µs<br>reuse_ratio = 50%</div>
    </div>

    <div style="color:#484f58;font-size:22px;display:flex;align-items:center;">→</div>

    <div class="pool-col">
      <h4>Step 3: 端口重分配</h4>
      <div class="circuit-slot slot-cold" style="background:#3d1a1a;border-color:#f8514933;">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">4</span>
        ✗ evicted (0→2)
      </div>
      <div class="circuit-slot slot-hot">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">3</span>
        hot ✓
      </div>
      <div class="circuit-slot slot-hot">
        <span class="slot-label label-src">0</span>→<span class="slot-label label-dst">1</span>
        hot ✓
      </div>
      <div class="circuit-slot slot-empty">槽 3 · 空闲</div>
      <div style="font-size:11px;color:#f85149;margin-top:6px;">
        池满时弹出最久未用的电路<br>
        OrderedDict.popitem(last=False)
      </div>
    </div>

  </div>
</div>

<!-- ====== 图 4: 重叠流水线对比 ====== -->
<div class="section">
  <div class="section-title">4. 通信/计算重叠 + OCS 预建立</div>

  <!-- EPS Overlap -->
  <div class="timeline" style="margin-bottom:14px;">
    <div class="timeline-title">📡 EPS overlap 模式（基准） — 通信延迟暴露在关键路径上</div>
    <div class="tl-row">
      <span class="tl-rank">Rank 0</span>
      <span class="tl-bar tl-scatter" style="width:90px;">scatter₀</span>
      <span class="tl-gap"></span>
      <span class="tl-overlap-bracket">overlap<br>zone ↓</span>
      <span class="tl-bar tl-scatter" style="width:90px;">scatter₁</span>
    </div>
    <div class="tl-row">
      <span class="tl-rank">Rank 1</span>
      <span class="tl-bar tl-scatter" style="width:90px;opacity:0.4;"></span>
      <span class="tl-gap"></span>
      <span class="tl-bar tl-scatter" style="width:90px;">scatter₀ 到达</span>
      <span class="tl-bar tl-compute" style="width:70px;">compute₀</span>
      <span class="tl-bar tl-gather" style="width:90px;">gather₀</span>
    </div>
    <div style="font-size:11px;color:#f85149;margin-top:10px;">
      ⚠ 每次 scatter/gather 都支付 EPS 延迟（例如 500µs）—— 即使计算已经重叠
    </div>
  </div>

  <!-- OCS Pipeline -->
  <div class="timeline">
    <div class="timeline-title">🔬 OCS ocs_pipeline 模式 — 预建立电路，热路径通信</div>
    <div class="tl-row">
      <span class="tl-rank">Rank 0</span>
      <span class="tl-bar tl-pre-exposed" style="width:40px;">pre₀</span>
      <span class="tl-bar tl-scatter" style="width:90px;">scatter₀</span>
      <span class="tl-gap"></span>
      <span class="tl-overlap-bracket">overlap<br>zone ↓</span>
      <span class="tl-bar tl-pre" style="width:28px;">pre₁</span>
      <span class="tl-bar tl-scatter" style="width:90px;">scatter₁</span>
    </div>
    <div class="tl-row">
      <span class="tl-rank">Rank 0</span>
      <span style="font-size:10px;color:#f85149;width:40px;text-align:center;">冷启动<br>50µs</span>
      <span style="width:90px;"></span>
      <span style="width:6px;"></span>
      <span style="font-size:10px;color:#3fb950;width:120px;text-align:center;">已隐藏于<br>compute₀ 之后</span>
    </div>
    <div class="tl-row" style="margin-top:4px;">
      <span class="tl-rank">Rank 1</span>
      <span class="tl-bar tl-scatter" style="width:90px;opacity:0.4;"></span>
      <span class="tl-gap"></span>
      <span class="tl-bar tl-scatter" style="width:90px;">scatter₀ 到达</span>
      <span class="tl-bar tl-compute" style="width:70px;">compute₀</span>
      <span class="tl-bar tl-gather" style="width:90px;">gather₀</span>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="legend-swatch" style="background:#f85149;"></div>冷预建立（暴露）</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#3fb950;"></div>热预建立（重叠）</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#58a6ff;"></div>Scatter</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#d29922;"></div>Compute</div>
      <div class="legend-item"><div class="legend-swatch" style="background:#bc8cff;"></div>Gather</div>
    </div>
  </div>
</div>

<!-- ====== 图 5: 实验设计 ====== -->
<div class="section">
  <div class="section-title">5. 实验设计: A/B 对照</div>
  <div class="exp-flow">
    <div class="exp-node">
      <div class="icon">📋</div>
      <b>同一 MoE 任务</b>
      <div class="desc">4 ranks, 8 experts<br>top-1 gating, 2 microbatches</div>
    </div>
    <span class="exp-arrow-char">→</span>
    <div class="exp-node" style="border-color:#d29922;">
      <div class="icon">📡</div>
      <b>Baseline: EPS</b>
      <div class="desc">overlap 流水线<br>flat delay = 500µs<br>无电路池</div>
    </div>
    <span class="exp-arrow-char">→</span>
    <div class="exp-node exp-vs">
      <div class="icon">⚖️</div>
      <b>对比</b>
      <div class="desc">compare_ocs.py<br>相同 pipeline<br>不同 transport</div>
    </div>
    <span class="exp-arrow-char">←</span>
    <div class="exp-node" style="border-color:#58a6ff;">
      <div class="icon">🔬</div>
      <b>OCS: ocs_pipeline</b>
      <div class="desc">同 overlap 流水线<br>8 电路池, 50µs 重配置<br>200Gbps 光带宽</div>
    </div>
    <span class="exp-arrow-char">→</span>
    <div class="exp-node">
      <div class="icon">📊</div>
      <b>输出</b>
      <div class="desc">JSON + HTML 报告<br>Delta 分析<br>电路复用率</div>
    </div>
  </div>
</div>

<!-- ====== 图 6: 结果摘要 ====== -->
<div class="section">
  <div class="section-title">6. 实测结果摘要</div>
  <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;">

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:18px;">
      <div>
        <div style="font-size:13px;color:#f0f6fc;margin-bottom:10px;">📡 EPS Baseline (overlap + 500µs delay)</div>
        <div style="font-size:26px;font-weight:700;">23,269 µs</div>
        <div style="font-size:12px;color:#8b949e;">comm = 100% · 71 个事件</div>
      </div>
      <div>
        <div style="font-size:13px;color:#f0f6fc;margin-bottom:10px;">🔬 OCS Pipeline (50µs reconfig)</div>
        <div style="font-size:26px;font-weight:700;color:#3fb950;">17,424 µs</div>
        <div style="font-size:12px;color:#8b949e;">OCS 额外开销 333µs · 81 个事件</div>
      </div>
    </div>

    <div style="background:#0d1117;border-radius:8px;padding:12px;margin-bottom:14px;">
      <div style="font-size:16px;font-weight:700;color:#3fb950;margin-bottom:8px;">
        Δ = −5,845 µs (−25.1%)
      </div>
      <div class="result-bar">
        <span class="result-label">EPS Baseline</span>
        <div class="result-track"><div class="result-fill" style="width:100%;background:#d29922;">23,269 µs</div></div>
        <span class="result-value">100%</span>
      </div>
      <div class="result-bar">
        <span class="result-label">OCS Pipeline</span>
        <div class="result-track"><div class="result-fill" style="width:74.9%;background:#3fb950;">17,424 µs</div></div>
        <span class="result-value" style="color:#3fb950;">74.9%</span>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;font-size:12px;">
      <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
        <div style="font-size:20px;font-weight:700;color:#3fb950;">96.7%</div>
        <div style="color:#8b949e;">电路复用率</div>
        <div style="color:#484f58;">87/90 请求命中热路径</div>
      </div>
      <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
        <div style="font-size:20px;font-weight:700;color:#d29922;">3 次</div>
        <div style="color:#8b949e;">冷建立</div>
        <div style="color:#484f58;">仅首个 microbatch 触发</div>
      </div>
      <div style="background:#0d1117;border-radius:6px;padding:10px;text-align:center;">
        <div style="font-size:20px;font-weight:700;color:#3fb950;">0 次</div>
        <div style="color:#8b949e;">端口重分配</div>
        <div style="color:#484f58;">8 个槽位足够 3 条电路</div>
      </div>
    </div>
  </div>
</div>

<!-- ====== Footer ====== -->
<div style="text-align:center;font-size:12px;color:#484f58;padding:20px 0;">
  MoE Communication Research Testbed · OCS Integration · 2026<br>
  <code style="font-size:11px;">python scripts/compare_ocs.py</code> 运行实验 &nbsp;|&nbsp;
  <code style="font-size:11px;">python scripts/draw_system_diagram.py</code> 生成本图
</div>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate MoE+OCS system architecture diagram HTML")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default depends on mode)",
    )
    parser.add_argument(
        "-m", "--mode", choices=["detail", "slide"], default="detail",
        help="Diagram mode: 'detail' = multi-section documentation (default), "
             "'slide' = single-slide presentation diagram",
    )
    args = parser.parse_args()

    if args.mode == "slide":
        default_name = "system_slide.html"
        html_content = SLIDE_HTML
        desc = "slide diagram"
    else:
        default_name = "system_overview.html"
        html_content = DETAIL_HTML
        desc = "detailed overview"

    output = args.output or os.path.join(PROJECT_DIR, "outputs", "diagrams", default_name)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, "w") as f:
        f.write(html_content)

    print(f"Diagram generated ({desc}): {output}")
    print(f"Open in browser: open {output}")


if __name__ == "__main__":
    main()
