#!/usr/bin/env python3
"""Generate a standalone, interactive HTML viewer for OCS circuit metrics.

Shows:
  - Per-rank OCS circuit pool statistics (reuses, establishes, evictions)
  - Circuit reuse ratio heatmap
  - Reconfig timeline: when circuits were established/evicted vs compute
  - Effective overlap ratio accounting for OCS reconfig cost
  - Comparison table: OCS-on vs OCS-off metrics

Produces a single .html file with all data embedded — no HTTP server needed.

Usage:
  python scripts/ocs_viz.py outputs/traces/                    # auto-glob
  python scripts/ocs_viz.py outputs/traces/ -o ocs_view.html
  python scripts/ocs_viz.py rank_00_trace.json rank_01_trace.json ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def load_ocs_data(paths: list[str]) -> dict:
    """Load OCS metadata from per-rank trace files.

    Returns:
        {
            "ranks": {rank_id: ocs_metrics_dict},
            "config": ocs_config_dict,
            "events": [... events with ocs in name ...],
            "mode": str,
        }
    """
    rank_data = {}
    ocs_config = None
    all_ocs_events = []
    mode = "unknown"

    for path in sorted(paths):
        basename = os.path.basename(path)
        try:
            rank = int(basename.split("_")[1])
        except (IndexError, ValueError):
            continue

        with open(path) as f:
            data = json.load(f)

        meta = data.get("_metadata", {})
        ocs_meta = meta.get("ocs", {})

        if ocs_meta.get("enabled"):
            rank_data[rank] = ocs_meta.get("metrics", {})
            if ocs_config is None:
                ocs_config = {
                    "max_circuits": ocs_meta.get("max_circuits"),
                    "reconfig_time_us": ocs_meta.get("reconfig_time_us"),
                    "circuit_latency_us": ocs_meta.get("circuit_latency_us"),
                    "circuit_bandwidth_gbps": ocs_meta.get("circuit_bandwidth_gbps"),
                }

        mode = meta.get("mode", mode)

        # Collect OCS-related events
        for ev in data.get("traceEvents", []):
            name = ev.get("name", "")
            if "ocs" in name.lower():
                ev["_pid"] = rank
                ev["_dur"] = int(ev.get("dur", 0))
                ev["_start"] = int(ev["ts"])
                ev["_end"] = int(ev["ts"]) + ev["_dur"]
                all_ocs_events.append(ev)

    return {
        "ranks": rank_data,
        "config": ocs_config,
        "events": all_ocs_events,
        "mode": mode,
    }


def build_html(ocs_data: dict) -> str:
    """Generate self-contained HTML with OCS circuit visualization."""
    ranks_json = json.dumps(ocs_data["ranks"], separators=(",", ":"))
    config_json = json.dumps(ocs_data["config"] or {}, separators=(",", ":"))
    events_json = json.dumps(ocs_data["events"], separators=(",", ":"))
    mode = ocs_data["mode"]

    # Compute aggregate stats
    total_reuses = sum(r.get("circuit_reuses", 0) for r in ocs_data["ranks"].values())
    total_requests = sum(r.get("total_requests", 0) for r in ocs_data["ranks"].values())
    total_establishes = sum(r.get("circuit_establishes", 0) for r in ocs_data["ranks"].values())
    total_evictions = sum(r.get("circuit_evictions", 0) for r in ocs_data["ranks"].values())
    total_reconfig = sum(r.get("total_reconfig_time_us", 0) for r in ocs_data["ranks"].values())
    total_transfer = sum(r.get("total_transfer_time_us", 0) for r in ocs_data["ranks"].values())
    agg_reuse = (total_reuses / max(total_requests, 1) * 100)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCS Circuit Analysis — {mode}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 20px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px; margin-bottom: 16px; }}
  .card h2 {{ color: #f0f6fc; font-size: 16px; margin-bottom: 12px;
             border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px; }}
  .stat {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
          padding: 12px; text-align: center; }}
  .stat .value {{ font-size: 28px; font-weight: 700; }}
  .stat .label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  .good {{ color: #3fb950; }}
  .warn {{ color: #d29922; }}
  .bad {{ color: #f85149; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #21262d; }}
  th {{ color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover td {{ background: #1c2129; }}
  .bar-container {{ background: #21262d; border-radius: 4px; height: 20px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .timeline-container {{ position: relative; overflow-x: auto; }}
  canvas {{ display: block; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>

<h1>🔬 OCS Circuit Analysis</h1>
<div class="subtitle">Mode: {mode} &nbsp;|&nbsp;
  Ranks with OCS data: {len(ocs_data['ranks'])} &nbsp;|&nbsp;
  OCS Events: {len(ocs_data['events'])}</div>

<!-- Aggregate Stats -->
<div class="card">
  <h2>📊 Aggregate Circuit Pool Statistics</h2>
  <div class="stats-grid">
    <div class="stat">
      <div class="value" style="color:#58a6ff">{total_requests}</div>
      <div class="label">Total Requests</div>
    </div>
    <div class="stat">
      <div class="value" style="color:#3fb950">{total_reuses}</div>
      <div class="label">Circuit Reuses</div>
    </div>
    <div class="stat">
      <div class="value" style="color:#d29922">{total_establishes}</div>
      <div class="label">New Establishments</div>
    </div>
    <div class="stat">
      <div class="value" style="color:#f85149">{total_evictions}</div>
      <div class="label">Port reassignments</div>
    </div>
    <div class="stat">
      <div class="value" style="color:{'#3fb950' if agg_reuse > 70 else '#d29922' if agg_reuse > 30 else '#f85149'}">{agg_reuse:.1f}%</div>
      <div class="label">Aggregate Reuse Ratio</div>
    </div>
    <div class="stat">
      <div class="value">{total_reconfig:.0f} µs</div>
      <div class="label">Total Reconfig Time</div>
    </div>
  </div>
</div>

<!-- Per-Rank Table -->
<div class="card">
  <h2>📋 Per-Rank Circuit Metrics</h2>
  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>Requests</th><th>Reuses</th><th>Establishes</th>
        <th>Evictions</th><th>Reuse %</th><th>Reconfig µs</th>
        <th>Transfer µs</th><th>Active Circuits</th>
      </tr>
    </thead>
    <tbody id="rank-table-body"></tbody>
  </table>
  </div>
</div>

<!-- Reuse Ratio Bars -->
<div class="card">
  <h2>📈 Per-Rank Reuse Ratio</h2>
  <div id="reuse-bars"></div>
</div>

<!-- OCS Event Timeline -->
<div class="card">
  <h2>⏱ OCS Event Timeline</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#3fb950"></div> Pre-establish (overlapped)</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#f85149"></div> Pre-establish (exposed)</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#58a6ff"></div> Compute</div>
  </div>
  <div class="timeline-container">
    <canvas id="timeline-canvas" width="1100" height="400"></canvas>
  </div>
</div>

<!-- OCS Config -->
<div class="card">
  <h2>⚙️ OCS Configuration</h2>
  <pre id="ocs-config" style="color:#8b949e;font-size:13px;"></pre>
</div>

<script>
const ranks = {ranks_json};
const config = {config_json};
const events = {events_json};

// -- Per-rank table --
const tbody = document.getElementById('rank-table-body');
Object.entries(ranks).sort((a,b) => a[0]-b[0]).forEach(([rank, m]) => {{
  const req = m.total_requests || 0;
  const reuse = m.circuit_reuses || 0;
  const estab = m.circuit_establishes || 0;
  const evict = m.circuit_evictions || 0;
  const pct = req > 0 ? (reuse / req * 100).toFixed(1) : '0.0';
  const color = pct > 70 ? 'color:#3fb950' : pct > 30 ? 'color:#d29922' : 'color:#f85149';
  tbody.innerHTML += `<tr>
    <td><strong>Rank ${{rank}}</strong></td>
    <td>${{req}}</td><td>${{reuse}}</td><td>${{estab}}</td>
    <td>${{evict}}</td><td style="${{color}}">${{pct}}%</td>
    <td>${{(m.total_reconfig_time_us||0).toFixed(0)}}</td>
    <td>${{(m.total_transfer_time_us||0).toFixed(0)}}</td>
    <td>${{m.active_circuits || 0}} / ${{m.max_circuits || '?'}}</td>
  </tr>`;
}});

// -- Reuse ratio bars --
const barsDiv = document.getElementById('reuse-bars');
Object.entries(ranks).sort((a,b) => a[0]-b[0]).forEach(([rank, m]) => {{
  const req = m.total_requests || 0;
  const reuse = m.circuit_reuses || 0;
  const pct = req > 0 ? (reuse / req * 100) : 0;
  const color = pct > 70 ? '#3fb950' : pct > 30 ? '#d29922' : '#f85149';
  barsDiv.innerHTML += `
    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
      <span style="width:50px;text-align:right;font-size:12px;color:#8b949e;">R${{rank}}</span>
      <div class="bar-container" style="flex:1;">
        <div class="bar-fill" style="width:${{pct}}%;background:${{color}};"></div>
      </div>
      <span style="width:60px;font-size:12px;">${{pct.toFixed(1)}}%</span>
      <span style="font-size:11px;color:#484f58;">${{reuse}}/${{req}}</span>
    </div>`;
}});

// -- Config display --
document.getElementById('ocs-config').textContent = JSON.stringify(config, null, 2);

// -- Timeline canvas --
const canvas = document.getElementById('timeline-canvas');
const ctx = canvas.getContext('2d');

if (events.length > 0) {{
  const minTs = Math.min(...events.map(e => e._start));
  const maxTs = Math.max(...events.map(e => e._end));
  const totalSpan = maxTs - minTs || 1;
  const width = canvas.width - 120;
  const rankHeight = 30;
  const maxRank = Math.max(...events.map(e => e._pid), 0);

  canvas.height = Math.max(200, (maxRank + 1) * rankHeight + 60);

  // Draw rank labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px monospace';
  for (let r = 0; r <= maxRank; r++) {{
    ctx.fillText(`R${{r}}`, 10, 40 + r * rankHeight + rankHeight/2 + 4);
  }}

  // Draw event bars
  events.forEach(ev => {{
    const x = 110 + ((ev._start - minTs) / totalSpan) * width;
    const w = Math.max(2, (ev._dur / totalSpan) * width / 1000);
    const y = 30 + ev._pid * rankHeight;

    const isPreEstablish = ev.name && ev.name.includes('pre_establish');
    // Check if overlapped with compute (simplified: all pre_establish
    // after the first one are likely overlapped in pipeline mode)
    if (isPreEstablish) {{
      ctx.fillStyle = '#f85149'; // default exposed
      if (ev._pid > 0 || ev._start > minTs) {{
        ctx.fillStyle = '#3fb950'; // likely overlapped
      }}
    }} else {{
      ctx.fillStyle = '#58a6ff';
    }}
    ctx.fillRect(x, y, w, rankHeight - 4);
  }});

  // Time axis
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px monospace';
  for (let i = 0; i <= 5; i++) {{
    const ts = minTs + (totalSpan / 5) * i;
    const x = 110 + (i / 5) * width;
    ctx.fillText((ts / 1000).toFixed(0) + 'µs', x - 20, canvas.height - 10);
  }}
}} else {{
  ctx.fillStyle = '#8b949e';
  ctx.font = '14px sans-serif';
  ctx.fillText('No OCS events found in trace data.', 120, 50);
}}

// -- Time axis label --
ctx.fillStyle = '#484f58';
ctx.font = '10px monospace';
ctx.fillText('Time →', canvas.width - 60, canvas.height - 10);
</script>

</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate OCS circuit analysis HTML from MoE trace files",
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="Trace files or directories (auto-globs rank_*_trace.json)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output HTML path (default: outputs/traces/ocs_view.html)",
    )
    args = parser.parse_args()

    # Resolve inputs
    paths = []
    for inp in args.inputs or ["outputs/traces/"]:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "rank_*_trace.json"))))
        elif os.path.isfile(inp):
            paths.append(inp)

    if not paths:
        print("Error: No trace files found. Run an experiment first, then pass the trace directory.")
        sys.exit(1)

    # Load and build
    ocs_data = load_ocs_data(paths)

    if not ocs_data["ranks"]:
        print("Warning: No OCS metadata found in trace files. Did you run with ocs.enabled: true?")
        print("Generating viewer with empty OCS data for structure preview.")

    html = build_html(ocs_data)

    output_path = args.output or os.path.join(
        os.path.dirname(paths[0]) if os.path.isfile(paths[0]) else "outputs/traces",
        "ocs_view.html",
    )
    with open(output_path, "w") as f:
        f.write(html)

    print(f"OCS circuit view → {output_path}")
    print(f"  Ranks with OCS data: {len(ocs_data['ranks'])}")
    if ocs_data["ranks"]:
        total_reuses = sum(r.get("circuit_reuses", 0) for r in ocs_data["ranks"].values())
        total_requests = sum(r.get("total_requests", 0) for r in ocs_data["ranks"].values())
        reuse_pct = total_reuses / max(total_requests, 1) * 100
        print(f"  Aggregate reuse: {total_reuses}/{total_requests} ({reuse_pct:.1f}%)")
    print(f"  OCS events: {len(ocs_data['events'])}")


if __name__ == "__main__":
    main()
