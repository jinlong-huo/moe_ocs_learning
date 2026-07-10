#!/usr/bin/env python3
"""Generate a standalone, interactive HTML timeline viewer from MoE trace files.

Shows:
  - Per-rank event lanes with color-coded events
  - Expert parallelism (EP) mapping: which experts live on each rank
  - Topology layout: pod/node/rank grouping
  - Stats: events, span, comm/compute breakdown, overlap ratio

Produces a single .html file with all trace data embedded -- no HTTP server
needed, just open the file in any browser.

Usage:
  python scripts/trace_viz.py outputs/traces/                    # auto-glob
  python scripts/trace_viz.py outputs/traces/ -o trace_view.html
  python scripts/trace_viz.py rank_00_trace.json rank_01_trace.json ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def load_traces(paths: list[str]) -> tuple[list[dict], str, dict | None]:
    """Load per-rank trace files. Returns (events, mode, metadata)."""
    all_events = []
    metadata = None
    for rank, path in enumerate(sorted(paths)):
        with open(path) as f:
            data = json.load(f)
        for ev in data.get("traceEvents", []):
            ev["pid"] = rank
            ev["_dur"] = int(ev.get("dur", 0))
            ev["_start"] = int(ev["ts"])
            ev["_end"] = int(ev["ts"]) + ev["_dur"]
            all_events.append(ev)
        # Grab metadata from first trace that has it
        if metadata is None and "_metadata" in data:
            metadata = data["_metadata"]

    has_overlap = any("scatter_wait" in e.get("name", "") for e in all_events)
    mode = "overlap" if has_overlap else "serial"
    return all_events, mode, metadata


def build_html(events: list[dict], mode: str, metadata: dict | None) -> str:
    """Generate a self-contained HTML page with embedded trace data and EP info."""
    events_json = json.dumps(events, separators=(",", ":"))
    meta_json = json.dumps(metadata, separators=(",", ":")) if metadata else "null"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoE Trace Viewer -- {mode.upper()}</title>
<style>
  :root {{
    --bg: #1a1a2e; --surface: #16213e; --text: #e0e0e0;
    --text2: #8890a0; --border: #2a2a4a; --accent: #4a6cf7;
    --pod0: #1a3a2e; --pod1: #1a2a3e;
    --node0: #2a4a30; --node1: #2a3a4a; --node2: #3a2a4a; --node3: #4a3a2a;
  }}
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
    background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100vh;
  }}

  /* -- header -- */
  .header{{
    background:var(--surface);border-bottom:1px solid var(--border);
    padding:8px 16px;display:flex;align-items:center;justify-content:space-between;
    flex-wrap:wrap;gap:8px;
  }}
  .header h1{{font-size:1.05rem;font-weight:600;}}
  .header .controls{{display:flex;gap:6px;align-items:center;flex-wrap:wrap;}}
  .header button{{
    background:#2a2a4a;color:var(--text);border:1px solid var(--border);
    padding:5px 12px;border-radius:6px;cursor:pointer;font-size:0.78rem;
  }}
  .header button:hover{{background:#3a3a5a;}}
  .header button.active{{background:var(--accent);}}
  .mode-badge{{
    display:inline-flex;padding:3px 12px;border-radius:12px;
    font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;
  }}
  .mode-serial{{background:#ff980020;color:#ff9800;}}
  .mode-overlap{{background:#4caf5020;color:#4caf50;}}

  /* -- stats bar -- */
  .stats{{
    display:flex;gap:16px;padding:7px 16px;background:var(--surface);
    border-bottom:1px solid var(--border);flex-wrap:wrap;font-size:0.78rem;
  }}
  .stat{{display:flex;flex-direction:column;align-items:center;}}
  .stat .val{{font-size:1.0rem;font-weight:600;}}
  .stat .lbl{{color:var(--text2);font-size:0.65rem;text-transform:uppercase;letter-spacing:0.03em;}}

  /* -- ep panel (collapsible) -- */
  .ep-panel{{
    background:var(--surface);border-bottom:1px solid var(--border);
    padding:10px 16px;display:none;font-size:0.75rem;
  }}
  .ep-panel.open{{display:block;}}
  .ep-panel h3{{font-size:0.85rem;margin-bottom:6px;color:var(--accent);}}
  .ep-grid{{
    display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
    gap:8px;
  }}
  .ep-card{{
    background:#1a1a30;border:1px solid var(--border);border-radius:8px;
    padding:8px 10px;
  }}
  .ep-card .rn{{font-weight:600;color:var(--accent);margin-bottom:3px;}}
  .ep-card .exps{{font-family:monospace;font-size:0.72rem;color:var(--text2);}}
  .ep-card .loc{{font-size:0.68rem;color:var(--text2);margin-top:2px;}}
  .ep-summary{{
    margin-bottom:8px;color:var(--text2);line-height:1.5;
  }}
  .ep-summary strong{{color:var(--text);}}

  /* -- canvas -- */
  .canvas-wrap{{flex:1;overflow:auto;position:relative;}}
  canvas{{display:block;}}

  /* -- legend -- */
  .legend{{
    display:flex;gap:12px;padding:6px 16px;background:var(--surface);
    border-top:1px solid var(--border);flex-wrap:wrap;font-size:0.73rem;
  }}
  .legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;}}

  /* -- tooltip -- */
  .tip{{
    position:fixed;pointer-events:none;background:#0d0d1a;border:1px solid #4a4a6a;
    border-radius:8px;padding:10px 14px;font-size:0.74rem;z-index:100;
    display:none;min-width:180px;box-shadow:0 4px 16px rgba(0,0,0,0.5);
  }}
  .tip .n{{font-weight:600;margin-bottom:4px;}}
  .tip .r{{display:flex;justify-content:space-between;gap:12px;}}
  .tip .r span:first-child{{color:var(--text2);}}
  .tip .ep{{font-size:0.68rem;color:var(--text2);margin-top:2px;}}
</style>
</head>
<body>

<div class="header">
  <h1>MoE Trace Viewer <span class="mode-badge mode-{mode}">{mode}</span></h1>
  <div class="controls">
    <button id="btnEP" onclick="toggleEP()">EP Layout</button>
    <button onclick="zoomFit()">Fit</button>
    <button onclick="zoomIn()">+</button>
    <button onclick="zoomOut()">-</button>
    <span style="font-size:0.7rem;color:var(--text2)">Scroll zoom · Drag pan · Hover details · F=fit</span>
  </div>
</div>

<div class="ep-panel" id="epPanel"></div>
<div class="stats" id="stats"></div>
<div class="canvas-wrap" id="wrap"><canvas id="c"></canvas></div>
<div class="legend" id="legend"></div>
<div class="tip" id="tip"></div>

<script>
// == Embedded data ======================================================
const DATA = {events_json};
const META = {meta_json};

const COLORS = {{
  route:'#4CAF50',scatter:'#42A5F5',scatter_wait:'#00BCD4',
  compute:'#FF9800',gather:'#AB47BC',combine:'#E91E63',
  all_to_all:'#EF5350',barrier:'#78909C',other:'#546E7A',
}};
const LANE_H=24, LANE_GAP=3, HEADER_H=36, PAD_L=180, PAD_R=30, PAD_T=12, PAD_B=12, MIN_BW=3;

// == Process ===========================================================
const events = DATA;
const byRank = new Map();
let minTs=Infinity, maxTs=-Infinity;
for (const e of events) {{
  if (e._start < minTs) minTs = e._start;
  if (e._end > maxTs) maxTs = e._end;
  const pid = e.pid;
  if (!byRank.has(pid)) byRank.set(pid, []);
  byRank.get(pid).push(e);
}}
for (const evts of byRank.values()) evts.sort((a,b)=>a._start-b._start);
const ranks = [...byRank.keys()].sort((a,b)=>a-b);

// == EP Metadata =======================================================
function buildEPInfo() {{
  if (!META) return null;
  const w = META.world_size;
  const nExp = META.num_experts;
  const epr = META.experts_per_rank;
  const topo = META.topology;

  // Build rank -> expert range mapping
  const rankExperts = [];
  for (let r=0; r<w; r++) {{
    const start = r * epr;
    const end = start + epr - 1;
    rankExperts.push({{rank:r, start, end, experts: Array.from({{length:epr}}, (_,i)=>start+i)}});
  }}

  // Build topology groupings
  let pods = [];
  if (topo) {{
    const numPods = topo.num_pods;
    const nodesPerPod = topo.nodes_per_pod;
    const ranksPerNode = topo.ranks_per_node;
    const ranksPerPod = nodesPerPod * ranksPerNode;

    for (let p=0; p<numPods; p++) {{
      const podRanks = [];
      for (let n=0; n<nodesPerPod; n++) {{
        const nodeRanks = [];
        const baseRank = p * ranksPerPod + n * ranksPerNode;
        for (let lr=0; lr<ranksPerNode; lr++) {{
          nodeRanks.push(baseRank + lr);
        }}
        podRanks.push({{nodeIdx:n, ranks:nodeRanks}});
      }}
      pods.push({{podIdx:p, nodes:podRanks}});
    }}
  }}

  return {{w, nExp, epr, topK: META.top_k, routing: META.routing_strategy, rankExperts, pods}};
}}

const epInfo = buildEPInfo();

// == EP Panel ==========================================================
function renderEPPanel() {{
  const panel = document.getElementById('epPanel');
  if (!epInfo) {{ panel.innerHTML = '<em>No EP metadata in trace files (re-run with latest code)</em>'; return; }}

  let html = '<h3>Expert Parallelism Layout</h3>';
  html += `<div class="ep-summary">
    <strong>${{epInfo.w}}</strong> ranks &times;
    <strong>${{epInfo.epr}}</strong> experts/rank =
    <strong>${{epInfo.nExp}}</strong> total experts &nbsp;|&nbsp;
    Top-<strong>${{epInfo.topK}}</strong> gating &nbsp;|&nbsp;
    Routing: <strong>${{epInfo.routing}}</strong>
    ${{META.qwen ? `&nbsp;|&nbsp;Model: <strong>Qwen3.6-35B-A3B</strong>` : ''}}
  </div>`;
  if (META.qwen) {{
    html += `<div class="ep-summary" style="color:#4a6cf7;font-size:0.73rem">
      Qwen Export: dim=${{META.qwen.hidden_dim}} intermediate=${{META.qwen.intermediate_dim}}
      experts_exported=${{META.qwen.experts_exported}} top_k=${{META.qwen.top_k}}
    </div>`;
  }}

  if (epInfo.pods && epInfo.pods.length > 1) {{
    // Topology view: grouped by pod > node
    const podColors = ['#1a4a2e','#1a2a4a'];
    for (const pod of epInfo.pods) {{
      html += `<div style="margin-bottom:10px;padding:8px;background:${{podColors[pod.podIdx]||'#1a1a30'}};border-radius:8px;border:1px solid var(--border)">`;
      html += `<div style="font-weight:700;color:var(--accent);margin-bottom:6px">Pod ${{pod.podIdx}}</div>`;
      html += '<div style="display:flex;gap:8px;flex-wrap:wrap">';
      for (const node of pod.nodes) {{
        html += `<div style="flex:1;min-width:200px;padding:6px 8px;background:#1a1a3080;border-radius:6px">`;
        html += `<div style="font-size:0.7rem;color:var(--text2);margin-bottom:3px">Node ${{node.nodeIdx}}</div>`;
        for (const r of node.ranks) {{
          const re = epInfo.rankExperts[r];
          html += `<div style="font-size:0.72rem;font-family:monospace;padding:2px 0">
            <span style="color:var(--accent);width:50px;display:inline-block">Rank ${{r}}</span>
            <span style="color:var(--text2)">experts ${{re.start}}-${{re.end}}</span>
          </div>`;
        }}
        html += '</div>';
      }}
      html += '</div></div>';
    }}
  }} else {{
    // Flat view: one card per rank
    html += '<div class="ep-grid">';
    for (const re of epInfo.rankExperts) {{
      html += `<div class="ep-card">
        <div class="rn">Rank ${{re.rank}}</div>
        <div class="exps">experts ${{re.start}}-${{re.end}} [${{re.experts.join(', ')}}]</div>
      </div>`;
    }}
    html += '</div>';
  }}

  panel.innerHTML = html;
}}

function toggleEP() {{
  const panel = document.getElementById('epPanel');
  const btn = document.getElementById('btnEP');
  panel.classList.toggle('open');
  btn.classList.toggle('active');
  if (panel.classList.contains('open')) {{
    renderEPPanel();
    setTimeout(render, 50);  // resize canvas
  }} else {{
    setTimeout(render, 50);
  }}
}}

// == Helpers ===========================================================
function color(name) {{
  for (const [k,c] of Object.entries(COLORS)) if (name.includes(k)) return c;
  return COLORS.other;
}}
function label(name) {{
  if (name.includes('combine')) return 'Combine';
  if (name.includes('route')) return 'Route';
  if (name.includes('scatter_wait')) return 'Scatter Wait';
  if (name.includes('scatter')) return 'Scatter';
  if (name.includes('compute')) return 'Compute';
  if (name.includes('gather')) return 'Gather';
  if (name.includes('all_to_all')) return 'All-to-All';
  if (name.includes('barrier')) return 'Barrier';
  return 'Other';
}}
function fmtUs(u) {{ return u>=1e6 ? (u/1e6).toFixed(2)+'s' : u>=1e3 ? (u/1e3).toFixed(2)+'ms' : u.toFixed(1)+'us'; }}
function pct(v) {{ return (v*100).toFixed(1)+'%'; }}

// == DOM ===============================================================
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
const wrap = document.getElementById('wrap');

let zoom = null;  // [startUs, endUs] or null=auto

// Compute pod/node grouping info for lane background coloring
function getGroupInfo() {{
  if (!epInfo || !epInfo.pods) return null;
  const rankToPod = {{}};
  const rankToNode = {{}};
  for (const pod of epInfo.pods) {{
    for (const node of pod.nodes) {{
      for (const r of node.ranks) {{
        rankToPod[r] = pod.podIdx;
        rankToNode[r] = node.nodeIdx;
      }}
    }}
  }}
  const podColors = ['rgba(26,74,46,', 'rgba(26,42,74,'];
  return {{rankToPod, rankToNode, podColors}};
}}

// == Stats =============================================================
function updateStats() {{
  let comm=0, comp=0, route=0, combine=0;
  let commEvts=[], compEvts=[];
  for (const e of events) {{
    if (e.name.includes('all_to_all')||e.name.includes('barrier')||e.name.includes('scatter_wait')||e.name.includes('gather_wait'))
      {{ comm+=e._dur; commEvts.push(e); }}
    else if (e.name.includes('compute')) {{ comp+=e._dur; compEvts.push(e); }}
    else if (e.name.includes('route')) route+=e._dur;
    else if (e.name.includes('combine')) combine+=e._dur;
  }}
  const total = maxTs-minTs;
  let overlap=0;
  for (const ce of commEvts) {{
    for (const xe of compEvts) {{
      const oS=Math.max(ce._start, xe._start), oE=Math.min(ce._end, xe._end);
      if (oS<oE) overlap += oE-oS;
    }}
  }}
  const ovr = comm>0 ? overlap/comm*100 : 0;

  let statsHTML =
    `<div class="stat"><span class="val">${{ranks.length}}</span><span class="lbl">Ranks</span></div>
     <div class="stat"><span class="val">${{events.length}}</span><span class="lbl">Events</span></div>
     <div class="stat"><span class="val">${{fmtUs(total)}}</span><span class="lbl">Span</span></div>
     <div class="stat"><span class="val">${{fmtUs(comm)}}</span><span class="lbl">Comm</span></div>
     <div class="stat"><span class="val">${{fmtUs(comp)}}</span><span class="lbl">Compute</span></div>
     <div class="stat"><span class="val">${{pct(comm/(comm+comp+route+combine||1))}}</span><span class="lbl">Comm %</span></div>
     <div class="stat"><span class="val">${{ovr.toFixed(1)}}%</span><span class="lbl">Overlap</span></div>`;

  if (epInfo) {{
    statsHTML +=
      `<div class="stat"><span class="val">${{epInfo.nExp}}</span><span class="lbl">Experts</span></div>
       <div class="stat"><span class="val">${{epInfo.epr}}</span><span class="lbl">Exp/Rank</span></div>
       <div class="stat"><span class="val">Top-${{epInfo.topK}}</span><span class="lbl">Gating</span></div>`;
  }}

  document.getElementById('stats').innerHTML = statsHTML;
  document.getElementById('legend').innerHTML = Object.entries(COLORS).map(([k,c]) =>
    `<span><i style="background:${{c}}"></i>${{label(k)}}</span>`).join('');
}}

// == Render ============================================================
function render() {{
  const groupInfo = getGroupInfo();
  const epPanelOpen = document.getElementById('epPanel').classList.contains('open');
  const epPanelH = epPanelOpen ? document.getElementById('epPanel').offsetHeight + 2 : 0;

  const numRanks = ranks.length;
  const totalH = HEADER_H + PAD_T + numRanks*(LANE_H+LANE_GAP) + PAD_B;
  const plotW = wrap.clientWidth - PAD_L - PAD_R;
  const dpr = window.devicePixelRatio||1;
  const cssW = wrap.clientWidth;

  cv.style.width = cssW+'px'; cv.style.height = totalH+'px';
  cv.width = cssW*dpr; cv.height = totalH*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);

  const dom = zoom||[minTs,maxTs];
  const span = dom[1]-dom[0]||1;
  const tx = t => PAD_L + ((t-dom[0])/span)*plotW;

  // bg
  ctx.fillStyle='#1a1a2e'; ctx.fillRect(0,0,cssW,totalH);

  // axis area
  ctx.fillStyle='#ffffff06'; ctx.fillRect(0,0,cssW,HEADER_H);

  // grid & ticks
  let tickUs;
  if (span>1e7) tickUs=5e5;
  else if (span>2e6) tickUs=1e5;
  else if (span>5e5) tickUs=50000;
  else if (span>2e5) tickUs=20000;
  else if (span>1e5) tickUs=10000;
  else if (span>50000) tickUs=5000;
  else if (span>10000) tickUs=2000;
  else tickUs=1000;

  ctx.strokeStyle='#ffffff10'; ctx.lineWidth=0.5;
  ctx.fillStyle='#8890a0'; ctx.font='9px system-ui'; ctx.textAlign='center';
  let first = Math.floor(dom[0]/tickUs)*tickUs;
  for (let t=first; t<=dom[1]; t+=tickUs) {{
    const x=tx(t);
    if (x<PAD_L||x>PAD_L+plotW) continue;
    ctx.beginPath(); ctx.moveTo(x,HEADER_H); ctx.lineTo(x,totalH); ctx.stroke();
    ctx.fillText((t/1e6).toFixed(1)+'s', x, HEADER_H-4);
  }}

  // rank labels + lane backgrounds
  ctx.font='11px system-ui'; ctx.textAlign='right';
  for (let i=0;i<numRanks;i++) {{
    const r=ranks[i];
    const y=HEADER_H+PAD_T+i*(LANE_H+LANE_GAP);

    // Lane background with pod/node grouping colors
    let bgColor = '#16213e';
    if (groupInfo && groupInfo.podColors) {{
      const pod = groupInfo.rankToPod[r];
      const alpha = 0.3 + (i%2)*0.05;
      bgColor = (groupInfo.podColors[pod]||'rgba(22,33,62,') + alpha + ')';
    }}
    ctx.fillStyle=bgColor;
    ctx.fillRect(PAD_L,y,plotW,LANE_H);

    // Pod separator lines
    if (groupInfo && i>0) {{
      const prevPod = groupInfo.rankToPod[ranks[i-1]];
      const curPod = groupInfo.rankToPod[r];
      if (prevPod !== curPod) {{
        ctx.strokeStyle='#ffffff20'; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.moveTo(0,y-1); ctx.lineTo(cssW,y-1); ctx.stroke();
      }}
    }}

    // Rank label with expert info
    let rankLabel = 'Rank '+r;
    if (epInfo) {{
      const re = epInfo.rankExperts[r];
      rankLabel += ' [e'+re.start+'-'+re.end+']';
    }}
    ctx.fillStyle='#c0c8d0'; ctx.fillText(rankLabel, PAD_L-8, y+LANE_H-6);

    // Node label (small, above rank)
    if (groupInfo) {{
      ctx.fillStyle='#667080'; ctx.font='8px system-ui';
      const nodeLabel = 'N'+groupInfo.rankToNode[r];
      ctx.fillText(nodeLabel, PAD_L-8, y+8);
      ctx.font='11px system-ui';
    }}

    // Events on this lane
    for (const e of byRank.get(r)||[]) {{
      const x1=tx(e._start), x2=tx(e._end);
      const w=Math.max(MIN_BW,x2-x1);
      ctx.fillStyle=color(e.name); ctx.fillRect(x1,y+3,w,LANE_H-6);
    }}
  }}

  cv._s = {{ranks,dom,span,plotW,tx,totalH}};
}}

// == Hover =============================================================
cv.addEventListener('mousemove',e=>{{
  const s=cv._s; if(!s) return;
  const r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  const idx=Math.floor((my-HEADER_H-PAD_T)/(LANE_H+LANE_GAP));
  if (idx<0||idx>=s.ranks.length) {{tip.style.display='none';return;}}
  const rank=s.ranks[idx];
  const evts=byRank.get(rank)||[];
  let hit=null;
  for (const ev of evts) {{
    const x1=s.tx(ev._start), x2=s.tx(ev._end);
    if (mx>=x1&&mx<=x2){{hit=ev;break;}}
  }}
  if (!hit) {{tip.style.display='none';return;}}
  const c=color(hit.name);
  let epExtra = '';
  if (epInfo) {{
    const re = epInfo.rankExperts[hit.pid];
    epExtra = `<div class="ep">Rank ${{hit.pid}}: experts ${{re.start}}-${{re.end}}</div>`;
  }}
  tip.innerHTML=`<div class="n" style="color:${{c}}">${{hit.name}}</div>
    <div class="r"><span>Duration</span><span>${{fmtUs(hit._dur)}}</span></div>
    <div class="r"><span>Category</span><span>${{label(hit.name)}}</span></div>
    ${{epExtra}}`;
  tip.style.display='block';
  tip.style.left=(e.clientX+16)+'px'; tip.style.top=(e.clientY-10)+'px';
}});
cv.addEventListener('mouseleave',()=>{{tip.style.display='none'}});

// == Zoom & Pan ========================================================
const zoomFit = ()=>{{zoom=null;render();}};
const zoomIn = ()=>{{
  const d=zoom||[minTs,maxTs]; const sp=d[1]-d[0]; const m=(d[0]+d[1])/2;
  zoom=[m-sp*0.3,m+sp*0.3]; render();
}};
const zoomOut = ()=>{{
  if(!zoom) return;
  const sp=zoom[1]-zoom[0], m=(zoom[0]+zoom[1])/2, ns=sp*1.5;
  const s=m-ns/2, e=m+ns/2;
  zoom=(s<=minTs&&e>=maxTs)?null:[Math.max(minTs,s),Math.min(maxTs,e)];
  render();
}};

let drag=false, ds=0, dd=null;
cv.addEventListener('mousedown',e=>{{drag=true;ds=e.clientX;dd=zoom?[...zoom]:[minTs,maxTs];cv.style.cursor='grabbing';}});
window.addEventListener('mouseup',()=>{{drag=false;cv.style.cursor='default';}});
window.addEventListener('mousemove',e=>{{
  if(!drag||!dd||!cv._s) return;
  const px=(e.clientX-ds), uspp=(dd[1]-dd[0])/cv._s.plotW;
  const shift=-px*uspp;
  zoom=[Math.max(minTs,dd[0]+shift),Math.min(maxTs,dd[1]+shift)];
  render();
}});

wrap.addEventListener('wheel',e=>{{
  e.preventDefault();
  const s=cv._s; if(!s) return;
  const r=cv.getBoundingClientRect();
  const mx=e.clientX-r.left;
  const dom=zoom||[minTs,maxTs], ds=dom[1]-dom[0];
  const tAt = dom[0]+((mx-PAD_L)/s.plotW)*ds;
  const f=e.deltaY>0?1.3:0.7;
  let ns=ds*f; ns=Math.max(500,Math.min(maxTs-minTs,ns));
  const ratio=(tAt-dom[0])/ds;
  let a=tAt-ratio*ns, b=a+ns;
  if (a<minTs){{a=minTs;b=a+ns;}}
  if (b>maxTs){{b=maxTs;a=b-ns;}}
  zoom=(a<=minTs&&b>=maxTs)?null:[a,b];
  render();
}},{{passive:false}});

// == Keyboard ==========================================================
window.addEventListener('keydown',e=>{{if(e.key.toLowerCase()==='f')zoomFit();}});
window.addEventListener('resize',render);

// == Init ==============================================================
updateStats();
render();
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate standalone MoE trace HTML viewer")
    parser.add_argument("inputs", nargs="*", help="Trace JSON files or a directory of rank_*_trace.json")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    paths = []
    for inp in args.inputs or []:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "rank_*_trace.json"))))
        else:
            paths.append(inp)

    if not paths:
        default_dir = "outputs/traces"
        paths = sorted(glob.glob(os.path.join(default_dir, "rank_*_trace.json")))
        if not paths:
            print("ERROR: No trace files found.", file=sys.stderr)
            sys.exit(1)

    events, mode, metadata = load_traces(paths)
    html = build_html(events, mode, metadata)

    if args.output:
        out = args.output
    else:
        trace_dir = os.path.dirname(os.path.abspath(paths[0]))
        out = os.path.join(trace_dir, "trace_viewer.html")

    with open(out, "w") as f:
        f.write(html)

    num_ranks = len(set(e["pid"] for e in events))
    topo_str = ""
    if metadata and metadata.get("topology"):
        t = metadata["topology"]
        topo_str = f", {t['num_pods']}p×{t['nodes_per_pod']}n×{t['ranks_per_node']}r topology"
    print(f"Trace viewer written -> {out}")
    print(f"  {len(events)} events, {num_ranks} ranks, mode={mode}{topo_str}")
    print(f"  Open this file in any browser (no server needed)")


if __name__ == "__main__":
    main()
