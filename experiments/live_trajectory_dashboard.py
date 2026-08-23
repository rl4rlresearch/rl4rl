#!/usr/bin/env python3
"""Read-only, locally served trajectory dashboard for live C0-C3 campaigns.

The server reads campaign logs only when a browser requests ``/api/data``.  It
does not import the experiment controller, acquire a campaign lock, or write to
any campaign artifact, so it is safe to use while controllers are running.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTORESEARCH = Path(
    "/private/tmp/rl4rl-v16-codex1644-confined-campaign-fresh-20260822c"
)
DEFAULT_OPENEVOLVE = REPO_ROOT / "data/c0c3/controlled-openevolve-transformer-v2-mps-campaign"

# These are display-only price weights, not a billing record.  They match the
# previous trajectory visualizer's token-cost convention and can be overridden.
DEFAULT_PRICE_PER_MILLION = {"input": 1.75, "cached_input": 0.175, "output": 14.0}


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parameter_at_seed(state: dict[str, Any]) -> int | None:
    candidates = state.get("candidates", {})
    if not isinstance(candidates, dict):
        return None
    for candidate in candidates.values():
        if not isinstance(candidate, dict) or candidate.get("created_opportunity") != 0:
            continue
        metrics = candidate.get("metrics", {})
        if isinstance(metrics, dict) and isinstance(metrics.get("parameters"), int):
            return metrics["parameters"]
    return None


def weighted_cost(usage: dict[str, Any], prices: dict[str, float]) -> float:
    input_tokens = float(usage.get("input_tokens", 0) or 0)
    cached_tokens = float(usage.get("cached_input_tokens", 0) or 0)
    output_tokens = float(usage.get("output_tokens", 0) or 0)
    uncached_tokens = max(0.0, input_tokens - cached_tokens)
    return (
        uncached_tokens * prices["input"]
        + cached_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000


def compact_label(run_id: str) -> str:
    parts = run_id.lower().split("-")
    block = next((part.upper() for part in parts if len(part) == 3 and part.startswith("b") and part[1:].isdigit()), "B??")
    condition = next((part.upper() for part in parts if part in {"c0", "c1", "c2", "c3"}), "C?")
    return f"{block}-{condition}"


def build_run(run_dir: Path, prices: dict[str, float]) -> dict[str, Any] | None:
    state = read_json(run_dir / "state.json", {})
    if not isinstance(state, dict):
        return None
    run_id = state.get("run_id")
    condition = state.get("condition")
    if not isinstance(run_id, str) or not isinstance(condition, str):
        return None
    events = iter_jsonl(run_dir / "events.jsonl")
    started: dict[int, datetime] = {}
    elapsed_seconds = 0.0
    seed_parameters = parameter_at_seed(state)
    best_parameters = seed_parameters
    points: list[dict[str, float | int]] = []
    for event in events:
        opportunity = event.get("opportunity")
        if not isinstance(opportunity, int):
            continue
        timestamp = parse_timestamp(event.get("timestamp"))
        if event.get("event") == "proposal_started" and timestamp is not None:
            started[opportunity] = timestamp
            continue
        if event.get("event") != "proposal_completed":
            continue
        if timestamp is not None and opportunity in started:
            elapsed_seconds += max(0.0, (timestamp - started[opportunity]).total_seconds())
        evaluation = event.get("evaluation", {})
        metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
        parameters = metrics.get("parameters") if isinstance(metrics, dict) else None
        if isinstance(evaluation, dict) and evaluation.get("valid") and isinstance(parameters, int):
            best_parameters = parameters if best_parameters is None else min(best_parameters, parameters)
        usage = event.get("usage_cumulative", {})
        usage = usage if isinstance(usage, dict) else {}
        if best_parameters is not None:
            points.append(
                {
                    "proposal": opportunity,
                    "active_hours": round(elapsed_seconds / 3600, 6),
                    "token_cost": round(weighted_cost(usage, prices), 6),
                    "parameters": best_parameters,
                }
            )
    if seed_parameters is not None:
        points.insert(
            0,
            {"proposal": 0, "active_hours": 0.0, "token_cost": 0.0, "parameters": seed_parameters},
        )
    usage = state.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    return {
        "run_id": run_id,
        "label": compact_label(run_id),
        "condition": condition.upper(),
        "status": state.get("status", "unknown"),
        "proposals_used": state.get("proposals_used", 0),
        "total_tokens": int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
        "lowest_parameters": min((point["parameters"] for point in points), default=None),
        "points": points,
    }


def campaign_data(campaign: Path, prices: dict[str, float]) -> dict[str, Any]:
    runs_root = campaign / "runs"
    runs = [build_run(path, prices) for path in sorted(runs_root.glob("*")) if path.is_dir()]
    factorial_runs = [
        run
        for run in runs
        if run is not None and run["condition"] in {"C0", "C1", "C2", "C3"}
    ]
    return {
        "campaign": str(campaign),
        "available": campaign.is_dir(),
        "runs": factorial_runs,
    }


def dashboard_data(autoresearch: Path, openevolve: Path, prices: dict[str, float]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "price_per_million": prices,
        "autoresearch": campaign_data(autoresearch, prices),
        "openevolve_v2": campaign_data(openevolve, prices),
    }


PAGE = r'''<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RL4RL live trajectories</title>
<style>
  :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; background:#0d1117; color:#e6edf3; }
  body { max-width:1500px; margin:0 auto; padding:24px; } h1,h2 { margin:0 0 7px; } .sub { color:#9da7b5; margin:0 0 16px; }
  button { padding:8px 12px; margin-right:10px; border-radius:6px; border:1px solid #49566b; background:#1a2332; color:inherit; cursor:pointer; }
  .stamp { color:#9da7b5; font-size:13px; } .section { margin-top:30px; border-top:1px solid #30363d; padding-top:20px; }
  .charts { display:grid; grid-template-columns:repeat(3,minmax(300px,1fr)); gap:16px; } .chart { min-height:310px; }
  table { width:100%; border-collapse:collapse; margin-top:15px; font-size:13px; } th,td { border-bottom:1px solid #30363d; padding:7px; text-align:right; } th:first-child,td:first-child { text-align:left; }
  .legend { display:flex; gap:12px; flex-wrap:wrap; margin:10px 0; } .key { display:inline-flex; align-items:center; gap:5px; font-size:12px; } .dot { width:10px; height:10px; border-radius:50%; }
  @media (max-width:980px) { .charts { grid-template-columns:1fr; } } 
</style>
<h1>RL4RL live trajectories</h1>
<p class="sub">Read-only dashboard. Refresh rereads the campaign logs; automatic refresh runs every 30 seconds.</p>
<button id="refresh">Refresh now</button><span class="stamp" id="stamp">Loading…</span>
<div id="content"></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script>
const conditionColors = {C0:['#60a5fa','#2563eb','#1d4ed8'],C1:['#fb923c','#ea580c','#c2410c'],C2:['#c084fc','#9333ea','#7e22ce'],C3:['#4ade80','#16a34a','#15803d']};
const charts=[];
function color(run) { const match=run.label.match(/^B(\d+)/); return (conditionColors[run.condition]||['#ddd'])[match ? (Number(match[1])-1)%3 : 0]; }
function chartOptions(xTitle) { return {responsive:true, maintainAspectRatio:false, parsing:false, plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${Number(c.raw.y).toLocaleString()} parameters`}}}, scales:{x:{type:'linear',min:0,title:{display:true,text:xTitle}},y:{min:0,title:{display:true,text:'Best valid parameters'}}}}; }
function makeChart(canvas, runs, xKey, title) { return new Chart(canvas,{type:'line',data:{datasets:runs.filter(r=>r.points.length).map(r=>({label:r.label,data:r.points.map(p=>({x:p[xKey],y:p.parameters})),borderColor:color(r),backgroundColor:color(r),borderWidth:2,pointRadius:2,tension:.12}))},options:chartOptions(title)}); }
function section(key, title, payload) {
  if (!payload.available) return `<section class="section"><h2>${title}</h2><p class="sub">Campaign directory is not available: ${payload.campaign}</p></section>`;
  const id=key.replace(/[^a-z0-9]/g,''); const legend=payload.runs.map(r=>`<span class="key"><i class="dot" style="background:${color(r)}"></i>${r.label}</span>`).join('');
  const rows=payload.runs.map(r=>`<tr><td>${r.label}</td><td>${r.condition}</td><td>${r.status}</td><td>${r.proposals_used}</td><td>${r.total_tokens.toLocaleString()}</td><td>${r.lowest_parameters===null?'—':r.lowest_parameters.toLocaleString()}</td></tr>`).join('');
  return `<section class="section"><h2>${title}</h2><p class="sub">${payload.campaign}</p><div class="legend">${legend}</div><div class="charts"><div class="chart"><canvas id="${id}-proposal"></canvas></div><div class="chart"><canvas id="${id}-cost"></canvas></div><div class="chart"><canvas id="${id}-time"></canvas></div></div><table><thead><tr><th>Run</th><th>Condition</th><th>Status</th><th>Proposals</th><th>Reported tokens</th><th>Lowest parameters</th></tr></thead><tbody>${rows}</tbody></table></section>`;
}
async function refresh() {
  document.getElementById('refresh').disabled=true;
  try {
    const data=await fetch('/api/data',{cache:'no-store'}).then(r=>r.json());
    charts.splice(0).forEach(c=>c.destroy());
    document.getElementById('stamp').textContent='Updated '+new Date(data.generated_at).toLocaleString();
    document.getElementById('content').innerHTML=section('autoresearch','Autoresearch',data.autoresearch)+section('openevolve','OpenEvolve v2',data.openevolve_v2);
    [['autoresearch',data.autoresearch],['openevolve',data.openevolve_v2]].forEach(([id,p])=>{ if(!p.available)return; charts.push(makeChart(document.getElementById(id+'-proposal'),p.runs,'proposal','Proposal')); charts.push(makeChart(document.getElementById(id+'-cost'),p.runs,'token_cost','Price-weighted token cost (USD)')); charts.push(makeChart(document.getElementById(id+'-time'),p.runs,'active_hours','Active wall-clock time (hours)')); });
  } catch (error) { document.getElementById('stamp').textContent='Refresh failed: '+error.message; }
  finally { document.getElementById('refresh').disabled=false; }
}
document.getElementById('refresh').addEventListener('click',refresh); refresh(); setInterval(refresh,30000);
</script>'''


def make_handler(autoresearch: Path, openevolve: Path, prices: dict[str, float]):
    class Handler(BaseHTTPRequestHandler):
        def send_payload(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self.send_payload(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/data":
                payload = json.dumps(dashboard_data(autoresearch, openevolve, prices)).encode("utf-8")
                self.send_payload(payload, "application/json; charset=utf-8")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"dashboard: {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a read-only live C0-C3 trajectory dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--autoresearch-campaign", type=Path, default=DEFAULT_AUTORESEARCH)
    parser.add_argument("--openevolve-campaign", type=Path, default=DEFAULT_OPENEVOLVE)
    parser.add_argument("--input-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["input"])
    parser.add_argument("--cached-input-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["cached_input"])
    parser.add_argument("--output-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = {"input": args.input_per_million, "cached_input": args.cached_input_per_million, "output": args.output_per_million}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.autoresearch_campaign, args.openevolve_campaign, prices))
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Autoresearch: {args.autoresearch_campaign}")
    print(f"OpenEvolve v2: {args.openevolve_campaign}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
