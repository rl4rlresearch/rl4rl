#!/usr/bin/env python3
# ruff: noqa: E501
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
DEFAULT_AUTORESEARCH_V17 = (
    REPO_ROOT / "data/c0c3/transformer-optimization-v1-7-source-only-campaign"
)
DEFAULT_OPENEVOLVE_V21 = (
    REPO_ROOT / "data/c0c3/controlled-openevolve-transformer-v2-1-mps-campaign"
)
DEFAULT_OPENEVOLVE_V21_NANOGPT = (
    REPO_ROOT / "data/c0c3/nanogpt-openevolve-v2-1-h100-campaign"
)

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


def metric_at_seed(state: dict[str, Any], objective_metric: str) -> float | None:
    candidates = state.get("candidates", {})
    if not isinstance(candidates, dict):
        return None
    for candidate in candidates.values():
        if not isinstance(candidate, dict) or candidate.get("created_opportunity") != 0:
            continue
        metrics = candidate.get("metrics", {})
        value = metrics.get(objective_metric) if isinstance(metrics, dict) else None
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
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


def build_run(
    run_dir: Path,
    prices: dict[str, float],
    *,
    objective_metric: str,
    objective_direction: str,
) -> dict[str, Any] | None:
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
    seed_objective = metric_at_seed(state, objective_metric)
    best_objective = seed_objective
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
        objective = metrics.get(objective_metric) if isinstance(metrics, dict) else None
        if (
            isinstance(evaluation, dict)
            and evaluation.get("valid")
            and isinstance(objective, int | float)
            and not isinstance(objective, bool)
        ):
            value = float(objective)
            if best_objective is None:
                best_objective = value
            elif objective_direction == "maximize":
                best_objective = max(best_objective, value)
            else:
                best_objective = min(best_objective, value)
        usage = event.get("usage_cumulative", {})
        usage = usage if isinstance(usage, dict) else {}
        if best_objective is not None:
            points.append(
                {
                    "proposal": opportunity,
                    "active_hours": round(elapsed_seconds / 3600, 6),
                    "token_cost": round(weighted_cost(usage, prices), 6),
                    "objective": best_objective,
                }
            )
    if seed_objective is not None:
        points.insert(
            0,
            {"proposal": 0, "active_hours": 0.0, "token_cost": 0.0, "objective": seed_objective},
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
        "best_objective": best_objective,
        "lowest_parameters": (
            best_objective if objective_metric == "parameters" else None
        ),
        "points": points,
    }


def campaign_data(campaign: Path, prices: dict[str, float]) -> dict[str, Any]:
    runs_root = campaign / "runs"
    task = read_json(campaign / "inputs/task.json", {})
    task = task if isinstance(task, dict) else {}
    objective_metric = str(task.get("objective_metric", "parameters"))
    objective_direction = str(task.get("objective_direction", "minimize"))
    runs = [
        build_run(
            path,
            prices,
            objective_metric=objective_metric,
            objective_direction=objective_direction,
        )
        for path in sorted(runs_root.glob("*"))
        if path.is_dir()
    ]
    factorial_runs = [
        run
        for run in runs
        if run is not None and run["condition"] in {"C0", "C1", "C2", "C3"}
    ]
    return {
        "campaign": str(campaign),
        "available": campaign.is_dir(),
        "objective_metric": objective_metric,
        "objective_direction": objective_direction,
        "runs": factorial_runs,
    }


def dashboard_data(campaigns: dict[str, Path], prices: dict[str, float]) -> dict[str, Any]:
    campaign_payloads = {
        key: campaign_data(path, prices) for key, path in campaigns.items()
    }
    payload = {
        "schema_version": "2.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "price_per_million": prices,
        "campaigns": campaign_payloads,
    }
    # Keep tabs loaded before the multi-campaign dashboard upgrade functional.
    # Those clients refresh in place and still read these two top-level keys.
    if "autoresearch_v16" in campaign_payloads:
        payload["autoresearch"] = campaign_payloads["autoresearch_v16"]
    if "openevolve_v2" in campaign_payloads:
        payload["openevolve_v2"] = campaign_payloads["openevolve_v2"]
    return payload


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
const conditionColors = {C0:['#93c5fd','#60a5fa','#3b82f6','#2563eb','#1d4ed8'],C1:['#fed7aa','#fdba74','#fb923c','#f97316','#c2410c'],C2:['#e9d5ff','#d8b4fe','#c084fc','#a855f7','#7e22ce'],C3:['#bbf7d0','#86efac','#4ade80','#22c55e','#15803d']};
const charts=[];
function color(run) { const family=conditionColors[run.condition]||['#ddd']; const match=run.label.match(/^B(\d+)/); return family[match ? (Number(match[1])-1)%family.length : 0]; }
function chartOptions(xTitle,yTitle) { return {responsive:true, maintainAspectRatio:false, parsing:false, plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${Number(c.raw.y).toLocaleString()} ${yTitle}`}}}, scales:{x:{type:'linear',min:0,title:{display:true,text:xTitle}},y:{min:0,title:{display:true,text:`Best valid ${yTitle}`}}}}; }
function makeChart(canvas, runs, xKey, title, objectiveMetric) { return new Chart(canvas,{type:'line',data:{datasets:runs.filter(r=>r.points.length).map(r=>({label:r.label,data:r.points.map(p=>({x:p[xKey],y:p.objective})),borderColor:color(r),backgroundColor:color(r),borderWidth:2,pointRadius:2,tension:.12}))},options:chartOptions(title,objectiveMetric)}); }
function section(key, title, payload) {
  if (!payload || !payload.available) { const campaign=payload?.campaign||'Not configured'; return `<section class="section"><h2>${title}</h2><p class="sub">Campaign directory is not available: ${campaign}</p></section>`; }
  const id=key.replace(/[^a-z0-9]/g,''); const legend=payload.runs.map(r=>`<span class="key"><i class="dot" style="background:${color(r)}"></i>${r.label}</span>`).join('');
  const rows=payload.runs.map(r=>`<tr><td>${r.label}</td><td>${r.condition}</td><td>${r.status}</td><td>${r.proposals_used}</td><td>${r.total_tokens.toLocaleString()}</td><td>${r.best_objective===null?'—':Number(r.best_objective).toLocaleString()}</td></tr>`).join('');
  return `<section class="section"><h2>${title}</h2><p class="sub">${payload.campaign}</p><div class="legend">${legend}</div><div class="charts"><div class="chart"><canvas id="${id}-proposal"></canvas></div><div class="chart"><canvas id="${id}-cost"></canvas></div><div class="chart"><canvas id="${id}-time"></canvas></div></div><table><thead><tr><th>Run</th><th>Condition</th><th>Status</th><th>Proposals</th><th>Reported tokens</th><th>Best ${payload.objective_metric}</th></tr></thead><tbody>${rows}</tbody></table></section>`;
}
async function refresh() {
  document.getElementById('refresh').disabled=true;
  try {
    const data=await fetch('/api/data',{cache:'no-store'}).then(r=>r.json());
    charts.splice(0).forEach(c=>c.destroy());
    document.getElementById('stamp').textContent='Updated '+new Date(data.generated_at).toLocaleString();
    const campaigns=data.campaigns||{};
    const sections=[['autoresearchv16','Autoresearch v1.6',campaigns.autoresearch_v16||data.autoresearch],['openevolvev2','OpenEvolve v2.0',campaigns.openevolve_v2||data.openevolve_v2],['autoresearchv17','Autoresearch v1.7 · addition',campaigns.autoresearch_v17],['openevolvev21','OpenEvolve v2.1 · addition',campaigns.openevolve_v21],['openevolvev21nanogpt','OpenEvolve v2.1 · nanoGPT H100',campaigns.openevolve_v21_nanogpt]];
    document.getElementById('content').innerHTML=sections.map(([id,title,p])=>section(id,title,p)).join('');
    sections.forEach(([id,_title,p])=>{ if(!p||!p.available)return; charts.push(makeChart(document.getElementById(id+'-proposal'),p.runs,'proposal','Proposal',p.objective_metric)); charts.push(makeChart(document.getElementById(id+'-cost'),p.runs,'token_cost','Price-weighted token cost (USD)',p.objective_metric)); charts.push(makeChart(document.getElementById(id+'-time'),p.runs,'active_hours','Active wall-clock time (hours)',p.objective_metric)); });
  } catch (error) { document.getElementById('stamp').textContent='Refresh failed: '+error.message; }
  finally { document.getElementById('refresh').disabled=false; }
}
document.getElementById('refresh').addEventListener('click',refresh); refresh(); setInterval(refresh,30000);
</script>'''


def make_handler(campaigns: dict[str, Path], prices: dict[str, float]):
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
                payload = json.dumps(dashboard_data(campaigns, prices)).encode("utf-8")
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
    parser.add_argument(
        "--autoresearch-v17-campaign", type=Path, default=DEFAULT_AUTORESEARCH_V17
    )
    parser.add_argument(
        "--openevolve-v21-campaign", type=Path, default=DEFAULT_OPENEVOLVE_V21
    )
    parser.add_argument(
        "--openevolve-v21-nanogpt-campaign",
        type=Path,
        default=DEFAULT_OPENEVOLVE_V21_NANOGPT,
    )
    parser.add_argument("--input-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["input"])
    parser.add_argument("--cached-input-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["cached_input"])
    parser.add_argument("--output-per-million", type=float, default=DEFAULT_PRICE_PER_MILLION["output"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = {"input": args.input_per_million, "cached_input": args.cached_input_per_million, "output": args.output_per_million}
    campaigns = {
        "autoresearch_v16": args.autoresearch_campaign,
        "openevolve_v2": args.openevolve_campaign,
        "autoresearch_v17": args.autoresearch_v17_campaign,
        "openevolve_v21": args.openevolve_v21_campaign,
        "openevolve_v21_nanogpt": args.openevolve_v21_nanogpt_campaign,
    }
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(campaigns, prices)
    )
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Autoresearch: {args.autoresearch_campaign}")
    print(f"OpenEvolve v2: {args.openevolve_campaign}")
    print(f"Autoresearch v1.7: {args.autoresearch_v17_campaign}")
    print(f"OpenEvolve v2.1: {args.openevolve_v21_campaign}")
    print(f"OpenEvolve v2.1 nanoGPT: {args.openevolve_v21_nanogpt_campaign}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
