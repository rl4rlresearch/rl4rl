"use client";

import { useMemo, useState } from "react";
import trajectoryData from "./data/trajectories.json";

type Point = {
  iteration: number;
  candidateId: string;
  architectureHash: string | null;
  parameterCount: number | null;
  bestValidParameterCount: number | null;
  valid: boolean;
  isNewArchitecture: boolean;
  publicAccuracy: number | null;
  architectureName: string;
  mechanismHypothesis: string;
  failureReason: string;
  nodeKinds: Record<string, number>;
};

type Run = {
  id: string;
  method: "AutoResearch" | "OpenEvolve";
  methodCode: "ar" | "oe";
  condition: "RD0" | "RD1" | "RD2" | "RD3";
  conditionLabel: string;
  horizon: 8 | 12 | 20;
  finalBestParameterCount: number | null;
  finalBestPublicAccuracy: number | null;
  validCount: number;
  uniqueArchitectureCount: number;
  improvementCount: number;
  proposalCount: number;
  trajectory: Point[];
};

type ViewMode = "best" | "proposal";

const COLORS = [
  "#4f7cff",
  "#f0844a",
  "#a56bf5",
  "#42b883",
  "#67a6ff",
  "#ff9f43",
  "#c88cf5",
  "#55d88b",
];
const HORIZONS = [8, 12, 20] as const;
const METHODS = ["ar", "oe"] as const;
const CONDITIONS = ["RD0", "RD1", "RD2", "RD3"] as const;
const runs = trajectoryData.runs as Run[];

function colorFor(run: Run) {
  const methodOffset = run.methodCode === "ar" ? 0 : 4;
  return COLORS[methodOffset + Number(run.condition.slice(-1))];
}

function dashFor(run: Run) {
  if (run.horizon === 8) return "5 5";
  if (run.horizon === 20) return "12 5";
  return undefined;
}

function shortLabel(run: Run) {
  return `${run.methodCode.toUpperCase()} · h${run.horizon} · ${run.condition}`;
}

function formatParams(value: number | null) {
  return value === null ? "—" : new Intl.NumberFormat("en-US").format(value);
}

function formatAccuracy(value: number | null) {
  return value === null ? "—" : value.toFixed(4);
}

function toggleSet<T>(current: Set<T>, value: T) {
  const next = new Set(current);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

export function TrajectoryDashboard() {
  const [horizons, setHorizons] = useState<Set<number>>(() => new Set(HORIZONS));
  const [methods, setMethods] = useState<Set<string>>(() => new Set(METHODS));
  const [conditions, setConditions] = useState<Set<string>>(() => new Set(CONDITIONS));
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const [viewMode, setViewMode] = useState<ViewMode>("best");
  const [focusRunId, setFocusRunId] = useState<string>(() => runs[0]?.id ?? "");
  const [selectedPointKey, setSelectedPointKey] = useState<string>("");

  const filteredRuns = useMemo(
    () =>
      runs.filter(
        (run) =>
          horizons.has(run.horizon) &&
          methods.has(run.methodCode) &&
          conditions.has(run.condition),
      ),
    [horizons, methods, conditions],
  );
  const visibleRuns = filteredRuns.filter((run) => !hidden.has(run.id));
  const focusRun = filteredRuns.find((run) => run.id === focusRunId) ?? filteredRuns[0] ?? runs[0];

  const summary = useMemo(() => {
    const incumbents = filteredRuns
      .map((run) => run.finalBestParameterCount)
      .filter((value): value is number => value !== null);
    return {
      proposals: filteredRuns.reduce((total, run) => total + run.proposalCount, 0),
      valid: filteredRuns.reduce((total, run) => total + run.validCount, 0),
      unique: filteredRuns.reduce((total, run) => total + run.uniqueArchitectureCount, 0),
      lowest: incumbents.length ? Math.min(...incumbents) : null,
    };
  }, [filteredRuns]);

  const chart = useMemo(() => {
    const width = 1040;
    const height = 470;
    const padding = { top: 24, right: 26, bottom: 62, left: 84 };
    const values = visibleRuns.flatMap((run) =>
      run.trajectory
        .map((point) =>
          viewMode === "best" ? point.bestValidParameterCount : point.parameterCount,
        )
        .filter((value): value is number => value !== null),
    );
    const maxX = Math.max(...visibleRuns.map((run) => run.horizon), 1);
    const rawMaxY = Math.max(...values, 1);
    const step = rawMaxY <= 1000 ? 100 : rawMaxY <= 10000 ? 1000 : 5000;
    const maxY = Math.ceil(rawMaxY / step) * step;
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    const x = (iteration: number) => padding.left + (iteration / maxX) * innerWidth;
    const y = (value: number) => padding.top + innerHeight - (value / maxY) * innerHeight;
    return { width, height, padding, innerWidth, innerHeight, maxX, maxY, x, y };
  }, [visibleRuns, viewMode]);

  const selectedPoint = useMemo(() => {
    if (!focusRun) return null;
    const explicit = focusRun.trajectory.find(
      (point) => `${focusRun.id}:${point.iteration}` === selectedPointKey,
    );
    return explicit ?? focusRun.trajectory.at(-1) ?? null;
  }, [focusRun, selectedPointKey]);

  function selectPoint(run: Run, point: Point) {
    setFocusRunId(run.id);
    setSelectedPointKey(`${run.id}:${point.iteration}`);
  }

  return (
    <main className="dashboard-shell">
      <header className="page-header">
        <div>
          <h1>Architecture search trajectories</h1>
          <p>Compare how memory and assumption challenges change parameter-minimizing search.</p>
        </div>
        <div className="dataset-stamp">{trajectoryData.runCount} completed runs</div>
      </header>

      <section className="filter-bar" aria-label="Trajectory filters">
        <FilterGroup label="Iterations">
          {HORIZONS.map((horizon) => (
            <FilterButton
              key={horizon}
              active={horizons.has(horizon)}
              onClick={() => setHorizons((current) => toggleSet(current, horizon))}
            >
              {horizon}
            </FilterButton>
          ))}
        </FilterGroup>
        <FilterGroup label="Framework">
          <FilterButton active={methods.has("ar")} onClick={() => setMethods((current) => toggleSet(current, "ar"))}>AutoResearch</FilterButton>
          <FilterButton active={methods.has("oe")} onClick={() => setMethods((current) => toggleSet(current, "oe"))}>OpenEvolve</FilterButton>
        </FilterGroup>
        <FilterGroup label="Condition">
          {CONDITIONS.map((condition) => (
            <FilterButton
              key={condition}
              active={conditions.has(condition)}
              onClick={() => setConditions((current) => toggleSet(current, condition))}
            >
              {condition}
            </FilterButton>
          ))}
        </FilterGroup>
      </section>

      <section className="summary-row" aria-label="Filtered dataset summary">
        <div><span>Visible matrix cells</span><strong>{filteredRuns.length}</strong></div>
        <div><span>Proposals</span><strong>{summary.proposals}</strong></div>
        <div><span>Unique architectures</span><strong>{summary.unique}</strong></div>
        <div><span>Lowest incumbent</span><strong>{formatParams(summary.lowest)}</strong></div>
      </section>

      <section className="run-controls" aria-label="Toggle individual trajectories">
        {filteredRuns.map((run) => {
          const active = !hidden.has(run.id);
          return (
            <button
              className={active ? "run-toggle active" : "run-toggle"}
              key={run.id}
              onClick={() => setHidden((current) => toggleSet(current, run.id))}
              aria-pressed={active}
              title={run.conditionLabel}
            >
              <span className="run-swatch" style={{ backgroundColor: colorFor(run) }} />
              {shortLabel(run)}
            </button>
          );
        })}
        {filteredRuns.length === 0 && <p className="empty-copy">Select at least one value in each filter group.</p>}
      </section>

      <section className="chart-panel" aria-labelledby="chart-title">
        <div className="panel-heading">
          <div>
            <h2 id="chart-title">Parameters vs. proposal</h2>
            <p>{visibleRuns.length} of {filteredRuns.length} filtered trajectories visible</p>
          </div>
          <div className="view-switch" aria-label="Chart value">
            <button className={viewMode === "best" ? "selected" : ""} onClick={() => setViewMode("best")} aria-pressed={viewMode === "best"}>Running best</button>
            <button className={viewMode === "proposal" ? "selected" : ""} onClick={() => setViewMode("proposal")} aria-pressed={viewMode === "proposal"}>Each proposal</button>
          </div>
        </div>
        <div className="chart-scroll">
          <svg
            className="trajectory-chart"
            viewBox={`0 0 ${chart.width} ${chart.height}`}
            role="img"
            aria-label="Line and point chart comparing parameter count by proposal"
          >
            {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
              const value = Math.round(chart.maxY * fraction);
              const y = chart.y(value);
              return (
                <g key={fraction}>
                  <line x1={chart.padding.left} x2={chart.width - chart.padding.right} y1={y} y2={y} className="grid-line" />
                  <text x={chart.padding.left - 14} y={y + 5} textAnchor="end" className="axis-label">{formatParams(value)}</text>
                </g>
              );
            })}
            {Array.from({ length: chart.maxX + 1 }, (_, iteration) => iteration)
              .filter((iteration) => iteration === 0 || iteration === chart.maxX || iteration % (chart.maxX > 12 ? 4 : 2) === 0)
              .map((iteration) => (
                <text key={iteration} x={chart.x(iteration)} y={chart.height - 28} textAnchor="middle" className="axis-label">{iteration}</text>
              ))}
            <line x1={chart.padding.left} x2={chart.padding.left} y1={chart.padding.top} y2={chart.height - chart.padding.bottom} className="axis-line" />
            <line x1={chart.padding.left} x2={chart.width - chart.padding.right} y1={chart.height - chart.padding.bottom} y2={chart.height - chart.padding.bottom} className="axis-line" />
            <text x={chart.padding.left + chart.innerWidth / 2} y={chart.height - 3} textAnchor="middle" className="axis-title">Proposal</text>
            <text transform={`translate(21 ${chart.padding.top + chart.innerHeight / 2}) rotate(-90)`} textAnchor="middle" className="axis-title">{viewMode === "best" ? "Best valid parameters" : "Candidate parameters"}</text>
            {visibleRuns.map((run) => {
              const points = run.trajectory.filter((point) =>
                (viewMode === "best" ? point.bestValidParameterCount : point.parameterCount) !== null,
              );
              const color = colorFor(run);
              const path = viewMode === "best"
                ? points
                    .map((point, pointIndex) => {
                      const value = point.bestValidParameterCount as number;
                      return `${pointIndex === 0 ? "M" : "L"} ${chart.x(point.iteration)} ${chart.y(value)}`;
                    })
                    .join(" ")
                : "";
              return (
                <g key={run.id}>
                  {path && <path d={path} fill="none" stroke={color} strokeWidth="3" strokeDasharray={dashFor(run)} strokeLinejoin="round" strokeLinecap="round" />}
                  {points.map((point) => {
                    const value = (viewMode === "best" ? point.bestValidParameterCount : point.parameterCount) as number;
                    const selected = selectedPointKey === `${run.id}:${point.iteration}`;
                    return (
                      <circle
                        className="chart-point"
                        key={point.iteration}
                        cx={chart.x(point.iteration)}
                        cy={chart.y(value)}
                        r={selected ? 7 : 4.5}
                        fill={point.valid || viewMode === "best" ? color : "var(--surface)"}
                        stroke={color}
                        strokeWidth={selected ? 3 : 2}
                        onClick={() => selectPoint(run, point)}
                        tabIndex={0}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") selectPoint(run, point);
                        }}
                      >
                        <title>{`${shortLabel(run)}, proposal ${point.iteration}: ${formatParams(value)} parameters`}</title>
                      </circle>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </div>
      </section>

      {focusRun && selectedPoint && (
        <section className="evidence-grid">
          <article className="detail-panel">
            <div className="detail-header">
              <div>
                <span>{shortLabel(focusRun)} · proposal {selectedPoint.iteration}</span>
                <h2>{selectedPoint.architectureName}</h2>
              </div>
              <span className={selectedPoint.valid ? "status valid" : "status invalid"}>{selectedPoint.valid ? "Structurally valid" : "Invalid / unevaluated"}</span>
            </div>
            <p>{selectedPoint.mechanismHypothesis}</p>
            <dl className="detail-metrics">
              <div><dt>Candidate parameters</dt><dd>{formatParams(selectedPoint.parameterCount)}</dd></div>
              <div><dt>Running best</dt><dd>{formatParams(selectedPoint.bestValidParameterCount)}</dd></div>
              <div><dt>Public accuracy</dt><dd>{formatAccuracy(selectedPoint.publicAccuracy)}</dd></div>
              <div><dt>New architecture</dt><dd>{selectedPoint.isNewArchitecture ? "Yes" : "No"}</dd></div>
            </dl>
            {selectedPoint.failureReason && <p className="failure-copy">{selectedPoint.failureReason}</p>}
            {Object.keys(selectedPoint.nodeKinds).length > 0 && (
              <div className="node-list" aria-label="Architecture node types">
                {Object.entries(selectedPoint.nodeKinds).map(([kind, count]) => <span key={kind}>{kind.replaceAll("_", " ")} · {count}</span>)}
              </div>
            )}
          </article>

          <article className="trajectory-table-panel">
            <div className="table-toolbar">
              <div>
                <h2>Proposal trajectory</h2>
                <p>{focusRun.conditionLabel}</p>
              </div>
              <label>
                <span>Focus run</span>
                <select value={focusRun.id} onChange={(event) => { setFocusRunId(event.target.value); setSelectedPointKey(""); }}>
                  {filteredRuns.map((run) => <option key={run.id} value={run.id}>{shortLabel(run)}</option>)}
                </select>
              </label>
            </div>
            <div className="table-scroll">
              <table>
                <thead><tr><th>Proposal</th><th>Parameters</th><th>Best</th><th>Valid</th><th>Accuracy</th><th>Architecture</th></tr></thead>
                <tbody>
                  {focusRun.trajectory.map((point) => (
                    <tr
                      key={point.iteration}
                      className={selectedPoint.iteration === point.iteration ? "selected-row" : ""}
                      onClick={() => selectPoint(focusRun, point)}
                    >
                      <td>{point.iteration === 0 ? "Seed" : point.iteration}</td>
                      <td>{formatParams(point.parameterCount)}</td>
                      <td>{formatParams(point.bestValidParameterCount)}</td>
                      <td>{point.valid ? "Yes" : "No"}</td>
                      <td>{formatAccuracy(point.publicAccuracy)}</td>
                      <td>{point.architectureName}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </section>
      )}

      <aside className="validity-note">
        <strong>Interpret carefully.</strong> Structural validity does not require non-zero task accuracy. Tiny incumbents can indicate objective collapse rather than useful AdderBoard performance.
      </aside>
    </main>
  );
}

function FilterGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="filter-group"><span>{label}</span><div>{children}</div></div>;
}

function FilterButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button className={active ? "filter-button active" : "filter-button"} onClick={onClick} aria-pressed={active}>{children}</button>;
}
