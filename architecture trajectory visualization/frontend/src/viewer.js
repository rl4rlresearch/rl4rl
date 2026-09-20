import "./style.css";
import { validateBundle } from "./bundle.js";
import { ArchitectureScene, colorFor } from "./scene.js";
import {
  normalizeRun,
  buildReplay,
  defaultEndpoint,
  comparisonOccurrence,
  diffGraphs,
  layoutGraph,
  evaluationMetrics,
  metricValue,
  metricDefinition,
  formatMetric,
} from "./replay.js";
const $ = (id) => document.getElementById(id);
const el = (tag, text, cls) => {
  const e = document.createElement(tag);
  if (text != null) e.textContent = String(text);
  if (cls) e.className = cls;
  return e;
};
const svg = (tag, attrs = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
};
const query = new URLSearchParams(location.search);
const state = {
  catalog: null,
  campaign: query.get("campaign"),
  run: null,
  sequence: { occurrences: [], ids: [] },
  index: 0,
  mode: query.get("mode") || "lineage",
  scope: query.get("scope") || "dashboard",
  endpoint: null,
  routes: Object.create(null),
  playing: false,
  playbackToken: 0,
  speed: 1,
  metric: null,
  evaluation: null,
  baseline: false,
  graph: null,
  layout: null,
  bundle: null,
  cache: new Map(),
  snapshotRequests: new Map(),
  pending: false,
  displayedOccurrence: null,
  token: 0,
  runToken: 0,
  catalogToken: 0,
  expanded: new Set(),
};
try {
  state.routes = Object.assign(
    Object.create(null),
    JSON.parse(query.get("route") || "{}"),
  );
} catch {}
if (!["lineage", "all", "incumbent"].includes(state.mode))
  state.mode = "lineage";
if (!["archive", "dashboard"].includes(state.scope)) state.scope = "dashboard";
let scene = null,
  timer = null,
  noticeTimer = null;
try {
  if (query.get("webgl") === "off") throw Error("List view requested");
  scene = new ArchitectureScene($("canvas-mount"), inspectNode);
} catch {
  $("component-list").hidden = false;
  $("view-button").setAttribute("aria-pressed", "true");
  for (const id of ["fit-button", "camera-button", "capture-button"])
    $(id).disabled = true;
  notice(
    "3D is unavailable. The component list contains the same source evidence.",
  );
}
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => ($("notice").hidden = true), 8000);
}
function options(select, items, value) {
  select.replaceChildren(
    ...items.map(([id, label]) => {
      const o = el("option", label);
      o.value = id;
      return o;
    }),
  );
  if (value != null) select.value = value;
}
function current() {
  return state.sequence.occurrences[state.index] || null;
}
function pause() {
  state.playing = false;
  state.playbackToken++;
  clearTimeout(timer);
  $("play-button").textContent = "Play";
  $("play-button").setAttribute("aria-label", "Play trajectory");
}
function download(content, name) {
  const a = el("a");
  a.download = name;
  a.href = content;
  a.click();
}
async function api(action, params = {}, signal) {
  const q = new URLSearchParams({
    scope: state.scope,
    campaign: state.campaign || "",
    run: state.run?.run_id || "",
    revision: state.run?.revision || "",
    ...params,
  });
  const response = await fetch(`/api/architectures/${action}?${q}`, { signal });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw Error(error.error || `Could not load ${action}`);
  }
  return response.json();
}
function syncURL() {
  if (!state.run) return;
  const p = new URLSearchParams({
    campaign: state.campaign,
    run: state.run.run_id,
    revision: state.run.revision || "",
    scope: state.scope,
    mode: state.mode,
    endpoint: state.endpoint || "",
    occurrence: current()?.id || "",
    position: String(state.index),
  });
  if (Object.keys(state.routes).length)
    p.set("route", JSON.stringify(state.routes));
  if (state.bundle) p.set("bundle", state.run.revision || state.run.run_id);
  if (query.get("webgl") === "off") p.set("webgl", "off");
  history.replaceState(null, "", `${location.pathname}?${p}`);
}
function campaignLabel(row) {
  const task = `${row.key} ${row.task_id || ""}`.toLowerCase();
  return /fashion/.test(task)
    ? "Fashion-MNIST"
    : /nanogpt/.test(task)
      ? "nanoGPT"
      : /kws/.test(task)
        ? "Tiny Keyword Spotting"
        : /har/.test(task)
          ? "UCI HAR"
          : /tiny/.test(task)
            ? "Tiny AdderBoard"
            : /addition|transformer/.test(task)
              ? "Addition"
              : row.label;
}
async function loadCatalog(restore = false) {
  pause();
  const ticket = ++state.catalogToken,
    requestedScope = state.scope;
  state.runToken++;
  state.token++;
  abortSnapshotRequests();
  state.run = null;
  state.graph = null;
  state.displayGraph = null;
  state.layout = null;
  state.displayedOccurrence = null;
  state.sequence = { occurrences: [], ids: [], warnings: [] };
  state.index = 0;
  scene?.clear();
  clearRecordUI("Loading campaigns…");
  renderTimeline();
  $("scene-message").hidden = false;
  $("scene-message").textContent = "Reading campaign catalog…";
  try {
    const catalog = await api("catalog", { scope: requestedScope });
    if (ticket !== state.catalogToken || requestedScope !== state.scope) return;
    state.catalog = catalog;
    const rows = state.catalog.campaigns;
    if (!rows.length)
      throw Error("No campaign data is prepared. See the preparation guide.");
    if (!rows.some((r) => r.key === state.campaign))
      state.campaign =
        rows.find((r) => r.available && /fashion/.test(r.key))?.key ||
        rows.find((r) => r.available)?.key ||
        rows[0].key;
    options(
      $("campaign-select"),
      rows.map((r) => [
        r.key,
        `${campaignLabel(r)}${r.version_label ? " / " + r.version_label : ""} · ${r.available ? r.runs.length + " runs" : "prepare data"}`,
      ]),
      state.campaign,
    );
    $("scope-select").value = state.scope;
    $("mode-select").value = state.mode;
    setCampaign(restore);
  } catch (error) {
    if (ticket !== state.catalogToken || requestedScope !== state.scope) return;
    $("scene-message").textContent = error.message;
    notice(error.message);
  }
}
function setCampaign(restore = false) {
  const c = state.catalog.campaigns.find((r) => r.key === state.campaign);
  const conditions = [
    ...new Set(c.runs.map((r) => r.condition).filter(Boolean)),
  ].sort();
  options(
    $("condition-select"),
    [["", "All conditions"], ...conditions.map((x) => [x, x])],
    "",
  );
  options(
    $("framework-select"),
    [
      ["", "All frameworks"],
      ...[...new Set(c.runs.map((r) => r.framework).filter(Boolean))].map(
        (x) => [x, x],
      ),
    ],
    "",
  );
  options(
    $("block-select"),
    [
      ["", "All blocks"],
      ...[...new Set(c.runs.map((r) => r.replicate).filter((x) => x != null))]
        .sort()
        .map((x) => [String(x), `Block ${x}`]),
    ],
    "",
  );
  fillRuns(restore ? query.get("run") : null);
  $("scope-description").textContent =
    `${state.scope === "dashboard" ? "Comparable dashboard scope" : "Full archive history"} · ${c.data_origin || "local bundle"}. ${c.availability_note || "Recorded proposal order; retained ≠ incumbent."}`;
}
function fillRuns(preferred) {
  const c = state.catalog.campaigns.find((r) => r.key === state.campaign),
    rows = c.runs
      .filter(
        (r) =>
          !$("condition-select").value ||
          r.condition === $("condition-select").value,
      )
      .filter(
        (r) =>
          !$("block-select").value ||
          String(r.replicate) === $("block-select").value,
      )
      .filter(
        (r) =>
          !$("framework-select").value ||
          r.framework === $("framework-select").value,
      );
  options(
    $("run-select"),
    rows.map((r) => [
      r.id,
      `${r.condition || "run"} · block ${r.replicate ?? "—"} · ${r.id}`,
    ]),
    rows.some((r) => r.id === preferred) ? preferred : rows[0]?.id,
  );
  if (rows.length) loadRun($("run-select").value);
  else {
    state.token++;
    state.runToken++;
    abortSnapshotRequests();
    state.run = null;
    state.sequence = { occurrences: [], ids: [] };
    state.graph = null;
    state.displayGraph = null;
    state.layout = null;
    state.displayedOccurrence = null;
    scene?.clear();
    clearRecordUI("No prepared run");
    $("scene-message").hidden = false;
    $("scene-message").textContent =
      c.availability_note || "No runs match this filter.";
    renderTimeline();
    $("history-list").replaceChildren();
  }
}
async function loadRun(runId) {
  pause();
  const ticket = ++state.runToken;
  state.token++;
  abortSnapshotRequests();
  state.run = null;
  state.graph = null;
  state.displayedOccurrence = null;
  state.sequence = { occurrences: [], ids: [] };
  state.index = 0;
  clearRecordUI("Loading run…");
  renderTimeline();
  $("model-title").textContent = "Loading run…";
  $("model-subtitle").textContent = runId;
  $("metrics").replaceChildren();
  $("diff-detail").replaceChildren();
  $("history-list").replaceChildren();
  $("canvas-mount").style.visibility = "hidden";
  state.cache.clear();
  state.expanded.clear();
  state.layout = null;
  state.displayGraph = null;
  state.evaluation = null;
  $("scene-message").hidden = false;
  $("scene-message").textContent = "Loading recorded proposals…";
  try {
    const raw = state.bundle
      ? state.bundle.run
      : await api("run", { run: runId });
    if (ticket !== state.runToken) return;
    state.run = normalizeRun(raw);
    state.endpoint = state.run.byId.has(query.get("endpoint"))
      ? query.get("endpoint")
      : defaultEndpoint(state.run);
    if (!state.run.finalIncumbentId && state.mode === "lineage")
      state.mode = "all";
    $("mode-select").value = state.mode;
    options(
      $("endpoint-select"),
      state.run.occurrences.map((o) => [
        o.id,
        `#${o.proposal} · ${o.candidate_id || o.status}`,
      ]),
      state.endpoint,
    );
    options(
      $("metric-select"),
      state.run.metric_definitions.map((d) => [
        d.key,
        `${d.label}${d.direction === "minimize" ? " ↓" : d.direction === "maximize" ? " ↑" : ""}`,
      ]),
    );
    state.metric =
      state.run.metric_definitions.find((d) => d.is_objective)?.key ||
      state.run.metric_definitions[0]?.key;
    $("metric-select").value = state.metric;
    rebuild(query.get("occurrence"));
    if (query.get("revision") && query.get("revision") !== raw.revision)
      notice(
        "The bookmark references a different data revision. Showing the prepared revision.",
      );
    if (scene) scene.fitted = false;
    await selectIndex(state.index);
    for (const k of ["occurrence", "endpoint", "revision", "run", "position"])
      query.delete(k);
  } catch (error) {
    if (ticket !== state.runToken) return;
    $("scene-message").textContent = error.message;
    notice(error.message);
  }
}
function rebuild(preserve = current()?.id) {
  if (!state.run) return;
  state.sequence = buildReplay(state.run, {
    mode: state.mode,
    endpointId: state.endpoint,
    routeParents: state.routes,
  });
  state.index = Math.max(
    0,
    state.sequence.ids.includes(preserve)
      ? state.sequence.ids.indexOf(preserve)
      : state.sequence.ids.length - 1,
  );
  if (query.get("position") && state.mode === "incumbent") {
    const i = Number(query.get("position"));
    if (state.sequence.ids[i] === preserve) state.index = i;
  }
  renderHistory();
  renderTimeline();
  if (!state.sequence.occurrences.length) {
    state.token++;
    abortSnapshotRequests();
    state.graph = null;
    state.displayGraph = null;
    state.layout = null;
    state.displayedOccurrence = null;
    scene?.clear();
    clearRecordUI("No recorded path", true);
    $("scene-message").hidden = false;
    $("scene-message").textContent =
      state.sequence.warnings[0] ||
      "No occurrences are recorded for this replay mode.";
  }
}
function selectHistory(id) {
  pause();
  if (!state.sequence.ids.includes(id)) {
    state.endpoint = id;
    state.mode = "lineage";
    $("mode-select").value = "lineage";
    $("endpoint-select").value = id;
    rebuild(id);
  }
  selectIndex(state.sequence.ids.indexOf(id));
}
function renderHistory() {
  if (!state.run) return;
  $("history-count").textContent = `${state.run.occurrences.length} records`;
  drawLineage();
  const list = $("history-list");
  if (list.dataset.selected !== current()?.id && !$("history-search").value) {
    list.dataset.selected = current()?.id || "";
    const i = state.run.occurrences.findIndex((o) => o.id === current()?.id),
      top = i * 56;
    if (top < list.scrollTop || top + 56 > list.scrollTop + list.clientHeight)
      list.scrollTop = Math.max(0, top - list.clientHeight / 2);
  }
  renderHistoryRows();
}
function renderHistoryRows() {
  if (!state.run) return;
  const list = $("history-list"),
    search = $("history-search").value.toLowerCase(),
    rows = state.run.occurrences.filter((o) =>
      `${o.proposal} ${o.candidate_id} ${o.status}`
        .toLowerCase()
        .includes(search),
    );
  const height = 56,
    begin = Math.max(0, Math.floor(list.scrollTop / height) - 3),
    end = Math.min(
      rows.length,
      begin + Math.ceil(list.clientHeight / height) + 7,
    ),
    nodes = [];
  const spacer = el("div");
  spacer.style.height = `${begin * height}px`;
  nodes.push(spacer);
  for (const o of rows.slice(begin, end)) {
    const b = el(
      "button",
      null,
      `history-row ${o.id === current()?.id ? "selected" : ""} ${state.sequence.ids.includes(o.id) ? "path-member" : ""}`,
    );
    b.style.height = "54px";
    b.dataset.occurrence = o.id;
    b.title = o.candidate_id || o.id;
    b.setAttribute("role", "listitem");
    const details = el("span");
    details.append(
      el("span", (o.candidate_id || "source pending").slice(0, 16), "row-id"),
      el(
        "span",
        `${o.status}${o.retained && o.status !== "retained" ? " · retained" : ""}`,
        "row-status",
      ),
    );
    b.append(
      el("span", o.proposal === 0 ? "S" : o.proposal, "row-proposal"),
      details,
      el(
        "span",
        o.retained ? "●" : o.status === "failed" ? "×" : "○",
        `row-symbol ${o.retained ? "retained" : o.status === "failed" ? "failed" : ""}`,
      ),
    );
    b.onclick = () => selectHistory(o.id);
    nodes.push(b);
  }
  const after = el("div");
  after.style.height = `${Math.max(0, rows.length - end) * height}px`;
  nodes.push(after);
  list.replaceChildren(...nodes);
}
function drawLineage() {
  const rows = state.run.occurrences,
    s = svg("svg", { viewBox: "0 0 220 60" }),
    pos = new Map(
      rows.map((o, i) => [
        o.id,
        {
          x: 6 + (i / Math.max(1, rows.length - 1)) * 208,
          y: state.sequence.ids.includes(o.id) ? 20 : 43,
        },
      ]),
    );
  for (const o of rows)
    for (const p of state.run.parentsById.get(o.id) || []) {
      const a = pos.get(p),
        b = pos.get(o.id);
      if (a && b)
        s.append(
          svg("path", {
            d: `M${a.x},${a.y} Q${a.x},${b.y} ${b.x},${b.y}`,
            fill: "none",
            stroke: state.sequence.ids.includes(o.id) ? "#647d6a" : "#354039",
            "stroke-width": 0.7,
          }),
        );
    }
  for (const o of rows) {
    const p = pos.get(o.id),
      c = svg("circle", {
        cx: p.x,
        cy: p.y,
        r: o.id === current()?.id ? 3.5 : 1.5,
        fill:
          o.id === current()?.id
            ? "#d9ede0"
            : o.retained
              ? "#9cbda7"
              : "#627669",
      });
    const t = svg("title");
    t.textContent = `#${o.proposal} ${o.candidate_id}`;
    c.append(t);
    c.onclick = () => selectHistory(o.id);
    s.append(c);
  }
  $("lineage-mini").replaceChildren(s);
}
function renderTimeline() {
  const rows = state.sequence.occurrences,
    o = current(),
    slider = $("timeline-slider");
  slider.max = Math.max(0, rows.length - 1);
  slider.value = state.index;
  slider.disabled = !rows.length;
  slider.setAttribute(
    "aria-valuetext",
    o
      ? `Proposal ${o.proposal}, ${o.candidate_id || o.status}, ${state.index + 1} of ${rows.length}`
      : "No proposal",
  );
  $("selection-label").textContent = o
    ? `Proposal ${o.proposal} · ${state.index + 1} / ${rows.length} · ${o.status}`
    : "No recorded proposals";
  $("timeline-start").textContent = rows.length ? `#${rows[0].proposal}` : "—";
  $("timeline-end").textContent = rows.length
    ? `#${rows.at(-1).proposal}`
    : "—";
  $("timeline-context").textContent =
    state.sequence.warnings?.[0] ||
    `${state.mode === "lineage" ? "Recorded ancestry" : state.mode === "incumbent" ? "Recorded incumbent changes" : "All proposal opportunities"} · gaps remain missing`;
  const markers = rows.map((r, i) => {
    const e = el(
      "span",
      null,
      `timeline-marker ${i === state.index ? "selected" : ""}`,
    );
    e.style.left = `${(i / Math.max(1, rows.length - 1)) * 100}%`;
    e.style.background =
      r.status === "failed" ? "#b27879" : r.retained ? "#92bba0" : "#61716b";
    return e;
  });
  $("timeline-markers").replaceChildren(...markers);
  for (const k of ["start", "previous", "next", "end", "play"])
    $(`${k}-button`).disabled = !rows.length;
  drawPlot();
}
function drawPlot() {
  const rows = state.sequence.occurrences,
    values = rows.map((o) =>
      metricValue(
        o,
        state.metric,
        o.id === current()?.id ? state.evaluation : null,
        { allowApproximate: true },
      ),
    ),
    valid = values.filter((v) => v !== null),
    s = svg("svg", { viewBox: "0 0 1000 55", preserveAspectRatio: "none" });
  let low = valid.length ? Math.min(...valid) : 0,
    high = valid.length ? Math.max(...valid) : 1;
  if (low === high) high = low + 1;
  const x = (i) => (i / Math.max(1, rows.length - 1)) * 1000,
    y = (v) => 47 - ((v - low) / (high - low)) * 36;
  values.forEach((v, i) => {
    if (v === null) return;
    if (i && values[i - 1] !== null)
      s.append(
        svg("path", {
          d: `M${x(i - 1)},${y(values[i - 1])} L${x(i)},${y(v)}`,
          fill: "none",
          stroke: "#8db29b",
          "stroke-width": 1.5,
          "vector-effect": "non-scaling-stroke",
        }),
      );
    s.append(
      svg("circle", {
        cx: x(i),
        cy: y(v),
        r: rows.length > 150 ? 1 : 1.7,
        fill: "#9ebb9f",
      }),
    );
  });
  s.append(
    svg("line", {
      x1: x(state.index),
      x2: x(state.index),
      y1: 0,
      y2: 54,
      stroke: "#d7e7d9",
      "stroke-width": 1,
      "stroke-dasharray": "3 3",
      "vector-effect": "non-scaling-stroke",
    }),
  );
  s.onclick = (e) => {
    pause();
    const r = s.getBoundingClientRect();
    selectIndex(
      Math.round(((e.clientX - r.left) / r.width) * (rows.length - 1)),
    );
  };
  const title = svg("title");
  title.textContent = `${metricDefinition(state.metric || "", state.run?.metric_definitions).label}. Plot positions approximate large integers; displayed values remain exact. Missing values remain gaps; click to select.`;
  s.append(title);
  $("metric-chart").replaceChildren(s);
}
function dl(node, entries) {
  node.replaceChildren(
    ...entries.flatMap(([k, v]) => [el("dt", k), el("dd", v ?? "Unavailable")]),
  );
}
function renderHeadlineMetric(o) {
  const d = metricDefinition(state.metric || "", state.run?.metric_definitions),
    value = evaluationMetrics(o, state.evaluation)[state.metric];
  $("model-metric").textContent = `${d.label}: ${formatMetric(value, d)}`;
}
function clearRecordUI(title, keepEndpoint = false) {
  setPending(false);
  $("model-title").textContent = title;
  for (const id of [
    "model-context",
    "model-subtitle",
    "model-metric",
    "status",
    "proposal-label",
    "change-caption",
    "source-label",
  ])
    $(id).textContent = "";
  for (const id of [
    "metrics",
    "diff-detail",
    "node-inspector",
    "provenance",
    "diagnostics",
    "operation-legend",
    "component-list",
    "categorical-evidence",
  ])
    $(id).replaceChildren();
  $("proposal-description").textContent = "No proposal selected.";
  $("parent-choice-label").hidden = true;
  $("evaluation-label").hidden = true;
  if (!keepEndpoint) options($("endpoint-select"), []);
  $("representation").textContent = "No architecture selected";
}
function renderMetadata(o) {
  const campaign = state.catalog?.campaigns.find(
    (c) => c.key === state.campaign,
  );
  $("model-context").textContent =
    `${campaign ? campaignLabel(campaign) : state.campaign} / ${state.run.condition || "Recorded run"}${state.run.replicate != null ? " · block " + state.run.replicate : ""}`;
  $("model-title").textContent =
    `Proposal ${o.proposal === 0 ? "0 · seed" : o.proposal}`;
  $("status").textContent =
    o.status + (o.retained && o.status !== "retained" ? " · retained" : "");
  $("status").className = `status ${o.status === "failed" ? "failed" : ""}`;
  renderHeadlineMetric(o);
  $("model-subtitle").textContent =
    o.candidate_id || "No candidate saved for this occurrence";
  $("proposal-label").textContent = `#${o.proposal}`;
  const measurements = evaluationMetrics(o, state.evaluation),
    defs = state.run.metric_definitions
      .filter((d) => Object.hasOwn(measurements, d.key))
      .slice(0, 7);
  dl(
    $("metrics"),
    defs.length
      ? defs.map((d) => [d.label, formatMetric(measurements[d.key], d)])
      : [["Measurements", "Unavailable"]],
  );
  $("proposal-description").textContent =
    o.mechanism ||
    o.hypothesis ||
    o.intended_edit ||
    "No proposal description was recorded.";
  const parents = state.run.parentsById.get(o.id) || [];
  $("parent-choice-label").hidden =
    parents.length < 2 && (o.parent_ids || []).length < 2;
  options(
    $("parent-select"),
    parents.map((id) => [
      id,
      `#${state.run.byId.get(id).proposal} · ${state.run.byId.get(id).candidate_id}`,
    ]),
    state.routes[o.id] || state.run.primaryParentById.get(o.id),
  );
  $("evaluation-label").hidden = (o.evaluations || []).length < 2;
  options(
    $("evaluation-select"),
    (o.evaluations || []).map((e) => [
      e.id,
      `${e.id === o.decision_evaluation_id ? "Decision · " : ""}${e.id}`,
    ]),
    state.evaluation || o.decision_evaluation_id,
  );
  dl($("provenance"), [
    ["Candidate", o.candidate_id],
    ["Source snapshot", o.source?.source_candidate_id],
    ["Incumbent after", o.incumbent_after],
    ["Decision", o.retention_decision],
    ["Parents", o.parent_ids?.join(", ") || "None recorded"],
    ["Evaluation", state.evaluation || o.decision_evaluation_id],
    ["Source revision", state.run.source_revision?.slice(0, 12)],
    ["Data revision", state.run.revision?.slice(0, 12)],
    ["Source hash", o.source?.source_hash?.slice(0, 16)],
    ["Scope", state.scope],
    ["Categorical review", o.categorical?.status || "Not joined"],
    ["Family", o.categorical?.family_id || "Unavailable"],
  ]);
  $("node-inspector").replaceChildren(
    el("h3", "Component evidence"),
    el("p", "Select a block or a component in the list."),
  );
  renderCategorical(o);
  $("source-label").textContent =
    `${state.bundle ? "Local bundle" : "GitHub"} ${state.run.source_revision?.slice(0, 7) || "unavailable"}`;
}

function renderCategorical(o) {
  const c = o.categorical,
    box = $("categorical-evidence");
  box.replaceChildren();
  if (!c) {
    box.append(el("p", "No publication joined for this occurrence."));
    return;
  }
  box.append(
    el("p", `${c.status} · ${c.reason || ""}`),
    el("p", c.source_verification_reason || ""),
  );
  if (c.family_id) box.append(el("p", `Family: ${c.family_id}`));
  box.append(
    el(
      "p",
      "Published metrics use the primary parent and full run; viewing a path does not recompute them.",
    ),
  );
  if (c.no_change_assumed)
    box.append(
      el(
        "p",
        "Carried-forward categorical totals do not prove unchanged topology.",
      ),
    );
  for (const [key, value] of Object.entries(c.metrics || {})) {
    if (key.endsWith("_cumulative"))
      box.append(el("p", `${key.replaceAll("_", " ")}: ${value}`));
  }
  if (c.changed_components?.length)
    box.append(
      el("p", `Primary-parent changes: ${c.changed_components.join(", ")}`),
    );
  if (c.fingerprint) {
    const details = el("details"),
      summary = el("summary", "Reviewed component labels");
    details.append(summary);
    for (const [key, value] of Object.entries(c.fingerprint))
      if (value !== "absent")
        details.append(el("p", `${key.replaceAll("_", " ")}: ${value}`));
    box.append(details);
  }
}

function snapshotKey(o) {
  return o
    ? `${state.scope}:${state.campaign}:${state.run?.run_id}:${state.run?.revision}:${o.id}`
    : null;
}
function abortSnapshotRequests(keep = new Set()) {
  for (const [key, request] of state.snapshotRequests) {
    if (!keep.has(key)) {
      request.controller.abort();
      state.snapshotRequests.delete(key);
    }
  }
}
function setPending(pending) {
  state.pending = pending;
  $("viewport").setAttribute("aria-busy", String(pending));
  $("component-list").inert = pending || !state.graph;
  $("capture-button").disabled = !scene || pending || !state.graph;
}
async function snapshot(o) {
  if (!o) return null;
  if (state.bundle) return state.bundle.snapshots[o.id] || null;
  const key = snapshotKey(o);
  if (state.cache.has(key)) return state.cache.get(key);
  if (state.snapshotRequests.has(key))
    return state.snapshotRequests.get(key).promise;
  // Pin every request to its original run. Shared prefetches can survive a
  // selection change, but must never populate another revision's cache.
  const runToken = state.runToken,
    request = { controller: new AbortController(), promise: null },
    params = {
      scope: state.scope,
      campaign: state.campaign,
      run: state.run.run_id,
      revision: state.run.revision,
      proposal: o.proposal,
    };
  request.promise = api("snapshot", params, request.controller.signal)
    .then((value) => {
      if (
        value.run_revision !== params.revision ||
        value.occurrence_id !== o.id ||
        value.candidate_id !== o.candidate_id ||
        (value.source_hash ?? null) !== (o.source?.source_hash ?? null)
      )
        throw Error(
          "Data revision changed. Select this run again to load matching source and measurements.",
        );
      if (runToken === state.runToken && !request.controller.signal.aborted) {
        state.cache.set(key, value);
        if (state.cache.size > 240)
          state.cache.delete(state.cache.keys().next().value);
      }
      return value;
    })
    .finally(() => {
      if (state.snapshotRequests.get(key) === request)
        state.snapshotRequests.delete(key);
    });
  state.snapshotRequests.set(key, request);
  return request.promise;
}
function prefetchAdjacent() {
  if (state.bundle || !state.run) return;
  // At most four requests: the next and previous stops plus their comparisons.
  // Work outside a new selection's pair is cancelled before it can accumulate.
  const candidates = [];
  for (const index of [state.index + 1, state.index - 1]) {
    const occurrence = state.sequence.occurrences[index];
    if (!occurrence) continue;
    candidates.push(
      occurrence,
      comparisonOccurrence(state.run, occurrence.id, {
        baseline: state.baseline,
        routeParents: state.routes,
      }),
    );
  }
  for (const occurrence of candidates) {
    if (!occurrence || state.snapshotRequests.size >= 4) continue;
    const key = snapshotKey(occurrence);
    if (state.cache.has(key) || state.snapshotRequests.has(key)) continue;
    // A failed speculative request is retried normally if the stop is selected.
    void snapshot(occurrence).catch(() => {});
  }
}
function projectedGraph(graph) {
  if (innerWidth <= 700) {
    const groups = (graph.groups || []).filter(
      (g) => g.container_node_id && !state.expanded.has(g.id),
    );
    const mapping = new Map(),
      summaries = new Map();
    for (const g of groups) {
      const members = graph.nodes.filter((n) => n.group_id === g.id);
      if (members.length < 3) continue;
      for (const n of members) mapping.set(n.id, g.container_node_id);
      summaries.set(g.container_node_id, { group: g, members });
    }
    if (mapping.size) {
      const nodes = graph.nodes
        .filter((n) => !mapping.has(n.id))
        .map((n) => {
          const s = summaries.get(n.id);
          return s
            ? {
                ...n,
                label: `${n.label} · ${s.members.length} components`,
                config: { ...n.config, components: s.members.length },
                _collapsedGroup: s.group.id,
              }
            : n;
        });
      const edges = [],
        seen = new Set();
      for (const e of graph.edges || []) {
        const source = mapping.get(e.source) || e.source,
          target = mapping.get(e.target) || e.target,
          key = `${source}:${target}:${e.type}`;
        if (source !== target && !seen.has(key)) {
          edges.push({ ...e, source, target });
          seen.add(key);
        }
      }
      graph = { ...graph, nodes, edges };
    }
  }
  if (graph.nodes.length <= 28) return graph;
  const groups = new Map();
  for (const n of graph.nodes) {
    const g = n.group_id || n.id;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(n);
  }
  const map = new Map(),
    nodes = [];
  for (const [id, members] of groups) {
    if (members.length < 3 || state.expanded.has(id)) {
      for (const n of members) {
        nodes.push(n);
        map.set(n.id, n.id);
      }
    } else {
      const name =
        graph.groups?.find((g) => g.id === id)?.label || id.split(":").at(-1);
      nodes.push({
        id: `group:${id}`,
        label: `${name} · ${members.length} components`,
        kind: "module",
        group_id: id,
        config: { components: members.length },
        evidence: members.flatMap((n) => n.evidence || []).slice(0, 6),
        _collapsedGroup: id,
      });
      for (const n of members) map.set(n.id, `group:${id}`);
    }
  }
  const edges = [],
    seen = new Set();
  for (const e of graph.edges || []) {
    const source = map.get(e.source),
      target = map.get(e.target),
      key = `${source}:${target}:${e.type}`;
    if (source && target && source !== target && !seen.has(key)) {
      edges.push({ ...e, source, target });
      seen.add(key);
    }
  }
  return { ...graph, nodes, edges };
}
async function selectIndex(
  index,
  { keepPlaying = false, keepEvaluation = false } = {},
) {
  if (!keepPlaying) pause();
  if (!state.run || !state.sequence.occurrences.length) return;
  state.index = Math.max(
    0,
    Math.min(state.sequence.occurrences.length - 1, index),
  );
  if (!keepEvaluation) state.evaluation = null;
  const o = current(),
    ticket = ++state.token,
    comparison = comparisonOccurrence(state.run, o.id, {
      baseline: state.baseline,
      routeParents: state.routes,
    }),
    previousVisible = Boolean(state.displayGraph?.nodes?.length && scene);
  abortSnapshotRequests(
    new Set([snapshotKey(o), snapshotKey(comparison)].filter(Boolean)),
  );
  state.graph = null;
  setPending(true);
  renderMetadata(o);
  renderHistory();
  renderTimeline();
  syncURL();
  // Keep the previous architecture on screen while the next descriptor loads.
  // Current metadata belongs to the requested stop; the caption makes the
  // temporary visual mismatch explicit and inspection/capture stay disabled.
  $("scene-message").hidden = previousVisible;
  $("scene-message").textContent = "Reading saved architecture…";
  $("representation").textContent = previousVisible
    ? `Loading proposal ${o.proposal} · previous architecture remains visible`
    : `Loading proposal ${o.proposal}…`;
  $("change-caption").textContent = "";
  $("diff-detail").replaceChildren();
  $("node-inspector").replaceChildren(
    el("h3", "Component evidence"),
    el("p", "Loading the selected architecture…"),
  );
  if (!previousVisible) {
    $("operation-legend").replaceChildren();
    $("component-list").replaceChildren();
  }
  try {
    const [graph, before] = await Promise.all([
      snapshot(o),
      snapshot(comparison),
    ]);
    if (ticket !== state.token) return;
    if (!graph || !graph.nodes?.length) {
      $("diagnostics").textContent = [
        ...(state.run.diagnostics || []),
        ...(o.source?.diagnostics || []),
        ...(graph?.extraction?.warnings || []),
      ].join(" · ");
      scene?.clear();
      state.displayGraph = null;
      state.layout = null;
      state.displayedOccurrence = null;
      $("scene-message").hidden = false;
      $("operation-legend").replaceChildren();
      $("component-list").replaceChildren();
      $("node-inspector").replaceChildren(
        el("p", "No supported components are available for this proposal."),
      );
      $("scene-message").textContent =
        "No supported architecture snapshot is available for this proposal. Its recorded outcome and ancestry remain visible.";
      $("representation").textContent = o.source?.available
        ? "Source available · extraction unsupported"
        : "Source unavailable";
      return;
    }
    state.graph = graph;
    const semantic = before?.nodes?.length ? diffGraphs(before, graph) : null,
      display = projectedGraph(graph),
      transition = state.displayGraph
        ? diffGraphs(state.displayGraph, display)
        : null,
      layout = layoutGraph(display, {
        previousLayout: state.layout,
        diff: transition,
      }),
      displayDiff = semantic
        ? diffGraphs(projectedGraph(before), display)
        : null;
    state.layout = layout;
    state.displayGraph = display;
    state.displayedOccurrence = o.id;
    scene?.setGraph(display, layout, displayDiff, transition, {
      durationMs: 850 / (keepPlaying ? state.speed : 1),
    });
    $("canvas-mount").style.visibility = "";
    renderComponents(display);
    $("node-inspector").replaceChildren(
      el("h3", "Component evidence"),
      el("p", "Select a block or a component in the list."),
    );
    $("scene-message").hidden = true;
    $("representation").textContent =
      graph.extraction?.completeness === "recorded"
        ? "Recorded IR · schematic scale"
        : "Source structure · partial · schematic scale";
    const desc = semantic
      ? `${semantic.added.length} added · ${semantic.removed.length} removed · ${semantic.changed.length} changed in source view`
      : "Comparison structure unavailable";
    $("change-caption").textContent = comparison
      ? `vs #${comparison.proposal} · ${desc}`
      : "Seed / no recorded parent";
    const items = [
      comparison
        ? `Compared with #${comparison.proposal} (${state.baseline ? "run baseline" : "selected parent"}).`
        : "No recorded comparison is available.",
      semantic ? desc : "Architecture change is unknown.",
      semantic
        ? `${semantic.edges.added.length} connections added; ${semantic.edges.removed.length} removed. ${semantic.renamed.length} matched renames.`
        : "",
      semantic?.sharingChanged ? "Shared-parameter relationships changed." : "",
      semantic?.recurrenceChanged
        ? "Known state / recurrence relationships changed."
        : "",
      ...(semantic?.warnings || []),
    ];
    if (semantic)
      for (const [label, ids, from] of [
        ["Added", semantic.added, graph],
        ["Removed", semantic.removed, before],
        ["Changed", semantic.changed, graph],
      ])
        if (ids.length)
          items.push(
            `${label}: ${ids
              .slice(0, 8)
              .map((id) => from.nodes.find((n) => n.id === id)?.label || id)
              .join(", ")}${ids.length > 8 ? "…" : ""}`,
          );
    $("diff-detail").replaceChildren(
      ...items.filter(Boolean).map((t) => el("p", t, "diff-item")),
    );
    $("diagnostics").textContent = [
      ...(state.run.diagnostics || []),
      ...(o.source?.diagnostics || []),
      ...(graph.extraction?.warnings || []),
      ...(state.sequence.warnings || []),
    ].join(" · ");
    const kinds = [...new Set(display.nodes.map((n) => n.kind))];
    $("operation-legend").replaceChildren(
      ...kinds.slice(0, 8).map((kind) => {
        const e = el("span", null, "legend-item"),
          swatch = el("i", null, "legend-swatch");
        swatch.style.background = colorFor(kind);
        e.append(swatch, el("span", kind.replaceAll("_", " ")));
        return e;
      }),
    );
  } catch (error) {
    if (ticket !== state.token || error.name === "AbortError") return;
    $("scene-message").hidden = previousVisible;
    $("scene-message").textContent = error.message;
    $("representation").textContent = previousVisible
      ? `Proposal ${o.proposal} unavailable · previous architecture remains visible`
      : "Architecture unavailable";
    $("node-inspector").replaceChildren(
      el("p", "Selected architecture unavailable."),
    );
    notice(error.message);
  } finally {
    if (ticket === state.token) {
      setPending(false);
      prefetchAdjacent();
    }
  }
}
function renderComponents(graph) {
  $("component-list").replaceChildren(
    ...graph.nodes.map((n) => {
      const b = el("button", null, "component-row");
      b.append(
        el("span", n.label || n.id),
        el(
          "small",
          `${n.kind} · ${n.confidence || "source evidence"}${n._collapsedGroup ? " · expandable group" : ""}`,
        ),
      );
      b.onclick = () => inspectNode(n);
      return b;
    }),
  );
}
function inspectNode(n) {
  if (state.pending || !state.graph) return;
  const box = $("node-inspector");
  box.replaceChildren(
    el("h3", n.label || n.id),
    el("p", `${n.kind} · ${n.confidence || "recorded evidence"}`),
  );
  if (n._collapsedGroup) {
    const b = el("button", "Expand components");
    b.onclick = () => {
      state.expanded.add(n._collapsedGroup);
      selectIndex(state.index);
    };
    box.append(b);
  }
  for (const [k, v] of Object.entries(n.config || {}))
    box.append(
      el("p", `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`),
    );
  for (const e of (n.evidence || []).slice(0, 6))
    box.append(
      el(
        "div",
        `${e.file || "IR"}${e.line ? `:${e.line}` : ""}`,
        "evidence-file",
      ),
      el("pre", e.expression || e.text || JSON.stringify(e)),
    );
  if (!n.evidence?.length)
    box.append(el("p", "No line-level evidence is attached."));
  if (innerWidth < 970) document.body.classList.add("show-details");
}
async function tick() {
  if (!state.playing) return;
  const ticket = state.playbackToken;
  if (state.index >= state.sequence.occurrences.length - 1) {
    pause();
    return;
  }
  await selectIndex(state.index + 1, { keepPlaying: true });
  if (state.playing && ticket === state.playbackToken)
    timer = setTimeout(tick, 1300 / state.speed);
}
$("campaign-select").onchange = () => {
  state.campaign = $("campaign-select").value;
  state.routes = Object.create(null);
  setCampaign();
};
$("condition-select").onchange = () => fillRuns();
$("framework-select").onchange = () => fillRuns();
$("block-select").onchange = () => fillRuns();
$("run-select").onchange = () => loadRun($("run-select").value);
$("scope-select").onchange = () => {
  state.scope = $("scope-select").value;
  loadCatalog();
};
$("mode-select").onchange = () => {
  pause();
  state.mode = $("mode-select").value;
  rebuild();
  selectIndex(state.index);
};
$("endpoint-select").onchange = () => {
  pause();
  state.endpoint = $("endpoint-select").value;
  state.mode = "lineage";
  $("mode-select").value = "lineage";
  rebuild();
  selectIndex(state.index);
};
$("history-search").oninput = () => {
  $("history-list").scrollTop = 0;
  renderHistoryRows();
};
$("history-list").onscroll = renderHistoryRows;
$("timeline-slider").oninput = (e) => selectIndex(Number(e.target.value));
$("start-button").onclick = () => selectIndex(0);
$("end-button").onclick = () => selectIndex(state.sequence.ids.length - 1);
$("previous-button").onclick = () => selectIndex(state.index - 1);
$("next-button").onclick = () => selectIndex(state.index + 1);
$("play-button").onclick = () => {
  if (state.playing) {
    pause();
    return;
  }
  state.playing = true;
  $("play-button").textContent = "Pause";
  $("play-button").setAttribute("aria-label", "Pause trajectory");
  if (state.index === state.sequence.ids.length - 1) state.index = -1;
  tick();
};
$("speed-select").onchange = (e) => (state.speed = Number(e.target.value));
$("metric-select").onchange = (e) => {
  state.metric = e.target.value;
  drawPlot();
  if (current()) renderHeadlineMetric(current());
};
$("comparison-select").onchange = (e) => {
  state.baseline = e.target.value === "baseline";
  selectIndex(state.index, { keepEvaluation: true });
};
$("parent-select").onchange = (e) => {
  state.routes[current().id] = e.target.value;
  const id = current().id;
  rebuild(id);
  selectIndex(state.index);
};
$("evaluation-select").onchange = (e) => {
  state.evaluation = e.target.value;
  selectIndex(state.index, { keepEvaluation: true });
};
$("fit-button").onclick = () => scene?.fit();
$("camera-button").onclick = () => {
  scene?.fit(!scene.top);
  $("camera-button").textContent = scene?.top ? "Perspective" : "Top view";
};
$("view-button").onclick = () => {
  const show = $("component-list").hidden;
  $("component-list").hidden = !show;
  $("view-button").setAttribute("aria-pressed", String(show));
};
$("history-toggle").onclick = () => {
  const show = document.body.classList.toggle("show-history");
  $("history-toggle").setAttribute("aria-pressed", String(show));
};
$("inspector-toggle").onclick = () => {
  const show =
    innerWidth < 970
      ? document.body.classList.toggle("show-details")
      : !document.body.classList.toggle("no-inspector");
  $("inspector-toggle").setAttribute("aria-pressed", String(show));
};
$("presentation-button").onclick = () => {
  const show = document.body.classList.toggle("present");
  $("presentation-button").setAttribute("aria-pressed", String(show));
  $("presentation-button").textContent = show ? "Exit presentation" : "Present";
  setTimeout(() => scene?.fit(), 80);
};
$("capture-button").onclick = () => {
  if (!scene || !state.graph || state.pending) {
    notice("Select an available architecture before capturing.");
    return;
  }
  const o = current();
  download(
    scene.capture({
      title: `RL4RL · ${$("campaign-select").selectedOptions[0]?.textContent} · #${o.proposal}`,
      subtitle: `${o.candidate_id} · ${o.status} · ${$("representation").textContent}`,
      metric: `${$("change-caption").textContent} | ${metricDefinition(state.metric, state.run.metric_definitions).label}: ${formatMetric(evaluationMetrics(o, state.evaluation)[state.metric], metricDefinition(state.metric, state.run.metric_definitions))}`,
      provenance: `${state.bundle ? "Local bundle" : "GitHub recorded data"} · ${state.run.source_revision?.slice(0, 12)}`,
    }),
    `rl4rl-proposal-${o.proposal}.png`,
  );
};
$("export-button").onclick = async () => {
  if (!state.run) return;
  pause();
  $("export-button").disabled = true;
  notice("Preparing portable run with source-backed descriptors…");
  try {
    const b = state.bundle || (await api("export")),
      url = URL.createObjectURL(
        new Blob([JSON.stringify(b)], { type: "application/json" }),
      );
    download(url, `${b.run.run_id}.architecture.json`);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notice(
      "Portable run exported. Open it with “Open bundle”; no upload is involved.",
    );
  } catch (e) {
    notice(e.message);
  } finally {
    $("export-button").disabled = false;
  }
};
$("import-button").onclick = () => $("bundle-input").click();
$("bundle-input").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    if (file.size > 100 * 1024 * 1024) throw Error("Bundle limit is 100 MB");
    const b = validateBundle(JSON.parse(await file.text()));
    state.catalogToken++;
    state.bundle = b;
    state.campaign = b.run.campaign_key || b.run.campaign;
    state.scope = b.run.scope || "dashboard";
    $("scope-select").value = state.scope;
    state.catalog = {
      campaigns: [
        {
          key: state.campaign,
          label: state.campaign,
          available: true,
          data_origin: "local read-only bundle",
          runs: [
            {
              id: b.run.run_id,
              condition: b.run.condition,
              replicate: b.run.replicate,
            },
          ],
        },
      ],
    };
    options(
      $("campaign-select"),
      [[state.campaign, `${state.campaign} · local bundle`]],
      state.campaign,
    );
    $("scope-select").disabled = true;
    setCampaign(true);
    notice("Local bundle opened read-only. Nothing was uploaded.");
  } catch (error) {
    notice(error.message);
  }
  e.target.value = "";
};
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(e.target.tagName))
    return;
  if (e.code === "Space") {
    e.preventDefault();
    $("play-button").click();
  }
  if (e.key === "ArrowRight") {
    e.preventDefault();
    selectIndex(state.index + 1);
  }
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    selectIndex(state.index - 1);
  }
  if (e.key === "Escape")
    document.body.classList.remove("show-history", "show-details");
});
let narrow = innerWidth <= 700;
window.addEventListener("resize", () => {
  renderHistoryRows();
  if (narrow !== innerWidth <= 700) {
    narrow = innerWidth <= 700;
    state.layout = null;
    if (scene) scene.fitted = false;
    selectIndex(state.index);
  }
});
window.addEventListener("beforeunload", () => {
  pause();
  abortSnapshotRequests();
  scene?.dispose();
});
if (query.has("bundle")) {
  $("scene-message").textContent =
    "This bookmark uses a local bundle. Choose “Open bundle” to load the matching file; it has not been uploaded.";
  $("campaign-select").replaceChildren(el("option", "Open local bundle"));
} else loadCatalog(true);
Object.defineProperty(window, "architectureReplayMotion", {
  get: () => scene?.inspectMotion() ?? null,
});
Object.defineProperty(window, "architectureReplayState", {
  get: () => ({
    campaign: state.campaign,
    scope: state.scope,
    dataScope: state.run?.scope,
    run: state.run?.run_id,
    occurrence: current()?.id,
    proposal: current()?.proposal,
    index: state.index,
    length: state.sequence.ids.length,
    mode: state.mode,
    graphOccurrence: state.graph?.occurrence_id,
    displayedOccurrence: state.displayedOccurrence,
    pending: state.pending,
    snapshotRequests: state.snapshotRequests.size,
    nodeCount: state.graph?.nodes?.length || 0,
    playing: state.playing,
    revision: state.run?.revision,
    metrics: evaluationMetrics(current(), state.evaluation),
    comparison: current()
      ? comparisonOccurrence(state.run, current().id, {
          baseline: state.baseline,
          routeParents: state.routes,
        })?.proposal
      : null,
  }),
});

let assetRevision = null;
setInterval(async () => {
  if (state.bundle || state.playing) return;
  try {
    const r = await fetch("/api/revision").then((x) => x.json());
    if (assetRevision && r.revision !== assetRevision) location.reload();
    assetRevision = r.revision;
  } catch {}
}, 6000);
