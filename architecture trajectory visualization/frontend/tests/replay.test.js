import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeRun,
  defaultEndpoint,
  buildReplay,
  reconcileSelection,
  comparisonOccurrence,
  evaluationMetrics,
  metricValue,
  metricDefinition,
  formatMetric,
  diffGraphs,
  layoutGraph,
} from "../src/replay.js";

const occurrence = (proposal, candidate, parents = [], extra = {}) => ({
  id: `run:${proposal}`,
  proposal,
  candidate_id: candidate,
  parent_ids: parents,
  metrics: {},
  ...extra,
});
const branching = () =>
  normalizeRun({
    run_id: "run",
    occurrences: [
      occurrence(0, "A", [], { status: "seed", incumbent_after: "A" }),
      occurrence(1, "B", ["A"], { retained: true, incumbent_after: "B" }),
      occurrence(2, "C", ["B"], { retained: false, incumbent_after: "B" }),
      occurrence(3, "D", ["B"], { retained: true, incumbent_after: "D" }),
    ],
  });
const node = (id, kind = "linear", width = 16) => ({
  id,
  label: id,
  kind,
  config: { width },
});
const graph = (nodes, edges = []) => ({ nodes, edges });
const edge = (source, target, type = "data_flow") => ({
  id: `${source}-${target}-${type}`,
  source,
  target,
  type,
});

test("lineage excludes rejected siblings while all-attempts preserves every opportunity", () => {
  const run = branching();
  assert.deepEqual(buildReplay(run).ids, ["run:0", "run:1", "run:3"]);
  assert.deepEqual(buildReplay(run, { mode: "all" }).ids, [
    "run:0",
    "run:1",
    "run:2",
    "run:3",
  ]);
  assert.deepEqual(buildReplay(run, { endpointId: "run:2" }).ids, [
    "run:0",
    "run:1",
    "run:2",
  ]);
  assert.equal(comparisonOccurrence(run, "run:3").id, "run:1");
  assert.equal(defaultEndpoint(run), "run:3");
});

test("revisiting source/candidate identity creates a later occurrence, not an ancestry cycle", () => {
  const run = normalizeRun({
    occurrences: [
      occurrence(0, "A"),
      occurrence(1, "B", ["A"]),
      occurrence(2, "A", ["B"]),
      occurrence(3, "C", ["A"]),
    ],
  });
  assert.deepEqual(buildReplay(run, { endpointId: "run:3" }).ids, [
    "run:0",
    "run:1",
    "run:2",
    "run:3",
  ]);
  assert.deepEqual(run.parentsById.get("run:3"), ["run:2"]);
});

test("portfolio retention never substitutes for a recorded incumbent decision", () => {
  const run = normalizeRun({
    occurrences: [
      occurrence(0, "A", [], { incumbent_after: "A" }),
      occurrence(1, "B", ["A"], {
        retained: true,
        retention_decision: "filled_open_portfolio_slot",
        incumbent_after: "A",
      }),
      occurrence(2, "C", ["B"], { retained: true, incumbent_after: "C" }),
    ],
  });
  assert.deepEqual(buildReplay(run, { mode: "incumbent" }).ids, [
    "run:0",
    "run:2",
  ]);
  assert.equal(run.occurrences[1].incumbent_occurrence_id, "run:0");
  const unknown = normalizeRun({
    occurrences: [
      occurrence(0, "A"),
      occurrence(1, "B", ["A"], { retained: true }),
    ],
  });
  assert.equal(unknown.finalIncumbentId, null);
  assert.deepEqual(buildReplay(unknown, { mode: "incumbent" }).ids, []);
});

test("explicit merge route preserves recorded parent order and comparison follows chosen parent", () => {
  const run = normalizeRun({
    occurrences: [
      occurrence(0, "A"),
      occurrence(1, "B", ["A"]),
      occurrence(2, "C", ["A"]),
      occurrence(3, "D", ["B", "C"]),
    ],
  });
  assert.deepEqual(buildReplay(run, { endpointId: "run:3" }).ids, [
    "run:0",
    "run:1",
    "run:3",
  ]);
  const routeParents = { "run:3": "run:2" };
  assert.deepEqual(
    buildReplay(run, { endpointId: "run:3", routeParents }).ids,
    ["run:0", "run:2", "run:3"],
  );
  assert.equal(
    comparisonOccurrence(run, "run:3", { routeParents }).id,
    "run:2",
  );
  assert.equal(
    comparisonOccurrence(run, "run:3", { baseline: true }).id,
    "run:0",
  );
});

test("missing/future parents remain unresolved and trailing incomplete opportunities remain visible", () => {
  const run = normalizeRun({
    occurrences: [
      occurrence(0, "A"),
      occurrence(1, "B", ["missing"], { parent_occurrence_ids: ["run:2"] }),
      occurrence(2, null, ["B"], {
        status: "interrupted",
        metrics: { accuracy: null },
      }),
    ],
  });
  assert.deepEqual(buildReplay(run, { mode: "all" }).ids, [
    "run:0",
    "run:1",
    "run:2",
  ]);
  assert.deepEqual(buildReplay(run).ids, ["run:1", "run:2"]);
  assert.match(buildReplay(run).warnings[0], /incomplete/);
  assert.equal(comparisonOccurrence(run, "run:1"), null);
  assert.equal(run.occurrences[2].candidate_id, null);
});

test("selection follows route membership; empty and duplicate data produce explicit outcomes", () => {
  const path = buildReplay(branching());
  assert.equal(reconcileSelection(path, "run:1"), "run:1");
  assert.equal(reconcileSelection(path, "run:2"), "run:3");
  assert.equal(reconcileSelection([], "missing"), null);
  assert.deepEqual(buildReplay(normalizeRun({})).ids, []);
  assert.throws(
    () =>
      normalizeRun({ occurrences: [occurrence(1, "A"), occurrence(1, "A")] }),
    /Duplicate occurrence/,
  );
});

test("an unresolved primary parent does not silently select a known secondary merge parent", () => {
  const run = normalizeRun({
    occurrences: [occurrence(0, "A"), occurrence(1, "B", ["missing", "A"])],
  });
  assert.deepEqual(buildReplay(run).ids, ["run:1"]);
  assert.equal(comparisonOccurrence(run, "run:1"), null);
  assert.deepEqual(
    buildReplay(run, { routeParents: { "run:1": "run:0" } }).ids,
    ["run:0", "run:1"],
  );
});

test("evaluation selection never mixes retry measurements or falls back from a missing chosen evaluation", () => {
  const item = occurrence(1, "B", ["A"], {
    decision_evaluation_id: "second",
    metrics: { accuracy: 0.9, memory: 20 },
    evaluations: [
      { id: "first", metrics: { accuracy: 0.9, memory: 20 } },
      { id: "second", metrics: { accuracy: 0.8 } },
    ],
  });
  assert.deepEqual(evaluationMetrics(item), { accuracy: 0.8 });
  assert.equal(metricValue(item, "memory"), null);
  assert.equal(metricValue(item, "memory", "first"), 20);
  assert.equal(metricValue(item, "accuracy", "missing"), null);
});

test("nonfinite/unsafe metrics stay unavailable numerically while exact integer strings display losslessly", () => {
  const item = occurrence(1, "A", [], {
    metrics: {
      nan: "nan",
      inf: Infinity,
      empty: "",
      zero: 0,
      big: "99999999999999999991",
      bool: false,
    },
  });
  for (const key of ["nan", "inf", "empty", "big", "bool"])
    assert.equal(metricValue(item, key), null);
  assert.equal(metricValue(item, "zero"), 0);
  assert.equal(
    metricValue(item, "big", null, { allowApproximate: true }),
    1e20,
  );
  assert.equal(formatMetric(item.metrics.big), "99,999,999,999,999,999,991");
  assert.equal(formatMetric(2 ** 60), "Unavailable");
  assert.equal(formatMetric(null), "Unavailable");
  assert.equal(
    formatMetric(1.23456, { unit: "bits/byte", precision: 3 }),
    "1.235 bits/byte",
  );
  assert.equal(formatMetric(0.0000001), "1.0000e-7");
  assert.equal(metricDefinition("val_bpb").label, "Validation bits per byte");
  assert.equal(metricDefinition("num_params_M").unit, "M parameters");
  assert.equal(
    metricDefinition("accuracy", [
      { key: "accuracy", direction: "minimize", label: "Error" },
    ]).label,
    "Error",
  );
});

test("rename-only graph preserves structure and matched node positions", () => {
  const before = graph(
    [node("embed", "embedding"), node("proj", "linear")],
    [edge("embed", "proj")],
  );
  const after = graph(
    [node("embed", "embedding"), node("renamed", "linear")],
    [edge("embed", "renamed")],
  );
  const diff = diffGraphs(before, after);
  assert.deepEqual(diff.added, []);
  assert.deepEqual(diff.removed, []);
  assert.deepEqual(diff.changed, []);
  assert.equal(diff.renamed.length, 1);
  assert.deepEqual(diff.edges, { added: [], removed: [] });
  const initial = layoutGraph(before);
  const layout = layoutGraph(after, { previousLayout: initial, diff });
  assert.deepEqual(layout.positions.renamed, initial.positions.proj);
  assert.deepEqual(layout.positions.embed, initial.positions.embed);
});

test("one configuration change is isolated; a source-hash change alone is not a structural change", () => {
  const before = { ...graph([node("a"), node("b")]), source_hash: "first" };
  const after = {
    ...graph([node("a"), node("b", "linear", 32)]),
    source_hash: "second",
  };
  const diff = diffGraphs(before, after);
  assert.deepEqual(diff.changed, ["b"]);
  assert.deepEqual(diff.added, []);
  assert.deepEqual(
    diffGraphs(before, { ...before, source_hash: "different" }).changed,
    [],
  );
});

test("shared-parameter ties and recurrent edges remain distinct from data flow", () => {
  const nodes = [node("a"), node("b")];
  const before = graph(nodes, [edge("a", "b")]);
  const after = graph(nodes, [
    edge("a", "b"),
    edge("a", "b", "shared_parameters"),
    edge("b", "a", "state"),
  ]);
  const diff = diffGraphs(before, after);
  assert.equal(diff.sharingChanged, true);
  assert.equal(diff.recurrenceChanged, true);
  assert.deepEqual(
    diff.edges.added.map((item) => item.type),
    ["shared_parameters", "state"],
  );
  const positions = layoutGraph(after).positions;
  assert.ok(positions.b.x > positions.a.x);
});

test("ambiguous renamed repeated blocks are labeled uncertain rather than asserted equivalent", () => {
  const diff = diffGraphs(
    graph([node("a"), node("b")]),
    graph([node("x"), node("y")]),
  );
  assert.equal(diff.matches.length, 2);
  assert.ok(diff.matches.every((item) => item.ambiguous));
  assert.ok(diff.warnings.length);
  assert.equal(diff.nodeStatus.x, "uncertain");
});

test("cyclic architectures produce finite layouts and unchanged nodes stay fixed on additions", () => {
  const before = graph(
    [node("a"), node("b")],
    [edge("a", "b"), edge("b", "a")],
  );
  const after = graph(
    [...before.nodes, node("c")],
    [...before.edges, edge("b", "c")],
  );
  const original = layoutGraph(before);
  const diff = diffGraphs(before, after);
  const changed = layoutGraph(after, { previousLayout: original, diff });
  assert.deepEqual(changed.positions.a, original.positions.a);
  assert.deepEqual(changed.positions.b, original.positions.b);
  assert.ok(
    changed.nodes.every((item) =>
      ["x", "y", "z"].every((axis) => Number.isFinite(item[axis])),
    ),
  );
  assert.deepEqual(layoutGraph(before), layoutGraph(before));
});

test("a long occurrence history and detailed graph retain complete chronological coverage", () => {
  const records = Array.from({ length: 1000 }, (_, index) =>
    occurrence(
      index,
      `candidate-${index}`,
      index ? [`candidate-${index - 1}`] : [],
    ),
  );
  const run = normalizeRun({ occurrences: records });
  assert.equal(buildReplay(run).ids.length, 1000);
  const nodes = Array.from({ length: 128 }, (_, index) =>
    node(`layer-${String(index).padStart(3, "0")}`),
  );
  const edges = nodes
    .slice(1)
    .map((item, index) => edge(nodes[index].id, item.id));
  const layout = layoutGraph(graph(nodes, edges));
  assert.equal(layout.nodes.length, 128);
  assert.equal(
    new Set(layout.nodes.map(({ x, y, z }) => `${x}:${y}:${z}`)).size,
    128,
  );
});

test("operation, repeat count, sharing, and output role are semantic edits", () => {
  const base = {
    nodes: [
      {
        id: "layer",
        kind: "activation",
        operation: "ReLU",
        repeat: { count: 2 },
        sharing: { group: "weights" },
        is_output: false,
      },
    ],
    edges: [],
  };
  for (const patch of [
    { operation: "SiLU" },
    { repeat: { count: 4 } },
    { sharing: { group: "other" } },
    { is_output: true },
  ]) {
    const after = { ...base, nodes: [{ ...base.nodes[0], ...patch }] };
    const d = diffGraphs(base, after);
    assert.deepEqual(d.changed, ["layer"]);
    if (patch.sharing) assert.equal(d.sharingChanged, true);
  }
});
test("inserted AST components do not steal the positional ID of a unique old component", () => {
  const before = {
    representation: "static_module_structure",
    nodes: [
      { id: "conv0", kind: "convolution", config: { width: 32 } },
      { id: "conv1", kind: "convolution", config: { width: 64 } },
    ],
    edges: [],
  };
  const after = {
    representation: "static_module_structure",
    nodes: [
      { id: "conv0", kind: "convolution", config: { width: 16 } },
      { id: "conv1", kind: "convolution", config: { width: 32 } },
      { id: "conv2", kind: "convolution", config: { width: 64 } },
    ],
    edges: [],
  };
  const d = diffGraphs(before, after);
  assert.deepEqual(d.added, ["conv0"]);
  assert.equal(d.changed.length, 0);
  assert.ok(d.matches.some((m) => m.before === "conv0" && m.after === "conv1"));
});
