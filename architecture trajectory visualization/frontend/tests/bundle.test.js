import test from "node:test";
import assert from "node:assert/strict";
import { validateBundle } from "../src/bundle.js";
function fixture() {
  return {
    schema_version: "architecture-bundle/1",
    run: {
      schema_version: "architecture-replay/1",
      run_id: "run",
      revision: "revision",
      metric_definitions: [],
      occurrences: [
        {
          id: "run:0",
          proposal: 0,
          candidate_id: "seed",
          source: { source_hash: "digest" },
        },
      ],
    },
    snapshots: {
      "run:0": {
        occurrence_id: "run:0",
        candidate_id: "seed",
        source_hash: "digest",
        nodes: [{ id: "input", kind: "input", evidence: [] }],
        edges: [],
      },
    },
  };
}
test("portable run validates exact occurrence/source identity without modifying values", () => {
  const b = fixture();
  b.run.occurrences[0].metrics = { cost: "9223372036854775807" };
  assert.equal(validateBundle(b), b);
  assert.equal(b.run.occurrences[0].metrics.cost, "9223372036854775807");
});
test("local imports reject schema, candidate, source, and graph endpoint mismatches", () => {
  const mutations = [
    (b) => (b.schema_version = "future"),
    (b) => (b.snapshots["run:0"].candidate_id = "other"),
    (b) => (b.snapshots["run:0"].source_hash = "other"),
    (b) =>
      b.snapshots["run:0"].edges.push({
        source: "input",
        target: "absent",
        type: "data_flow",
      }),
    (b) => b.snapshots["run:0"].nodes.push({ id: "input", kind: "input" }),
    (b) => delete b.snapshots["run:0"],
    (b) => (b.snapshots.extra = b.snapshots["run:0"]),
  ];
  for (const mutate of mutations) {
    const b = fixture();
    mutate(b);
    assert.throws(() => validateBundle(b));
  }
});
test("source/proposal text stays inert data, not parsed markup or code", () => {
  const b = fixture();
  b.run.occurrences[0].mechanism = "<img src=x onerror=alert(1)>";
  b.snapshots["run:0"].nodes[0].label = "<script>throw Error()</script>";
  assert.equal(
    validateBundle(b).run.occurrences[0].mechanism,
    b.run.occurrences[0].mechanism,
  );
});
