import { normalizeRun, diffGraphs } from "./replay.js";

/** Validate local exports before any UI state is replaced. No source is executed. */
export function validateBundle(bundle) {
  if (
    bundle?.schema_version !== "architecture-bundle/1" ||
    bundle.run?.schema_version !== "architecture-replay/1" ||
    !Array.isArray(bundle.run.occurrences) ||
    !bundle.snapshots ||
    typeof bundle.snapshots !== "object"
  )
    throw Error(
      "Expected architecture-bundle/1 with a versioned run and snapshots.",
    );
  if (bundle.run.occurrences.length > 10000)
    throw Error("Bundle exceeds 10,000 occurrences.");
  if (
    !bundle.run.run_id ||
    !bundle.run.revision ||
    !Array.isArray(bundle.run.metric_definitions)
  )
    throw Error("Missing run identity, revision, or metric definitions.");
  const run = normalizeRun(bundle.run);
  const occurrenceIds = new Set(run.occurrences.map((o) => o.id));
  if (Object.keys(bundle.snapshots).some((id) => !occurrenceIds.has(id)))
    throw Error("Snapshot does not belong to this run.");
  for (const occurrence of run.occurrences) {
    if (!Number.isInteger(occurrence.proposal) || occurrence.proposal < 0)
      throw Error("Invalid proposal number.");
    const graph = bundle.snapshots[occurrence.id];
    if (!graph || graph.occurrence_id !== occurrence.id)
      throw Error("Missing or mismatched occurrence snapshot.");
    if (graph.candidate_id !== occurrence.candidate_id)
      throw Error("Snapshot candidate identity mismatch.");
    if (
      !Array.isArray(graph.nodes) ||
      !Array.isArray(graph.edges) ||
      graph.nodes.length > 5000 ||
      graph.edges.length > 30000
    )
      throw Error("Invalid or oversized graph.");
    if (
      graph.nodes.some(
        (node) =>
          typeof node.id !== "string" ||
          typeof node.kind !== "string" ||
          (node.evidence && !Array.isArray(node.evidence)),
      )
    )
      throw Error("Invalid component identity or evidence.");
    const ids = new Set(graph.nodes.map((node) => node.id));
    if (
      graph.edges.some(
        (edge) =>
          !ids.has(edge.source) ||
          !ids.has(edge.target) ||
          typeof edge.type !== "string",
      )
    )
      throw Error("Unknown edge endpoint or relationship.");
    diffGraphs(null, graph);
    if (
      (graph.source_hash ?? null) !== (occurrence.source?.source_hash ?? null)
    )
      throw Error("Source digest mismatch.");
  }
  return bundle;
}
