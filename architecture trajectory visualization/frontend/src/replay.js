/** Pure replay, evidence-safe metric, graph matching, and layout helpers. */

const OWN = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const list = (value) => (Array.isArray(value) ? value : []);
const idOf = (value) => {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "object") return idOf(value.candidate_id ?? value.id);
  return String(value);
};
const finiteOrder = (value, fallback) => {
  if (value === null || value === undefined || value === "") return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
};

/**
 * Prepared runs contain one occurrence per proposal opportunity, with lifecycle
 * and evaluation retries already joined by the backend. Candidate IDs are NOT
 * occurrence IDs. In particular, visiting the same candidate twice keeps two
 * separate nodes in the temporal ancestry graph.
 */
export function normalizeRun(rawRun = {}) {
  const runId = String(rawRun.run_id ?? rawRun.id ?? "run");
  const occurrences = list(rawRun.occurrences ?? rawRun.points)
    .map((raw, index) => {
      const proposal = raw.proposal ?? raw.iteration ?? index;
      return {
        ...raw,
        id: String(
          raw.id ?? raw.occurrence_id ?? `${runId}:proposal:${proposal}`,
        ),
        run_id: raw.run_id ?? runId,
        proposal,
        sequence: finiteOrder(raw.sequence, finiteOrder(proposal, index)),
        candidate_id: idOf(raw.candidate_id),
        parent_ids: list(raw.parent_ids).map(idOf).filter(Boolean),
        metrics:
          raw.metrics && typeof raw.metrics === "object" ? raw.metrics : {},
        evaluations: list(raw.evaluations),
        _inputIndex: index,
      };
    })
    .sort((a, b) => a.sequence - b.sequence || a._inputIndex - b._inputIndex);

  const byId = new Map();
  const parentsById = new Map();
  const primaryParentById = new Map();
  const unresolvedParentsById = new Map();
  const latestByCandidate = new Map();
  const incumbentChanges = [];
  const warnings = [];
  let recordedIncumbent = null;
  let incumbentOccurrenceId = null;
  let hasRecordedIncumbent = false;

  for (const occurrence of occurrences) {
    if (byId.has(occurrence.id)) {
      throw new Error(
        `Duplicate occurrence ID: ${occurrence.id}. Join lifecycle events before replay.`,
      );
    }
    const explicit = list(occurrence.parent_occurrence_ids);
    const parentCount = Math.max(occurrence.parent_ids.length, explicit.length);
    const parents = [];
    const unresolved = [];
    for (let index = 0; index < parentCount; index += 1) {
      const candidateId = occurrence.parent_ids[index] ?? null;
      const explicitId = idOf(explicit[index]);
      const explicitParent = explicitId ? byId.get(explicitId) : null;
      let parent =
        explicitParent &&
        (!candidateId || explicitParent.candidate_id === candidateId)
          ? explicitParent
          : null;
      if (explicitId && !parent) {
        warnings.push(
          `${occurrence.id}: parent occurrence ${explicitId} is not a matching earlier occurrence.`,
        );
      }
      if (!parent && candidateId)
        parent = latestByCandidate.get(candidateId) ?? null;
      if (parent) {
        if (!parents.includes(parent.id)) parents.push(parent.id);
      } else {
        unresolved.push(candidateId ?? explicitId ?? "unknown");
      }
      if (index === 0) primaryParentById.set(occurrence.id, parent?.id ?? null);
    }
    parentsById.set(occurrence.id, parents);
    unresolvedParentsById.set(occurrence.id, unresolved);
    byId.set(occurrence.id, occurrence);
    if (occurrence.candidate_id)
      latestByCandidate.set(occurrence.candidate_id, occurrence);

    // Portfolio retention is intentionally irrelevant here: only recorded
    // incumbent_after establishes an incumbent change.
    const nextIncumbent = idOf(occurrence.incumbent_after);
    if (nextIncumbent) {
      hasRecordedIncumbent = true;
      if (nextIncumbent !== recordedIncumbent || !incumbentOccurrenceId) {
        const incumbent = latestByCandidate.get(nextIncumbent);
        recordedIncumbent = nextIncumbent;
        incumbentOccurrenceId = incumbent?.id ?? null;
        if (incumbent) {
          incumbentChanges.push({
            occurrence_id: incumbent.id,
            decision_occurrence_id: occurrence.id,
            candidate_id: nextIncumbent,
          });
        } else {
          warnings.push(
            `${occurrence.id}: recorded incumbent ${nextIncumbent} has no known occurrence.`,
          );
        }
      }
    }
    occurrence.incumbent_occurrence_id = incumbentOccurrenceId;
    delete occurrence._inputIndex;
  }

  return {
    ...rawRun,
    run_id: runId,
    occurrences,
    byId,
    parentsById,
    primaryParentById,
    unresolvedParentsById,
    incumbentChanges,
    hasRecordedIncumbent,
    finalIncumbentId: incumbentOccurrenceId,
    warnings,
  };
}

export function defaultEndpoint(run) {
  return run.finalIncumbentId ?? run.occurrences.at(-1)?.id ?? null;
}

function routeParent(run, occurrenceId, routeParents = {}) {
  const parents = run.parentsById.get(occurrenceId) ?? [];
  const chosen =
    routeParents instanceof Map
      ? routeParents.get(occurrenceId)
      : routeParents[occurrenceId];
  if (parents.includes(chosen)) return chosen;
  return run.primaryParentById?.has(occurrenceId)
    ? run.primaryParentById.get(occurrenceId)
    : (parents[0] ?? null);
}

/** One path is traversed lazily; the Cartesian product of merge routes is never built. */
export function buildReplay(
  run,
  { mode = "lineage", endpointId, routeParents = {} } = {},
) {
  if (mode === "all" || mode === "attempts" || mode === "all-attempts") {
    return {
      occurrences: run.occurrences,
      ids: run.occurrences.map((item) => item.id),
      warnings: [],
    };
  }
  if (mode === "incumbent") {
    const occurrences = run.incumbentChanges.map((change) =>
      run.byId.get(change.occurrence_id),
    );
    return {
      occurrences,
      ids: occurrences.map((item) => item.id),
      decisionIds: run.incumbentChanges.map(
        (change) => change.decision_occurrence_id,
      ),
      warnings: occurrences.length
        ? []
        : ["No resolvable incumbent changes were recorded for this run."],
    };
  }
  if (mode !== "lineage") throw new Error(`Unknown replay mode: ${mode}`);
  const endpoint = endpointId ?? defaultEndpoint(run);
  if (!endpoint) return { occurrences: [], ids: [], warnings: [] };
  if (!run.byId.has(endpoint))
    return {
      occurrences: [],
      ids: [],
      warnings: [`Unknown endpoint: ${endpoint}`],
    };
  const reversed = [];
  const seen = new Set();
  const warnings = [];
  let current = endpoint;
  while (current) {
    if (seen.has(current)) {
      warnings.push(`Cyclic occurrence ancestry stopped at ${current}.`);
      break;
    }
    seen.add(current);
    reversed.push(run.byId.get(current));
    const unresolved = run.unresolvedParentsById.get(current) ?? [];
    if (unresolved.length)
      warnings.push(
        `${current}: unresolved parent ${unresolved.join(", ")}; lineage is incomplete.`,
      );
    current = routeParent(run, current, routeParents);
  }
  const occurrences = reversed.reverse();
  return { occurrences, ids: occurrences.map((item) => item.id), warnings };
}

/** Keep a shared occurrence when changing routes; otherwise use the new endpoint. */
export function reconcileSelection(sequence, currentId) {
  const ids = Array.isArray(sequence)
    ? sequence.map((item) => (typeof item === "string" ? item : item.id))
    : (sequence.ids ?? sequence.occurrences?.map((item) => item.id) ?? []);
  return ids.includes(currentId) ? currentId : (ids.at(-1) ?? null);
}

/** Comparison follows actual ancestry, never the preceding all-attempts stop. */
export function comparisonOccurrence(
  run,
  id,
  { baseline = false, routeParents = {} } = {},
) {
  if (!run.byId.has(id)) return null;
  if (baseline) {
    const seed = run.occurrences.find(
      (item) => item.status === "seed" || Number(item.proposal) === 0,
    );
    return seed && seed.id !== id ? seed : null;
  }
  return run.byId.get(routeParent(run, id, routeParents)) ?? null;
}

function evaluationIdentity(evaluation) {
  return idOf(
    evaluation.id ?? evaluation.evaluation_id ?? evaluation.attempt_id,
  );
}

/** Evaluation selection is all-or-nothing: never fill missing retry metrics from another attempt. */
export function evaluationMetrics(occurrence, evaluationId = null) {
  if (!occurrence) return {};
  const evaluations = list(occurrence.evaluations);
  const selectedId = idOf(evaluationId ?? occurrence.decision_evaluation_id);
  if (selectedId) {
    const selected = evaluations.find(
      (item) => evaluationIdentity(item) === selectedId,
    );
    return selected ? (selected.metrics ?? {}) : {};
  }
  if (occurrence.metrics && typeof occurrence.metrics === "object")
    return occurrence.metrics;
  if (evaluations.length === 1) return evaluations[0].metrics ?? {};
  return {};
}

/** Exact raw values remain on the occurrence; approximate plotting requires an explicit opt-in. */
export function metricValue(
  occurrence,
  key,
  evaluationId = null,
  { allowApproximate = false } = {},
) {
  const raw = evaluationMetrics(occurrence, evaluationId)[key];
  if (
    raw === null ||
    raw === undefined ||
    raw === "" ||
    typeof raw === "boolean"
  )
    return null;
  if (!["number", "string", "bigint"].includes(typeof raw)) return null;
  if (
    typeof raw === "string" &&
    !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw.trim())
  )
    return null;
  const number = Number(raw);
  if (!Number.isFinite(number)) return null;
  if (
    !allowApproximate &&
    Number.isInteger(number) &&
    !Number.isSafeInteger(number)
  )
    return null;
  return number;
}

const METRIC_DEFAULTS = {
  accuracy: { label: "Accuracy", unit: "", direction: "maximize" },
  public_accuracy: {
    label: "Public accuracy",
    unit: "",
    direction: "maximize",
  },
  validation_accuracy: {
    label: "Validation accuracy",
    unit: "",
    direction: "maximize",
  },
  validation_cross_entropy: {
    label: "Validation cross-entropy",
    unit: "",
    direction: "minimize",
  },
  val_bpb: {
    label: "Validation bits per byte",
    unit: "bits/byte",
    direction: "minimize",
  },
  parameters: {
    label: "Parameters",
    unit: "parameters",
    direction: null,
    precision: 0,
  },
  trainable_parameters: {
    label: "Trainable parameters",
    unit: "parameters",
    direction: null,
    precision: 0,
  },
  num_params_M: { label: "Parameters", unit: "M parameters", direction: null },
  peak_vram_mb: { label: "Peak VRAM", unit: "MB", direction: "minimize" },
  recurrent_steps: { label: "Recurrent steps", unit: "steps", direction: null },
  mean_recurrent_steps: {
    label: "Mean recurrent steps",
    unit: "steps",
    direction: null,
  },
  inference_cost: { label: "Inference cost", unit: "", direction: null },
};

export function metricDefinition(key, definitions = []) {
  const explicit = Array.isArray(definitions)
    ? definitions.find((item) => item.key === key)
    : definitions?.[key];
  return {
    key,
    label: key.replaceAll("_", " "),
    unit: "",
    direction: null,
    ...(METRIC_DEFAULTS[key] ?? {}),
    ...(explicit ?? {}),
  };
}

/** Decimal integer strings are grouped directly and never coerced through Number. */
export function formatMetric(value, definition = {}) {
  if (
    value === null ||
    value === undefined ||
    value === "" ||
    typeof value === "boolean"
  )
    return "Unavailable";
  const unit = definition.unit ? ` ${definition.unit}` : "";
  if (
    typeof value === "bigint" ||
    (typeof value === "string" && /^[+-]?\d+$/.test(value.trim()))
  ) {
    const raw = String(value).trim();
    const negative = raw.startsWith("-");
    const digits = raw.replace(/^[+-]/, "").replace(/^0+(?=\d)/, "");
    return `${negative ? "-" : ""}${digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}${unit}`;
  }
  if (typeof value !== "number" && typeof value !== "string")
    return "Unavailable";
  if (
    typeof value === "string" &&
    !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())
  )
    return "Unavailable";
  const number = Number(value);
  if (!Number.isFinite(number)) return "Unavailable";
  if (Number.isInteger(number) && !Number.isSafeInteger(number)) {
    return typeof value === "string" ? `${value.trim()}${unit}` : "Unavailable";
  }
  const maximumFractionDigits = Math.max(
    0,
    Math.min(
      12,
      Number.isInteger(definition.precision) ? definition.precision : 4,
    ),
  );
  if (number !== 0 && Math.abs(number) < 10 ** -maximumFractionDigits) {
    return `${number.toExponential(Math.min(maximumFractionDigits, 4))}${unit}`;
  }
  return `${new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(number)}${unit}`;
}

function stable(value) {
  if (value === undefined) return "null";
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stable(value[key])}`)
    .join(",")}}`;
}

function semanticNode(node) {
  return stable({
    kind: node.kind ?? "unknown",
    operation: node.operation ?? null,
    repeat: node.repeat ?? null,
    sharing: node.sharing ?? null,
    is_output: node.is_output ?? null,
    config: node.config ?? {},
    shape: node.shape ?? null,
    parameter_count: node.parameter_count ?? null,
    recurrence: node.recurrence ?? null,
  });
}

function edgeSignature(edge, remap = new Map()) {
  return stable({
    source: remap.get(String(edge.source)) ?? String(edge.source),
    target: remap.get(String(edge.target)) ?? String(edge.target),
    type: edge.type ?? "unknown",
    config: edge.config ?? null,
    attributes: edge.attributes ?? null,
    source_port: edge.source_port ?? null,
    target_port: edge.target_port ?? null,
  });
}

function graphParts(graph) {
  const nodes = list(graph?.nodes).map((node) => ({
    ...node,
    id: String(node.id),
  }));
  const ids = new Set(nodes.map((node) => node.id));
  if (ids.size !== nodes.length)
    throw new Error("Architecture graph contains duplicate node IDs.");
  const edges = list(graph?.edges).filter(
    (edge) => ids.has(String(edge.source)) && ids.has(String(edge.target)),
  );
  return { nodes, edges };
}

/**
 * Match recorded IDs, or unique structure before positional AST IDs. Structurally
 * indistinguishable renames are paired deterministically but explicitly marked
 * ambiguous; source hashes alone never imply a topology edit.
 */
export function diffGraphs(before, after) {
  const oldGraph = graphParts(before);
  const newGraph = graphParts(after);
  const oldNodes = new Map(oldGraph.nodes.map((node) => [node.id, node]));
  const newNodes = new Map(newGraph.nodes.map((node) => [node.id, node]));
  const unmatchedOld = new Set(oldNodes.keys());
  const unmatchedNew = new Set(newNodes.keys());
  const matches = [];
  const warnings = [];
  const match = (oldId, newId, reason, ambiguous = false) => {
    matches.push({ before: oldId, after: newId, reason, ambiguous });
    unmatchedOld.delete(oldId);
    unmatchedNew.delete(newId);
  };
  const positionalIds =
    before?.representation === "static_module_structure" ||
    after?.representation === "static_module_structure";
  if (!positionalIds)
    for (const id of [...unmatchedOld])
      if (newNodes.has(id)) match(id, id, "stable-id");

  const signatureGroups = (ids, nodes, signature) => {
    const groups = new Map();
    for (const id of [...ids].sort()) {
      const key = signature(nodes.get(id));
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(id);
    }
    return groups;
  };
  const neighborhood = (node, graph, nodes) => {
    const neighbors = graph.edges
      .flatMap((edge) => {
        if (String(edge.source) === node.id)
          return [
            `out:${edge.type}:${semanticNode(nodes.get(String(edge.target)))}`,
          ];
        if (String(edge.target) === node.id)
          return [
            `in:${edge.type}:${semanticNode(nodes.get(String(edge.source)))}`,
          ];
        return [];
      })
      .sort();
    return `${semanticNode(node)}:${stable(neighbors)}`;
  };
  const oldSignatures = signatureGroups(unmatchedOld, oldNodes, (node) =>
    neighborhood(node, oldGraph, oldNodes),
  );
  const newSignatures = signatureGroups(unmatchedNew, newNodes, (node) =>
    neighborhood(node, newGraph, newNodes),
  );
  for (const [signature, oldIds] of oldSignatures) {
    const newIds = newSignatures.get(signature) ?? [];
    if (oldIds.length === 1 && newIds.length === 1)
      match(oldIds[0], newIds[0], "unique-structure");
  }
  if (positionalIds)
    for (const id of [...unmatchedOld])
      if (unmatchedNew.has(id)) {
        const signature = semanticNode(oldNodes.get(id));
        const oldCount = oldGraph.nodes.filter(
            (n) => semanticNode(n) === signature,
          ).length,
          newCount = newGraph.nodes.filter(
            (n) => semanticNode(n) === signature,
          ).length;
        const ambiguous =
          oldCount !== newCount &&
          (oldCount > 1 || newCount > 1) &&
          semanticNode(newNodes.get(id)) === signature;
        match(
          id,
          id,
          ambiguous ? "ambiguous-positional-id" : "source-position",
          ambiguous,
        );
        if (
          ambiguous &&
          !warnings.includes(
            "Repeated source components changed count; their exact correspondence is uncertain.",
          )
        )
          warnings.push(
            "Repeated source components changed count; their exact correspondence is uncertain.",
          );
      }
  const oldSemantic = signatureGroups(unmatchedOld, oldNodes, semanticNode);
  const newSemantic = signatureGroups(unmatchedNew, newNodes, semanticNode);
  for (const [signature, oldIds] of oldSemantic) {
    const newIds = newSemantic.get(signature) ?? [];
    const ambiguous = oldIds.length > 1 || newIds.length > 1;
    const count = Math.min(oldIds.length, newIds.length);
    for (let index = 0; index < count; index += 1) {
      match(
        oldIds[index],
        newIds[index],
        ambiguous ? "ambiguous-structure" : "unique-structure",
        ambiguous,
      );
    }
    if (ambiguous && count)
      warnings.push(
        `${count} component matches are structurally indistinguishable; correspondence is uncertain.`,
      );
  }

  const renamed = matches
    .filter(
      ({ before: oldId, after: newId }) =>
        oldId !== newId ||
        oldNodes.get(oldId).label !== newNodes.get(newId).label,
    )
    .map(({ before: oldId, after: newId }) => ({
      before: oldId,
      after: newId,
      from: oldNodes.get(oldId).label ?? oldId,
      to: newNodes.get(newId).label ?? newId,
    }));
  const changed = matches
    .filter(
      ({ before: oldId, after: newId }) =>
        semanticNode(oldNodes.get(oldId)) !== semanticNode(newNodes.get(newId)),
    )
    .map((item) => item.after);
  const remap = new Map(matches.map((item) => [item.before, item.after]));
  const oldEdgeGroups = new Map();
  for (const edge of oldGraph.edges) {
    const key = edgeSignature(edge, remap);
    if (!oldEdgeGroups.has(key)) oldEdgeGroups.set(key, []);
    oldEdgeGroups.get(key).push(edge);
  }
  const addedEdges = [];
  for (const edge of newGraph.edges) {
    const existing = oldEdgeGroups.get(edgeSignature(edge));
    if (existing?.length) existing.pop();
    else addedEdges.push(edge);
  }
  const removedEdges = [...oldEdgeGroups.values()].flat();
  const added = [...unmatchedNew];
  const removed = [...unmatchedOld];
  const nodeStatus = Object.fromEntries(
    newGraph.nodes.map((node) => [node.id, "unchanged"]),
  );
  for (const item of matches)
    if (item.ambiguous) nodeStatus[item.after] = "uncertain";
  for (const id of changed) nodeStatus[id] = "changed";
  for (const id of added) nodeStatus[id] = "added";
  const typeChanged = (pattern) =>
    [...addedEdges, ...removedEdges].some((edge) =>
      pattern.test(edge.type ?? ""),
    );
  return {
    matches,
    added,
    removed,
    changed,
    renamed,
    edges: { added: addedEdges, removed: removedEdges },
    nodeStatus,
    sharingChanged:
      typeChanged(/shared|sharing|tied/) ||
      matches.some(
        (m) =>
          stable(oldNodes.get(m.before).sharing) !==
          stable(newNodes.get(m.after).sharing),
      ),
    recurrenceChanged:
      typeChanged(/state|recur|control/) ||
      matches.some(
        (m) =>
          stable(oldNodes.get(m.before).recurrence) !==
          stable(newNodes.get(m.after).recurrence),
      ),
    warnings,
  };
}

function hashString(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

const FLOW_TYPES = new Set([
  "data_flow",
  "dataflow",
  "data",
  "flow",
  "residual",
  "residual_flow",
  "routing",
  "routing_flow",
]);

// Tarjan SCCs let recurrent components occupy one rank without pretending their
// graph is acyclic. State/shared/containment relations do not imply flow order.
function flowRanks(nodes, edges) {
  const adjacency = new Map(nodes.map((node) => [node.id, []]));
  for (const edge of edges)
    if (FLOW_TYPES.has(edge.type))
      adjacency.get(String(edge.source)).push(String(edge.target));
  for (const neighbors of adjacency.values()) neighbors.sort();
  const indices = new Map();
  const low = new Map();
  const stack = [];
  const onStack = new Set();
  const component = new Map();
  const components = [];
  let index = 0;
  function visit(id) {
    indices.set(id, index);
    low.set(id, index);
    index += 1;
    stack.push(id);
    onStack.add(id);
    for (const next of adjacency.get(id)) {
      if (!indices.has(next)) {
        visit(next);
        low.set(id, Math.min(low.get(id), low.get(next)));
      } else if (onStack.has(next))
        low.set(id, Math.min(low.get(id), indices.get(next)));
    }
    if (low.get(id) === indices.get(id)) {
      const group = [];
      let member;
      do {
        member = stack.pop();
        onStack.delete(member);
        component.set(member, components.length);
        group.push(member);
      } while (member !== id);
      components.push(group);
    }
  }
  for (const id of [...adjacency.keys()].sort())
    if (!indices.has(id)) visit(id);
  const outgoing = components.map(() => new Set());
  const incoming = components.map(() => 0);
  for (const [source, targets] of adjacency) {
    for (const target of targets) {
      const from = component.get(source);
      const to = component.get(target);
      if (from !== to && !outgoing[from].has(to)) {
        outgoing[from].add(to);
        incoming[to] += 1;
      }
    }
  }
  const ranks = components.map(() => 0);
  const queue = incoming.flatMap((degree, group) =>
    degree === 0 ? [group] : [],
  );
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const group = queue[cursor];
    for (const next of outgoing[group]) {
      ranks[next] = Math.max(ranks[next], ranks[group] + 1);
      incoming[next] -= 1;
      if (incoming[next] === 0) queue.push(next);
    }
  }
  return new Map(nodes.map((node) => [node.id, ranks[component.get(node.id)]]));
}

function schematicBand(kind) {
  if (/input|embed|token/.test(kind)) return 0;
  if (/conv|pool|patch|temporal/.test(kind)) return 1;
  if (/attention|recurrent|rnn|lstm|gru|state|routing/.test(kind)) return 2;
  if (/feed|mlp|linear|dense|norm|activation|residual|fusion/.test(kind))
    return 3;
  if (/output|readout|head|classifier/.test(kind)) return 4;
  return 2;
}

/**
 * Coordinates are schematic, not parameter volume or inferred tensor order.
 * Keep prior matched coordinates exactly; newly added nodes use vacant slots.
 */
export function layoutGraph(
  graph,
  { previousLayout = null, diff = null } = {},
) {
  const { nodes, edges } = graphParts(graph);
  const hasFlow = edges.some((edge) => FLOW_TYPES.has(edge.type));
  const ranks = hasFlow ? flowRanks(nodes, edges) : null;
  const previous = previousLayout?.positions ?? {};
  const matchedFrom = new Map(
    list(diff?.matches).map((item) => [item.after, item.before]),
  );
  const positions = Object.create(null);
  const occupied = new Set();
  // A group's shallow vertical offset must not put labels over another group.
  const positionKey = ({ x, z }) => `${x.toFixed(3)}:${z.toFixed(3)}`;
  const sorted = [...nodes].sort((a, b) => a.id.localeCompare(b.id));

  for (const node of sorted) {
    const oldId = matchedFrom.get(node.id) ?? node.id;
    const position =
      previous instanceof Map
        ? previous.get(oldId)
        : OWN(previous, oldId)
          ? previous[oldId]
          : null;
    if (
      position &&
      ["x", "y", "z"].every((axis) => Number.isFinite(position[axis]))
    ) {
      positions[node.id] = { x: position.x, y: position.y, z: position.z };
      occupied.add(positionKey(position));
    }
  }
  for (const node of sorted) {
    if (OWN(positions, node.id)) continue;
    const rank = hasFlow
      ? ranks.get(node.id)
      : schematicBand(String(node.kind ?? "unknown").toLowerCase());
    const row = hasFlow ? Math.floor(rank / 6) : 0;
    const column = hasFlow ? (row % 2 ? 5 - (rank % 6) : rank % 6) : rank;
    const x = column * 9;
    const y = 0;
    const preferred = hasFlow ? row * 3 : (hashString(node.id) % 5) - 2;
    let slot = preferred;
    let attempt = 0;
    let position = { x, y, z: slot * 3.4 };
    while (occupied.has(positionKey(position))) {
      attempt += 1;
      slot =
        preferred +
        (attempt % 2 ? Math.ceil(attempt / 2) : -Math.ceil(attempt / 2));
      position = { x, y, z: slot * 3.4 };
    }
    positions[node.id] = position;
    occupied.add(positionKey(position));
  }
  const placed = nodes.map((node) => ({ ...node, ...positions[node.id] }));
  const min = {};
  const max = {};
  const center = {};
  const size = {};
  for (const axis of ["x", "y", "z"]) {
    const coordinates = placed.map((node) => node[axis]);
    min[axis] = coordinates.length ? Math.min(...coordinates) - 1 : -1;
    max[axis] = coordinates.length ? Math.max(...coordinates) + 1 : 1;
    center[axis] = (min[axis] + max[axis]) / 2;
    size[axis] = max[axis] - min[axis];
  }
  return {
    nodes: placed,
    positions,
    bounds: { min, max, center, size },
    schematic: !hasFlow,
  };
}
