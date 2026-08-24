import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the completed trajectory dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>RL4RL Architecture Trajectories<\/title>/i);
  assert.match(html, /Architecture search trajectories/);
  assert.match(html, /24<!-- --> completed runs/);
  assert.match(html, /<span>Proposals<\/span><strong>320<\/strong>/);
  assert.match(html, /Parameters vs\. proposal/);
  assert.match(html, /AutoResearch/);
  assert.match(html, /OpenEvolve/);
  assert.match(html, /RD0/);
  assert.match(html, /RD3/);
  assert.match(html, /og\.png/);
  assert.doesNotMatch(html, /Building your site|react-loading-skeleton/);
});

test("ships the exact 24-cell, 320-proposal matrix", async () => {
  const raw = await readFile(
    new URL("../app/data/trajectories.json", import.meta.url),
    "utf8",
  );
  const payload = JSON.parse(raw);
  assert.equal(payload.runCount, 24);
  assert.equal(payload.proposalCount, 320);
  assert.equal(payload.runs.length, 24);

  const identities = new Set(
    payload.runs.map(
      (run) => `${run.methodCode}:${run.horizon}:${run.condition}`,
    ),
  );
  assert.equal(identities.size, 24);
  for (const method of ["ar", "oe"]) {
    for (const horizon of [8, 12, 20]) {
      for (const condition of ["RD0", "RD1", "RD2", "RD3"]) {
        assert.ok(identities.has(`${method}:${horizon}:${condition}`));
      }
    }
  }
  assert.equal(
    payload.runs.reduce((total, run) => total + run.proposalCount, 0),
    320,
  );
});
