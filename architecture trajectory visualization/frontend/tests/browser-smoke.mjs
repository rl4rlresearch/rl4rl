/** Run against the existing Python dashboard. Requires agent-browser on PATH. */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import assert from "node:assert/strict";
const exec = promisify(execFile),
  binary = process.env.AGENT_BROWSER || "agent-browser";
const session = `${process.env.BROWSER_SESSION || "rl4rl-architecture-e2e"}-${process.pid}`;
const base = process.env.DASHBOARD_URL || "http://127.0.0.1:8765";
const output = resolve(
  process.env.BROWSER_OUTPUT || "../../outputs/architecture-viewer",
);
await mkdir(output, { recursive: true });
const results = [];
async function browser(...args) {
  const { stdout } = await exec(
    binary,
    [
      "--session",
      session,
      "--json",
      ...(process.env.BROWSER_EXECUTABLE
        ? ["--executable-path", process.env.BROWSER_EXECUTABLE]
        : []),
      ...args,
    ],
    { maxBuffer: 8e6, timeout: 45000 },
  );
  const response = JSON.parse(stdout);
  if (!response.success) throw Error(response.error);
  return response.data;
}
async function evaluate(js) {
  return (await browser("eval", js)).result;
}
async function wait(js) {
  await browser("wait", "--fn", js);
}
async function state() {
  return evaluate("window.architectureReplayState");
}
async function check(name, fn) {
  const start = performance.now();
  const details = await fn();
  results.push({
    name,
    passed: true,
    milliseconds: Math.round(performance.now() - start),
    details,
  });
  console.log(`PASS ${name}`);
}
await browser("open", `${base}/architectures`);
await browser("set", "viewport", "1600", "1000");
await wait("window.architectureReplayState?.nodeCount > 0");
await check("Real default data and source-backed graph", async () => {
  const s = await state();
  assert.equal(s.graphOccurrence, s.occurrence);
  assert.ok(s.run.includes("fashion"));
  assert.ok(s.length > 5);
  return s;
});
await check("Rapid scrub cancels stale graph responses", async () => {
  await browser("select", "#mode-select", "all");
  const result = await evaluate(
    `new Promise((resolve,reject)=>{const slider=document.getElementById('timeline-slider'),start=performance.now();for(let i=0;i<40;i++){slider.value=String(i);slider.dispatchEvent(new Event('input',{bubbles:true}));}const check=()=>{const s=window.architectureReplayState;if(s.proposal===39&&s.graphOccurrence===s.occurrence)resolve({milliseconds:performance.now()-start,state:s,caption:document.getElementById('change-caption').textContent});else if(performance.now()-start>15000)reject(Error('Scrub did not settle'));else requestAnimationFrame(check)};check()})`,
  );
  assert.equal(result.state.proposal, 39);
  assert.ok(result.caption.startsWith(`vs #${result.state.comparison}`));
  return result;
});
await check("Displayed metrics and parent match saved occurrence", async () => {
  const result = await evaluate(
    `(async()=>{const s=window.architectureReplayState,p=new URLSearchParams({campaign:s.campaign,run:s.run});const run=await fetch('/api/architectures/run?'+p).then(r=>r.json());const o=run.occurrences.find(o=>o.id===s.occurrence);return {sameMetrics:JSON.stringify(o.metrics)===JSON.stringify(s.metrics),parent:run.occurrences.find(p=>p.id===o.parent_occurrence_ids[0])?.proposal,comparison:s.comparison,retention:o.retention_decision}})()`,
  );
  assert.equal(result.sameMetrics, true);
  assert.equal(result.parent, result.comparison);
  return result;
});
await check("Rejected endpoint is selectable as its own lineage", async () => {
  const id = await evaluate(
    `(async()=>{const s=window.architectureReplayState;const run=await fetch('/api/architectures/run?'+new URLSearchParams({campaign:s.campaign,run:s.run})).then(r=>r.json());return run.occurrences.find(o=>o.status==='rejected').id})()`,
  );
  await browser("select", "#endpoint-select", id);
  await wait(
    `window.architectureReplayState?.graphOccurrence===${JSON.stringify(id)}`,
  );
  const s = await state();
  assert.equal(s.mode, "lineage");
  assert.equal(s.occurrence, id);
  return s;
});
await check("URL restores selected occurrence and replay mode", async () => {
  const before = await state();
  await browser("reload");
  await wait("window.architectureReplayState?.nodeCount > 0");
  const after = await state();
  assert.equal(after.occurrence, before.occurrence);
  assert.equal(after.mode, before.mode);
  return { proposal: after.proposal, mode: after.mode };
});
await check("Playback stops at last opportunity", async () => {
  await browser("select", "#mode-select", "all");
  await browser("click", "#end-button");
  await wait("window.architectureReplayState?.proposal===200");
  await browser("click", "#previous-button");
  await browser("select", "#speed-select", "4");
  await browser("click", "#play-button");
  await wait(
    "window.architectureReplayState?.proposal===200 && !window.architectureReplayState?.playing",
  );
  const s = await state();
  assert.equal(s.graphOccurrence, s.occurrence);
  return { proposal: s.proposal, playing: s.playing };
});
await check("Every prepared task opens and renders its real seed", async () => {
  const campaigns = await evaluate(
    `Array.from(document.getElementById('campaign-select').options).filter(o=>!o.textContent.includes('prepare data')).map(o=>({key:o.value,label:o.textContent}))`,
  );
  const summary = [];
  for (const c of campaigns) {
    await browser("select", "#campaign-select", c.key);
    await wait(
      `window.architectureReplayState?.campaign===${JSON.stringify(c.key)} && !!window.architectureReplayState?.run && window.architectureReplayState?.length>0`,
    );
    await browser("select", "#mode-select", "all");
    await browser("click", "#start-button");
    await wait(
      `window.architectureReplayState?.proposal===0 && window.architectureReplayState?.nodeCount>0 && window.architectureReplayState?.graphOccurrence===window.architectureReplayState?.occurrence`,
    );
    const s = await state();
    summary.push({
      campaign: c.label,
      run: s.run,
      occurrences: s.length,
      seedNodes: s.nodeCount,
    });
  }
  return summary;
});
await check(
  "Rapid scope changes cannot install an older catalog or run",
  async () => {
    await evaluate(
      `(()=>{const c=document.getElementById('scope-select');for(let i=0;i<10;i++){c.value=i%2?'dashboard':'archive';c.dispatchEvent(new Event('change',{bubbles:true}));}})()`,
    );
    await wait(
      'window.architectureReplayState?.scope==="dashboard" && window.architectureReplayState?.dataScope==="dashboard" && window.architectureReplayState?.nodeCount>0',
    );
    const s = await state();
    assert.equal(s.scope, s.dataScope);
    assert.equal(s.graphOccurrence, s.occurrence);
    return { scope: s.scope, dataScope: s.dataScope, proposal: s.proposal };
  },
);
await check(
  "Presentation PNG preserves provenance and viewport controls",
  async () => {
    await browser("click", "#presentation-button");
    await browser("click", "#fit-button");
    await browser(
      "download",
      "#capture-button",
      `${output}/architecture-capture.png`,
    );
    await browser("screenshot", `${output}/presentation.png`);
    const body = await evaluate('document.body.classList.contains("present")');
    assert.equal(body, true);
    return { capture: `${output}/architecture-capture.png` };
  },
);
await check(
  "Detailed desktop labels remain readable without overlap",
  async () => {
    await browser("open", `${base}/architectures`);
    await browser("set", "viewport", "1440", "900");
    await wait(
      "window.architectureReplayState?.nodeCount>0 && document.getElementById('canvas-mount').dataset.transition!=='active'",
    );
    await browser("select", "#metric-select", "validation_accuracy");
    const details = await evaluate(
      `(()=>{const labels=[...document.querySelectorAll('.node-label')].filter(e=>getComputedStyle(e).visibility==='visible'&&e.getBoundingClientRect().width>0).map(e=>e.getBoundingClientRect().toJSON());let overlaps=0;for(let i=0;i<labels.length;i++)for(let j=i+1;j<labels.length;j++){const a=labels[i],b=labels[j];if(a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top)overlaps++;}return{labels:labels.length,overlaps,graphNodes:window.architectureReplayState.nodeCount}})()`,
    );
    assert.ok(details.labels >= 3);
    assert.equal(details.overlaps, 0);
    await browser("screenshot", `${output}/desktop-final.png`);
    return details;
  },
);
await check(
  "Narrow layout retains its timeline and a readable grouped model",
  async () => {
    await browser("set", "viewport", "390", "844");
    await wait(
      "(()=>{const m=document.getElementById('canvas-mount'),bounds=m.getBoundingClientRect(),labels=[...document.querySelectorAll('.node-label')];return labels.length===3&&m.dataset.transition!=='active'&&labels.every(e=>{const r=e.getBoundingClientRect();return getComputedStyle(e).visibility==='visible'&&r.left>=bounds.left&&r.right<=bounds.right&&r.top>=bounds.top&&r.bottom<=bounds.bottom})})()",
    );
    const details = await evaluate(
      `(()=>{const slider=document.getElementById('timeline-slider').getBoundingClientRect(),mount=document.getElementById('canvas-mount').getBoundingClientRect(),labels=[...document.querySelectorAll('.node-label')];return{controlsAboveChart:document.getElementById('selection-label').getBoundingClientRect().bottom<=document.getElementById('metric-chart').getBoundingClientRect().top,allBlocksVisible:labels.every(e=>{const r=e.getBoundingClientRect();return getComputedStyle(e).visibility==='visible'&&r.left>=mount.left&&r.right<=mount.right&&r.top>=mount.top&&r.bottom<=mount.bottom}),overflow:document.documentElement.scrollWidth>innerWidth,sliderTop:slider.top,sliderBottom:slider.bottom,height:innerHeight,sourceNodes:window.architectureReplayState.nodeCount,renderedNodes:document.querySelectorAll('.node-label').length}})()`,
    );
    assert.equal(details.overflow, false);
    assert.equal(details.allBlocksVisible, true);
    assert.equal(details.controlsAboveChart, true);
    assert.ok(details.sliderTop > 0 && details.sliderBottom <= details.height);
    await browser("screenshot", `${output}/mobile-final.png`);
    return details;
  },
);
const errors = await browser("errors");
assert.equal(errors.errors?.length || 0, 0, JSON.stringify(errors.errors));
results.push({ name: "Browser errors", details: errors });
await writeFile(
  `${output}/browser-verification.json`,
  JSON.stringify(results, null, 2) + "\n",
);
console.log(`Saved verification to ${output}/browser-verification.json`);

await browser("close");
