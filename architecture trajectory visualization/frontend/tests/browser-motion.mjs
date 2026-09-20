/** Focused real-data morph checks against the running Python dashboard. */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import assert from "node:assert/strict";

const exec = promisify(execFile);
const binary = process.env.AGENT_BROWSER || "agent-browser";
const session = `${process.env.BROWSER_SESSION || "rl4rl-motion"}-${process.pid}`;
const base = process.env.DASHBOARD_URL || "http://127.0.0.1:8765";
const output = resolve(
  process.env.BROWSER_OUTPUT || "../../outputs/architecture-viewer",
);
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
const evaluate = async (js) => (await browser("eval", js)).result;
const wait = async (js) => browser("wait", "--fn", js);
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
const settled =
  "window.architectureReplayState?.graphOccurrence===window.architectureReplayState?.occurrence && window.architectureReplayState?.nodeCount>0 && !window.architectureReplayState?.pending && !window.architectureReplayMotion?.active";

try {
  await mkdir(output, { recursive: true });
  await browser("open", `${base}/architectures`);
  await browser("set", "viewport", "1440", "900");
  await wait(settled);
  await browser("select", "#mode-select", "all");
  await browser("click", "#start-button");
  await wait(`window.architectureReplayState?.proposal===0 && (${settled})`);
  await check(
    "Real Fashion seed morph keeps matched objects and camera while edges stay attached",
    async () => {
      const result = await evaluate(`(async()=>{
      const before=window.architectureReplayMotion;
      const slider=document.getElementById('timeline-slider');
      slider.value='1';slider.dispatchEvent(new Event('input',{bubbles:true}));
      const start=performance.now(),samples=[];
      await new Promise((resolve,reject)=>{const frame=()=>{
        const state=window.architectureReplayState,motion=window.architectureReplayMotion;
        if(state.proposal===1&&state.graphOccurrence===state.occurrence){samples.push(motion);if(!motion.active&&samples.length>1)return resolve();}
        if(performance.now()-start>10000)return reject(Error('Morph did not settle'));
        requestAnimationFrame(frame);
      };requestAnimationFrame(frame)});
      const after=samples.at(-1),oldUIDs=new Set(before.nodes.map(n=>n.uid));
      const firstByUID=new Map(samples[0].nodes.map(n=>[n.uid,n]));
      return {task:window.architectureReplayState.campaign,frames:samples.length,activeFrames:samples.filter(s=>s.active).length,persistent:after.nodes.filter(n=>oldUIDs.has(n.uid)).length,movingNodes:after.nodes.filter(n=>{const p=firstByUID.get(n.uid);return p&&JSON.stringify([p.position,p.scale,p.opacity])!==JSON.stringify([n.position,n.scale,n.opacity])}).length,maxAttachmentError:Math.max(...samples.map(s=>s.attachmentError)),cameraBefore:before.camera,cameraAfter:after.camera,retiringAfter:after.retiring,settled:!after.active};
    })()`);
      assert.match(result.task, /fashion/);
      assert.ok(result.persistent >= 5);
      assert.ok(result.activeFrames >= 3);
      assert.ok(result.movingNodes > 0);
      assert.ok(result.maxAttachmentError < 1e-4);
      assert.deepEqual(result.cameraBefore, result.cameraAfter);
      assert.equal(result.retiringAfter, 0);
      assert.equal(result.settled, true);
      return result;
    },
  );
  await check(
    "Rapid retargeting settles on the newest occurrence without moving the camera",
    async () => {
      const result = await evaluate(`(async()=>{
      const before=window.architectureReplayMotion.camera,slider=document.getElementById('timeline-slider');
      for(const i of [2,3,6,4,8,11,12]){slider.value=String(i);slider.dispatchEvent(new Event('input',{bubbles:true}));await new Promise(r=>setTimeout(r,55));}
      const start=performance.now();await new Promise((resolve,reject)=>{const frame=()=>{const s=window.architectureReplayState,m=window.architectureReplayMotion;if(s.proposal===12&&s.graphOccurrence===s.occurrence&&!s.pending&&!m.active)return resolve();if(performance.now()-start>10000)return reject(Error('Retarget did not settle'));requestAnimationFrame(frame)};frame()});
      const state=window.architectureReplayState,motion=window.architectureReplayMotion;
      return{proposal:state.proposal,sameOccurrence:state.graphOccurrence===state.occurrence,pending:state.pending,requests:state.snapshotRequests,cameraBefore:before,cameraAfter:motion.camera,retiring:motion.retiring,attachmentError:motion.attachmentError};
    })()`);
      assert.equal(result.proposal, 12);
      assert.equal(result.sameOccurrence, true);
      assert.equal(result.pending, false);
      assert.ok(result.requests <= 4);
      assert.deepEqual(result.cameraBefore, result.cameraAfter);
      assert.equal(result.retiring, 0);
      assert.ok(result.attachmentError < 1e-4);
      return result;
    },
  );
  await check(
    "A delayed snapshot preserves the previous scene with an explicit pending caption",
    async () => {
      const result = await evaluate(`(async()=>{
      const previous=window.architectureReplayState.graphOccurrence,oldFetch=window.fetch;
      window.fetch=async(...args)=>{if(String(args[0]).includes('/api/architectures/snapshot'))await new Promise(r=>setTimeout(r,900));return oldFetch(...args)};
      try{const slider=document.getElementById('timeline-slider');slider.value='88';slider.dispatchEvent(new Event('input',{bubbles:true}));await new Promise(r=>setTimeout(r,80));
      return{previous,state:window.architectureReplayState,caption:document.getElementById('representation').textContent,canvasVisibility:getComputedStyle(document.getElementById('canvas-mount')).visibility,canvasCount:document.querySelectorAll('#canvas-mount canvas').length,captureDisabled:document.getElementById('capture-button').disabled,listInert:document.getElementById('component-list').inert,visibleObjects:window.architectureReplayMotion.nodes.length};}finally{window.fetch=oldFetch;}
    })()`);
      assert.equal(result.state.pending, true);
      assert.equal(result.state.displayedOccurrence, result.previous);
      assert.equal(result.state.graphOccurrence, undefined);
      assert.match(
        result.caption,
        /Loading proposal 88.*previous architecture remains visible/,
      );
      assert.equal(result.canvasVisibility, "visible");
      assert.equal(result.canvasCount, 1);
      assert.equal(result.captureDisabled, true);
      assert.equal(result.listInert, true);
      assert.ok(result.visibleObjects > 0);
      await wait(
        `window.architectureReplayState?.proposal===88 && (${settled})`,
      );
      return result;
    },
  );
  await check(
    "Reduced motion applies the new architecture with no intermediate animation",
    async () => {
      await browser("set", "media", "dark", "reduced-motion");
      const result = await evaluate(`(async()=>{
      const slider=document.getElementById('timeline-slider');slider.value='89';slider.dispatchEvent(new Event('input',{bubbles:true}));
      const start=performance.now();return new Promise((resolve,reject)=>{const frame=()=>{const s=window.architectureReplayState,m=window.architectureReplayMotion;if(s.proposal===89&&s.graphOccurrence===s.occurrence)return resolve({reduced:matchMedia('(prefers-reduced-motion: reduce)').matches,active:m.active,retiring:m.retiring,proposal:s.proposal,attachmentError:m.attachmentError});if(performance.now()-start>10000)return reject(Error('Reduced motion did not load'));requestAnimationFrame(frame)};requestAnimationFrame(frame)});
    })()`);
      assert.equal(result.reduced, true);
      assert.equal(result.active, false);
      assert.equal(result.retiring, 0);
      assert.equal(result.proposal, 89);
      assert.ok(result.attachmentError < 1e-4);
      return result;
    },
  );
  const errors = await browser("errors");
  assert.equal(errors.errors?.length || 0, 0, JSON.stringify(errors.errors));
  results.push({ name: "Browser errors", details: errors });
  await writeFile(
    `${output}/browser-motion-verification.json`,
    JSON.stringify(results, null, 2) + "\n",
  );
  console.log(
    `Saved verification to ${output}/browser-motion-verification.json`,
  );
} finally {
  await browser("close").catch(() => {});
}
