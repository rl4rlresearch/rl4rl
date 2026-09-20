/** Optional local demo: agent-browser + ffmpeg with MJPEG decode and libvpx. */
import { execFile, spawn } from "node:child_process";
import { once } from "node:events";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
  copyFile,
} from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { promisify } from "node:util";
const exec = promisify(execFile);
const binary = process.env.AGENT_BROWSER || "agent-browser";
const ffmpeg = process.env.FFMPEG || "ffmpeg";
const session = `rl4rl-demo-${process.pid}`;
const base = process.env.DASHBOARD_URL || "http://127.0.0.1:8765";
const output = fileURLToPath(
  new URL("../../../outputs/architecture-viewer/", import.meta.url),
);
const movie = resolve(output, "architecture-replay-demo.webm");
const fps = 12;
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
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
    { timeout: 45000, maxBuffer: 8e6 },
  );
  const result = JSON.parse(stdout);
  if (!result.success) throw Error(result.error);
  return result.data;
}
async function state() {
  return (await browser("eval", "window.architectureReplayState")).result;
}
async function encode(frames) {
  // JPEG avoids the minimal Playwright ffmpeg build's missing PNG/fps support.
  // Repeated frames preserve measured capture timing instead of fabricating it.
  const child = spawn(
    ffmpeg,
    [
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-f",
      "image2pipe",
      "-vcodec",
      "mjpeg",
      "-framerate",
      String(fps),
      "-i",
      "pipe:0",
      "-c:v",
      "libvpx",
      "-pix_fmt",
      "yuv420p",
      "-b:v",
      "3M",
      "-deadline",
      "realtime",
      "-cpu-used",
      "4",
      movie,
    ],
    { stdio: ["pipe", "ignore", "pipe"] },
  );
  let errors = "";
  child.stderr.on("data", (chunk) => {
    errors += chunk;
  });
  child.stdin.on("error", () => {});
  const completion = once(child, "close");
  let written = 0;
  for (let i = 0; i < frames.length; i++) {
    const end =
      i + 1 < frames.length ? frames[i + 1].time : frames[i].time + 1000;
    const count = Math.max(
      1,
      Math.round(((end - frames[i].time) / 1000) * fps),
    );
    const jpeg = await readFile(frames[i].path);
    for (let n = 0; n < count; n++) {
      if (child.exitCode !== null) throw Error(`ffmpeg failed: ${errors}`);
      if (!child.stdin.write(jpeg)) await once(child.stdin, "drain");
      written++;
    }
  }
  child.stdin.end();
  const [code] = await completion;
  if (code !== 0) throw Error(`ffmpeg failed: ${errors}`);
  return { encodedFrames: written, durationSeconds: written / fps };
}
await mkdir(output, { recursive: true });
const temporary = await mkdtemp(resolve(output, ".demo-frames-"));
try {
  await exec(ffmpeg, ["-version"], { timeout: 10000 });
  await browser(
    "open",
    `${base}/architectures?campaign=openevolve_v21_fashion_mnist&mode=all`,
  );
  await browser("set", "viewport", "1440", "900");
  await browser("wait", "--fn", "window.architectureReplayState?.nodeCount>0");
  await browser("select", "#mode-select", "all");
  await browser(
    "eval",
    "(()=>{const s=document.getElementById('timeline-slider');s.value='11';s.dispatchEvent(new Event('input',{bubbles:true}));})()",
  );
  await browser(
    "wait",
    "--fn",
    "window.architectureReplayState?.proposal===11 && window.architectureReplayState?.nodeCount>0",
  );
  await browser("select", "#metric-select", "validation_accuracy");
  await browser("select", "#speed-select", "2");
  await browser("click", "#presentation-button");
  await browser("click", "#fit-button");
  await browser("click", "#start-button");
  await browser(
    "wait",
    "--fn",
    "window.architectureReplayState?.proposal===0 && window.architectureReplayState?.nodeCount>0",
  );
  const initial = await state();
  if (initial.campaign !== "openevolve_v21_fashion_mnist")
    throw Error("Wrong campaign selected");
  // Let the camera warm-up transition finish before the opening seed frame.
  await sleep(800);
  await browser(
    "eval",
    "window.__demoStop=setInterval(()=>{const s=window.architectureReplayState;if(s?.proposal>=15&&s.playing){document.getElementById('play-button').click();clearInterval(window.__demoStop);}},30)",
  );
  const frames = [];
  const started = Date.now();
  let latest = initial;
  do {
    const path = resolve(
      temporary,
      `${String(frames.length).padStart(4, "0")}.jpg`,
    );
    await browser(
      "screenshot",
      path,
      "--screenshot-format",
      "jpeg",
      "--screenshot-quality",
      "90",
    );
    frames.push({
      path,
      time: Date.now() - started,
      proposal: latest.proposal,
    });
    if (frames.length === 1) {
      await copyFile(path, resolve(output, "demo-start-frame.jpg"));
      await browser("click", "#play-button");
    }
    latest = await state();
    if (Date.now() - started > 90000)
      throw Error("Demo playback exceeded 90 seconds");
    await sleep(120);
  } while (
    latest.proposal < 15 ||
    latest.playing ||
    latest.graphOccurrence !== latest.occurrence
  );
  await sleep(800);
  const end = resolve(temporary, "end.jpg");
  await browser(
    "screenshot",
    end,
    "--screenshot-format",
    "jpeg",
    "--screenshot-quality",
    "90",
  );
  frames.push({
    path: end,
    time: Date.now() - started,
    proposal: latest.proposal,
  });
  await browser("screenshot", resolve(output, "demo-end-frame.png"));
  await copyFile(end, resolve(output, "demo-end-frame.jpg"));
  const encoded = await encode(frames);
  await browser("open", pathToFileURL(movie).href);
  await browser(
    "wait",
    "--fn",
    "document.querySelector('video')?.readyState>=2",
  );
  const playback = (
    await browser(
      "eval",
      "(()=>{const v=document.querySelector('video');return {width:v.videoWidth,height:v.videoHeight,duration:v.duration,error:v.error?.message||null};})()",
    )
  ).result;
  if (
    playback.error ||
    playback.width !== 1440 ||
    playback.height !== 900 ||
    !(playback.duration > 1)
  )
    throw Error("Browser could not decode the captured movie");
  await writeFile(
    resolve(output, "architecture-replay-demo.json"),
    JSON.stringify(
      {
        initial,
        final: latest,
        capturedFrames: frames.length,
        fps,
        ...encoded,
        playback,
        method:
          "Real dashboard playback captured as JPEG frames; measured timing preserved at 12 fps.",
      },
      null,
      2,
    ) + "\n",
  );
  console.log(JSON.stringify({ movie, ...encoded, playback }));
} finally {
  await browser("close").catch(() => {});
  await rm(temporary, { recursive: true, force: true });
}
