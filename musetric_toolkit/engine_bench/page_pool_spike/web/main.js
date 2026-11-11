import { SR, Q, PAGE, STEM_URLS } from './common.js';

const T = (window.T = {});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ctx = null, node = null, worker = null, ref = null, gainNode = null, stressWorkers = [], jankTimer = null;
let seq = 0, gen = 0, frames = [0, 0, 0];
const pendingAck = new Map(), pendingSound = new Map(), pendingMsg = new Map();
let lastAnchor = null; const events = []; let anchorsRx = 0;
let rng = 12345; const rnd = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 4294967296; };
T.seedRng = (s) => { rng = s >>> 0; };

const once = (target, type, pred) => new Promise((resolve) => { const h = (e) => { if (!pred || pred(e.data)) { target.removeEventListener(type, h); resolve(e.data); } }; target.addEventListener(type, h); });
let reqId = 0;
const ask = (w, msg, tr = []) => new Promise((resolve, reject) => { const id = ++reqId; const h = (e) => { if (e.data.id === id) { w.removeEventListener('message', h); e.data.t === 'error' ? reject(new Error(e.data.message)) : resolve(e.data); } }; w.addEventListener('message', h); w.postMessage({ ...msg, id }, tr); });
const nodeAsk = (type, extra = {}, reply = type) => new Promise((resolve) => { const h = (e) => { if (e.data.t === reply) { node.port.removeEventListener('message', h); resolve(e.data); } }; node.port.addEventListener('message', h); node.port.postMessage({ t: type, ...extra }); });
const workerAsk = (type, extra = {}) => new Promise((resolve) => { const h = (e) => { if (e.data.t === type) { worker.removeEventListener('message', h); resolve(e.data); } }; worker.addEventListener('message', h); worker.postMessage({ t: type, ...extra }); });

const wireNode = () => {
  node.port.onmessage = (e) => {
    const m = e.data;
    if (m.t === 'ack') { pendingAck.get(m.seq)?.(m); pendingAck.delete(m.seq); }
    else if (m.t === 'sound') { m.recvMs = performance.now(); pendingSound.get(m.seq)?.(m); pendingSound.delete(m.seq); }
    else if (m.t === 'anchor') { lastAnchor = m; anchorsRx++; }
    else if (m.t === 'state') events.push({ ...m, ms: performance.now() });
    else if (m.t === 'error') { console.error('WORKLET ERROR ' + m.message); events.push({ ...m, ms: performance.now() }); }
    else pendingMsg.get(m.t)?.(m);
  };
  node.port.addEventListener?.('message', () => {});
};
const nodeWait = (t) => new Promise((resolve) => { pendingMsg.set(t, (m) => { pendingMsg.delete(t); resolve(m); }); });

T.env = async () => {
  const c = new AudioContext();
  const r = { ua: navigator.userAgent, sampleRate: c.sampleRate, baseLatency: c.baseLatency, outputLatency: c.outputLatency, state: c.state, hasPlaybackStats: 'playbackStats' in c, hasWorklet: !!c.audioWorklet, hasAudioDecoder: typeof AudioDecoder !== 'undefined', visibility: document.visibilityState, coi: self.crossOriginIsolated,
    hasSAB: typeof SharedArrayBuffer !== 'undefined', cores: navigator.hardwareConcurrency };
  if (c.playbackStats) { try { r.playbackStatsKeys = Object.keys(c.playbackStats.toJSON ? c.playbackStats.toJSON() : {}); r.playbackStatsProto = Object.getOwnPropertyNames(Object.getPrototypeOf(c.playbackStats)); } catch (e) { r.playbackStatsErr = String(e); } }
  try { const a = await navigator.gpu?.requestAdapter?.({ powerPreference: 'high-performance' }); r.gpu = a ? { vendor: a.info?.vendor, arch: a.info?.architecture, desc: a.info?.description, device: a.info?.device } : null; } catch (e) { r.gpuErr = String(e); }
  await c.close();
  return r;
};

T.setup = async ({ kind = 'pages', stretchers = 1, stagger = true, latencyHint = 'interactive', bug = '' } = {}) => {
  await T.teardown();
  seq = 0; gen = 0; lastAnchor = null; events.length = 0; anchorsRx = 0;
  ctx = new AudioContext({ latencyHint, sampleRate: SR });
  await ctx.resume();
  gainNode = ctx.createGain(); gainNode.gain.value = 0; gainNode.connect(ctx.destination);
  if (kind === 'whole') {
    ref = new Worker('/web/ref-worker.js', { type: 'module' });
    await ctx.audioWorklet.addModule('/web/whole-processor.js');
    node = new AudioWorkletNode(ctx, 'whole-processor', { numberOfInputs: 0, outputChannelCount: [2] }); node.connect(gainNode); wireNode();
    const t0 = performance.now();
    const w = await ask(ref, { t: 'wholeDecodeOnly' });
    frames = w.stems.map((s) => s[0].length);
    const ready = nodeWait('ready');
    node.port.postMessage({ t: 'stems', stems: w.stems }, w.stems.flatMap((s) => [s[0].buffer, s[1].buffer]));
    await ready;
    ref.terminate(); ref = null;
    return { kind, frames, decodeMs: w.ms, setupMs: performance.now() - t0 };
  }
  worker = new Worker('/web/media-worker.js', { type: 'module' });
  const mc = new MessageChannel();
  const workerReady = once(worker, 'message', (m) => m.t === 'ready');
  worker.postMessage({ t: 'init', port: mc.port1 }, [mc.port1]);
  const wr = await workerReady; frames = wr.frames;
  await ctx.audioWorklet.addModule('/web/page-processor.js');
  node = new AudioWorkletNode(ctx, 'page-processor', { numberOfInputs: 0, outputChannelCount: [2], processorOptions: { nStems: 3, nFrames: frames } });
  node.connect(gainNode); wireNode();
  const text = await (await fetch('/vendor/SignalsmithStretch.mjs')).text();
  const prefix = text.slice(0, text.indexOf('function registerWorkletProcessor'));
  const b64 = /data:application\/octet-stream;base64,([A-Za-z0-9+/=]+)/.exec(prefix)[1];
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const wasmModule = await WebAssembly.compile(bytes);
  const ready = nodeWait('ready');
  let cloned = true;
  try { node.port.postMessage({ t: 'init', port: mc.port2, wasmSource: prefix, wasmModule }, [mc.port2]); } catch (e) { cloned = String(e); }
  const rd = await ready;
  if (stretchers !== 1 || !stagger) await T.cmd('stretchers', { count: stretchers, stagger });
  if (bug) node.port.postMessage({ t: 'bug', name: bug });
  return { kind, frames, poolBytes: wr.poolBytes, poolSize: wr.poolSize, wasm: rd, moduleCloned: cloned };
};

T.teardown = async () => {
  if (jankTimer) { clearInterval(jankTimer); jankTimer = null; }
  for (const w of stressWorkers) w.terminate(); stressWorkers = [];
  if (worker) { worker.terminate(); worker = null; }
  if (ref) { ref.terminate(); ref = null; }
  if (node) { node.disconnect(); node = null; }
  if (ctx) { await ctx.close(); ctx = null; }
};

T.cmd = (t, data = {}) => { const s = ++seq; const p = new Promise((r) => pendingAck.set(s, r)); node.port.postMessage({ t, seq: s, ...data }); return p; };
T.play = () => T.cmd('play');
T.stop = () => T.cmd('stop');
T.tempo = (v) => { if (worker) worker.postMessage({ t: 'tempo', value: v }); return T.cmd('tempo', { value: v }); };
T.setLoop = (on, from = 0, to = 0) => { gen++; if (worker) worker.postMessage({ t: 'loop', on, from, to, gen }); return T.cmd('loop', { on, from, to, gen }); };
T.seek = (frame) => {
  const s = ++seq; gen++;
  const ack = new Promise((r) => pendingAck.set(s, r)); const sound = new Promise((r) => pendingSound.set(s, r));
  const t0 = performance.now();
  if (worker) worker.postMessage({ t: 'seek', frame, gen });
  node.port.postMessage({ t: 'seek', seq: s, frame, gen });
  return { ack, sound, t0, seq: s };
};
T.state = () => ({ lastAnchor, anchorsRx, events: events.slice(-20), frames, gen });
T.getFrames = () => frames;

const stats = async () => {
  const n = await nodeAsk('stats');
  const w = worker ? await workerAsk('stats') : null;
  let ps = null; try { ps = ctx.playbackStats ? ctx.playbackStats.toJSON() : null; } catch { }
  return { node: n, worker: w, playbackStats: ps, ctxTime: ctx.currentTime, perf: performance.now() };
};
T.stats = stats;

const loadRef = async () => { if (!ref) { ref = new Worker('/web/ref-worker.js', { type: 'module' }); } return ask(ref, { t: 'load' }); };
T.loadRef = loadRef;
T.capture = async (nq, script, { controls = false, stretchInput = false } = {}) => {
  await loadRef();
  const armed = nodeWait('armed'); node.port.postMessage({ t: 'capture', nq, input: stretchInput }); await armed;
  await script();
  const dumpP = nodeWait('dump'); node.port.postMessage({ t: 'dump' }); const d = await dumpP;
  const tr = [d.l.buffer, d.r.buffer, d.pos.buffer, d.state.buffer, d.fade.buffer, d.loopFrom.buffer, d.loopTo.buffer, d.seekAt.buffer, d.frame.buffer];
  if (d.inCap) tr.push(d.inCap.buffer, d.inN.buffer, d.inHo.buffer, d.inOn.buffer);
  const res = await ask(ref, { t: 'compare', cap: d, controls, stretchInput, stretchers: T.nStretch ?? 1 }, tr);
  return { quanta: d.n, ...res.out };
};
const waitResident = async (ms = 3000) => { const t = performance.now(); while (performance.now() - t < ms) { const s = await nodeAsk('stats'); if (s.motion !== 2) return; await sleep(20); } };
T.scenario = async (name, { controls = false, bug = '' } = {}) => {
  await T.setup({ kind: 'pages', bug });
  const total = frames[0];
  let script, nq = 7500;
  if (name === 'plain') script = async () => { T.seek(100); await T.play(); await sleep(16000); };
  else if (name === 'seeks') {
    script = async () => {
      T.seek(0); await T.play(); await sleep(1500);
      const targets = [];
      for (let k = 1; k <= 6; k++) targets.push(k * 32768 - 300 - Math.floor(rnd() * 100));
      targets.push(0, total - 3 * SR);
      for (let k = 0; k < 6; k++) targets.push(Math.floor(rnd() * (total - 2 * SR)));
      for (const t of targets) { T.seek(t); await sleep(1100); }
      for (let k = 0; k < 6; k++) { const a = T.state().lastAnchor?.song ?? 0; T.seek(Math.max(0, a + (k % 2 ? 40000 : -30000))); await sleep(450); }
    };
    nq = 9500;
  } else if (name === 'loop') {
    script = async () => { const from = 5 * SR + 1234, to = 9 * SR + 4321; await T.setLoop(true, from, to); T.seek(from - Math.round(0.3 * SR)); await T.play(); await sleep(15000); };
  } else if (name === 'loopBoundary') {
    script = async () => { const from = 3 * PAGE - 77, to = 6 * PAGE + 55; await T.setLoop(true, from, to); T.seek(from - 3000); await T.play(); await sleep(15000); };
  } else throw new Error('scenario ' + name);
  const out = await T.capture(nq, script, { controls });
  const st = await stats();
  await T.teardown();
  return { name, bug, ...out, nodeStats: st.node.st, workerStats: summarizeWorker(st.worker) };
};

T.stretchScenario = async (name, tempo, stretchers = 1) => {
  await T.setup({ kind: 'pages', stretchers }); T.nStretch = stretchers;
  await T.tempo(tempo);
  const total = frames[0];
  let script;
  if (name === 'plain') script = async () => { T.seek(1000); await T.play(); await sleep(12000); };
  else if (name === 'loop') script = async () => { const from = 5 * SR + 1234, to = 9 * SR + 4321; await T.setLoop(true, from, to); T.seek(from - SR); await T.play(); await sleep(14000); };
  else if (name === 'seeks') script = async () => { T.seek(0); await T.play(); await sleep(2000); for (let k = 0; k < 8; k++) { T.seek(Math.floor(rnd() * (total - 3 * SR))); await sleep(1500); } };
  const out = await T.capture(8000, script, { stretchInput: true });
  const st = await stats(); await T.teardown();
  return { name, tempo, stretchers, quanta: out.quanta, main: { rolling: out.main.rolling, covered: out.main.covered }, input: out.input, output: out.output, nodeStats: st.node.st };
};

const summarizeWorker = (w) => {
  if (!w) return null; const s = w.stats; const q = (a, p) => { if (!a.length) return null; const b = [...a].sort((x, y) => x - y); return +b[Math.min(b.length - 1, Math.floor(p * b.length))].toFixed(2); };
  return { decoded: s.decoded, decodeMs: { median: q(s.decodeMs, 0.5), p95: q(s.decodeMs, 0.95), max: q(s.decodeMs, 1) }, fetchMs: { median: q(s.decodeFetchMs, 0.5), p95: q(s.decodeFetchMs, 0.95) }, posted: s.posted, retsRx: s.retsRx, evictsTx: s.evictsTx,
    staleDropped: s.staleDropped, pagesStaleAtWorker: s.pagesStaleAtWorker, idleNoVictim: s.idleNoVictim, poolMin: s.poolMin, free: w.free, resident: w.resident, seekToFirstPostMs: { median: q(s.seekToFirstPostMs, 0.5), p95: q(s.seekToFirstPostMs, 0.95), n: s.seekToFirstPostMs.length } };
};

const pct = (a, p) => { const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
const dist = (a) => (a.length ? { n: a.length, min: +Math.min(...a).toFixed(2), p50: +pct(a, 0.5).toFixed(2), p90: +pct(a, 0.9).toFixed(2), p95: +pct(a, 0.95).toFixed(2), p99: +pct(a, 0.99).toFixed(2), max: +Math.max(...a).toFixed(2) } : null);
T.seekLatency = async ({ kind = 'pages', mode = 'cold', n = 200, tempo = 1, stretchers = 1 } = {}) => {
  const su = await T.setup({ kind, stretchers });
  if (tempo !== 1) await T.tempo(tempo);
  const total = frames[0];
  T.seek(0); await T.play(); await sleep(2000);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const cur = T.state().lastAnchor?.song ?? 0;
    let target;
    if (mode === 'cold') { do { target = Math.floor(rnd() * (total - 4 * SR)); } while (Math.abs(target - cur) < 14 * SR); }
    else target = Math.min(total - 2 * SR, Math.max(0, cur + Math.floor((rnd() * 9 - 3) * SR)));
    const s = T.seek(target);
    const r = await Promise.race([s.sound, sleep(4000).then(() => null)]);
    if (r) rows.push({ target, latFrames: r.soundFrame - r.handlerFrame, latMs: (r.soundFrame - r.handlerFrame) / SR * 1000, wallMs: r.recvMs - s.t0, warm: r.warm });
    else rows.push({ target, timeout: true });
    await sleep(mode === 'cold' ? 700 : 300);
  }
  const ok = rows.filter((r) => !r.timeout);
  const st = await stats();
  await T.teardown();
  return { kind, mode, tempo, stretchers, setup: su, timeouts: rows.length - ok.length, warmFlagged: ok.filter((r) => r.warm).length, latMs: dist(ok.map((r) => r.latMs)), wallMs: dist(ok.map((r) => r.wallMs)),
    latMsColdFlag: dist(ok.filter((r) => !r.warm).map((r) => r.latMs)), latMsWarmFlag: dist(ok.filter((r) => r.warm).map((r) => r.latMs)), wallMsColdFlag: dist(ok.filter((r) => !r.warm).map((r) => r.wallMs)), wallMsWarmFlag: dist(ok.filter((r) => r.warm).map((r) => r.wallMs)),
    worker: kind === 'pages' ? summarizeWorker(st.worker) : null, rows };
};

const blockMain = (ms, every) => { jankTimer = setInterval(() => { const e = performance.now() + ms; while (performance.now() < e); }, every); };
const stressUp = (n) => { const src = 'let x=0;for(;;){x+=Math.sqrt(x+1);if(x>1e12)x=0;}'; const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' })); for (let i = 0; i < n; i++) stressWorkers.push(new Worker(url)); };
const diff = (a, b) => { if (!a || !b) return null; const o = {}; for (const k of Object.keys(b)) if (typeof b[k] === 'number' && typeof a[k] === 'number') o[k] = +(b[k] - a[k]).toFixed(4); return o; };
T.underrun = async ({ name, tempo = 1, stretchers = 1, loop = false, seekEveryMs = 0, seekMode = 'cold', jank = false, stress = 0, durationMs = 20000, warmupMs = 3000, stagger = true, latencyHint = 'interactive' }) => {
  const su = await T.setup({ kind: 'pages', stretchers, stagger, latencyHint });
  if (tempo !== 1) await T.tempo(tempo);
  const total = frames[0];
  if (loop) { const from = 20 * SR + 777, to = 24 * SR + 777; await T.setLoop(true, from, to); T.seek(from - SR / 2); } else T.seek(10 * SR);
  await T.play();
  await sleep(warmupMs);
  if (stress) stressUp(stress);
  if (jank) blockMain(200, 1000);
  const s0 = await stats(); const ps0 = s0.playbackStats;
  const t0 = performance.now(); let nSeeks = 0;
  while (performance.now() - t0 < durationMs) {
    if (seekEveryMs) {
      const cur = T.state().lastAnchor?.song ?? 0;
      const target = seekMode === 'cold' ? Math.floor(rnd() * (total - 4 * SR)) : Math.min(total - 2 * SR, Math.max(0, cur + Math.floor((rnd() * 6 - 2) * SR)));
      T.seek(target); nSeeks++; await sleep(seekEveryMs);
    } else await sleep(250);
  }
  const s1 = await stats();
  const ctxDur = s1.ctxTime - s0.ctxTime;
  await T.teardown();
  const n0 = s0.node.st, n1 = s1.node.st; const d = {}; for (const k of Object.keys(n1)) d[k] = +(n1[k] - n0[k]).toFixed(3);
  return { name, tempo, stretchers, loop, seekEveryMs, seekMode, jank, stress, stagger, latencyHint, nSeeks, wallMs: performance.now() - t0, ctxSeconds: ctxDur, outputLatency: null, playbackStatsStart: ps0, playbackStatsEnd: s1.playbackStats, playbackStatsDelta: diff(ps0, s1.playbackStats),
    spike: d, quantaExpected: Math.round(ctxDur * SR / Q), worker: summarizeWorker(s1.worker), setupStretch: su.wasm, heapMoved: n1.heapMoved };
};

T.messageCost = async ({ playMs = 20000 } = {}) => {
  await T.setup({ kind: 'pages' });
  T.seek(0); await T.play(); await sleep(3000);
  node.port.postMessage({ t: 'resetStats' });
  worker.postMessage({ t: 'resetStats' });
  const s0 = await stats(); const t0 = performance.now();
  await sleep(playMs);
  const s1 = await stats(); const secs = (performance.now() - t0) / 1000;
  const n = s1.node.st;

  const res = { secs, messages: { pages: n.pages, rets: n.rets, evicts: n.evicts, anchorsSent: n.anchors, anchorToMainAndWorker: n.anchors * 2, perSecond: +((n.pages + n.rets + n.anchors * 2) / secs).toFixed(2) },
    handlerMsSum: n.hMs, handlerCount: n.hCount, heavyHandlers: n.hHeavy, procMsSum: n.procMs, quanta: n.quanta, procMeanUs: +(n.procMs / n.quanta * 1000).toFixed(2), procHeavyQuanta: n.procHeavy, bufQuanta: n.bufQuanta, errors: n.errors, workerPosted: summarizeWorker(s1.worker) };
  const b = await ask(worker, { t: 'bench', n: 64, batches: 30 });
  res.bench = { batches: b.res.length, perBatchMs: dist(b.res.map((x) => x.spanMs)), perMessageMsMean: +(b.res.reduce((a, x) => a + x.spanMs, 0) / b.res.reduce((a, x) => a + x.count, 0)).toFixed(4) };
  const s2 = await stats();
  res.afterBench = { bufQuanta: s2.node.st.bufQuanta - s1.node.st.bufQuanta, procHeavy: s2.node.st.procHeavy - s1.node.st.procHeavy, benchMsgs: s2.node.st.benchMsgs - s1.node.st.benchMsgs, quanta: s2.node.st.quanta - s1.node.st.quanta };
  await T.teardown();
  return res;
};

T.memory = async ({ kind, playMs = 12000 }) => {
  if (kind === 'idle') { await sleep(playMs); return { kind }; }
  const su = await T.setup({ kind });
  T.seek(0); await T.play();
  if (kind === 'pages') { for (let i = 0; i < 8; i++) { await sleep(1000); T.seek(Math.floor(rnd() * (frames[0] - 4 * SR))); } await sleep(playMs - 8000); }
  else { for (let i = 0; i < 8; i++) { await sleep(1000); T.seek(Math.floor(rnd() * (frames[0] - 4 * SR))); } await sleep(playMs - 8000); }
  const st = kind === 'pages' ? await stats() : null;
  const out = { kind, setup: su, resident: st?.worker?.resident, free: st?.worker?.free };
  await T.teardown();
  return out;
};

window.ready = true;
