const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const T = (window.T = {});
let ctx, node, gain;
let last = {};
const wait = (type) =>
  new Promise((res) => {
    const h = (e) => {
      if (e.data.t === type) {
        node.port.removeEventListener('message', h);
        res(e.data);
      }
    };
    node.port.addEventListener('message', h);
    node.port.start?.();
  });
T.setup = async (opts = {}) => {
  await T.teardown();
  const o = { ...opts };
  ctx = new AudioContext(o);
  await ctx.resume();
  await ctx.audioWorklet.addModule('/web/probe-processor.js');
  node = new AudioWorkletNode(ctx, 'probe', { numberOfInputs: 0, outputChannelCount: [2] });
  gain = ctx.createGain();
  gain.gain.value = 0;
  node.connect(gain);
  gain.connect(ctx.destination);
  node.port.start?.();
  await sleep(300);
  return T.info();
};
T.info = () => ({
  state: ctx.state,
  sampleRate: ctx.sampleRate,
  baseLatency: ctx.baseLatency,
  outputLatency: ctx.outputLatency,
  renderQuantumSize: ctx.renderQuantumSize,
  sinkId: typeof ctx.sinkId === 'object' ? JSON.stringify(ctx.sinkId) : ctx.sinkId,
  hasPlaybackStats: !!ctx.playbackStats,
  currentTime: ctx.currentTime,
});
T.teardown = async () => {
  if (node) {
    node.disconnect();
    node = null;
  }
  if (ctx) {
    try {
      await ctx.close();
    } catch {}
    ctx = null;
  }
};
T.record = async (ms) => {
  node.port.postMessage({ t: 'start' });
  await sleep(ms);
  node.port.postMessage({ t: 'stop' });
  const p = wait('dump');
  node.port.postMessage({ t: 'dump' });
  const d = await p;
  return d;
};
const analyze = (d) => {
  const n = d.n,
    c = d.cap;
  const ms = (i) => c[3 * i],
    fr = (i) => c[3 * i + 1];
  const bursts = [];
  let start = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || ms(i) - ms(i - 1) >= 1) {
      bursts.push({ t: ms(start), q: i - start });
      start = i;
    }
  }
  let gaps = 0,
    nonContig = 0;
  for (let i = 1; i < n; i++) if (fr(i) - fr(i - 1) !== d.len) nonContig++;
  const sizes = bursts.map((b) => b.q);
  const dt = [];
  for (let i = 1; i < bursts.length; i++) dt.push(bursts[i].t - bursts[i - 1].t);
  const hist = (a) => {
    const h = {};
    for (const v of a) h[v] = (h[v] || 0) + 1;
    return h;
  };
  const pct = (a, p) => {
    const s = [...a].sort((x, y) => x - y);
    return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : null;
  };
  const wall = (ms(n - 1) - ms(0)) / 1e3;
  return {
    quanta: n,
    wallSec: wall,
    quantaPerSec: n / wall,
    expectedQuantaPerSec: d.sr / d.len,
    renderQuantumLen: d.len,
    nonContiguous: nonContig,
    bursts: bursts.length,
    burstSizeHist: hist(sizes),
    burstSizeP50: pct(sizes, 0.5),
    burstSizeP99: pct(sizes, 0.99),
    burstSizeMax: Math.max(...sizes),
    gapMsHist: hist(dt),
    gapMsP50: pct(dt, 0.5),
    gapMsP95: pct(dt, 0.95),
    gapMsP99: pct(dt, 0.99),
    gapMsMax: Math.max(...dt),
    msPerQuantumMean: (wall * 1e3) / n,
  };
};
T.callbacks = async ({ seconds = 10 } = {}) => {
  const d = await T.record(seconds * 1e3);
  return { info: T.info(), ...analyze(d), raw: Array.from(d.cap.slice(0, 3 * 4e3)), sr: d.sr };
};
T.rate = async (seconds) => {
  const a = await countNow();
  const t0 = performance.now();
  await sleep(seconds * 1e3);
  const b = await countNow();
  const wall = (performance.now() - t0) / 1e3;
  return {
    quanta: b.total - a.total,
    wallSec: +wall.toFixed(3),
    quantaPerSec: +((b.total - a.total) / wall).toFixed(1),
    ctxTimeRate: +((b.time - a.time) / wall).toFixed(4),
  };
};
const countNow = async () => {
  const p = wait('count');
  node.port.postMessage({ t: 'count' });
  return p;
};
T.stall = async (ms) => {
  node.port.postMessage({ t: 'stall', ms });
  await sleep(ms + 400);
};
T.suspendResume = async () => {
  const t0 = performance.now();
  await ctx.suspend();
  const t1 = performance.now();
  await ctx.resume();
  return { suspendMs: +(t1 - t0).toFixed(1), resumeMs: +(performance.now() - t1).toFixed(1), state: ctx.state };
};
T.bug560233271 = async ({ stallMs = 1500, baselineSec = 4, afterSec = 6, doSuspend = true } = {}) => {
  const out = { info: T.info() };
  out.baseline = await T.rate(baselineSec);
  if (stallMs) {
    out.stall = { ms: stallMs };
    await T.stall(stallMs);
    out.afterStall = await T.rate(baselineSec);
  }
  if (doSuspend) {
    out.sr = await T.suspendResume();
    await sleep(500);
    out.infoAfter = T.info();
    out.afterResume = await T.rate(afterSec);
    out.afterResume2 = await T.rate(afterSec);
  }
  return out;
};
T.clock = async ({ seconds = 60, stepMs = 100 } = {}) => {
  const rows = [];
  const t0 = performance.now();
  while (performance.now() - t0 < seconds * 1e3) {
    const ts = ctx.getOutputTimestamp();
    rows.push([
      +(performance.now() - t0).toFixed(1),
      ts.contextTime,
      ts.performanceTime,
      ctx.outputLatency,
      ctx.currentTime,
      ctx.baseLatency,
    ]);
    await sleep(stepMs);
  }
  return rows;
};
T.renderSize = async (hint) => {
  const r = { hint };
  try {
    const c = new AudioContext(hint === void 0 ? {} : { renderSizeHint: hint });
    r.created = true;
    r.renderQuantumSize = c.renderQuantumSize;
    r.baseLatency = c.baseLatency;
    r.sampleRate = c.sampleRate;
    await c.resume();
    await c.audioWorklet.addModule('/web/probe-processor.js');
    const n = new AudioWorkletNode(c, 'probe', { numberOfInputs: 0, outputChannelCount: [2] });
    const g = c.createGain();
    g.gain.value = 0;
    n.connect(g);
    g.connect(c.destination);
    n.port.start?.();
    const p = new Promise((res) => {
      n.port.onmessage = (e) => e.data.t === 'count' || res(e.data);
    });
    n.port.postMessage({ t: 'start' });
    await sleep(600);
    n.port.postMessage({ t: 'dump' });
    const d = await p;
    r.workletLen = d.len;
    r.workletQuanta = d.n;
    await c.close();
  } catch (e) {
    r.error = String(e);
  }
  return r;
};
T.stallCheck = async (ms) => {
  const ps0 = ctx.playbackStats?.toJSON?.();
  node.port.postMessage({ t: 'start' });
  await sleep(1500);
  node.port.postMessage({ t: 'stall', ms });
  await sleep(ms + 1500);
  node.port.postMessage({ t: 'stop' });
  const p = wait('dump');
  node.port.postMessage({ t: 'dump' });
  const d = await p;
  let maxGap = 0;
  for (let i = 1; i < d.n; i++) maxGap = Math.max(maxGap, d.cap[3 * i] - d.cap[3 * i - 3]);
  return { maxGapMs: maxGap, ps0, ps1: ctx.playbackStats?.toJSON?.(), ctxTime: ctx.currentTime };
};
window.ready = true;
