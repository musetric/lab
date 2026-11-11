import * as MB from '/mediabunny.mjs';
import { compare, firstNonZero, runControls } from '/web/compare.js';
import { indexFromBuffer, indexFromRanges } from '/web/mp4index.js';

const FRAME = 1024, PAGE_PACKETS = 32, PAGE_FRAMES = FRAME * PAGE_PACKETS, SR = 48000;
const now = () => performance.now();
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const stat = (a) => ({ n: a.length, min: +Math.min(...a).toFixed(3), median: +pct(a, 0.5).toFixed(3), p95: +pct(a, 0.95).toFixed(3), max: +Math.max(...a).toFixed(3), mean: +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(3) });
const bufCache = new Map();
const load = async (url) => { if (!bufCache.has(url)) bufCache.set(url, await (await fetch(url)).arrayBuffer()); return bufCache.get(url); };
const hex = async (ab) => [...new Uint8Array(await crypto.subtle.digest('SHA-256', ab))].map((b) => b.toString(16).padStart(2, '0')).join('');
const hashChannels = async (chs) => { const n = chs[0].length; const x = new Float32Array(n * chs.length); chs.forEach((c, i) => x.set(c, i * n)); return hex(x.buffer); };
let seed = 12345; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const fetchRange = async (url, s, e) => { const r = await fetch(url, { headers: { range: `bytes=${s}-${e - 1}` } }); return r.arrayBuffer(); };
const counter = { n: 0, reqs: [] }; const origFetch = self.fetch.bind(self);
const countingFetch = async (...a) => { const r = await origFetch(...a); const h = a[1]?.headers ?? (a[0] instanceof Request ? a[0].headers : undefined); const range = h ? (h.get ? h.get('range') : h.range ?? h.Range) : null; if (!r.body) return r; const reader = r.body.getReader(); const rec = { range, got: 0 }; counter.reqs.push(rec);
  const body = new ReadableStream({ async pull(c) { const { done, value } = await reader.read(); if (done) { c.close(); return; } rec.got += value.length; counter.n += value.length; c.enqueue(value); }, cancel(why) { return reader.cancel(why); } }); return new Response(body, { status: r.status, statusText: r.statusText, headers: r.headers }); };
const stats = async () => (await fetch('/__stats')).json();
const resetStats = () => fetch('/__reset');

const getConfig = async (buf) => {
  const input = new MB.Input({ formats: [MB.MP4], source: new MB.BufferSource(buf) });
  const track = await input.getPrimaryAudioTrack(); const cfg = await track.getDecoderConfig();
  const out = { cfg, channels: await track.getNumberOfChannels(), sampleRate: await track.getSampleRate() }; input.dispose(); return out;
};
const wholeDecode = async (buf) => {
  const input = new MB.Input({ formats: [MB.MP4], source: new MB.BufferSource(buf) });
  try {
    const track = await input.getPrimaryAudioTrack(); const n = await track.getNumberOfChannels(); const cfg = await track.getDecoderConfig();
    const support = await AudioDecoder.isConfigSupported(cfg); if (!support.supported || !support.config) throw new Error('unsupported ' + cfg.codec);
    const sink = new MB.EncodedPacketSink(track);
    const chunks = Array.from({ length: n }, () => []); let frames = 0, err; const sizes = new Map();
    const decoder = new AudioDecoder({ output: (ad) => { sizes.set(ad.numberOfFrames, (sizes.get(ad.numberOfFrames) ?? 0) + 1);
      for (let c = 0; c < n; c++) { const ch = new Float32Array(ad.numberOfFrames); ad.copyTo(ch, { planeIndex: c, format: 'f32-planar' }); chunks[c].push(ch); } frames += ad.numberOfFrames; ad.close(); }, error: (e) => { err = e; } });
    try { decoder.configure(support.config); for await (const p of sink.packets()) decoder.decode(p.toEncodedAudioChunk()); await decoder.flush(); if (err) throw err; } finally { decoder.close(); }
    const channels = chunks.map((cs) => { const r = new Float32Array(frames); let o = 0; for (const c of cs) { r.set(c, o); o += c.length; } return r; });
    return { channels, frames, outputSizes: Object.fromEntries(sizes) };
  } finally { input.dispose(); }
};

class PageDecoder {
  constructor(cfg, frameSamples = FRAME) { this.cfg = cfg; this.fs = frameSamples; this.cur = null; this.err = null; this.make(); }
  make() { this.err = null; this.decoder = new AudioDecoder({ output: (ad) => this.onOut(ad), error: (e) => { this.err = e; } }); this.decoder.configure(this.cfg); }
  onOut(ad) {
    const cur = this.cur; if (!cur) { ad.close(); return; }
    if (cur.firstOut === undefined) cur.firstOut = now() - cur.t0;
    const n = ad.numberOfFrames, ch = [];
    for (let c = 0; c < ad.numberOfChannels; c++) { const a = new Float32Array(n); ad.copyTo(a, { planeIndex: c, format: 'f32-planar' }); ch.push(a); }
    cur.parts.push(ch); cur.frames += n; cur.sizes.add(n); ad.close();
  }
  async run(packets, firstIndex, dropFrames, mode = 'reset', noFlush = false) {
    const tR = now(); let tReset = 0, tConfigure = 0;
    if (mode === 'reset') { this.decoder.reset(); tReset = now() - tR; const t1 = now(); this.decoder.configure(this.cfg); tConfigure = now() - t1; }
    else if (mode === 'new') { try { this.decoder.close(); } catch { } this.make(); tConfigure = now() - tR; }
    const tConf = now() - tR;
    const cur = this.cur = { parts: [], frames: 0, sizes: new Set(), t0: now(), firstOut: undefined };
    for (let i = 0; i < packets.length; i++) this.decoder.decode(new EncodedAudioChunk({ type: 'key', timestamp: Math.round((firstIndex + i) * this.fs * 1e6 / SR), data: packets[i] }));
    const tQueue = now() - cur.t0;
    if (noFlush) { const dl = now() + 1000; while (cur.firstOut === undefined && now() < dl) await new Promise((r) => setTimeout(r, 0)); this.cur = null; return { firstOut: cur.firstOut, tConf, tReset, tConfigure }; }
    await this.decoder.flush(); const tFlush = now() - cur.t0;
    if (this.err) throw this.err;
    const nCh = cur.parts[0]?.length ?? 2; const total = cur.frames; const keep = total - dropFrames;
    const channels = []; for (let c = 0; c < nCh; c++) { const r = new Float32Array(Math.max(0, keep)); let o = -dropFrames; for (const p of cur.parts) { const a = p[c]; if (o + a.length > 0) { const s = Math.max(0, -o); r.set(a.subarray(s), Math.max(0, o)); } o += a.length; } channels.push(r); }
    this.cur = null;
    return { channels, totalFrames: total, outSizes: [...cur.sizes], tConf, tReset, tConfigure, tQueue, tFlush, firstOut: cur.firstOut, tAll: now() - tR };
  }
  close() { try { this.decoder.close(); } catch { } }
}
const slicePackets = (buf, idx, a, b) => { const r = []; for (let i = a; i < b; i++) r.push(new Uint8Array(buf, idx[i].offset, idx[i].size)); return r; };
const pageDecode = (pd, buf, idx, first, count, P, mode, noFlush) => { const start = Math.max(0, first - P); return pd.run(slicePackets(buf, idx, start, first + count), start, (first - start) * FRAME, mode, noFlush); };

const cmds = {};
cmds.env = async () => {
  const buf = await load('/data/chord.mp4'); const { cfg } = await getConfig(buf);
  const r = { ua: navigator.userAgent, crossOriginIsolated: self.crossOriginIsolated, hasAudioDecoder: typeof AudioDecoder !== 'undefined', hasEncodedAudioChunk: typeof EncodedAudioChunk !== 'undefined', hasMeasureMemory: typeof performance.measureUserAgentSpecificMemory === 'function', hasCrypto: !!self.crypto?.subtle, cfg: { codec: cfg.codec, sampleRate: cfg.sampleRate, numberOfChannels: cfg.numberOfChannels, description: cfg.description ? [...(ArrayBuffer.isView(cfg.description) ? new Uint8Array(cfg.description.buffer, cfg.description.byteOffset, cfg.description.byteLength) : new Uint8Array(cfg.description))] : null } };
  if (r.hasAudioDecoder) {
    try { r.supportMp4a = (await AudioDecoder.isConfigSupported(cfg)).supported; } catch (e) { r.supportMp4a = 'throws ' + e; }
    try { r.supportMp4aNoDesc = (await AudioDecoder.isConfigSupported({ codec: 'mp4a.40.2', sampleRate: 48000, numberOfChannels: 2 })).supported; } catch (e) { r.supportMp4aNoDesc = 'throws ' + e; }
    try { r.supportFlacNoDesc = (await AudioDecoder.isConfigSupported({ codec: 'flac', sampleRate: 48000, numberOfChannels: 2 })).supported; } catch (e) { r.supportFlacNoDesc = 'throws ' + e; }
  }
  return r;
};
cmds.controls = async () => runControls();

cmds.indexcheck = async ({ name }) => {
  const buf = await load(`/data/${name}.mp4`); const idx = indexFromBuffer(buf).packets; const input = new MB.Input({ formats: [MB.MP4], source: new MB.BufferSource(buf) });
  const track = await input.getPrimaryAudioTrack(); const sink = new MB.EncodedPacketSink(track); let i = 0, mism = 0, tsMism = 0, maxTsErr = 0; const flags = new Set();
  for await (const p of sink.packets()) { const e = idx[i]; const d = new Uint8Array(buf, e.offset, e.size); flags.add(p.type); if (p.data.byteLength !== e.size || p.data.some((v, k) => v !== d[k])) mism++; const te = Math.abs(p.timestamp - i * FRAME / SR); maxTsErr = Math.max(maxTsErr, te); if (te > 1e-6) tsMism++; i++; }
  input.dispose(); return { name, mediabunnyPackets: i, ownPackets: idx.length, byteMismatch: mism, tsMismatch: tsMism, maxTsErrSec: maxTsErr, types: [...flags] };
};

const readSource = async (name) => { const wav = new DataView(await load(`/data/${name}.wav`)); const n = (wav.byteLength - 44) / 6; const L = new Float32Array(n), R = new Float32Array(n);
  const s24 = (o) => ((wav.getUint8(o) | (wav.getUint8(o + 1) << 8) | (wav.getUint8(o + 2) << 16)) << 8 >> 8);
  for (let i = 0; i < n; i++) { L[i] = s24(44 + i * 6) / 8388608; R[i] = s24(44 + i * 6 + 3) / 8388608; } return [L, R]; };

cmds.correctness = async ({ name, Ps = [0, 1, 2, 3], withSource = true }) => {
  const buf = await load(`/data/${name}.mp4`); const idx = indexFromBuffer(buf).packets; const { cfg } = await getConfig(buf);
  const tW = now(); const whole = await wholeDecode(buf); const wholeMs = now() - tW;
  const res = { name, packets: idx.length, wholeFrames: whole.frames, wholeFirstNonZero: firstNonZero(whole.channels), wholeOutputSizes: whole.outputSizes, wholeMs, wholeSha256: await hashChannels(whole.channels), perP: {}, windows: {} };
  const nPages = Math.ceil(whole.frames / PAGE_FRAMES); res.pages = nPages;
  const pd = new PageDecoder(cfg);
  for (const P of Ps) {
    const order = shuffle([...Array(nPages).keys()]); const rows = new Array(nPages); const concat = [new Float32Array(whole.frames), new Float32Array(whole.frames)];
    for (const k of order) {
      const a = k * PAGE_FRAMES, b = Math.min(whole.frames, a + PAGE_FRAMES);
      const o = await pageDecode(pd, buf, idx, k * PAGE_PACKETS, (b - a) / FRAME, P, 'reset');
      const c = compare(whole.channels.map((x) => x.subarray(a, b)), o.channels);
      rows[k] = { k, ok: c.ok, maxDiff: c.maxDiff, diffCount: c.diffCount, bitDiff: c.bitDiffCount, firstNonZero: firstNonZero(o.channels), frames: o.channels[0].length, firstDiff: c.firstDiff, outSizes: o.outSizes.join(','), reasons: c.ok ? undefined : c.reasons };
      for (let ch = 0; ch < 2; ch++) concat[ch].set(o.channels[ch], a);
    }
    const cAll = compare(whole.channels, concat);
    const bad = rows.filter((r) => !r.ok);
    res.perP[P] = { pagesOk: rows.length - bad.length, pages: nPages, maxDiff: Math.max(...rows.map((r) => r.maxDiff)), totalDiffSamples: rows.reduce((s, r) => s + r.diffCount, 0), page0FirstNonZero: rows[0].firstNonZero, concatOk: cAll.ok, concatFrames: concat[0].length, concatDiff: cAll.diffCount, concatMaxDiff: cAll.maxDiff, badFirstDiffHistogram: Object.fromEntries(bad.reduce((m, r) => m.set(r.firstDiff, (m.get(r.firstDiff) ?? 0) + 1), new Map())), badPages: bad.slice(0, 4).map((r) => ({ k: r.k, diffCount: r.diffCount, maxDiff: r.maxDiff, firstDiff: r.firstDiff })), firstNonZeroPerPage: rows.map((r) => r.firstNonZero).slice(0, 8) };
    if (P > 0) {
      const wins = []; for (let t = 0; t < 60; t++) { const len = 1 + Math.floor(rnd() * 64); const first = Math.floor(rnd() * (idx.length - len)); wins.push([first, len]); }
      let nbad = 0, diff = 0, maxd = 0; for (const [first, len] of wins) { const o = await pageDecode(pd, buf, idx, first, len, P, 'reset'); const c = compare(whole.channels.map((x) => x.subarray(first * FRAME, (first + len) * FRAME)), o.channels); if (!c.ok) { nbad++; diff += c.diffCount; maxd = Math.max(maxd, c.maxDiff); } }
      res.windows[P] = { windows: wins.length, bad: nbad, diffSamples: diff, maxDiff: maxd };
    }
  }
  for (const mode of ['none', 'new']) {
    let nbad = 0, diff = 0, maxd = 0; const firstDiffs = [];
    for (let k = 0; k < nPages; k++) { const a = k * PAGE_FRAMES, b = Math.min(whole.frames, a + PAGE_FRAMES); const o = await pageDecode(pd, buf, idx, k * PAGE_PACKETS, (b - a) / FRAME, 0, mode); const c = compare(whole.channels.map((x) => x.subarray(a, b)), o.channels); if (!c.ok) { nbad++; diff += c.diffCount; maxd = Math.max(maxd, c.maxDiff); if (firstDiffs.length < 3) firstDiffs.push(c.firstDiff); } }
    (res.seqP0 ??= {})[mode] = { pages: nPages, bad: nbad, diffSamples: diff, maxDiff: maxd, firstDiffs };
  }
  pd.close();
  if (withSource) {
    const src = (await readSource(name))[0]; const n = src.length; const a = whole.channels[0]; const win = 8192, s0 = Math.min(n - win - 5000, 30000); let best = -1, bestE = Infinity;
    for (let L = 0; L <= 3000; L++) { let e = 0; for (let i = s0; i < s0 + win; i++) { const d = src[i] - a[i + L]; e += d * d; } if (e < bestE) { bestE = e; best = L; } }
    let sig = 0; for (let i = s0; i < s0 + win; i++) sig += src[i] * src[i];
    const thr = (x) => { for (let i = 0; i < x.length; i++) if (Math.abs(x[i]) > 1e-3) return i; return -1; };
    res.delay = { bestLagFrames: best, snrDbAtBestLag: +(10 * Math.log10(sig / bestE)).toFixed(1), sourceFirstAbove1e3: thr(src), decodedFirstAbove1e3: thr(a), sourceFrames: n, decodedFrames: whole.frames, decodedMinusSource: whole.frames - n };
  }
  return res;
};

cmds.benchDecoder = async ({ name = 'mix240', pagesN = 40 }) => {
  const buf = await load(`/data/${name}.mp4`); const idx = indexFromBuffer(buf).packets; const { cfg } = await getConfig(buf); const nPages = Math.floor(idx.length / PAGE_PACKETS) - 1;
  const out = { name };
  const tNew = [], tCfg = [], tRst = [], tRstCfg = [];
  for (let i = 0; i < 40; i++) { const t0 = now(); const d = new AudioDecoder({ output: (a) => a.close(), error: () => {} }); const t1 = now(); d.configure(cfg); const t2 = now(); tNew.push(t1 - t0); tCfg.push(t2 - t1); const t3 = now(); d.reset(); const t4 = now(); d.configure(cfg); const t5 = now(); tRst.push(t4 - t3); tRstCfg.push(t5 - t4); d.close(); }
  out.construct = stat(tNew); out.configureCall = stat(tCfg); out.resetCall = stat(tRst); out.reconfigureCall = stat(tRstCfg);
  const pd = new PageDecoder(cfg);
  const ttfo = [], ttfoNew = [], seekFlush = []; let noFlushNever = 0;
  for (let i = 0; i < pagesN; i++) { const k = 1 + Math.floor(rnd() * (nPages - 1)); const first = k * PAGE_PACKETS;
    const o = await pageDecode(pd, buf, idx, first, 1, 2, 'reset', true); if (o.firstOut === undefined) noFlushNever++; else ttfo.push(o.firstOut);
    const o2 = await pageDecode(pd, buf, idx, first, 1, 2, 'new', true); if (o2.firstOut !== undefined) ttfoNew.push(o2.firstOut);
    const o3 = await pageDecode(pd, buf, idx, first, 1, 2, 'reset'); seekFlush.push(o3.tAll); }
  out.firstOutputAfterSeek_reset_noFlush = ttfo.length ? stat(ttfo) : null; out.firstOutputNeverWithoutFlush = noFlushNever; out.firstOutputAfterSeek_newDecoder_noFlush = ttfoNew.length ? stat(ttfoNew) : null; out.seekThenFlush3packets = stat(seekFlush);
  for (const P of [0, 2]) { const t = [], ts = [], tq = []; for (let i = 0; i < pagesN; i++) { const k = Math.floor(rnd() * nPages); const o = await pageDecode(pd, buf, idx, k * PAGE_PACKETS, PAGE_PACKETS, P, 'reset'); t.push(o.tAll); tq.push(o.tQueue); ts.push((PAGE_PACKETS + P) / (o.tAll / 1000)); }
    out[`page_P${P}_ms`] = stat(t); out[`page_P${P}_packetsPerSec`] = stat(ts); out[`page_P${P}_queueMs`] = stat(tq); }
  const t0 = now(); const w = await wholeDecode(buf); const tw = now() - t0; out.whole = { ms: +tw.toFixed(0), packets: idx.length, packetsPerSec: +(idx.length / (tw / 1000)).toFixed(0), realtimeX: +((w.frames / SR) / (tw / 1000)).toFixed(0) };
  pd.close(); return out;
};

cmds.benchParallel = async ({ names = ['chord', 'sweep', 'clicks'], iterations = 20 }) => {
  const bufs = await Promise.all(names.map((n) => load(`/data/${n}.mp4`))); const idxs = bufs.map((b) => indexFromBuffer(b).packets); const cfg = (await getConfig(bufs[0])).cfg;
  const pds = names.map(() => new PageDecoder(cfg)); const nPages = Math.floor(idxs[0].length / PAGE_PACKETS) - 1; const P = 2;
  const ks = Array.from({ length: iterations }, () => Math.floor(rnd() * nPages)); const seqMs = [], parMs = [];
  for (let r = 0; r < 2; r++) for (const k of ks) {
    let t0 = now(); for (let s = 0; s < names.length; s++) await pageDecode(pds[s], bufs[s], idxs[s], k * PAGE_PACKETS, PAGE_PACKETS, P, 'reset'); const a = now() - t0;
    t0 = now(); await Promise.all(names.map((_, s) => pageDecode(pds[s], bufs[s], idxs[s], k * PAGE_PACKETS, PAGE_PACKETS, P, 'reset'))); const b = now() - t0; if (r) { seqMs.push(a); parMs.push(b); } }
  pds.forEach((p) => p.close()); return { stems: names, iterations, P, sequential3_ms: stat(seqMs), parallel3_oneWorker_ms: stat(parMs) };
};
cmds.loopPages = async ({ name, iterations, P = 2, startAt }) => {
  const buf = await load(`/data/${name}.mp4`); const idx = indexFromBuffer(buf).packets; const cfg = (await getConfig(buf)).cfg; const pd = new PageDecoder(cfg); const nPages = Math.floor(idx.length / PAGE_PACKETS) - 1;
  await pageDecode(pd, buf, idx, 0, PAGE_PACKETS, P, 'reset');
  while (startAt && Date.now() < startAt) await new Promise((r) => setTimeout(r, 0));
  const t0 = now(); for (let i = 0; i < iterations; i++) await pageDecode(pd, buf, idx, (i * 7 % nPages) * PAGE_PACKETS, PAGE_PACKETS, P, 'reset'); const ms = now() - t0; pd.close(); return { name, iterations, ms };
};

cmds.benchSeek = async ({ name = 'mix240' }) => {
  const url = `/data/${name}.mp4`; const buf = await load(url); const idxAll = indexFromBuffer(buf).packets; const dur = idxAll.length * FRAME / SR; const out = { name, durationSec: dur, fileBytes: buf.byteLength };
  const runMb = async (mk, label) => {
    const rows = [];
    for (const f of [0.1, 0.5, 0.9, 0.9, 0.5, 0.1]) { await resetStats(); counter.n = 0; counter.reqs = []; const t0 = now(); const input = new MB.Input({ formats: [MB.MP4], source: mk() }); const track = await input.getPrimaryAudioTrack(); const sink = new MB.EncodedPacketSink(track); const tSetup = now() - t0;
      const t1 = now(); const pk = await sink.getPacket(f * dur - 2 * FRAME / SR); const tGet = now() - t1; let p = pk, n = 0; const t2 = now(); while (p && n < 33) { p = await sink.getNextPacket(p); n++; } const tNext = now() - t2; const s = await stats();
      rows.push({ at: f, setupMs: +tSetup.toFixed(2), getPacketMs: +tGet.toFixed(2), next33Ms: +tNext.toFixed(2), requests: s.requests, requestedBytes: s.bytes, sentBytes: s.sent, consumedBytes: counter.n, reqs: counter.reqs.map((q) => q.range + ' got ' + q.got), ts: pk?.timestamp }); input.dispose(); }
    const input = new MB.Input({ formats: [MB.MP4], source: mk() }); const track = await input.getPrimaryAudioTrack(); const sink = new MB.EncodedPacketSink(track); await sink.getPacket(dur * 0.99); const warm = [];
    await resetStats(); counter.n = 0; counter.reqs = []; for (let i = 0; i < 30; i++) { const t0 = now(); await sink.getPacket(rnd() * dur * 0.95); warm.push(now() - t0); } const s = await stats(); const warmConsumed = counter.n, warmReqs = counter.reqs.length; input.dispose();
    out[label] = { coldRows: rows, warmGetPacketMs: stat(warm), warmRequestsFor30: s.requests, warmBytesFor30: s.sent, warmConsumedBytes: warmConsumed, warmClientRequests: warmReqs };
  };
  await runMb(() => new MB.BufferSource(buf), 'mediabunny_BufferSource');
  self.fetch = countingFetch; try { await runMb(() => new MB.UrlSource(url), 'mediabunny_UrlSource'); } finally { self.fetch = origFetch; }
  const bi = []; for (let i = 0; i < 10; i++) { const t0 = now(); indexFromBuffer(buf); bi.push(now() - t0); } out.ownIndexFromBufferMs = stat(bi);
  await resetStats(); const t0 = now(); const ri = await indexFromRanges((s, e) => fetchRange(url, s, e), buf.byteLength); out.ownIndexFromRanges = { ms: +(now() - t0).toFixed(1), requests: ri.requests, bytes: ri.bytes, fragments: ri.fragments, packets: ri.packets.length };
  out.indexJsonBytes = JSON.stringify(ri.packets.map((p) => p.size)).length; out.indexBytesAsVarint16 = ri.packets.length * 2;
  const idx = ri.packets; const { cfg } = await getConfig(buf); const pd = new PageDecoder(cfg); const nPages = Math.floor(idx.length / PAGE_PACKETS) - 1; const P = 2;
  const fetchPage = async (k) => { const a = Math.max(0, k * PAGE_PACKETS - P), b = k * PAGE_PACKETS + PAGE_PACKETS; const s = idx[a].offset, e = idx[b - 1].offset + idx[b - 1].size; const ab = await fetchRange(url, s, e);
    return { packets: idx.slice(a, b).map((p) => new Uint8Array(ab, p.offset - s, p.size)), start: a }; };
  const tf = [], te = []; for (let i = 0; i < 40; i++) { const k = 1 + Math.floor(rnd() * (nPages - 1)); const t0 = now(); const pg = await fetchPage(k); tf.push(now() - t0); const o = await pd.run(pg.packets, pg.start, P * FRAME, 'reset'); te.push(now() - t0); if (o.channels[0].length !== PAGE_FRAMES) throw new Error('page length ' + o.channels[0].length); }
  out.ownIndex_rangeFetchMs = stat(tf); out.ownIndex_e2ePageMs = stat(te);
  const input = new MB.Input({ formats: [MB.MP4], source: new MB.UrlSource(url) }); const track = await input.getPrimaryAudioTrack(); const sink = new MB.EncodedPacketSink(track); const tm = [];
  for (let i = 0; i < 30; i++) { const k = 1 + Math.floor(rnd() * (nPages - 1)); const t0 = now(); let p = await sink.getPacket((k * PAGE_PACKETS - P) * FRAME / SR + 1e-6); const pk = []; for (let n = 0; p && n < PAGE_PACKETS + P; n++) { pk.push(p.data.slice()); p = await sink.getNextPacket(p); }
    const o = await pd.run(pk, k * PAGE_PACKETS - P, P * FRAME, 'reset'); tm.push(now() - t0); if (o.channels[0].length !== PAGE_FRAMES) throw new Error('mb page length ' + o.channels[0].length); }
  out.mediabunny_e2ePageMs_sameInputWarm = stat(tm); input.dispose(); pd.close(); return out;
};

cmds.uaMemory = async () => { if (typeof performance.measureUserAgentSpecificMemory !== 'function') return null; try { const m = await performance.measureUserAgentSpecificMemory(); return { bytes: m.bytes }; } catch (e) { return 'throws ' + e; } };
cmds.memWhole = async ({ name = 'mix240', stems = 3 }) => {
  const buf = await load(`/data/${name}.mp4`); const t0 = now();
  const res = await Promise.all(Array.from({ length: stems }, () => wholeDecode(buf)));
  const ms = now() - t0; const frames = res[0].frames; let sum = 0; for (const r of res) sum += r.channels[0][frames >> 1];
  const ua = await cmds.uaMemory(); return { ms: +ms.toFixed(0), frames, decodedBytesHeld: res.length * res[0].channels.length * frames * 4, ua, sum };
};
cmds.memPages = async ({ name = 'mix240', stems = 3, ringPages = 5, P = 2, preload = false }) => {
  const url = `/data/${name}.mp4`; const size = Number((await fetch(url, { method: 'HEAD' })).headers.get('content-length')); const idx = (await indexFromRanges((s, e) => fetchRange(url, s, e), size)).packets;
  const cfg = (await getConfig(await fetchRange(url, 0, 4096))).cfg ?? null;
  const nPages = Math.ceil(idx.length / PAGE_PACKETS); const t0 = now(); let sum = 0, frames = 0;
  const one = async () => { const pd = new PageDecoder(cfg); const ring = [];
    for (let k = 0; k < nPages; k++) { const a = Math.max(0, k * PAGE_PACKETS - P), b = Math.min(idx.length, (k + 1) * PAGE_PACKETS); const s = idx[a].offset, e = idx[b - 1].offset + idx[b - 1].size; const ab = await fetchRange(url, s, e);
      const o = await pd.run(idx.slice(a, b).map((p) => new Uint8Array(ab, p.offset - s, p.size)), a, (k * PAGE_PACKETS - a) * FRAME, 'reset'); ring.push(o.channels); if (ring.length > ringPages) ring.shift(); sum += o.channels[0][o.channels[0].length >> 1]; frames += o.channels[0].length; }
    pd.close(); return ring.length; };
  await Promise.all(Array.from({ length: stems }, one)); const ms = now() - t0; const ua = await cmds.uaMemory();
  return { ms: +ms.toFixed(0), pagesPerStem: nPages, framesDecodedTotal: frames, ringBytesHeldPerStem: ringPages * 2 * PAGE_FRAMES * 4, ua, sum };
};
cmds.memIdle = async () => ({ ua: await cmds.uaMemory() });

const crc8 = (b, s, e) => { let c = 0; for (let i = s; i < e; i++) { c ^= b[i]; for (let k = 0; k < 8; k++) c = c & 0x80 ? ((c << 1) ^ 0x07) & 0xff : (c << 1) & 0xff; } return c; };
let crc16t; const crc16 = (b, s, e) => { if (!crc16t) { crc16t = new Uint16Array(256); for (let n = 0; n < 256; n++) { let c = n << 8; for (let k = 0; k < 8; k++) c = c & 0x8000 ? ((c << 1) ^ 0x8005) & 0xffff : (c << 1) & 0xffff; crc16t[n] = c; } } let c = 0; for (let i = s; i < e; i++) c = ((c << 8) & 0xffff) ^ crc16t[((c >> 8) ^ b[i]) & 0xff]; return c; };
const flacHeader = (b, p) => {
  if (b[p] !== 0xff || (b[p + 1] & 0xfe) !== 0xf8) return null; const bs = b[p + 2] >> 4, sr = b[p + 2] & 15; let q = p + 4; const first = b[q]; let extra = 0; let mask = 0x80; if (first >= 0xfc) extra = first >= 0xfe ? 6 : 5; else if (first >= 0xf8) extra = 4; else if (first >= 0xf0) extra = 3; else if (first >= 0xe0) extra = 2; else if (first >= 0xc0) extra = 1;
  let no = extra ? first & (0x3f >> extra) : first; for (let i = 1; i <= extra; i++) no = no * 64 + (b[q + i] & 0x3f); q += extra + 1; let blocksize = [0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768][bs];
  if (bs === 6) { blocksize = b[q] + 1; q += 1; } else if (bs === 7) { blocksize = (b[q] << 8 | b[q + 1]) + 1; q += 2; } if (sr === 12) q += 1; else if (sr === 13 || sr === 14) q += 2;
  if (crc8(b, p, q) !== b[q]) return null; return { len: q + 1 - p, frameNo: no, blocksize }; };
const flacIndex = (u8) => { let p = 4, last = false; while (!last) { last = !!(u8[p] & 0x80); const len = (u8[p + 1] << 16) | (u8[p + 2] << 8) | u8[p + 3]; p += 4 + len; }
  const meta = p; const frames = []; let cur = flacHeader(u8, p); if (!cur || cur.frameNo !== 0) throw new Error('no first frame'); let start = p; let crcBad = 0;
  for (;;) { let q = start + cur.len, nxt = null; const want = cur.frameNo + 1; while (q + 2 < u8.length) { if (u8[q] === 0xff && (u8[q + 1] & 0xfe) === 0xf8) { const h = flacHeader(u8, q); if (h && h.frameNo === want) { nxt = { q, h }; break; } } q++; }
    const end = nxt ? nxt.q : u8.length; if (crc16(u8, start, end - 2) !== ((u8[end - 2] << 8) | u8[end - 1])) crcBad++; frames.push({ offset: start, size: end - start, blocksize: cur.blocksize }); if (!nxt) break; start = nxt.q; cur = nxt.h; }
  return { metaEnd: meta, frames, crcBad }; };
cmds.flac = async ({ name, pagesN = 40 }) => {
  const ab = await load(`/data/${name}.flac`); const u8 = new Uint8Array(ab); const res = { name, bytes: ab.byteLength };
  const fi = flacIndex(u8); res.frames = fi.frames.length; res.crc16Bad = fi.crcBad; res.blocksizes = [...new Set(fi.frames.map((f) => f.blocksize))]; res.metaEnd = fi.metaEnd;
  const streamInfo = u8.slice(8, 8 + 34); const variants = { raw34: streamInfo, fLaC_hdr_streaminfo: u8.slice(0, 8 + 34), allMetadata: u8.slice(0, fi.metaEnd) };
  res.variants = {}; const src = await readSource(name); const BS = 4096;
  let good = null;
  for (const [vn, desc] of Object.entries(variants)) {
    const cfg = { codec: 'flac', sampleRate: 48000, numberOfChannels: 2, description: desc }; const v = {};
    try { v.supported = (await AudioDecoder.isConfigSupported(cfg)).supported; } catch (e) { v.supported = 'throws ' + e.message; }
    if (v.supported === true) { try { const pd = new PageDecoder(cfg, BS); const all = fi.frames.slice(0, 16).map((f) => new Uint8Array(ab, f.offset, f.size)); const o = await pd.run(all, 0, 0, 'reset'); const c = compare(src.map((x) => x.subarray(0, o.channels[0].length)), o.channels); v.first16FramesOk = c.ok; v.reasons = c.reasons; v.outSizes = o.outSizes; v.frames = o.channels[0].length; pd.close(); if (c.ok && !good) good = vn; } catch (e) { v.error = String(e.message ?? e); } }
    res.variants[vn] = v;
  }
  res.usedVariant = good; if (!good) return res;
  const cfg = { codec: 'flac', sampleRate: 48000, numberOfChannels: 2, description: variants[good] }; const pd = new PageDecoder(cfg, BS);
  const FP = 8; const nPages = Math.ceil(fi.frames.length / FP); const order = shuffle([...Array(nPages).keys()]); let bad = 0, diff = 0, maxd = 0, tot = 0; const times = [];
  for (const k of order) { const fr = fi.frames.slice(k * FP, (k + 1) * FP); const a = k * FP * BS; const o = await pd.run(fr.map((f) => new Uint8Array(ab, f.offset, f.size)), k * FP, 0, 'reset'); times.push(o.tAll);
    const c = compare(src.map((x) => x.subarray(a, a + o.channels[0].length)), o.channels); tot += o.channels[0].length; if (!c.ok || a + o.channels[0].length > src[0].length) { bad++; diff += c.diffCount; maxd = Math.max(maxd, c.maxDiff); } }
  res.pagesAll = { pages: nPages, bad, diffSamples: diff, maxDiff: maxd, totalFrames: tot, sourceFrames: src[0].length, lengthMatch: tot === src[0].length, msPerPage: stat(times) };
  const pd2 = new PageDecoder(cfg, BS); const w = await pd2.run(fi.frames.map((f) => new Uint8Array(ab, f.offset, f.size)), 0, 0, 'reset'); res.whole = { ok: compare(src, w.channels).ok, frames: w.channels[0].length, ms: +w.tAll.toFixed(0) }; pd2.close(); pd.close();
  return res;
};

self.onmessage = async (e) => { const { n, cmd, args } = e.data; try { const result = await cmds[cmd](args ?? {}); self.postMessage({ n, result }); } catch (err) { self.postMessage({ n, error: String(err?.stack ?? err) }); } };
