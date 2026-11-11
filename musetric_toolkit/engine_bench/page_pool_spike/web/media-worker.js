import { Q, SR, PAGE, PACKET, PRE_ROLL, AHEAD_FRAMES, BEHIND_FRAMES, PIN_MAX_FRAMES, PAGES_PER_STEM, STEMS, AAC_CONFIG, STEM_URLS, wrapFrame } from './common.js';
import { indexFromBuffer } from './mp4index.js';

const now = () => performance.now();
let dp = null, main = null;
const stems = [];
const free = [];
const mirror = [];
const state = { gen: 0, cur: 0, tempo: 1, loop: null, playing: false };
const stats = { decoded: 0, decodeMs: [], decodeFetchMs: [], posted: 0, retsRx: 0, evictsTx: 0, anchorsRx: 0, staleDropped: 0, pagesStaleAtWorker: 0, idleNoVictim: 0, poolMin: 1e9, seekToFirstPostMs: [] };
let seekT0 = 0, seekFirst = false;
let nPagesAll = 0;
let running = false, again = false;
let poolSize = 0;
let paused = false;

const pageOf = (f) => Math.floor(f / PAGE);

const wantedList = () => {
  const out = []; const seen = new Set();
  const add = (f) => { if (f < 0) return; const p = pageOf(f); if (p >= nPagesAll || seen.has(p)) return; seen.add(p); out.push(p); };
  const loop = state.loop;
  const map = (f) => (loop && f >= loop.to ? wrapFrame(loop.from, loop.to, f) : f);
  const cur = state.cur;
  for (let f = cur - 10000; f < cur + 14000; f += 8192) add(map(f));
  add(map(cur + 13999));
  for (let f = cur; f < cur + AHEAD_FRAMES * Math.max(1, 1); f += 8192) add(map(f));
  if (loop) {
    const len = loop.to - loop.from;
    if (len <= PIN_MAX_FRAMES) { for (let f = loop.from; f < loop.to; f += PAGE) add(f); add(loop.to - 1); }
    else { for (let f = loop.from; f < loop.from + AHEAD_FRAMES; f += PAGE) add(f); for (let f = loop.to - AHEAD_FRAMES; f < loop.to; f += PAGE) add(f); add(loop.to - 1); }
  }
  for (let f = cur; f > cur - BEHIND_FRAMES; f -= 8192) add(f);
  return out.slice(0, PAGES_PER_STEM - 2);
};

const makeDecoder = (st) => {
  st.ctx = null;
  st.dec = new AudioDecoder({
    output: (ad) => {
      const c = st.ctx;
      if (!c) { ad.close(); return; }
      const n = ad.numberOfFrames, from = c.seen; c.seen += n;
      if (from >= c.skip) {
        const at = from - c.skip;
        if (at + n <= PAGE) { ad.copyTo(c.l.subarray(at, at + n), { planeIndex: 0, format: 'f32-planar' }); ad.copyTo(c.r.subarray(at, at + n), { planeIndex: 1, format: 'f32-planar' }); c.got += n; }
      }
      ad.close();
    },
    error: (e) => { st.err = e; },
  });
};

const decodePage = async (si, k, buf) => {
  const st = stems[si];
  const l = new Float32Array(buf, 0, PAGE), r = new Float32Array(buf, PAGE * 4, PAGE);
  const first = k * (PAGE / PACKET);
  const start = Math.max(0, first - PRE_ROLL);
  const end = Math.min(first + PAGE / PACKET, st.nPackets);
  if (first >= st.nPackets) { l.fill(0); r.fill(0); return 0; }
  const t0 = now();
  const rs = st.index[start].offset, re = st.index[end - 1].offset + st.index[end - 1].size;
  const resp = await fetch(st.url, { headers: { range: `bytes=${rs}-${re - 1}` } });
  const bytes = await resp.arrayBuffer();
  const t1 = now();
  st.err = null;
  st.dec.reset(); st.dec.configure(AAC_CONFIG);
  const c = st.ctx = { l, r, seen: 0, skip: (first - start) * PACKET, got: 0 };
  for (let i = start; i < end; i++) {
    st.dec.decode(new EncodedAudioChunk({ type: 'key', timestamp: Math.round(i * PACKET * 1e6 / SR), data: new Uint8Array(bytes, st.index[i].offset - rs, st.index[i].size) }));
  }
  await st.dec.flush();
  st.ctx = null;
  if (st.err) throw st.err;
  if (c.got < PAGE) { l.fill(0, c.got); r.fill(0, c.got); }
  stats.decoded++; stats.decodeMs.push(now() - t0); stats.decodeFetchMs.push(t1 - t0);
  return c.got;
};

const step = async () => {
  const want = wantedList();
  const wantSet = new Set(want);
  const gen0 = state.gen;
  const curPage = pageOf(state.cur);
  for (const k of want) {
    const missing = []; let busy = false;
    for (let s = 0; s < STEMS; s++) { const m = mirror[s].get(k); if (!m) missing.push(s); else if (m.evicting) busy = true; }
    if (busy || !missing.length) continue;
    if (free.length < missing.length) {
      let pending = 0; const cand = [];
      for (let s = 0; s < STEMS; s++) for (const [p, m] of mirror[s]) { if (m.evicting) pending++; else if (!wantSet.has(p)) cand.push([s, p]); }
      const need = missing.length - free.length - pending;
      if (need <= 0) return false;
      if (!cand.length) { stats.idleNoVictim++; return false; }
      cand.sort((a, b) => Math.abs(b[1] - curPage) - Math.abs(a[1] - curPage));
      const items = [];
      for (const [s, p] of cand.slice(0, Math.max(need, 3))) { mirror[s].get(p).evicting = true; items.push(s, p); }
      stats.evictsTx += items.length / 2;
      dp.postMessage({ t: 'evict', items });
      return false;
    }
    const bufs = missing.map(() => free.pop());
    stats.poolMin = Math.min(stats.poolMin, free.length);
    try { await Promise.all(missing.map((s, i) => decodePage(s, k, bufs[i]))); } catch (e) { for (const b of bufs) free.push(b); main.postMessage({ t: 'error', message: String(e && e.stack || e) }); return false; }
    if (state.gen !== gen0) { stats.staleDropped += bufs.length; for (const b of bufs) free.push(b); return true; }
    missing.forEach((s, i) => {
      mirror[s].set(k, { gen: gen0, evicting: false });
      dp.postMessage({ t: 'page', stem: s, page: k, gen: gen0, frames: Math.min(PAGE, stems[s].frames - k * PAGE), buf: bufs[i] }, [bufs[i]]);
      stats.posted++;
    });
    if (!seekFirst) { seekFirst = true; stats.seekToFirstPostMs.push(now() - seekT0); }
    return true;
  }
  return false;
};

const kick = async () => {
  if (paused) return;
  if (running) { again = true; return; }
  running = true;
  try { do { again = false; while (!paused && await step()) { } } while (again && !paused); } catch (e) { main.postMessage({ t: 'error', message: String(e && e.stack || e) }); } finally { running = false; }
};

const onDirect = (m) => {
  if (m.t === 'anchor') {
    stats.anchorsRx++;
    if (m.gen === state.gen) { state.cur = m.song; state.tempo = m.tempo; state.playing = m.motion !== 0; }
    kick();
  } else if (m.t === 'ret') {
    stats.retsRx++;
    for (const it of m.items) {
      free.push(it.buf);
      const e = mirror[it.stem].get(it.page);
      if (e && e.gen === it.gen) mirror[it.stem].delete(it.page);
      if (m.reason === 'stale') stats.pagesStaleAtWorker++;
    }
    for (let i = 0; i < m.refused.length; i += 2) { const e = mirror[m.refused[i]].get(m.refused[i + 1]); if (e) e.evicting = false; }
    kick();
  } else if (m.t === 'benchDone') {
    benchResolve?.(m);
  }
};

let benchResolve = null;
const bench = async (nMsgs, batches) => {
  const bufs = Array.from({ length: nMsgs }, () => new ArrayBuffer(PAGE * 8));
  let pool = bufs; const res = [];
  for (let b = 0; b < batches; b++) {
    const done = new Promise((r) => { benchResolve = r; });
    for (let i = 0; i < nMsgs; i++) { const buf = pool[i]; dp.postMessage({ t: 'bench', i, n: nMsgs, buf }, [buf]); }
    const m = await done; pool = m.bufs; res.push({ spanMs: m.last - m.first, count: m.count });
  }
  benchResolve = null;
  return res;
};

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.t === 'init') {
      main = self; dp = m.port; dp.onmessage = (ev) => onDirect(ev.data);
      for (const url of STEM_URLS) {
        const buf = await (await fetch(url)).arrayBuffer();
        const { packets } = indexFromBuffer(buf);
        const st = { url, index: packets, nPackets: packets.length, frames: packets.length * PACKET, nPages: Math.ceil(packets.length * PACKET / PAGE) };
        makeDecoder(st); st.dec.configure(AAC_CONFIG);
        stems.push(st); mirror.push(new Map());
      }
      nPagesAll = Math.max(...stems.map((s) => s.nPages));
      poolSize = PAGES_PER_STEM * STEMS;
      for (let i = 0; i < poolSize; i++) free.push(new ArrayBuffer(PAGE * 8));
      postMessage({ t: 'ready', frames: stems.map((s) => s.frames), poolSize, poolBytes: poolSize * PAGE * 8 });
      kick();
    } else if (m.t === 'seek') {
      state.gen = m.gen; state.cur = m.frame; seekT0 = now(); seekFirst = false;
      kick();
    } else if (m.t === 'loop') {
      state.gen = m.gen; state.loop = m.on ? { from: m.from, to: m.to } : null; kick();
    } else if (m.t === 'tempo') { state.tempo = m.value; }
    else if (m.t === 'pause') { paused = m.on; if (!paused) kick(); }
    else if (m.t === 'stats') {
      const resident = mirror.map((mm) => mm.size);
      postMessage({ t: 'stats', stats: { ...stats }, free: free.length, resident, state: { ...state, loop: state.loop } });
    } else if (m.t === 'resetStats') { stats.decoded = 0; stats.decodeMs = []; stats.decodeFetchMs = []; stats.posted = 0; stats.retsRx = 0; stats.evictsTx = 0; stats.anchorsRx = 0; stats.staleDropped = 0; stats.pagesStaleAtWorker = 0; stats.idleNoVictim = 0; stats.poolMin = 1e9; stats.seekToFirstPostMs = []; }
    else if (m.t === 'bench') { const r = await bench(m.n, m.batches); postMessage({ t: 'bench', id: m.id, res: r }); }
  } catch (err) {
    postMessage({ t: 'error', message: String(err && err.stack || err) });
  }
};
