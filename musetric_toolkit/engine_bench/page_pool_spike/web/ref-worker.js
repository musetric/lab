import { Q, PACKET, SR, AAC_CONFIG, STEM_URLS, compareCapture, cloneCap, refMixQuantum } from './common.js';
import { indexFromBuffer } from './mp4index.js';
let whole = null;
const decodeWhole = async (url) => {
  const buf = await (await fetch(url)).arrayBuffer();
  const { packets } = indexFromBuffer(buf);
  const chunks = [[], []]; let frames = 0, err = null;
  const dec = new AudioDecoder({ output: (ad) => { for (let c = 0; c < 2; c++) { const a = new Float32Array(ad.numberOfFrames); ad.copyTo(a, { planeIndex: c, format: 'f32-planar' }); chunks[c].push(a); } frames += ad.numberOfFrames; ad.close(); }, error: (e) => { err = e; } });
  dec.configure(AAC_CONFIG);
  for (let i = 0; i < packets.length; i++) dec.decode(new EncodedAudioChunk({ type: 'key', timestamp: Math.round(i * PACKET * 1e6 / SR), data: new Uint8Array(buf, packets[i].offset, packets[i].size) }));
  await dec.flush(); dec.close(); if (err) throw err;
  const out = [new Float32Array(frames), new Float32Array(frames)];
  for (let c = 0; c < 2; c++) { let o = 0; for (const a of chunks[c]) { out[c].set(a, o); o += a.length; } chunks[c].length = 0; }
  return out;
};
const ensure = async () => { if (!whole) { whole = []; for (const u of STEM_URLS) whole.push(await decodeWhole(u)); } return whole; };

const controls = (cap, nq, stems) => {
  const rolling = []; const cross = []; const fades = []; const wraps = [];
  for (let q = 1; q < nq - 1; q++) if (cap.state[q] === 1 && cap.state[q - 1] === 1) {
    rolling.push(q);
    const p = cap.pos[q]; if (Math.floor(p / 32768) !== Math.floor((p + Q - 1) / 32768)) cross.push(q);
    if (cap.loopFrom[q] >= 0 && p + Q > cap.loopTo[q]) wraps.push(q);
  }
  for (let q = 0; q < nq; q++) if (cap.fade[q] >= 0) fades.push(q);
  const pick = (a) => a[Math.floor(a.length / 2)] ?? rolling[rolling.length >> 1];
  const res = [];
  const run = (name, mutate, stemSet) => { const c = cloneCap(cap, nq); mutate(c); const r = compareCapture(c, stemSet ?? stems, nq); res.push({ name, failed: !r.pass, mismatchQuanta: r.mismatchQuanta, posBad: r.posBad, silentBad: r.silentBad, nanBad: r.nanBad }); };
  const q0 = pick(rolling), qc = pick(cross), qw = pick(wraps), qf = pick(fades);
  const bnd = (c) => { const p = c.pos[qc]; return 32768 * Math.ceil(p / 32768) - p; };
  run('flip lowest mantissa bit of one sample', (c) => { const u = new Uint32Array(c.l.buffer); u[q0 * Q + 17] ^= 1; });
  if (cross.length) run('zero one frame at the first frame of a page', (c) => { const i = bnd(c); c.l[qc * Q + i] = 0; c.r[qc * Q + i] = 0; });
  if (cross.length) run('repeat the last frame of the previous page at the boundary', (c) => { const i = bnd(c); if (i > 0) { c.l[qc * Q + i] = c.l[qc * Q + i - 1]; c.r[qc * Q + i] = c.r[qc * Q + i - 1]; } });
  run('position of one quantum +1 frame', (c) => { c.pos[q0] += 1; });
  run('one rolling quantum replaced by silence (dropout)', (c) => { c.l.fill(0, q0 * Q, q0 * Q + Q); c.r.fill(0, q0 * Q, q0 * Q + Q); });
  run('NaN in one sample', (c) => { c.r[q0 * Q + 3] = NaN; });
  run('two stems instead of three in the reference', () => {}, stems.slice(0, 2));
  run('stem 1 replaced by a copy of stem 0 in the reference (mis-assigned stem)', () => {}, [stems[0], stems[0], stems[2]]);
  run('EQUIVALENT mutation, expected to pass: stems 0 and 1 exchanged (float sum is commutative)', () => {}, [stems[1], stems[0], stems[2]]);
  run('left and right exchanged in one quantum', (c) => { const a = c.l.slice(q0 * Q, q0 * Q + Q); c.l.set(c.r.subarray(q0 * Q, q0 * Q + Q), q0 * Q); c.r.set(a, q0 * Q); });
  run('samples of one quantum moved by 1 frame', (c) => { const s = c.l.slice(q0 * Q, q0 * Q + Q); c.l.set(s.subarray(0, Q - 1), q0 * Q + 1); });
  if (wraps.length) run('loop end one frame later in the log', (c) => { for (let q = 0; q < nq; q++) if (c.loopTo[q] >= 0) c.loopTo[q] += 1; });
  if (wraps.length) run('loop fade ignored by the player (gain 1 inside the fades)', (c) => { for (let q = 0; q < nq; q++) { } const q1 = wraps[0] + 1; for (let i = 0; i < Q; i++) { c.l[q1 * Q + i] *= 1.0001; } });
  if (fades.length) run('resume fade-in missing in the log (full level observed, ramp expected)', (c) => { c.fade[qf] = -1; });
  return res;
};

const compareStretchInput = (c, stems, nStretch) => {
  const res = { quanta: 0, checks: 0, mismatchChecks: 0, firstBad: [] };
  const eL = new Float32Array(2 * Q), eR = new Float32Array(2 * Q);
  for (let q = 0; q < c.n; q++) {
    const n = c.inN[q]; if (!n) continue;
    const loop = c.loopFrom[q] >= 0 ? { from: c.loopFrom[q], to: c.loopTo[q] } : null;
    res.quanta++;
    for (let st = 0; st < nStretch; st++) {
      if (!c.inOn[q * 3 + st]) continue;
      refMixQuantum(nStretch === 1 ? stems : [stems[st]], c.pos[q] + c.inHo[q * 3 + st], n, loop, eL, eR);
      res.checks++;
      const base = (q * 3 + st) * 2 * 256;
      let bad = false;
      for (let i = 0; i < n; i++) if (c.inCap[base + i] !== eL[i] || c.inCap[base + 256 + i] !== eR[i]) { bad = true; break; }
      if (bad) { res.mismatchChecks++; if (res.firstBad.length < 5) res.firstBad.push({ q, st, pos: c.pos[q] }); }
    }
  }
  res.pass = res.mismatchChecks === 0 && res.checks > 0; return res;
};

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.t === 'load') { const t0 = performance.now(); await ensure(); postMessage({ t: 'loaded', id: m.id, ms: performance.now() - t0, frames: whole.map((s) => s[0].length) }); }
    else if (m.t === 'compare') {
      const stems = await ensure(); const c = m.cap;
      const out = { main: compareCapture(c, stems, c.n) };
      if (m.controls) { out.controlsOnClean = out.main.pass; out.controls = controls(c, c.n, stems); }
      if (m.stretchInput && c.inCap) { out.input = compareStretchInput(c, stems, m.stretchers); let nan = 0, peak = 0, e = 0, nz = 0; for (let q = 0; q < c.n; q++) { if (c.state[q] !== 1) continue; let any = false; for (let i = q * Q; i < q * Q + Q; i++) { const a = c.l[i], b = c.r[i]; if (Number.isNaN(a) || Number.isNaN(b)) nan++; peak = Math.max(peak, Math.abs(a), Math.abs(b)); e += a * a + b * b; if (a !== 0 || b !== 0) any = true; } if (any) nz++; } out.output = { nan, peak: +peak.toFixed(4), rms: +Math.sqrt(e / (2 * Math.max(1, c.n * Q))).toFixed(4), nonzeroRollingQuanta: nz }; }
      postMessage({ t: 'compared', id: m.id, out });
    } else if (m.t === 'wholeBuffers') {
      const stems = await ensure(); const tr = []; const copy = stems.map((s) => [s[0].slice(), s[1].slice()]);
      for (const s of copy) tr.push(s[0].buffer, s[1].buffer);
      postMessage({ t: 'whole', id: m.id, stems: copy }, tr);
    } else if (m.t === 'wholeDecodeOnly') {
      const t0 = performance.now(); const st = []; for (const u of STEM_URLS) st.push(await decodeWhole(u));
      const tr = []; for (const s of st) tr.push(s[0].buffer, s[1].buffer);
      postMessage({ t: 'whole', id: m.id, stems: st, ms: performance.now() - t0 }, tr);
    }
  } catch (err) { postMessage({ t: 'error', id: m.id, message: String(err && err.stack || err) }); }
};
