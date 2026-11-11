export const SR = 48000;
export const Q = 128;
export const PAGE = 32768;
export const PACKET = 1024;
export const PRE_ROLL = 2;
export const AHEAD_FRAMES = 8 * SR;
export const BEHIND_FRAMES = 4 * SR;
export const PIN_MAX_FRAMES = 30 * SR;
export const PAGES_PER_STEM = 26;
export const STEMS = 3;
export const FADE = 256;
export const AAC_CONFIG = { codec: 'mp4a.40.2', sampleRate: 48000, numberOfChannels: 2, description: new Uint8Array([17, 144]) };
export const STEM_URLS = ['/data/stemA.mp4', '/data/stemB.mp4', '/data/stemC.mp4'];

const ramp = (d) => (d < FADE ? Math.max(0, d) / FADE : 1);
export const rampGain = ramp;
export const rangeGain = (from, to, frame) => {
  if (frame >= to) return 0;
  if (frame < from) return 1;
  return Math.min(ramp(frame - from), ramp(to - frame));
};
export const wrapFrame = (from, to, frame) => (frame < to ? frame : from + ((frame - from) % (to - from)));

export const refMixQuantum = (stems, u, n, loop, outL, outR) => {
  outL.fill(0, 0, n); outR.fill(0, 0, n);
  for (const s of stems) {
    for (let i = 0; i < n; i++) {
      const uf = u + i;
      let f = uf;
      let g = 1;
      if (loop) { f = wrapFrame(loop.from, loop.to, uf); g = rangeGain(loop.from, loop.to, f); }
      if (f < 0) continue;
      const a = s[0][f] ?? 0;
      const b = s[1][f] ?? 0;
      outL[i] += a * 1 * g;
      outR[i] += b * 1 * g;
    }
  }
};

export const compareCapture = (cap, stems, nq) => {
  const res = { quanta: nq, rolling: 0, mismatchQuanta: 0, mismatchSamples: 0, maxAbs: 0, firstBad: [], posBad: 0, silentBad: 0, nanBad: 0,
    covered: { pageCross: 0, wrap: 0, fadeIn: 0, buffering: 0, loopFade: 0, seeks: 0 } };
  const eL = new Float32Array(Q), eR = new Float32Array(Q);
  let prevRolling = false, prevEnd = -1, prevLoopKey = '', heldPos = null;
  for (let q = 0; q < nq; q++) {
    const st = cap.state[q];
    const l = cap.l.subarray(q * Q, q * Q + Q), r = cap.r.subarray(q * Q, q * Q + Q);
    if (st !== 1) {
      if (st === 2) { res.covered.buffering++; if (heldPos !== null && !cap.seekAt[q] && cap.pos[q] !== heldPos) { res.posBad++; if (res.firstBad.length < 8) res.firstBad.push({ q, kind: 'held', got: cap.pos[q], want: heldPos }); } heldPos = cap.pos[q]; }
      for (let i = 0; i < Q; i++) if (l[i] !== 0 || r[i] !== 0) { res.silentBad++; break; }
      prevRolling = false; continue;
    }
    res.rolling++;
    const p = cap.pos[q];
    if (heldPos !== null && !cap.seekAt[q] && p !== heldPos) { res.posBad++; if (res.firstBad.length < 8) res.firstBad.push({ q, kind: 'resume', got: p, want: heldPos }); }
    heldPos = null;
    const lf = cap.loopFrom[q], lt = cap.loopTo[q];
    const loop = lf >= 0 ? { from: lf, to: lt } : null;
    const key = `${lf}:${lt}`;
    if (prevRolling && !cap.seekAt[q] && key === prevLoopKey) {
      if (p !== prevEnd) { res.posBad++; if (res.firstBad.length < 8) res.firstBad.push({ q, kind: 'pos', got: p, want: prevEnd }); }
    }
    if (cap.seekAt[q]) res.covered.seeks++;
    refMixQuantum(stems, p, Q, loop, eL, eR);
    const f = cap.fade[q];
    if (f >= -0) {
      res.covered.fadeIn++;
      for (let i = 0; i < Q; i++) { const g = rampGain(f + i); eL[i] = eL[i] * g; eR[i] = eR[i] * g; }
    }
    let bad = 0;
    for (let i = 0; i < Q; i++) {
      const a = l[i], b = r[i];
      if (Number.isNaN(a) || Number.isNaN(b)) res.nanBad++;
      const dl = Math.abs(a - eL[i]), dr = Math.abs(b - eR[i]);
      if (a !== eL[i] || b !== eR[i]) { bad++; res.maxAbs = Math.max(res.maxAbs, dl, dr); }
    }
    if (bad) { res.mismatchQuanta++; res.mismatchSamples += bad; if (res.firstBad.length < 8) res.firstBad.push({ q, kind: 'sample', pos: p, bad }); }
    if (Math.floor(p / PAGE) !== Math.floor((p + Q - 1) / PAGE)) res.covered.pageCross++;
    if (loop && p + Q > lt) res.covered.wrap++;
    if (loop && (p - lf < FADE || lt - p <= Q + FADE)) res.covered.loopFade++;
    prevRolling = true;
    prevEnd = loop ? wrapFrame(lf, lt, p + Q) : p + Q;
    prevLoopKey = key;
  }
  res.pass = res.mismatchQuanta === 0 && res.posBad === 0 && res.silentBad === 0 && res.nanBad === 0 && res.rolling > 0;
  return res;
};

export const cloneCap = (c, nq) => ({ l: c.l.slice(0, nq * Q), r: c.r.slice(0, nq * Q), pos: c.pos.slice(0, nq), state: c.state.slice(0, nq), fade: c.fade.slice(0, nq),
  loopFrom: c.loopFrom.slice(0, nq), loopTo: c.loopTo.slice(0, nq), seekAt: c.seekAt.slice(0, nq) });
