import fs from 'node:fs';
import { createModule, createProcessor, createRangeFrames, render, SR, Q } from './common.mjs';

const burstLen = 480;
const burstAmp = 0.8;
const burstEnergy = (() => {
  let e = 0;
  for (let i = 0; i < burstLen; i += 1) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / burstLen);
    e += (burstAmp * w * Math.sin((2 * Math.PI * 1000 * i) / SR)) ** 2;
  }
  return e * 2;
})();
export const makeClickTrack = (centres, n) => {
  const t = [new Float32Array(n), new Float32Array(n)];
  for (const c of centres) {
    const s = c - burstLen / 2;
    for (let i = 0; i < burstLen; i += 1) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / burstLen);
      const v = burstAmp * w * Math.sin((2 * Math.PI * 1000 * i) / SR);
      if (s + i >= 0 && s + i < n) {
        t[0][s + i] = v;
        t[1][s + i] = v;
      }
    }
  }
  return t;
};
export const makeToneTrack = (n) => {
  const t = [new Float32Array(n), new Float32Array(n)];
  const fs_ = [200, 400, 600, 1000];
  for (let i = 0; i < n; i += 1) {
    let v = 0;
    for (const f of fs_) v += 0.12 * Math.sin((2 * Math.PI * f * i) / SR);
    t[0][i] = v;
    t[1][i] = v;
  }
  return t;
};

export const buildSegments = (hist, evLog, bypassOf, mode) => {
  const starts = new Set([0, ...evLog.filter((e) => e.type === 'seek').map((e) => e.k)]);
  for (let i = 1; i < hist.length; i += 1) {
    if (mode === 'stream' && (hist[i].ho !== hist[i - 1].ho || (bypassOf(hist[i - 1]) && !bypassOf(hist[i])))) starts.add(i);
  }
  const s = [...starts].sort((a, b) => a - b);
  return s.map((ks, i) => ({ ks, ke: i + 1 < s.length ? s[i + 1] : hist.length }));
};
const invertProduct = (hist, seg, U) => {
  const { ks, ke } = seg;
  const first = hist[ks];
  if (U < first.u) return Q * ks + (U - first.u) / Math.max(1e-9, first.n / Q);
  for (let k = ks; k < ke; k += 1) {
    const h = hist[k];
    const nextU = h.u + h.n;
    if (U >= h.u && U < nextU) return Q * k + (U - h.u) / (h.n / Q);
  }
  return null;
};
export const expectedTime = (hist, segs, U, model, D, inLat) => {
  for (const seg of segs) {
    const first = hist[seg.ks];
    let t;
    if (model === 'product' || first.bypass) t = invertProduct(hist, seg, U);
    else {
      const tp = invertProduct(hist, seg, U - first.ho + inLat);
      t = tp === null ? null : tp + D;
    }
    if (t !== null && t >= seg.ks * Q && t < seg.ke * Q) return { t, seg };
  }
  return null;
};

export const measureClick = (rec, tExp, tempo, W = 5000) => {
  const n = rec[0].length;
  const a = Math.max(0, Math.round(tExp) - W);
  const b = Math.min(n, Math.round(tExp) + W);
  if (b - a < W) return { found: false, why: 'out-of-range' };
  let e = 0;
  let m = 0;
  for (let i = a; i < b; i += 1) {
    const x = rec[0][i];
    const y = rec[1][i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) return { found: false, why: 'nan' };
    const p = x * x + y * y;
    e += p;
    m += p * i;
  }
  const ideal = burstEnergy / tempo;
  if (e < 0.25 * ideal) return { found: false, why: 'missing', ratio: e / ideal };
  return { found: true, offset: m / e - tExp, ratio: e / ideal };
};

export const detectorControls = () => {
  const res = [];
  const tempo = 0.75;
  const n = 12 * SR;
  const rec = [new Float32Array(n), new Float32Array(n)];
  const exp = [];
  for (let j = 0; j < 20; j += 1) {
    const c = 12000 + j * 24000 + (j % 3) * 7;
    exp.push(c);
    const len = Math.round(burstLen / tempo);
    for (let i = 0; i < len; i += 1) {
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / len);
      const v = burstAmp * w * Math.sin((2 * Math.PI * 1000 * i) / SR);
      const s = c - Math.floor(len / 2) + i;
      if (s >= 0 && s < n) {
        rec[0][s] = v;
        rec[1][s] = v;
      }
    }
  }
  const run = (name, r, expect, f) => {
    const out = exp.map((c) => measureClick(r, c, tempo));
    res.push({ name, expect, got: f(out) });
  };
  const meanOff = (o) => o.filter((x) => x.found).reduce((s, x) => s + x.offset, 0) / o.filter((x) => x.found).length;
  run('ideal', rec, '0 offset (|.|<=1 sample)', (o) => `mean ${meanOff(o).toFixed(2)}, max |off| ${Math.max(...o.map((x) => Math.abs(x.offset))).toFixed(2)}, all found ${o.every((x) => x.found)}`);
  const sh = [new Float32Array(n), new Float32Array(n)];
  sh[0].set(rec[0].subarray(0, n - 20), 20);
  sh[1].set(rec[1].subarray(0, n - 20), 20);
  run('shifted +20 samples', sh, '+20', (o) => `mean ${meanOff(o).toFixed(2)}`);
  const sh2 = [new Float32Array(n), new Float32Array(n)];
  sh2[0].set(rec[0].subarray(20), 0);
  sh2[1].set(rec[1].subarray(20), 0);
  run('shifted -20 samples', sh2, '-20', (o) => `mean ${meanOff(o).toFixed(2)}`);
  const miss = [Float32Array.from(rec[0]), Float32Array.from(rec[1])];
  miss[0].fill(0, 12000 + 5 * 24000 - 600, 12000 + 5 * 24000 + 700);
  miss[1].fill(0, 12000 + 5 * 24000 - 600, 12000 + 5 * 24000 + 700);
  run('click #5 missing', miss, 'found=false for #5 only', (o) => `not found: ${o.map((x, i) => (x.found ? '' : i + ':' + x.why)).filter(Boolean).join(',')}`);
  const nan = [Float32Array.from(rec[0]), Float32Array.from(rec[1])];
  nan[0][12000 + 7 * 24000 + 300] = NaN;
  run('NaN in click #7 window', nan, "flagged 'nan' for #7", (o) => `not found: ${o.map((x, i) => (x.found ? '' : i + ':' + x.why)).filter(Boolean).join(',')}`);
  const halfAmp = [Float32Array.from(rec[0], (v) => v * 0.3), Float32Array.from(rec[1], (v) => v * 0.3)];
  run('amplitude x0.3 (energy 9%)', halfAmp, 'flagged missing (energy < 25%)', (o) => `found: ${o.filter((x) => x.found).length}/${o.length}`);
  return res;
};

const stats = (a) => {
  if (!a.length) return { n: 0 };
  const s = [...a].sort((x, y) => x - y);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const abs = a.map(Math.abs).sort((x, y) => x - y);
  return { n: a.length, median: q(0.5), minv: s[0], maxv: s[s.length - 1], maxAbs: abs[abs.length - 1], p95abs: abs[Math.min(abs.length - 1, Math.floor(0.95 * abs.length))] };
};

export const variants = {
  V0: { mode: 'const', split: false },
  V1: { mode: 'stream', split: false },
  V2: { mode: 'stream', split: true },
  V3: { mode: 'const', split: true },
};

const secs = 40;

let modPool;
const getMod = async () => (modPool ??= await createModule());

export const runScenario = async (vname, sc) => {
  const m = await getMod();
  const v = variants[vname];
  const p = createProcessor(m, SR, v);
  const n = secs * SR;
  const range = sc.range ? createRangeFrames({ fromFrame: sc.range[0], toFrame: sc.range[1] }, 'cycle') : undefined;
  const track = makeClickTrack(sc.clicks, n);
  const r = render({
    procs: [p],
    groups: [[track]],
    range,
    startFrame: sc.start,
    nQ: sc.nQ,
    tempo: sc.tempo0,
    transpose: sc.transpose,
    events: sc.events ?? [],
    record: true,
  });
  return { r, p, track, range };
};
