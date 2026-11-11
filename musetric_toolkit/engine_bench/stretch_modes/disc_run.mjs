import fs from 'node:fs';
fs.mkdirSync('results', { recursive: true });
import { makeToneTrack, variants } from './accuracy.mjs';
import { createModule, createProcessor, createRangeFrames, render, SR, Q } from './common.mjs';
const m = await createModule();
const tone = makeToneTrack(40 * SR);
const W = 240;
const rmsSeries = (rec) => { const n = Math.floor(rec[0].length / W); const o = new Float64Array(n); for (let i = 0; i < n; i++) { let e = 0; for (let j = i*W; j < (i+1)*W; j++) e += rec[0][j]**2 + rec[1][j]**2; o[i] = Math.sqrt(e / (2*W)); } return o; };
const maxJump = (rec, a, b) => { let mx = 0; for (let i = Math.max(1,a); i < b; i++) mx = Math.max(mx, Math.abs(rec[0][i]-rec[0][i-1])); return mx; };
const db = (x) => 20 * Math.log10(Math.max(x, 1e-9));
const run = (vn, kind, tp, tr) => {
  const p = createProcessor(m, SR, variants[vn]);
  let o = { procs: [p], groups: [[tone]], transpose: tr, nQ: 3750, record: true };
  let tEv;
  if (kind === 'seek') { const n0 = Math.floor(tp*Q); o = { ...o, startFrame: 3*SR, tempo: tp, events: [{ k: 1500, type: 'seek', frame: 3*SR + Math.round(1500*Q*tp) + 8*SR }] }; tEv = 1500*Q; }
  if (kind === 'wrap') { o = { ...o, startFrame: Math.round(10.5*SR), tempo: tp, range: createRangeFrames({fromFrame:10*SR,toFrame:14*SR},'cycle') }; }
  if (kind === 'tempo') { o = { ...o, startFrame: 5*SR, tempo: 1.0, events: [{ k: 1500, type: 'tempo', value: tp }] }; tEv = 1500*Q; }
  const r = render(o);
  if (kind === 'wrap') { const h = r.hist.find((x) => x.u + x.n >= 14*SR); tEv = h.k * Q; }
  const rms = rmsSeries(r.rec);
  const ei = Math.floor(tEv / W);
  const a = Math.floor(1.0*SR/W), b = ei - Math.floor(0.4*SR/W);
  let steadyMed = 0; { const s = Array.from(rms.slice(a, b)).sort((x,y)=>x-y); steadyMed = s[Math.floor(s.length/2)]; }
  let stepBase = 0; for (let i = a+1; i < b; i++) stepBase = Math.max(stepBase, Math.abs(db(rms[i]) - db(rms[i-1])));
  const jumpBase = maxJump(r.rec, a*W, b*W);
  const lo = ei - Math.floor(0.25*SR/W), hi = ei + Math.floor(0.25*SR/W);
  let minRms = 1e9, step = 0; for (let i = lo; i < hi; i++) { minRms = Math.min(minRms, rms[i]); if (i>lo) step = Math.max(step, Math.abs(db(rms[i]) - db(rms[i-1]))); }
  const jump = maxJump(r.rec, lo*W, hi*W);
  let below = 0; for (let i = lo; i < hi; i++) if (db(rms[i]) < db(steadyMed) - 20) below++;
  return { vn, kind, tp, tr, dropDb: db(minRms) - db(steadyMed), stepDb: step, stepBase, stepExcess: step - stepBase, jumpRatio: jump / jumpBase, ms_below20: below*5 };
};
const rows = [];
for (const kind of ['seek','wrap','tempo']) for (const tp of [0.75, 1.5]) for (const vn of Object.keys(variants)) rows.push(run(vn, kind, tp, 3));
fs.writeFileSync('results/disc.json', JSON.stringify(rows));
for (const r of rows) console.log(r.kind, 't'+r.tp, r.vn, 'dip', r.dropDb.toFixed(1), 'dB; step', r.stepDb.toFixed(1), '(base', r.stepBase.toFixed(1)+')', 'jump x', r.jumpRatio.toFixed(2), 'below-20dB', r.ms_below20, 'ms');
