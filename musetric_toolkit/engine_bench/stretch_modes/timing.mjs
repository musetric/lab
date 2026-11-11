import fs from 'node:fs';
fs.mkdirSync('results', { recursive: true });
import os from 'node:os';
import { createModule, createProcessor, createRangeFrames, makeStems, render, SR, Q, BUDGET_US } from './common.mjs';
try { os.setPriority(0, os.constants.priority.PRIORITY_ABOVE_NORMAL); } catch {}
const quick = process.argv.includes('--quick');
const REPS = quick ? 1 : 2;
const mods = [await createModule(), await createModule(), await createModule()];
const st = makeStems(100);
const rec = st.lead;
const WARM = 375, N = 22500;
const range = createRangeFrames({ fromFrame: 10 * SR, toFrame: 14 * SR }, 'cycle');
const defs = [
  ['V0-1', 'const', false, 1, false], ['V0-3', 'const', false, 3, false], ['V0-3stag', 'const', false, 3, true],
  ['V1-1', 'stream', false, 1, false], ['V1-3', 'stream', false, 3, false], ['V1-3stag', 'stream', false, 3, true],
  ['V2-1', 'stream', true, 1, false], ['V2-3', 'stream', true, 3, false],
  ['V3-1', 'const', true, 1, false], ['V3-3', 'const', true, 3, false],
];
const run = (def, tempo, tr, useRange, fine = false, nq = N + WARM) => {
  const [, mode, split, count, stag] = def;
  const procs = [];
  for (let i = 0; i < count; i++) procs.push(createProcessor(mods[i], SR, { mode, split, probe: fine ? { js: 0, seek: 0, proc: 0, copy: 0 } : null }));
  const groups = count === 1 ? [[st.chord, st.drums, st.lead]] : [[st.chord, st.drums], [st.lead], [rec]];
  const startQ = procs.map((_, i) => (stag ? i * 4 : 0));
  const r = render({ procs, groups, range: useRange ? range : undefined, startFrame: useRange ? 10 * SR : 5 * SR, nQ: nq, tempo, transpose: tr, timing: fine ? 'fine' : 'total', startQ });
  const cut = (a) => a.subarray(WARM + 12);
  return { total: cut(r.times), fine: r.fine && { js: cut(r.fine.js), seek: cut(r.fine.seek), proc: cut(r.fine.proc), copy: cut(r.fine.copy) } };
};
const q = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
const summ = (arrs) => {
  const n = arrs.reduce((s, a) => s + a.length, 0);
  const all = new Float64Array(n); let o = 0; for (const a of arrs) { all.set(a, o); o += a.length; }
  const s = Float64Array.from(all).sort();
  let h50 = 0, h100 = 0, sum = 0; for (const v of all) { if (v > BUDGET_US * 0.5) h50++; if (v > BUDGET_US) h100++; sum += v; }
  return { n, mean: sum / n, med: q(s, 0.5), p99: q(s, 0.99), p999: q(s, 0.999), max: s[n - 1], over50: h50 / n, over100: h100 / n };
};
{ const t = performance.now(); let x = 0; for (let i = 0; i < 1e6; i++) x += performance.now(); console.log('performance.now cost ns', ((performance.now() - t) * 1e6 / 1e6).toFixed(0)); }
for (const d of [defs[0], defs[3], defs[1], defs[4]]) for (const ur of [false, true]) run(d, 0.75, 3, ur, false, 3000);
console.log('warm done');
const out = { rows: [], fine: [], budget: BUDGET_US, quick };
const tempos = quick ? [0.75] : [0.5, 0.75, 1.25, 1.5];
const trs = quick ? [3] : [0, 3];
for (const ur of [false, true]) for (const tr of trs) for (const tp of tempos) {
  const acc = new Map(defs.map((d) => [d[0], []]));
  for (let rep = 0; rep < REPS; rep++) {
    const order = rep % 2 === 0 ? defs : [...defs].reverse();
    for (const d of order) acc.get(d[0]).push(run(d, tp, tr, ur).total);
  }
  for (const d of defs) {
    const s = summ(acc.get(d[0]));
    const reps = acc.get(d[0]).map((a) => summ([a]));
    out.rows.push({ v: d[0], tempo: tp, tr, range: ur, ...s, repMed: reps.map((r) => r.med), repP99: reps.map((r) => r.p99) });
  }
  console.log('done', ur, tr, tp, new Date().toISOString());
  fs.writeFileSync(quick ? 'results/timing_quick.json' : 'results/timing.json', JSON.stringify(out));
}
for (const ur of [false, true]) for (const tp of [0.75, 1.25]) for (const d of defs.filter((x) => ['V0-1', 'V0-3', 'V1-1', 'V1-3', 'V2-3', 'V3-3'].includes(x[0]))) {
  const r = run(d, tp, 3, ur, true);
  const f = r.fine;
  out.fine.push({ v: d[0], tempo: tp, tr: 3, range: ur, total: summ([r.total]), js: summ([f.js]), seek: summ([f.seek]), proc: summ([f.proc]), copy: summ([f.copy]) });
}
console.log('fine done');
fs.writeFileSync(quick ? 'results/timing_quick.json' : 'results/timing.json', JSON.stringify(out));
