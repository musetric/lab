import { createModule, createProcessor, createRangeFrames, makeStems, render, SR, Q, BUDGET_US } from '/stretch_modes/common.mjs';
const defs = [
  ['V0-1', 'const', false, 1, false],
  ['V0-3', 'const', false, 3, false],
  ['V0-3stag', 'const', false, 3, true],
  ['V1-1', 'stream', false, 1, false],
  ['V1-3', 'stream', false, 3, false],
  ['V1-3stag', 'stream', false, 3, true],
  ['V2-1', 'stream', true, 1, false],
  ['V2-3', 'stream', true, 3, false],
  ['V3-1', 'const', true, 1, false],
  ['V3-3', 'const', true, 3, false],
];
let mods, st, rec, range;
const WARM = 375;
let sab;
const pacer = (periodUs) => {
  sab ??= new Int32Array(new SharedArrayBuffer(4));
  const t0 = performance.now();
  return (k) => {
    const w = t0 + (k * periodUs) / 1e3 - performance.now();
    if (w > 0.05) Atomics.wait(sab, 0, 0, w);
  };
};
const run = (def, tempo, tr, useRange, { fine = false, nq = 22500 + WARM, paced = false } = {}) => {
  const [, mode, split, count, stag] = typeof def === 'string' ? defs.find((d) => d[0] === def) : def;
  const procs = [];
  for (let i = 0; i < count; i++)
    procs.push(
      createProcessor(mods[i], SR, { mode, split, probe: fine ? { js: 0, seek: 0, proc: 0, copy: 0 } : null }),
    );
  const groups = count === 1 ? [[st.chord, st.drums, st.lead]] : [[st.chord, st.drums], [st.lead], [rec]];
  const startQ = procs.map((_, i) => (stag ? i * 4 : 0));
  const r = render({
    procs,
    groups,
    range: useRange ? range : void 0,
    startFrame: useRange ? 10 * SR : 5 * SR,
    nQ: nq,
    tempo,
    transpose: tr,
    timing: fine ? 'fine' : 'total',
    startQ,
    pace: paced ? pacer(BUDGET_US) : void 0,
  });
  const cut = (a) => a.subarray(WARM + 12);
  return {
    total: cut(r.times),
    fine: r.fine && { js: cut(r.fine.js), seek: cut(r.fine.seek), proc: cut(r.fine.proc), copy: cut(r.fine.copy) },
  };
};
const q = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
const summ = (a) => {
  const s = Float64Array.from(a).sort();
  let h50 = 0,
    h100 = 0,
    sum = 0;
  for (const v of a) {
    if (v > BUDGET_US * 0.5) h50++;
    if (v > BUDGET_US) h100++;
    sum += v;
  }
  const n = a.length;
  return {
    n,
    mean: sum / n,
    med: q(s, 0.5),
    p99: q(s, 0.99),
    p999: q(s, 0.999),
    max: s[n - 1],
    over50: h50,
    over100: h100,
  };
};
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === 'init') {
      mods = [await createModule(), await createModule(), await createModule()];
      postMessage({ id: -1, log: 'mods' });
      st = makeStems(100);
      postMessage({ id: -1, log: 'stems' });
      rec = st.lead;
      range = createRangeFrames({ fromFrame: 10 * SR, toFrame: 14 * SR }, 'cycle');
      const t = performance.now();
      let x = 0;
      for (let i = 0; i < 1e5; i++) x += performance.now();
      const timerNs = ((performance.now() - t) * 1e6) / 1e5;
      for (const d of [defs[0], defs[3], defs[1], defs[4]])
        for (const ur of [false, true]) run(d, 0.75, 3, ur, { nq: 3e3 });
      postMessage({ id: m.id, result: { timerNs, ok: true } });
    } else if (m.cmd === 'run') {
      const t0 = performance.now();
      const r = run(m.v, m.tempo, m.tr, m.range, { fine: !!m.fine, nq: (m.nq ?? 22500) + WARM, paced: !!m.paced });
      const out = {
        v: m.v,
        tempo: m.tempo,
        tr: m.tr,
        range: m.range,
        paced: !!m.paced,
        wallMs: performance.now() - t0,
        ...summ(r.total),
      };
      if (r.fine)
        out.fine = { js: summ(r.fine.js), seek: summ(r.fine.seek), proc: summ(r.fine.proc), copy: summ(r.fine.copy) };
      postMessage({ id: m.id, result: out });
    }
  } catch (err) {
    postMessage({ id: m.id, error: String(err.stack ?? err) });
  }
};
