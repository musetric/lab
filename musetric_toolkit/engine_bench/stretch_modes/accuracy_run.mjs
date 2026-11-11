import fs from 'node:fs';
fs.mkdirSync('results', { recursive: true });
import { runScenario, expectedTime, measureClick, buildSegments, variants, detectorControls, makeToneTrack } from './accuracy.mjs';
import { createModule, createProcessor, createRangeFrames, render, SR, Q } from './common.mjs';

const clickList = (first, step, secs) => { const a = []; for (let t = first; t < secs; t += step) a.push(Math.round(t * SR)); return a; };

const analyze = async (vname, sc) => {
  const { r, p } = await runScenario(vname, sc);
  const D = p.state.outputLatency, inLat = p.state.inputLatency;
  const hist = r.hist;
  const bypassOf = (h) => h.bypass;
  const segsP = buildSegments(hist, r.evLog, bypassOf, 'const');
  const segsS = buildSegments(hist, r.evLog, bypassOf, 'stream');
  const out = [];
  const models = variants[vname].mode === 'stream' ? ['product', 'stream'] : ['product'];
  const len = sc.range ? sc.range[1] - sc.range[0] : 0;
  for (const model of models) {
    const segs = model === 'product' ? segsP : segsS;
    segs.forEach((seg, si) => {
      const first = hist[seg.ks], last = hist[seg.ke - 1];
      const uHi = last.u + last.n;
      for (const c of sc.clicks) {
        if (sc.range && (c < sc.range[0] || c >= sc.range[1])) continue;
        const reps = sc.range ? Math.ceil((uHi - c) / len) + 1 : 1;
        for (let m = 0; m < Math.max(1, reps); m += 1) {
          const U = c + m * len;
          const e = expectedTime(hist, [seg], U, model, D, inLat);
          if (!e) continue;
          const tempo = hist[Math.min(seg.ke - 1, Math.max(seg.ks, Math.floor(e.t / Q)))].tempo;
          const mres = measureClick(r.rec, e.t, tempo, sc.W ?? 5000);
          out.push({ model, seg: si, ks: seg.ks, U, c, wrap: m, tExp: e.t, age: e.t - seg.ks * Q, ...mres });
        }
      }
    });
  }
  return { rows: out, D, inLat, hist, evLog: r.evLog, rec: r.rec, tempoAt: (k) => hist[k].tempo };
};

const secs = 40;
const results = { controls: detectorControls(), steady: [], wrap: [], seek: [], tempo: [], disc: [] };
const tempos = [0.5, 0.75, 1.25, 1.5];
for (const vn of Object.keys(variants)) {
  for (const tr of [0, 3]) for (const tp of tempos) {
    let a = await analyze(vn, { clicks: clickList(0.137, 0.5, secs), start: 5 * SR, nQ: 3000, tempo0: tp, transpose: tr });
    results.steady.push({ vn, tr, tp, D: a.D, rows: a.rows });
    a = await analyze(vn, { clicks: clickList(10.137, 0.5, 14), range: [10 * SR, 14 * SR], start: 13 * SR, nQ: 6000, tempo0: tp, transpose: tr });
    results.wrap.push({ vn, tr, tp, rows: a.rows });
  }
  for (const tr of [3]) for (const tp of [0.75, 1.25]) for (const d of [0.03, 0.12, 0.2, 0.35]) {
    const S = Math.round((10.1 - d) * SR);
    const a = await analyze(vn, { clicks: clickList(0.1, 0.25, secs), start: 3 * SR, nQ: 1750, tempo0: tp, transpose: tr, events: [{ k: 1000, type: 'seek', frame: S }], W: 3000 });
    results.seek.push({ vn, tr, tp, d, S, rows: a.rows });
  }
  const cases = [[1.0, 0.75, 3], [0.75, 1.5, 3], [1.25, 0.5, 3], [1.0, 0.75, 0], [1.5, 1.0, 3]];
  for (const [t0, t1, tr] of cases) {
    const a = await analyze(vn, { clicks: clickList(0.137, 0.5, secs), start: 5 * SR, nQ: 3000, tempo0: t0, transpose: tr, events: [{ k: 1125, type: 'tempo', value: t1 }] });
    results.tempo.push({ vn, t0, t1, tr, kChange: 1125, rows: a.rows });
  }
  console.log('done', vn);
}
fs.writeFileSync('results/accuracy.json', JSON.stringify(results));
