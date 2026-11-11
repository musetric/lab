import fs from 'node:fs';
fs.mkdirSync('results', { recursive: true });
import { runScenario, expectedTime, measureClick, buildSegments, variants } from './accuracy.mjs';
import { SR, Q } from './common.mjs';
const clickList = (first, step, secs) => { const a = []; for (let t = first; t < secs; t += step) a.push(Math.round(t * SR)); return a; };
const rows = [];
for (const vn of ['V0', 'V1', 'V2', 'V3']) for (const [t0, t1] of [[1, 0.75], [0.75, 1.5], [1.5, 1], [1.25, 0.5]]) {
  for (let off = 0; off < 48; off += 4) {
    const kc = 1125 + off;
    const sc = { clicks: clickList(0.137, 0.5, 40), start: 5 * SR, nQ: 2000, tempo0: t0, transpose: 3, events: [{ k: kc, type: 'tempo', value: t1 }], W: 4000 };
    const { r, p } = await runScenario(vn, sc);
    const D = p.state.outputLatency, inLat = p.state.inputLatency;
    const segs = buildSegments(r.hist, r.evLog, (h) => h.bypass, 'stream');
    for (const model of variants[vn].mode === 'stream' ? ['product', 'stream'] : ['product']) {
      for (const seg of segs) for (const c of sc.clicks) {
        const e = expectedTime(r.hist, [seg], c, model, D, inLat);
        if (!e) continue;
        const dt = e.t - kc * Q;
        if (dt < -6000 || dt > 24000) continue;
        const res = measureClick(r.rec, e.t, r.hist[Math.min(1999, Math.floor(e.t / Q))].tempo, 4000);
        rows.push({ vn, t0, t1, model, dtMs: dt / 48, found: res.found, off: res.found ? res.offset / 48 : null, ratio: res.ratio, why: res.why });
      }
    }
  }
}
fs.writeFileSync('results/tempo_sweep.json', JSON.stringify(rows));
const buckets = [[-125, -0.001, 'before'], [0, 62, '0-62ms'], [62, 125, '62-125'], [125, 250, '125-250'], [250, 500, '250-500']];
for (const model of ['product', 'stream']) for (const vn of ['V0', 'V1', 'V2', 'V3']) {
  const rs = rows.filter((r) => r.vn === vn && r.model === model);
  if (!rs.length) continue;
  const cells = buckets.map(([a, b, n]) => { const g = rs.filter((r) => r.dtMs >= a && r.dtMs < b); const ok = g.filter((r) => r.found); const mx = ok.length ? Math.max(...ok.map((r) => Math.abs(r.off))) : NaN; const md = ok.length ? ok.map((r) => r.off).sort((x, y) => x - y)[Math.floor(ok.length / 2)] : NaN; return `${n}: med ${md.toFixed(2)} max|.| ${mx.toFixed(2)} n${g.length} miss${g.length - ok.length}`; });
  console.log(model, vn, '|', cells.join(' | '));
}
