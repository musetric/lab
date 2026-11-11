import { readFileSync, existsSync } from 'node:fs';
const ids = ['S0', 'S1', 'S2', 'S2c', 'S3', 'S4', 'S4b', 'S5', 'S6', 'S7a', 'S7b'];
const load = (mode, id) =>
  [1, 2, 3].map((r) => `results/${mode}_${id}_r${r}.json`).filter(existsSync).map((f) => JSON.parse(readFileSync(f, 'utf8'))[0]);
const med = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const range = (xs, f = (x) => x) => `${f(Math.min(...xs))}-${f(Math.max(...xs))}`;
const us = (ms) => (ms * 1000).toFixed(1);
const out = [];
out.push('### Allocation per quantum (process() only; handlers listed separately)\n');
out.push('| Scenario | mean B/q (3 runs) | median B/q | p99 B/q | max B/q (one quantum) | handler bytes/msg (median, max) |');
out.push('|---|---|---|---|---|---|');
for (const id of ids) {
  const runs = load('alloc', id);
  if (!runs.length) continue;
  const mean = runs.map((r) => r.processBytesPerQuantumMean);
  const h = Object.entries(runs[0].handlers ?? {})
    .filter(([t]) => t !== 'mount' && t !== 'unmount')
    .map(([t, v]) => `${t} x${v.count}: ${v.bytes.median.toFixed(0)} / ${v.bytes.max.toFixed(0)}`)
    .join('; ');
  out.push(`| ${id} | ${range(mean, (x) => x.toFixed(0))} | ${med(runs.map((r) => r.processBytes.median)).toFixed(0)} | ${med(runs.map((r) => r.processBytes.p99)).toFixed(0)} | ${med(runs.map((r) => r.processBytes.max)).toFixed(0)} | ${h || '-'} |`);
}
out.push('\n### Time per quantum (us; median of 3 fresh-process runs, range in brackets)\n');
out.push('| Scenario | median | p99 | p99.9 | max | >1.33 ms (50 % budget) | >2.67 ms (budget) | minor GC (n, total ms, max ms) | major GC | realtime x |');
out.push('|---|---|---|---|---|---|---|---|---|---|');
for (const id of ids) {
  const runs = load('time', id);
  if (!runs.length) continue;
  const f = (k) => `${us(med(runs.map((r) => r.quantumMs[k])))} [${range(runs.map((r) => r.quantumMs[k]), us)}]`;
  const g = (kind) => {
    const gs = runs.map((r) => r.gc[kind] ?? { count: 0, totalMs: 0, maxMs: 0 });
    return `${med(gs.map((x) => x.count))}, ${med(gs.map((x) => x.totalMs)).toFixed(1)}, ${med(gs.map((x) => x.maxMs)).toFixed(2)}`;
  };
  out.push(`| ${id} | ${f('median')} | ${f('p99')} | ${f('p999')} | ${f('max')} | ${med(runs.map((r) => r.overHalfBudget))} | ${med(runs.map((r) => r.overBudget))} | ${g('minor')} | ${g('major')} | ${med(runs.map((r) => r.realtimeFactor)).toFixed(0)} |`);
}
out.push('\n### Message handlers, time (ms; sync part / incl. promise machinery), 3 runs pooled by median\n');
out.push('| Scenario | type | count | sync median | sync p99 | sync max | total median | total max | slowest (ms @quantum) |');
out.push('|---|---|---|---|---|---|---|---|---|');
for (const id of ids) {
  const runs = load('time', id);
  if (!runs.length) continue;
  for (const t of Object.keys(runs[0].handlers ?? {})) {
    if (t === 'mount' || t === 'unmount') continue;
    const hs = runs.map((r) => r.handlers[t]);
    const s = (k, m) => med(hs.map((h) => h[k][m]));
    const slow = hs[0].slowest.slice(0, 3).map((x) => `${x.ms.toFixed(2)}@${x.atQuantum}`).join(', ');
    out.push(`| ${id} | ${t} | ${hs[0].count} | ${s('syncMs', 'median').toFixed(4)} | ${s('syncMs', 'p99').toFixed(4)} | ${s('syncMs', 'max').toFixed(3)} | ${s('totalMs', 'median').toFixed(4)} | ${s('totalMs', 'max').toFixed(3)} | ${slow} |`);
  }
}
console.log(out.join('\n'));
