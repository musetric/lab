import fs from 'node:fs';

const file = process.argv[2];
if (!file) {
  console.error('usage: node report_stretch.mjs out/stretch_timing-<label>-<block>.json');
  process.exit(2);
}
const json = JSON.parse(fs.readFileSync(file, 'utf8'));
const rows = new Map();
for (const r of json.results) {
  const key = `${r.range ? 'R' : '-'} t${r.tempo} ${r.paced ? 'paced' : 'tight'} ${r.v}`;
  if (!rows.has(key)) rows.set(key, []);
  rows.get(key).push(r);
}
const join = (runs, pick, width) => runs.map(pick).join('/').padStart(width);
console.log('cfg'.padEnd(26), 'reps  med(us)  p99 (reps)        p99.9   max   >50%  >100%(per run)');
for (const [key, runs] of rows) {
  console.log(
    key.padEnd(26),
    String(runs.length).padStart(3),
    join(runs, (r) => r.med.toFixed(0), 10),
    join(runs, (r) => r.p99.toFixed(0), 14),
    join(runs, (r) => r.p999.toFixed(0), 12),
    join(runs, (r) => r.max.toFixed(0), 12),
    join(runs, (r) => r.over50, 10),
    runs.map((r) => r.over100).join('/'),
  );
}
const temps = json.results.map((r) => r.after.cpuMaxC);
console.log(
  'cpuMaxC range',
  Math.min(...temps),
  Math.max(...temps),
  'thermal',
  [...new Set(json.results.map((r) => r.after.thermalStatus))],
  'battery',
  json.results[0].before.batteryC,
  '->',
  json.results.at(-1).after.batteryC,
);
