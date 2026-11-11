import fs from 'node:fs';

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node report_stepper.mjs out/stepper-*.json');
  process.exit(2);
}
for (const file of files) {
  const json = JSON.parse(fs.readFileSync(file, 'utf8'));
  const byId = new Map();
  for (const r of json.results) {
    if (!byId.has(r.id)) byId.set(r.id, []);
    byId.get(r.id).push(r);
  }
  console.log('==', file, json.label, 'paced', json.paced);
  for (const [id, runs] of byId) {
    const us = (key) => runs.map((r) => (r.quantumMs[key] * 1000).toFixed(0)).join('/');
    console.log(
      id.padEnd(4),
      'runs',
      runs.length,
      'med',
      us('median'),
      'p99',
      us('p99'),
      'p99.9',
      us('p999'),
      'max',
      us('max'),
      'over50%',
      runs.map((r) => r.quantumMs.over50).join('/'),
      'over100%',
      runs.map((r) => r.quantumMs.over100).join('/'),
      'n',
      runs[0].quantumMs.n,
    );
  }
  const temps = json.results.map((r) => r.after.cpuMaxC);
  console.log('cpu max C', Math.min(...temps), Math.max(...temps), 'thermal', [
    ...new Set(json.results.map((r) => r.after.thermalStatus)),
  ]);
}
