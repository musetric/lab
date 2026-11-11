import fs from 'node:fs';
const r = JSON.parse(fs.readFileSync('results/timing.json','utf8'));
const f = (x,d=0)=>x.toFixed(d);
const row = (x)=>`${x.v.padEnd(9)} med ${f(x.med,1).padStart(6)} mean ${f(x.mean,1).padStart(6)} p99 ${f(x.p99).padStart(5)} p99.9 ${f(x.p999).padStart(5)} max ${f(x.max).padStart(5)} >50% ${(x.over50*100).toFixed(3)} >100% ${(x.over100*100).toFixed(3)}`;
for (const ur of [false,true]) for (const tr of [3,0]) for (const tp of [0.5,0.75,1.25,1.5]) {
  console.log(`\nrange=${ur} tr=${tr} tempo=${tp}`);
  for (const x of r.rows.filter(x=>x.range===ur&&x.tr===tr&&x.tempo===tp)) console.log(' ',row(x));
}
console.log('\nFINE');
for (const x of r.fine) console.log(x.range?'R':'-',x.tempo,x.v.padEnd(6),'tot med/mean',f(x.total.med,1),f(x.total.mean,1),'p99',f(x.total.p99),'| js mean',f(x.js.mean,1),'p99',f(x.js.p99),'| seek mean',f(x.seek.mean,1),'| wasm-process mean',f(x.proc.mean,1),'med',f(x.proc.med,1),'p99',f(x.proc.p99),'max',f(x.proc.max),'| copy',f(x.copy.mean,1));
