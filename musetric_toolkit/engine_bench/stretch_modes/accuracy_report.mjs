import fs from 'node:fs';
const R = JSON.parse(fs.readFileSync('results/accuracy.json','utf8'));
const ms = (x) => (x / 48).toFixed(2);
const st = (a) => { if (!a.length) return null; const s=[...a].sort((x,y)=>x-y); const ab=a.map(Math.abs).sort((x,y)=>x-y); return {n:a.length, med:s[Math.floor(s.length/2)], max:ab[ab.length-1]}; };
const line = (arr) => { const s = st(arr); return s ? `${ms(s.med)} / ${ms(s.max)} (n=${s.n})` : 'n/a'; };
console.log('CONTROLS'); for (const c of R.controls) console.log(' ', c.name, '|', c.expect, '|', c.got);
console.log('\nSTEADY (age>=150ms): median / max|offset| ms, [bad = found false count]');
for (const model of ['product','stream']) {
  console.log('model', model);
  for (const tr of [0,3]) for (const tp of [0.5,0.75,1.25,1.5]) {
    const cells = ['V0','V1','V2','V3'].map((vn) => {
      const e = R.steady.find((x)=>x.vn===vn&&x.tr===tr&&x.tp===tp);
      const rows = e.rows.filter((r)=>r.model===model);
      if (!rows.length) return '-';
      const ok = rows.filter((r)=>r.found&&r.age>=7200);
      const bad = rows.filter((r)=>r.age>=7200&&!r.found).length;
      return `${vn}: ${line(ok.map((r)=>r.offset))}${bad?` bad${bad}`:''}`;
    });
    console.log(` tr${tr} t${tp}  ${cells.join(' | ')}`);
  }
}
console.log('\nSTEADY early (age<150ms) first click');
for (const model of ['product','stream']) for (const vn of ['V0','V1','V2','V3']) {
  const o=[]; for (const e of R.steady) if(e.vn===vn) for (const r of e.rows) if(r.model===model&&r.age<7200) o.push(r.found?ms(r.offset):('x:'+r.why));
  if (o.length) console.log(' ',model,vn,o.join(' '));
}
console.log('\nWRAP: clicks by index after wrap (j0 first, j1 second, later) and pre-wrap; median / max|off| ms');
for (const model of ['product','stream']) {
  console.log('model', model);
  for (const tr of [0,3]) for (const tp of [0.5,0.75,1.25,1.5]) {
    const cells = ['V0','V1','V2','V3'].map((vn) => {
      const e = R.wrap.find((x)=>x.vn===vn&&x.tr===tr&&x.tp===tp);
      const rows = e.rows.filter((r)=>r.model===model&&r.age>=7200);
      if(!rows.length) return '-';
      const j = (r)=> Math.round((r.c-(10*48000+0.137*48000))/24000);
      const g = (f)=> rows.filter((r)=>r.wrap>=1&&f(j(r))&&r.found).map((r)=>r.offset);
      const bad = rows.filter((r)=>r.wrap>=1&&!r.found).length;
      return `${vn}: j0 ${line(g(x=>x===0))}; j1 ${line(g(x=>x===1))}; j2+ ${line(g(x=>x>=2))}; pre ${line(rows.filter((r)=>r.wrap===0&&r.found).map(r=>r.offset))}${bad?` bad${bad}`:''}`;
    });
    console.log(` tr${tr} t${tp}\n    ${cells.join('\n    ')}`);
  }
}
console.log('\nSEEK: first click after seek (d = song s from target to click): offset ms (ratio) ; later clicks >=150ms median/max');
for (const model of ['product','stream']) {
  console.log('model', model);
  for (const tp of [0.75,1.25]) for (const vn of ['V0','V1','V2','V3']) {
    const cells = [0.03,0.12,0.2,0.35].map((d)=>{
      const e = R.seek.find((x)=>x.vn===vn&&x.tp===tp&&x.d===d);
      const rows = e.rows.filter((r)=>r.model===model&&r.seg>=1).sort((a,b)=>a.tExp-b.tExp);
      const f = rows[0];
      return f ? `d${d}:${f.found?ms(f.offset):'x'+f.why}(${f.ratio?.toFixed(2)})` : 'd'+d+':none';
    });
    const e = R.seek.filter((x)=>x.vn===vn&&x.tp===tp);
    const late = e.flatMap((x)=>x.rows.filter((r)=>r.model===model&&r.seg>=1&&r.age>=7200&&r.found).map((r)=>r.offset));
    console.log(` t${tp} ${vn}: ${cells.join(' ')} | later ${line(late)}`);
  }
}
console.log('\nTEMPO CHANGE: transition(<250ms after change, ms-offsets) and settled(>=250ms) median/max');
for (const model of ['product','stream']) {
  console.log('model', model);
  for (const [t0,t1,tr] of [[1,0.75,3],[0.75,1.5,3],[1.25,0.5,3],[1,0.75,0],[1.5,1,3]]) for (const vn of ['V0','V1','V2','V3']) {
    const e = R.tempo.find((x)=>x.vn===vn&&x.t0===t0&&x.t1===t1&&x.tr===tr);
    const rows = e.rows.filter((r)=>r.model===model&&r.age>=7200);
    const tc = e.kChange*128;
    const pre = rows.filter((r)=>r.tExp<tc&&r.found).map(r=>r.offset);
    const tran = rows.filter((r)=>r.tExp>=tc&&r.tExp<tc+12000);
    const set = rows.filter((r)=>r.tExp>=tc+12000&&r.found).map(r=>r.offset);
    console.log(` ${t0}->${t1} tr${tr} ${vn}: pre ${line(pre)} | trans ${tran.map(r=>r.found?ms(r.offset):'x'+r.why).join(',')} | settled ${line(set)}`);
  }
}
