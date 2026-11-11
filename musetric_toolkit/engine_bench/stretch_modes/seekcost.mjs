import { createModule, createProcessor, createRangeFrames, makeStems, render, SR, Q } from './common.mjs';
const m = await createModule(); const st = makeStems(40);
const range = createRangeFrames({fromFrame:10*SR,toFrame:14*SR},'cycle');
for (const ur of [false,true]) for (const mode of ['const','stream']) {
  const p = createProcessor(m, SR, {mode});
  const evs = []; for (let i=1;i<=40;i++) evs.push({k:200+i*150,type:'seek',frame:(ur?10:5)*SR+i*7919});
  const r = render({procs:[p],groups:[[st.chord,st.drums,st.lead]],range:ur?range:undefined,startFrame:ur?10*SR:5*SR,nQ:6500,tempo:0.75,transpose:3,events:evs,timing:'total'});
  const at = evs.map(e=>r.times[e.k]).sort((a,b)=>a-b);
  const normal = Array.from(r.times.slice(300,5000)).sort((a,b)=>a-b);
  console.log('range',ur,mode,'seek-quantum us: med',at[20].toFixed(0),'max',at[39].toFixed(0),'| normal med',normal[Math.floor(normal.length/2)].toFixed(1),'p99',normal[Math.floor(normal.length*.99)].toFixed(0));
}
