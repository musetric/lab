import { createModule, createProcessor, makeStems, render, SR } from './common.mjs';
const m = await createModule();
console.log(Object.keys(m).filter(k=>k.startsWith('_')).join(' '));
const st = makeStems(20);
for (const mode of ['const','stream']) {
  const p = createProcessor(m, SR, {mode});
  const t=performance.now();
  const r = render({procs:[p], groups:[[st.chord,st.drums,st.lead]], startFrame:48000, nQ:3750, tempo:0.75, transpose:3, timing:'total', record:true});
  console.log(mode, performance.now()-t, p.state.inputLatency, p.state.outputLatency, r.hist[100], Math.max(...r.rec[0].slice(60000,70000)));
}
