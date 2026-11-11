import fs from 'node:fs';
import path from 'node:path';
import * as device from './device.mjs';
import { here, startServer } from './server.mjs';

const [label, block] = process.argv.slice(2);
if (!label || !block) {
  console.error('usage: node stretch_timing.mjs <chrome|webview> <main|tempo125|paced|webview|quick>');
  process.exit(2);
}
const target = device.targetOf(label);
const all = ['V0-1', 'V0-3', 'V0-3stag', 'V1-1', 'V1-3', 'V1-3stag', 'V2-1', 'V2-3', 'V3-1', 'V3-3'];
const jobs = [];
const add = (variants, options) => {
  for (let rep = 0; rep < (options.reps ?? 2); rep++) {
    for (const v of rep % 2 ? [...variants].reverse() : variants) {
      jobs.push({ v, tempo: 0.75, tr: 3, range: false, paced: false, nq: 22500, ...options, rep });
    }
  }
};
if (block === 'main') {
  add(all, { range: false });
  add(all, { range: true });
}
if (block === 'tempo125') add(['V0-3', 'V0-3stag', 'V1-3', 'V1-3stag'], { tempo: 1.25 });
if (block === 'paced') add(['V0-3', 'V0-3stag', 'V1-3', 'V1-3stag', 'V2-3', 'V1-1'], { paced: true, nq: 9000 });
if (block === 'webview') add(['V0-3', 'V0-3stag', 'V1-3', 'V1-3stag', 'V3-3'], { range: false });
if (block === 'quick') add(['V0-3', 'V1-3stag'], { nq: 4000, reps: 1 });

const tag = `stretch_timing-${label}-${block}`;
const results = [];
const { server, port } = await startServer({
  isolated: true,
  routes: { '/stretch_modes/common.mjs': path.join(here, '../stretch_modes/common.mjs') },
});
device.mapPort(port);
let opened;
try {
  device.wake();
  console.log('pre', JSON.stringify(await device.gate()));
  opened = await device.openTarget(target, `http://localhost:${port}/web/timing.html`);
  const { page } = opened;
  if (target === 'chrome') {
    device.front('chrome');
    await device.sleep(1500);
  }
  await page.waitForFunction(() => window.ready);
  device.assertFront(target);
  const init = await page.evaluate(() => window.send({ cmd: 'init' }));
  console.log('init', JSON.stringify(init));
  for (const j of jobs) {
    device.assertFront(target);
    const before = device.env();
    if (before.thermalStatus >= 2) {
      console.log('thermal stop', JSON.stringify(before));
      break;
    }
    const r = await page.evaluate((m) => window.send({ cmd: 'run', ...m }), {
      v: j.v,
      tempo: j.tempo,
      tr: j.tr,
      range: j.range,
      paced: j.paced,
      nq: j.nq,
    });
    const after = device.env();
    results.push({ ...r, rep: j.rep, before, after });
    const us = (v) => v.toFixed(0);
    console.log(
      `${j.v.padEnd(9)} t${j.tempo} R${+j.range} P${+j.paced} rep${j.rep}`,
      `med ${us(r.med)} p99 ${us(r.p99)} p99.9 ${us(r.p999)} max ${us(r.max)}`,
      `>50% ${r.over50} >100% ${r.over100} wall ${(r.wallMs / 1000).toFixed(1)}s`,
      `cpu ${before.cpuMaxC}->${after.cpuMaxC} th${after.thermalStatus}`,
    );
    fs.writeFileSync(
      device.outDir(`${tag}.json`),
      JSON.stringify({ label, block, device: device.versions(), init, results }, null, 1),
    );
  }
} finally {
  await opened?.close();
  device.cleanup();
  server.close();
}
process.exit(0);
