import fs from 'node:fs';
import * as device from './device.mjs';
import { startServer } from '../aac_pages/server.mjs';

const [label, taskArg] = process.argv.slice(2);
if (!label || !taskArg) {
  console.error('usage: node decode_pages.mjs <chrome|webview> <task[,task]>');
  console.error('tasks: env controls indexcheck correctness bench parallel flac');
  process.exit(2);
}
const tasks = taskArg.split(',');
const names = ['chord', 'noise', 'clicks', 'sweep', 'mix240'];

const save = (task, obj) => {
  const file = device.outDir(`decode_pages-${label}-${task}.json`);
  fs.writeFileSync(file, JSON.stringify(obj, null, 1));
  console.log('saved', file);
};
const call = (page, id, cmd, args) => page.evaluate(([i, c, a]) => window.callWorker(i, c, a), [id, cmd, args]);

const { server, port } = await startServer();
device.mapPort(port);
let target;
try {
  device.wake();
  console.log('pre', await device.gate());
  target = await device.openTarget(device.targetOf(label), `http://localhost:${port}/web/index.html`);
  const { page } = target;
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  for (const task of tasks) {
    const started = Date.now();
    const before = device.env();
    console.log(task, JSON.stringify(before));
    const out = {};
    if (task === 'env') out.env = await call(page, 'w', 'env');
    if (task === 'controls') out.controls = await call(page, 'w', 'controls');
    if (task === 'indexcheck') {
      for (const n of names) out[n] = await call(page, 'w', 'indexcheck', { name: n });
    }
    if (task === 'correctness') {
      for (const n of names.slice(0, 4)) {
        out[n] = await call(page, 'w', 'correctness', { name: n });
        const perP = out[n].perP;
        const rows = Object.entries(perP).map(([k, v]) => `P${k}:${v.pagesOk}/${v.pages} max=${v.maxDiff}`);
        console.log(n, 'wholeFrames', out[n].wholeFrames, rows.join(' '));
      }
    }
    if (task === 'bench') {
      out.decoder = await call(page, 'w', 'benchDecoder', { name: 'mix240' });
      out.seek = await call(page, 'w', 'benchSeek', { name: 'mix240' });
      out.seekChord = await call(page, 'w', 'benchSeek', { name: 'chord' });
      out.seekMfra = await call(page, 'w', 'benchSeek', { name: 'mix240_mfra' });
    }
    if (task === 'parallel') {
      out.oneWorker = await call(page, 'w', 'benchParallel', { names: ['chord', 'sweep', 'clicks'], iterations: 20 });
      const iterations = 30;
      const stems = ['chord', 'sweep', 'clicks'];
      const seq = await call(page, 'a', 'loopPages', { name: 'chord', iterations: iterations * 3 });
      for (const s of stems) await call(page, 'w' + s, 'env');
      const startAt = Date.now() + 1500;
      const par = await Promise.all(
        stems.map((s) => call(page, 'w' + s, 'loopPages', { name: s, iterations, startAt })),
      );
      out.threeWorkers = {
        pagesPerStem: iterations,
        sequential_oneStem_x3_ms: +seq.ms.toFixed(0),
        parallel_three_workers_each_ms: par.map((p) => +p.ms.toFixed(0)),
        parallelWallApprox_ms: Math.max(...par.map((p) => p.ms)),
      };
    }
    if (task === 'flac') {
      for (const n of names.slice(0, 4).concat('mix240')) {
        out[n] = await call(page, 'w', 'flac', { name: n });
        console.log(n, JSON.stringify(out[n].pagesAll ?? out[n].variants));
      }
    }
    save(task, {
      label,
      device: device.versions(),
      before,
      after: device.env(),
      seconds: (Date.now() - started) / 1000,
      ...out,
    });
  }
} finally {
  await target?.close();
  device.cleanup();
  server.close();
}
process.exit(0);
