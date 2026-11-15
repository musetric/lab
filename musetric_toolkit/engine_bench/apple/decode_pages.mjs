import fs from 'node:fs';
import path from 'node:path';
import * as D from './device.mjs';
import { session, stands } from './session.mjs';

const [target, taskArg] = process.argv.slice(2);
if (!taskArg) {
  console.error('usage: node decode_pages.mjs <target> <env,controls,indexcheck,correctness,bench,parallel,flac|mem>');
  process.exit(2);
}
const tasks = taskArg.split(',');
const names = ['chord', 'noise', 'clicks', 'sweep', 'mix240'];
const aacPages = path.join(stands, 'aac_pages');

await session(
  target,
  async ({ open, meta, env, gate, device }) => {
    console.log('pre', JSON.stringify(await gate()));
    if (tasks.includes('mem')) {
      const res = {};
      const scenarios = [
        ['idle', 'memIdle', {}],
        ['whole3', 'memWhole', { name: 'mix240', stems: 3 }],
        ['pages3_ring5', 'memPages', { name: 'mix240', stems: 3, ringPages: 5 }],
        ['whole1', 'memWhole', { name: 'mix240', stems: 1 }],
        ['pages1_ring5', 'memPages', { name: 'mix240', stems: 1, ringPages: 5 }],
      ];
      for (const [label, cmd, args] of scenarios) {
        for (let rep = 0; rep < 2; rep++) {
          const before = await gate();
          const page = await open('/web/index.html');
          await page.waitForFunction(() => Boolean(window.ready));
          const footprint = new D.Footprint(target, device.responsiblePid, 250);
          await D.sleep(1500);
          const from = Date.now();
          const r = await page.evaluate(([c, a]) => window.callWorker('m', c, a), [cmd, args]);
          await D.sleep(1500);
          const peaks = footprint.peaks(from, Date.now());
          await footprint.stop();
          const top = peaks[0] ?? {};
          Object.assign(r, { before, peaks, pagePeakPolledMB: top.polledMaxMB, pageLifetimeMaxMB: top.lifetimeMaxMB });
          (res[label] ??= []).push(r);
          console.log(label, rep, 'page peak polled', top.polledMaxMB, 'lifetime max', top.lifetimeMaxMB, 'ms', r.ms);
        }
      }
      fs.writeFileSync(D.outDir(`decode_pages-${target}-mem.json`), JSON.stringify({ ...meta(), ...res }, null, 1));
      return;
    }
    const page = await open('/web/index.html');
    await page.waitForFunction(() => Boolean(window.ready));
    const call = (id, cmd, args) => page.evaluate(([i, c, a]) => window.callWorker(i, c, a), [id, cmd, args]);
    for (const task of tasks) {
      const started = Date.now();
      const before = env();
      console.log(task, JSON.stringify(before));
      const out = {};
      if (task === 'env') out.env = await call('w', 'env');
      if (task === 'controls') out.controls = await call('w', 'controls');
      if (task === 'indexcheck') {
        for (const n of names) out[n] = await call('w', 'indexcheck', { name: n });
      }
      if (task === 'correctness') {
        for (const n of names.slice(0, 4)) {
          out[n] = await call('w', 'correctness', { name: n });
          const rows = Object.entries(out[n].perP).map(([k, v]) => `P${k}:${v.pagesOk}/${v.pages} max=${v.maxDiff}`);
          console.log(n, 'wholeFrames', out[n].wholeFrames, rows.join(' '));
        }
      }
      if (task === 'bench') {
        out.decoder = await call('w', 'benchDecoder', { name: 'mix240' });
        out.seek = await call('w', 'benchSeek', { name: 'mix240' });
        out.seekChord = await call('w', 'benchSeek', { name: 'chord' });
        out.seekMfra = await call('w', 'benchSeek', { name: 'mix240_mfra' });
      }
      if (task === 'parallel') {
        out.oneWorker = await call('w', 'benchParallel', { names: ['chord', 'sweep', 'clicks'], iterations: 20 });
        const iterations = 30;
        const stems = ['chord', 'sweep', 'clicks'];
        const seq = await call('a', 'loopPages', { name: 'chord', iterations: iterations * 3 });
        for (const s of stems) await call('w' + s, 'env');
        const startAt = Date.now() + 1500;
        const par = await Promise.all(stems.map((s) => call('w' + s, 'loopPages', { name: s, iterations, startAt })));
        out.threeWorkers = {
          pagesPerStem: iterations,
          sequential_oneStem_x3_ms: +seq.ms.toFixed(0),
          parallel_three_workers_each_ms: par.map((p) => +p.ms.toFixed(0)),
          parallelWallApprox_ms: Math.max(...par.map((p) => p.ms)),
        };
      }
      if (task === 'flac') {
        for (const n of names.slice(0, 4).concat('mix240')) {
          try {
            out[n] = await call('w', 'flac', { name: n });
          } catch (error) {
            out[n] = { error: String(error.message).split('\n')[0] };
          }
          console.log(n, JSON.stringify(out[n].pagesAll ?? out[n].variants ?? out[n].error));
        }
      }
      const file = D.outDir(`decode_pages-${target}-${task}.json`);
      fs.writeFileSync(file, JSON.stringify({ ...meta(), before, seconds: (Date.now() - started) / 1000, ...out }, null, 1));
      console.log('saved', file, `${((Date.now() - started) / 1000).toFixed(0)} s`);
    }
  },
  {
    roots: [aacPages],
    routes: { '/mediabunny.mjs': path.join(aacPages, 'node_modules/mediabunny/dist/bundles/mediabunny.min.mjs') },
    isolated: true,
  },
);
