import fs from 'node:fs';
import path from 'node:path';
import * as D from './device.mjs';
import { Overloads } from './overloads.mjs';
import { session, stands } from './session.mjs';

const [target, task, argJson] = process.argv.slice(2);
const arg = argJson ? JSON.parse(argJson) : {};
const tasks = ['env', 'correctness', 'controls', 'stretch', 'latency', 'underrun', 'memory'];
if (!tasks.includes(task)) {
  console.error(`usage: node page_pool.mjs <target> <${tasks.join('|')}> [json args]`);
  process.exit(2);
}
const save = (name, obj) => {
  const file = D.outDir(`page_pool-${target}-${name}.json`);
  fs.writeFileSync(file, JSON.stringify(obj, null, 1));
};

await session(
  target,
  async ({ open, call, meta, env, gate, device }) => {
    const fresh = async () => {
      const page = await open('/web/index.html');
      await page.waitForFunction(() => Boolean(window.ready));
      await D.sleep(1000);
      return page;
    };
    const T = (page, fn, a) => call(page, ([f, x]) => window.T[f](x), [fn, a]);
    console.log('pre', JSON.stringify(await gate()));
    const base = meta();
    if (task === 'env') {
      const page = await fresh();
      save('env', { ...base, env: await T(page, 'env'), after: env() });
    }
    if (task === 'correctness' || task === 'controls' || task === 'stretch') {
      const page = await fresh();
      const out = { ...base, rows: [] };
      if (task === 'correctness') {
        for (const sc of arg.scenarios ?? ['plain', 'seeks', 'loop', 'loopBoundary']) {
          const r = await call(page, ([n]) => window.T.scenario(n, {}), [sc]);
          out[sc] = r;
          console.log(sc, 'pass', r.main.pass, 'rolling', r.main.rolling, 'mismatchQ', r.main.mismatchQuanta, 'posBad', r.main.posBad,
            'silentBad', r.main.silentBad, JSON.stringify(r.main.covered), 'bufQ', r.nodeStats.bufQuanta);
        }
      }
      if (task === 'controls') {
        for (const sc of ['seeks', 'loop']) {
          out[sc] = await call(page, ([n]) => window.T.scenario(n, { controls: true }), [sc]);
          console.log(sc, 'clean', out[sc].controlsOnClean,
            (out[sc].controls ?? []).map((c) => `${c.failed ? 'FAIL-as-expected' : 'MISSED'}:${c.name}`).join(' | '));
        }
        for (const bug of ['edge', 'loopGain', 'swap', 'shift']) {
          out['bug_' + bug] = await call(page, ([n, b]) => window.T.scenario(n, { bug: b }), [bug === 'loopGain' ? 'loop' : 'plain', bug]);
          console.log('bug', bug, 'pass(expect false)', out['bug_' + bug].main.pass, 'mismatchQ', out['bug_' + bug].main.mismatchQuanta);
        }
      }
      if (task === 'stretch') {
        for (const [name, tempo, n] of [['plain', 0.75, 1], ['plain', 0.75, 3], ['loop', 0.75, 1], ['loop', 1.5, 3], ['seeks', 0.75, 1], ['seeks', 1.25, 3]]) {
          const r = await call(page, ([a, b, c]) => window.T.stretchScenario(a, b, c), [name, tempo, n]);
          out.rows.push(r);
          console.log(name, tempo, n, 'input pass', r.input?.pass, 'mismatchQ', r.input?.mismatchQuanta, 'bufQ', r.nodeStats?.bufQuanta, 'errors', r.nodeStats?.errors);
        }
      }
      out.after = env();
      save(task, out);
    }
    if (task === 'latency') {
      const n = arg.n ?? 200;
      const out = { ...base, rounds: [] };
      const order = [['whole', 'warm'], ['pages', 'warm'], ['pages', 'cold'], ['whole', 'cold']];
      for (let round = 0; round < (arg.rounds ?? 1); round++) {
        for (const [kind, mode] of round % 2 ? [...order].reverse() : order) {
          const before = await gate();
          const page = await fresh();
          const r = await call(page, ([k, m, nn]) => window.T.seekLatency({ kind: k, mode: m, n: nn }), [kind, mode, n]);
          delete r.rows;
          Object.assign(r, { before, after: env(), round });
          out.rounds.push(r);
          console.log(kind, mode, 'lat', JSON.stringify(r.latMs), 'wall', JSON.stringify(r.wallMs), 'timeouts', r.timeouts, 'warmFlag', r.warmFlagged);
          save('latency' + (arg.tag ?? ''), out);
        }
      }
    }
    if (task === 'underrun') {
      const overloads = new Overloads(target);
      const dur = arg.durationMs ?? 20000;
      const hint = arg.latencyHint ?? 'interactive';
      const out = { ...base, rows: [] };
      const defs = arg.defs ?? [
        { name: 'plain' },
        { name: 'tempo075_1st', tempo: 0.75, stretchers: 1 },
        { name: 'tempo075_3st', tempo: 0.75, stretchers: 3 },
        { name: 'tempo075_3st_aligned', tempo: 0.75, stretchers: 3, stagger: false },
        { name: 'loop4s', loop: true },
        { name: 'loop4s_tempo075_3st', loop: true, tempo: 0.75, stretchers: 3 },
        { name: 'seeks500_cold', seekEveryMs: 500, seekMode: 'cold' },
        { name: 'seeks500_near', seekEveryMs: 500, seekMode: 'near' },
        { name: 'jank200_1s', jank: true },
        { name: 'jank200_1s_tempo075_3st', jank: true, tempo: 0.75, stretchers: 3 },
        { name: 'whole_plain', kind: 'whole' },
        { name: 'whole_seeks500_cold', kind: 'whole', seekEveryMs: 500, seekMode: 'cold' },
        { name: 'whole_jank200_1s', kind: 'whole', jank: true },
        { name: 'whole_loop4s', kind: 'whole', loop: true },
      ];
      try {
        for (const d of defs) {
          const before = await gate();
          const page = await fresh();
          const from = Date.now();
          const r = await call(page, (a) => window.T.underrun(a), { ...d, durationMs: dur, latencyHint: hint });
          const to = Date.now();
          await D.sleep(1500);
          r.overloads = overloads.count(from, to);
          r.ctxBehindWallMs = +(r.wallMs - r.ctxSeconds * 1000).toFixed(1);
          Object.assign(r, { before, after: env() });
          out.rows.push(r);
          console.log(d.name.padEnd(26), 'ps', JSON.stringify(r.playbackStatsDelta), 'overloads', JSON.stringify(r.overloads),
            'gaps>=6/11/21/41ms', [r.spike?.gap6, r.spike?.gap11, r.spike?.gap21, r.spike?.gap41].join('/'), 'gapMax', r.gapMaxMs,
            'ctx-wall ms', r.ctxBehindWallMs, 'bufQ', r.spike?.bufQuanta, 'q', r.spike?.quanta, 'exp', r.quantaExpected,
            'procMeanUs', r.spike?.procMs ? +((r.spike.procMs / r.spike.quanta) * 1000).toFixed(1) : '-', 'heavy', r.spike?.procHeavy);
          save('underrun' + (arg.tag ?? ''), out);
        }
      } finally {
        overloads.stop();
      }
    }
    if (task === 'memory') {
      const out = { ...base, runs: [] };
      const kinds = arg.kinds ?? ['idle', 'whole', 'pages'];
      for (let rep = 0; rep < (arg.reps ?? 2); rep++) {
        for (const kind of rep % 2 ? [...kinds].reverse() : kinds) {
          const before = await gate();
          const page = await fresh();
          const footprint = new D.Footprint(target, device.responsiblePid);
          await D.sleep(2500);
          const idle = footprint.peaks(0, Date.now());
          const from = Date.now();
          const r = await call(page, ([k]) => window.T.memory({ kind: k, playMs: 14000 }), [kind]);
          await D.sleep(1500);
          const peaks = footprint.peaks(from, Date.now());
          await footprint.stop();
          const top = peaks[0] ?? {};
          const row = {
            kind, rep, ...r, before, after: env(),
            idleProcs: idle,
            peakProcs: peaks,
            pagePid: top.pid,
            pagePeakPolledMB: top.polledMaxMB,
            pageLifetimeMaxMB: top.lifetimeMaxMB,
            nSamples: footprint.samples.length,
          };
          out.runs.push(row);
          console.log(kind, rep, 'idle', JSON.stringify(idle.map((p) => p.polledMaxMB)), 'page peak polled', top.polledMaxMB,
            'lifetime max', top.lifetimeMaxMB, 'samples', footprint.samples.length);
          save('memory' + (arg.tag ?? ''), out);
        }
      }
    }
  },
  { roots: [path.join(stands, 'page_pool_spike')] },
);
