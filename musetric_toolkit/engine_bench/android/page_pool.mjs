import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import * as L from './device.mjs';
import { here, startServer } from './server.mjs';
const [task, argJson] = process.argv.slice(2);
const arg = argJson ? JSON.parse(argJson) : {};
if (!['env', 'correctness', 'controls', 'stretch', 'latency', 'underrun', 'memory'].includes(task)) {
  console.error('usage: node page_pool.mjs <env|correctness|controls|stretch|latency|underrun|memory> [json args]');
  process.exit(2);
}
const save = (name, obj) => {
  const f = L.outDir(`page_pool-${name}.json`);
  fs.writeFileSync(f, JSON.stringify(obj, null, 1));
  console.log('saved', f);
};
const { server, port } = await startServer({ roots: [path.resolve(here, '../page_pool_spike')] });
L.mapPort(port);
const sleep = L.sleep;

const memProcs = () =>
  new Promise((resolve) =>
    execFile(
      L.ADB,
      [
        'shell',
        'ps -A -o PID,NAME | grep com.android.chrome | while read p n; do echo "$p $n"; dumpsys meminfo $p | grep "TOTAL PSS"; done',
      ],
      { maxBuffer: 1 << 26 },
      (e, out) => {
        const procs = [];
        const lines = (out ?? '').split(/\r?\n/);
        for (let i = 0; i < lines.length; i++) {
          const m = /^(\d+) (\S+)/.exec(lines[i]);
          if (!m) continue;
          const t = /TOTAL PSS:\s+(\d+)\s+TOTAL RSS:\s+(\d+)/.exec(lines[i + 1] ?? '');
          if (!t || /zygote/.test(m[2])) continue;
          procs.push({
            name: m[2]
              .replace('com.android.chrome', 'chrome')
              .replace(/:org\.chromium\.content\.app\.SandboxedProcessService\d+/, ''),
            pid: +m[1],
            pssMB: +(+t[1] / 1024).toFixed(1),
            rssMB: +(+t[2] / 1024).toFixed(1),
          });
        }
        resolve(procs);
      },
    ),
  );
const fresh = async () => {
  L.chromeFresh();
  await sleep(3500);
  const browser = await L.connect();
  const page = await L.openPage(browser, `http://localhost:${port}/web/index.html`);
  await page.waitForFunction(() => window.ready);
  L.front('chrome');
  await sleep(1000);
  L.assertFront('chrome');
  await L.tapActivation();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') console.log('console error', m.text().slice(0, 200));
  });
  page.setDefaultTimeout(900000);
  return { browser, page };
};
const done = async ({ browser, page }) => {
  try {
    await page.close();
    await browser.close();
  } catch {}
};
const T = (page, fn, a) => page.evaluate(([f, x]) => window.T[f](x), [fn, a]);
const envBase = () => ({ device: L.versions(), before: L.env() });
try {
  L.wake();
  const g = await L.gate();
  console.log('pre', JSON.stringify(g));
  const meta = envBase();
  if (task === 'env') {
    const s = await fresh();
    const out = { ...meta, env: await T(s.page, 'env') };
    out.after = L.env();
    save('env', out);
    await done(s);
  }
  if (task === 'correctness' || task === 'controls' || task === 'stretch') {
    const s = await fresh();
    const out = { ...meta, rows: [] };
    if (task === 'correctness')
      for (const sc of arg.scenarios ?? ['plain', 'seeks', 'loop', 'loopBoundary']) {
        const r = await s.page.evaluate(([n]) => window.T.scenario(n, {}), [sc]);
        out[sc] = r;
        console.log(
          sc,
          'pass',
          r.main.pass,
          'rolling',
          r.main.rolling,
          'mismatchQ',
          r.main.mismatchQuanta,
          'posBad',
          r.main.posBad,
          'silentBad',
          r.main.silentBad,
          JSON.stringify(r.main.covered),
          'bufQ',
          r.nodeStats.bufQuanta,
        );
      }
    if (task === 'controls') {
      for (const sc of ['seeks', 'loop']) {
        out[sc] = await s.page.evaluate(([n]) => window.T.scenario(n, { controls: true }), [sc]);
        console.log(
          sc,
          'clean',
          out[sc].controlsOnClean,
          (out[sc].controls ?? []).map((c) => `${c.failed ? 'FAIL-as-expected' : 'MISSED'}:${c.name}`).join(' | '),
        );
      }
      for (const bug of ['edge', 'loopGain', 'swap', 'shift']) {
        out['bug_' + bug] = await s.page.evaluate(
          ([n, b]) => window.T.scenario(n, { bug: b }),
          [bug === 'loopGain' ? 'loop' : 'plain', bug],
        );
        console.log(
          'bug',
          bug,
          'pass(expect false)',
          out['bug_' + bug].main.pass,
          'mismatchQ',
          out['bug_' + bug].main.mismatchQuanta,
          'posBad',
          out['bug_' + bug].main.posBad,
        );
      }
    }
    if (task === 'stretch')
      for (const [name, tempo, n] of [
        ['plain', 0.75, 1],
        ['plain', 0.75, 3],
        ['loop', 0.75, 1],
        ['loop', 1.5, 3],
        ['seeks', 0.75, 1],
        ['seeks', 1.25, 3],
      ]) {
        const r = await s.page.evaluate(([a, b, c]) => window.T.stretchScenario(a, b, c), [name, tempo, n]);
        out.rows.push(r);
        console.log(
          name,
          tempo,
          n,
          'input pass',
          r.input?.pass,
          'mismatchQ',
          r.input?.mismatchQuanta,
          'bufQ',
          r.nodeStats?.bufQuanta,
          'errors',
          r.nodeStats?.errors,
        );
      }
    out.after = L.env();
    save(task, out);
    await done(s);
  }
  if (task === 'latency') {
    const n = arg.n ?? 200;
    const out = { ...meta, rounds: [] };
    const order = [
      ['whole', 'warm'],
      ['pages', 'warm'],
      ['pages', 'cold'],
      ['whole', 'cold'],
    ];
    for (let round = 0; round < (arg.rounds ?? 1); round++)
      for (const [kind, mode] of round % 2 ? [...order].reverse() : order) {
        await L.gate();
        const s = await fresh();
        const before = L.env();
        const r = await s.page.evaluate(
          ([k, m, nn]) => window.T.seekLatency({ kind: k, mode: m, n: nn }),
          [kind, mode, n],
        );
        delete r.rows;
        r.before = before;
        r.after = L.env();
        r.round = round;
        out.rounds.push(r);
        console.log(
          kind,
          mode,
          'lat',
          JSON.stringify(r.latMs),
          'wall',
          JSON.stringify(r.wallMs),
          'timeouts',
          r.timeouts,
          'warmFlag',
          r.warmFlagged,
          'cpu',
          before.cpuMaxC,
          '->',
          r.after.cpuMaxC,
        );
        save('latency', out);
        await done(s);
      }
  }
  if (task === 'underrun') {
    const dur = arg.durationMs ?? 20000;
    const out = { ...meta, rows: [] };
    const hint = arg.latencyHint ?? 'interactive';
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
    for (const d of defs) {
      const th = await L.gate();
      const s = await fresh();
      L.assertFront('chrome');
      const r = await s.page.evaluate((a) => window.T.underrun(a), { ...d, durationMs: dur, latencyHint: hint });
      r.before = th;
      r.after = L.env();
      r.focusAfter = r.after.focus;
      out.rows.push(r);
      console.log(
        d.name.padEnd(26),
        'ps',
        JSON.stringify(r.playbackStatsDelta),
        'bufQ',
        r.spike?.bufQuanta,
        'q',
        r.spike?.quanta,
        'exp',
        r.quantaExpected,
        'procMeanUs',
        r.spike?.procMs ? +((r.spike.procMs / r.spike.quanta) * 1000).toFixed(1) : '-',
        'heavy',
        r.spike?.procHeavy,
        'cpu',
        th.cpuMaxC,
        '->',
        r.after.cpuMaxC,
        'th',
        r.after.thermalStatus,
      );
      save('underrun' + (arg.tag ?? ''), out);
      await done(s);
    }
  }
  if (task === 'memory') {
    const out = { ...meta, runs: [] };
    const kinds = arg.kinds ?? ['idle', 'whole', 'pages'];
    const reps = arg.reps ?? 2;
    for (let rep = 0; rep < reps; rep++)
      for (const kind of rep % 2 ? [...kinds].reverse() : kinds) {
        await L.gate();
        const s = await fresh();
        const idle = await memProcs();
        const samples = [];
        let go = true;
        const poll = (async () => {
          while (go) {
            samples.push({ t: Date.now(), procs: await memProcs() });
            await sleep(250);
          }
        })();
        const r = await s.page.evaluate(([k]) => window.T.memory({ kind: k, playMs: 14000 }), [kind]);
        await sleep(1500);
        go = false;
        await poll;
        const final = await memProcs();
        const sum = (ps) => +ps.reduce((a, p) => a + p.pssMB, 0).toFixed(1);
        const sumR = (ps) => +ps.reduce((a, p) => a + p.rssMB, 0).toFixed(1);
        const rendererPeak = Math.max(
          0,
          ...samples.map((x) => Math.max(0, ...x.procs.filter((p) => /sandboxed/.test(p.name)).map((p) => p.pssMB))),
        );
        const rendererPeakRss = Math.max(
          0,
          ...samples.map((x) => Math.max(0, ...x.procs.filter((p) => /sandboxed/.test(p.name)).map((p) => p.rssMB))),
        );
        const row = {
          kind,
          rep,
          ...r,
          idleSumMB: sum(idle),
          idleProcs: idle,
          peakSumMB: Math.max(sum(idle), ...samples.map((x) => sum(x.procs))),
          peakRendererPssMB: rendererPeak,
          peakRendererRssMB: rendererPeakRss,
          peakSumRssMB: Math.max(sumR(idle), ...samples.map((x) => sumR(x.procs))),
          peakRendererRssMB: rendererPeakRss,
          peakSumRssMB: Math.max(sumR(idle), ...samples.map((x) => sumR(x.procs))),
          finalSumMB: sum(final),
          finalProcs: final,
          nSamples: samples.length,
          after: L.env(),
        };
        out.runs.push(row);
        console.log(
          kind,
          rep,
          'idleSum',
          row.idleSumMB,
          'peakSum',
          row.peakSumMB,
          'peakRenderer',
          rendererPeak,
          'final',
          row.finalSumMB,
          'n',
          samples.length,
        );
        save('memory', out);
        await done(s);
      }
  }
} finally {
  L.cleanup();
  server.close();
}
process.exit(0);
