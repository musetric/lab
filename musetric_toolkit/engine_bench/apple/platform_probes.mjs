import fs from 'node:fs';
import path from 'node:path';
import * as D from './device.mjs';
import { session, stands } from './session.mjs';

const [target, task, argJson] = process.argv.slice(2);
const arg = argJson ? JSON.parse(argJson) : {};
const tasks = ['renderSize', 'callbacks', 'bug', 'clock', 'stallCheck', 'background'];
if (!tasks.includes(task)) {
  console.error(`usage: node platform_probes.mjs <target> <${tasks.join('|')}> [json args]`);
  process.exit(2);
}
const save = (name, obj) => fs.writeFileSync(D.outDir(`platform_probes-${target}-${name}.json`), JSON.stringify(obj, null, 1));

const fit = (xs, ys) => {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxy / sxx;
  return { slope: b, icpt: my - b * mx };
};
const q = (a, p) => {
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};
const clockStats = (rows) => {
  const ok = rows.filter((r) => r[2] > 0);
  const f = fit(ok.map((r) => r[1]), ok.map((r) => r[2]));
  const res = ok.map((r) => r[2] - (f.icpt + f.slope * r[1]));
  const ol = rows.map((r) => r[3]);
  const stuck = ok.filter((r, i) => i > 0 && r[1] === ok[i - 1][1]).length;
  return {
    n: rows.length,
    valid: ok.length,
    perfMsPerCtxSec: ok.length > 2 ? +f.slope.toFixed(4) : null,
    residualMs: ok.length > 2
      ? {
          min: +Math.min(...res).toFixed(3),
          p5: +q(res, 0.05).toFixed(3),
          p50: +q(res, 0.5).toFixed(3),
          p95: +q(res, 0.95).toFixed(3),
          max: +Math.max(...res).toFixed(3),
          spread: +(Math.max(...res) - Math.min(...res)).toFixed(3),
        }
      : null,
    outputLatencyMs: {
      min: +(Math.min(...ol) * 1000).toFixed(2),
      median: +(q(ol, 0.5) * 1000).toFixed(2),
      max: +(Math.max(...ol) * 1000).toFixed(2),
      distinct: new Set(ol.map((v) => +v.toFixed(4))).size,
      first: ol[0],
      last: ol.at(-1),
    },
    baseLatencyMs: +(rows[0][5] * 1000).toFixed(2),
    ctxTimeStuckSamples: stuck,
  };
};

await session(
  target,
  async ({ open, call, meta, env, gate, bridge, remote }) => {
    const fresh = async () => {
      const page = await open('/web/probe.html');
      await page.waitForFunction(() => Boolean(window.ready));
      await D.sleep(1000);
      return page;
    };
    const setup = (page, h) => call(page, (hh) => window.T.setup(typeof hh === 'object' ? hh : { latencyHint: hh }), h);
    console.log('pre', JSON.stringify(await gate()));
    if (task === 'stallCheck') {
      const page = await fresh();
      const out = { ...meta() };
      await setup(page, 0.02);
      out.stall = await call(page, (ms) => window.T.stallCheck(ms), arg.stallMs ?? 1500);
      out.after = env();
      console.log(JSON.stringify(out.stall));
      save('stallCheck', out);
    }
    if (task === 'renderSize') {
      const page = await fresh();
      const out = { ...meta(), rows: [] };
      for (const h of [undefined, 'default', 'hardware', 64, 128, 256, 512]) {
        const r = await call(page, (hh) => window.T.renderSize(hh ?? undefined), h ?? null);
        out.rows.push(r);
        console.log(JSON.stringify(r));
      }
      out.after = env();
      save('renderSize', out);
    }
    if (task === 'callbacks') {
      const out = { ...meta(), rounds: [] };
      const hints = arg.hints ?? [0.02, 'interactive', 'playback'];
      for (let round = 0; round < (arg.rounds ?? 2); round++) {
        for (const h of round % 2 ? [...hints].reverse() : hints) {
          const before = await gate();
          const page = await fresh();
          await setup(page, h);
          const r = await call(page, (x) => window.T.callbacks(x), { seconds: arg.seconds ?? 15 });
          Object.assign(r, { hint: h, round, before, after: env() });
          out.rounds.push(r);
          console.log(String(h).padEnd(12), 'info', JSON.stringify(r.info), '| burst p50/p99/max', r.burstSizeP50, r.burstSizeP99,
            r.burstSizeMax, 'hist', JSON.stringify(r.burstSizeHist), 'gap p50/p95/p99/max ms', r.gapMsP50, r.gapMsP95, r.gapMsP99,
            r.gapMsMax, 'q/s', r.quantaPerSec.toFixed(1), 'noncontig', r.nonContiguous);
          save('callbacks', out);
        }
      }
    }
    if (task === 'bug') {
      const out = { ...meta(), trials: [] };
      const trials = arg.trials ?? [
        { name: 'stall1500_susres', stallMs: 1500, doSuspend: true },
        { name: 'nostall_susres(control)', stallMs: 0, doSuspend: true },
        { name: 'stall1500_nosuspend(control)', stallMs: 1500, doSuspend: false },
        { name: 'stall200_susres', stallMs: 200, doSuspend: true },
        { name: 'stall5000_susres', stallMs: 5000, doSuspend: true },
      ];
      for (const h of arg.hints ?? [0.02, 'interactive']) {
        for (let rep = 0; rep < (arg.reps ?? 2); rep++) {
          for (const t of rep % 2 ? [...trials].reverse() : trials) {
            const before = await gate();
            const page = await fresh();
            await setup(page, h);
            const r = await call(page, (x) => window.T.bug560233271(x), t);
            Object.assign(r, { name: t.name, hint: h, rep, before, after: env() });
            out.trials.push(r);
            console.log(String(h).padEnd(11), t.name.padEnd(30), 'rep', rep, 'q/s base', r.baseline.quantaPerSec, 'afterStall',
              r.afterStall?.quantaPerSec ?? '-', 'afterResume', r.afterResume?.quantaPerSec ?? '-', r.afterResume2?.quantaPerSec ?? '-',
              'ctxRate', r.afterResume?.ctxTimeRate ?? '-', 'susres', JSON.stringify(r.sr ?? {}));
            save('bug', out);
          }
        }
      }
    }
    if (task === 'clock') {
      const out = { ...meta(), runs: [] };
      for (const h of arg.hints ?? [0.02, 'interactive']) {
        const before = await gate();
        const page = await fresh();
        await setup(page, h);
        const info0 = await page.evaluate(() => window.T.info());
        const rows1 = await call(page, (x) => window.T.clock(x), { seconds: arg.seconds ?? 60, stepMs: 100 });
        const sr = await call(page, () => window.T.suspendResume());
        await D.sleep(500);
        const info1 = await page.evaluate(() => window.T.info());
        const rows2 = await call(page, (x) => window.T.clock(x), { seconds: arg.seconds2 ?? 30, stepMs: 100 });
        const r = { hint: h, info0, info1, sr, before, after: env(), beforeSR: clockStats(rows1), afterSR: clockStats(rows2), firstRowsAfter: rows2.slice(0, 5) };
        out.runs.push(r);
        console.log(String(h), 'baseLat', r.beforeSR.baseLatencyMs, 'before', JSON.stringify(r.beforeSR.residualMs),
          JSON.stringify(r.beforeSR.outputLatencyMs), 'slope', r.beforeSR.perfMsPerCtxSec, '| after', JSON.stringify(r.afterSR.residualMs),
          JSON.stringify(r.afterSR.outputLatencyMs), 'slope', r.afterSR.perfMsPerCtxSec, JSON.stringify(sr));
        out.runs.at(-1).rows1 = rows1.filter((_, i) => i % 5 === 0);
        out.runs.at(-1).rows2 = rows2.filter((_, i) => i % 5 === 0);
        save('clock', out);
      }
    }
    if (task === 'background') {
      if (!bridge) {
        throw new Error('background runs on the iPhone only');
      }
      const out = { ...meta(), runs: [] };
      const trials = arg.trials ?? [
        ...[0.02, 'interactive'].flatMap((h) => [false, true].map((sound) => ({ hint: h, sound, away: 10 }))),
        { hint: 0.02, sound: false, away: 60 },
      ];
      for (const { hint: h, sound, away } of trials) {
        const before = await gate();
        const page = await fresh();
        await setup(page, h);
        await page.evaluate((on) => window.T.tone(on), sound);
        const arm = await page.evaluate(() => window.T.bgArm());
        await D.sleep(2000);
        const away0 = Date.now();
        bridge.home(away);
        await D.sleep(away * 1000 + 3000);
        const awayMs = Date.now() - away0;
        const reloaded = await remote.page('/web/probe.html', { since: away0, timeoutMs: 1000 }).catch(() => null);
        const alive = await Promise.race([page.evaluate(() => true), D.sleep(8000).then(() => false)]);
        if (!alive) {
          const r = { hint: h, sound, away, awayMs, arm, before, after: env(), pageAnswered: false, newPageLoaded: Boolean(reloaded && reloaded.client.id !== page.client.id) };
          out.runs.push(r);
          console.log(String(h).padEnd(11), sound ? 'tone  ' : 'silent', `away ${away}s`, 'the page did not answer after the return; a new copy loaded:', r.newPageLoaded);
          save('background', out);
          continue;
        }
        const r = await page.evaluate((x) => window.T.bgReturn(x), { timeoutMs: 5000 });
        await page.evaluate(() => window.T.tone(false));
        const c0 = arm.count;
        const c1 = r.countAtReturn;
        r.renderedPerWall = c1 ? +((c1.frame - c0.frame) / r.info.sampleRate / ((c1.now - c0.now) / 1000)).toFixed(4) : null;
        Object.assign(r, { hint: h, sound, away, awayMs, arm, before, after: env() });
        out.runs.push(r);
        console.log(String(h).padEnd(11), sound ? 'tone  ' : 'silent', `away ${away}s`, 'state at return', r.stateAtReturn, 'rendered/wall while away', r.renderedPerWall,
          'resume', JSON.stringify(r.resume), 'rate after', JSON.stringify(r.rate), 'events', r.log.map((e) => `${e.type}:${e.state}`).join(','));
        save('background', out);
      }
    }
  },
  { roots: [path.join(stands, 'android')], isolated: true },
);
