import fs from 'node:fs';
const L = process.env.DESKTOP ? await import('./desktop.mjs') : await import('./device.mjs');
import { startServer } from './server.mjs';
const [task, argJson] = process.argv.slice(2);
const arg = argJson ? JSON.parse(argJson) : {};
if (!['renderSize', 'callbacks', 'bug', 'clock', 'stallCheck', 'background'].includes(task)) {
  console.error('usage: node platform_probes.mjs <renderSize|callbacks|bug|clock|stallCheck|background> [json args]');
  process.exit(2);
}
const save = (name, obj) => {
  fs.writeFileSync(L.outDir(`platform_probes-${name}.json`), JSON.stringify(obj, null, 1));
};
const { server, port } = await startServer({ isolated: true });
L.mapPort(port);
const sleep = L.sleep;
const probeUrl = `http://localhost:${port}/web/probe.html`;
const fresh = async (target = 'chrome') => {
  if (target === 'webview') {
    L.sh(`am force-stop ${L.webviewPackage()}`);
    const opened = await L.openTarget('webview', probeUrl);
    await opened.page.waitForFunction(() => window.ready);
    L.assertFront('webview');
    await L.tapActivation();
    return { page: opened.page, close: opened.close };
  }
  L.chromeFresh();
  await sleep(3500);
  const browser = await L.connect();
  const page = await L.openPage(browser, probeUrl);
  await page.waitForFunction(() => window.ready);
  L.front('chrome');
  await sleep(1000);
  L.assertFront('chrome');
  await L.tapActivation();
  page.setDefaultTimeout(900000);
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  const close = async () => {
    try {
      await page.close();
      await browser.close();
    } catch {}
  };
  return { browser, page, close };
};
const done = (s) => s.close();
const call = (page, fn, a) => page.evaluate(([f, x]) => window.T[f](x), [fn, a]);
const meta = () => ({ device: L.versions(), before: L.env() });
const fit = (xs, ys) => {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n,
    my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0,
    sxy = 0;
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
  const x = ok.map((r) => r[1]);
  const y = ok.map((r) => r[2]);
  const f = fit(x, y);
  const res = ok.map((r) => r[2] - (f.icpt + f.slope * r[1]));
  const ol = rows.map((r) => r[3]);
  const uniq = [...new Set(ol.map((v) => +v.toFixed(4)))];
  const stuck = ok.filter((r, i) => i > 0 && r[1] === ok[i - 1][1]).length;
  return {
    n: rows.length,
    valid: ok.length,
    perfMsPerCtxSec: +f.slope.toFixed(4),
    residualMs: {
      min: +Math.min(...res).toFixed(3),
      p5: +q(res, 0.05).toFixed(3),
      p50: +q(res, 0.5).toFixed(3),
      p95: +q(res, 0.95).toFixed(3),
      max: +Math.max(...res).toFixed(3),
      spread: +(Math.max(...res) - Math.min(...res)).toFixed(3),
    },
    outputLatencyMs: {
      min: +(Math.min(...ol) * 1000).toFixed(2),
      median: +(q(ol, 0.5) * 1000).toFixed(2),
      max: +(Math.max(...ol) * 1000).toFixed(2),
      distinct: uniq.length,
      first: ol[0],
      last: ol.at(-1),
    },
    baseLatencyMs: +(rows[0][5] * 1000).toFixed(2),
    ctxTimeStuckSamples: stuck,
  };
};
try {
  L.wake();
  await L.gate();
  if (task === 'stallCheck') {
    const s = await fresh();
    const out = { ...meta() };
    await s.page.evaluate(() => window.T.setup({ latencyHint: 0.02 }));
    out.stall = await s.page.evaluate((ms) => window.T.stallCheck(ms), arg.stallMs ?? 1500);
    out.after = L.env();
    console.log(JSON.stringify(out.stall));
    save('stallCheck', out);
    await done(s);
  }
  if (task === 'renderSize') {
    const s = await fresh();
    const out = { ...meta(), rows: [] };
    for (const h of [undefined, 'default', 'hardware', 64, 128, 256, 512]) {
      const r = await s.page.evaluate((hh) => window.T.renderSize(hh), h === undefined ? undefined : h);
      out.rows.push(r);
      console.log(JSON.stringify(r));
    }
    out.ua = await s.page.evaluate(() => navigator.userAgent);
    out.after = L.env();
    save('renderSize', out);
    await done(s);
  }
  if (task === 'callbacks') {
    const out = { ...meta(), rounds: [] };
    const hints = arg.hints ?? [0.02, 'interactive', 'playback'];
    const secs = arg.seconds ?? 15;
    for (let round = 0; round < (arg.rounds ?? 2); round++)
      for (const h of round % 2 ? [...hints].reverse() : hints) {
        await L.gate();
        const s = await fresh();
        const before = L.env();
        await s.page.evaluate((hh) => window.T.setup(typeof hh === 'object' ? hh : { latencyHint: hh }), h);
        const r = await s.page.evaluate((x) => window.T.callbacks(x), { seconds: secs });
        r.hint = h;
        r.round = round;
        r.before = before;
        r.after = L.env();
        out.rounds.push(r);
        console.log(
          String(h).padEnd(12),
          'info',
          JSON.stringify(r.info),
          '| burst p50/p99/max',
          r.burstSizeP50,
          r.burstSizeP99,
          r.burstSizeMax,
          'hist',
          JSON.stringify(r.burstSizeHist),
          'gap p50/p95/p99/max ms',
          r.gapMsP50,
          r.gapMsP95,
          r.gapMsP99,
          r.gapMsMax,
          'q/s',
          r.quantaPerSec.toFixed(1),
          'noncontig',
          r.nonContiguous,
        );
        save('callbacks', out);
        await done(s);
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
    const hints = arg.hints ?? [0.02, 'interactive'];
    for (const h of hints)
      for (let rep = 0; rep < (arg.reps ?? 2); rep++)
        for (const t of rep % 2 ? [...trials].reverse() : trials) {
          await L.gate();
          const s = await fresh();
          const before = L.env();
          await s.page.evaluate((hh) => window.T.setup(typeof hh === 'object' ? hh : { latencyHint: hh }), h);
          const r = await s.page.evaluate((x) => window.T.bug560233271(x), t);
          r.name = t.name;
          r.hint = h;
          r.rep = rep;
          r.before = before;
          r.after = L.env();
          out.trials.push(r);
          console.log(
            String(h).padEnd(11),
            t.name.padEnd(30),
            'rep',
            rep,
            'q/s base',
            r.baseline.quantaPerSec,
            'afterStall',
            r.afterStall?.quantaPerSec ?? '-',
            'afterResume',
            r.afterResume?.quantaPerSec ?? '-',
            r.afterResume2?.quantaPerSec ?? '-',
            'ctxRate',
            r.afterResume?.ctxTimeRate ?? '-',
            'susres',
            JSON.stringify(r.sr ?? {}),
            'baseLat',
            r.info.baseLatency,
            '->',
            r.infoAfter?.baseLatency ?? '-',
          );
          save('bug', out);
          await done(s);
        }
  }
  if (task === 'clock') {
    const out = { ...meta(), runs: [] };
    for (const h of arg.hints ?? [0.02, 'interactive']) {
      await L.gate();
      const s = await fresh();
      const before = L.env();
      await s.page.evaluate((hh) => window.T.setup(typeof hh === 'object' ? hh : { latencyHint: hh }), h);
      const info0 = await s.page.evaluate(() => window.T.info());
      const rows1 = await s.page.evaluate((x) => window.T.clock(x), { seconds: arg.seconds ?? 60, stepMs: 100 });
      const sr = await s.page.evaluate(() => window.T.suspendResume());
      await sleep(500);
      const info1 = await s.page.evaluate(() => window.T.info());
      const rows2 = await s.page.evaluate((x) => window.T.clock(x), { seconds: arg.seconds2 ?? 30, stepMs: 100 });
      const r = {
        hint: h,
        info0,
        info1,
        sr,
        before,
        after: L.env(),
        beforeSR: clockStats(rows1),
        afterSR: clockStats(rows2),
        firstRowsAfter: rows2.slice(0, 5),
      };
      out.runs.push(r);
      console.log(
        String(h),
        'baseLat',
        r.beforeSR.baseLatencyMs,
        'before',
        JSON.stringify(r.beforeSR.residualMs),
        JSON.stringify(r.beforeSR.outputLatencyMs),
        'slope',
        r.beforeSR.perfMsPerCtxSec,
        '| after',
        JSON.stringify(r.afterSR.residualMs),
        JSON.stringify(r.afterSR.outputLatencyMs),
        'slope',
        r.afterSR.perfMsPerCtxSec,
        JSON.stringify(sr),
      );
      out.runs.at(-1).rows1 = rows1.filter((_, i) => i % 5 === 0);
      out.runs.at(-1).rows2 = rows2.filter((_, i) => i % 5 === 0);
      save('clock', out);
      await done(s);
    }
  }
  if (task === 'background') {
    const target = arg.target ?? 'chrome';
    const out = { ...meta(), target, runs: [] };
    const awaySec = arg.seconds ?? 10;
    for (let rep = 0; rep < (arg.reps ?? 1); rep++)
      for (const h of arg.hints ?? [0.02, 'interactive'])
        for (const sound of rep % 2 ? [true, false] : [false, true]) {
          await L.gate();
          const s = await fresh(target);
          const before = L.env();
          await s.page.evaluate((hh) => window.T.setup(typeof hh === 'object' ? hh : { latencyHint: hh }), h);
          await s.page.evaluate((on) => window.T.tone(on), sound);
          const arm = await s.page.evaluate(() => window.T.bgArm());
          await sleep(2000);
          await L.toBackground();
          const away0 = Date.now();
          await sleep(awaySec * 1000);
          await L.toFront(target);
          const awayMs = Date.now() - away0;
          await sleep(500);
          const r = await s.page.evaluate((x) => window.T.bgReturn(x), { timeoutMs: 5000 });
          await s.page.evaluate(() => window.T.tone(false));
          const c0 = arm.count;
          const c1 = r.countAtReturn;
          r.renderedPerWall = c1
            ? +((c1.frame - c0.frame) / r.info.sampleRate / ((c1.now - c0.now) / 1000)).toFixed(4)
            : null;
          Object.assign(r, { hint: h, sound, awaySec, awayMs, rep, arm, before, after: L.env() });
          out.runs.push(r);
          console.log(
            String(h).padEnd(11),
            sound ? 'tone  ' : 'silent',
            'state at return',
            r.stateAtReturn,
            'rendered/wall while away',
            r.renderedPerWall,
            'resume',
            JSON.stringify(r.resume),
            'rate after',
            JSON.stringify(r.rate),
            'outputLatency',
            r.info.outputLatency,
            'events',
            r.log.map((e) => `${e.type}:${e.state}`).join(','),
          );
          save(`background-${target}${arg.tag ?? ''}`, out);
          await done(s);
        }
  }
} finally {
  L.cleanup();
  server.close();
}
process.exit(0);
