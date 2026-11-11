import fs from 'node:fs'; import os from 'node:os'; import path from 'node:path'; import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { startServer } from './server.mjs';

const root = import.meta.dirname; const outDir = path.join(root, 'out'); fs.mkdirSync(outDir, { recursive: true });
const tasks = (process.argv[2] ?? 'env').split(','); const arg = process.argv[3] ? JSON.parse(process.argv[3]) : {};
const save = (name, obj) => { const f = path.join(outDir, `${name}.json`); fs.writeFileSync(f, JSON.stringify(obj, null, 1)); console.log('saved', f); };
const { server, port } = await startServer(); const base = `http://127.0.0.1:${port}`;
const dirs = [];
const launch = async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pps-')); dirs.push(dir);
  const ctx = await chromium.launchPersistentContext(dir, { channel: 'msedge', headless: false, viewport: { width: 800, height: 500 },
    args: ['--force_high_performance_gpu', '--autoplay-policy=no-user-gesture-required', '--enable-precise-memory-info', '--window-position=0,0'] });
  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message)); page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('console', m.type(), m.text()); });
  await page.goto(`${base}/web/index.html`); await page.waitForFunction(() => window.ready);
  return { ctx, page, dir };
};
const ps = (script) => JSON.parse(execFileSync('powershell', ['-NoProfile', '-Command', script], { encoding: 'utf8', maxBuffer: 1 << 26 }) || 'null');
const procTree = (dir) => {
  const key = path.basename(dir);
  const s = `$all = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine; $roots = $all | Where-Object { $_.CommandLine -and $_.CommandLine -like '*${key}*' }; $ids = New-Object System.Collections.Generic.HashSet[int]; foreach($r in $roots){ [void]$ids.Add($r.ProcessId) }; $changed=$true; while($changed){ $changed=$false; foreach($p in $all){ if($ids.Contains([int]$p.ParentProcessId) -and -not $ids.Contains([int]$p.ProcessId)){ [void]$ids.Add([int]$p.ProcessId); $changed=$true } } }; $out = @(); foreach($id in $ids){ $g = Get-Process -Id $id -ErrorAction SilentlyContinue; if($g){ $c = ($all | Where-Object {$_.ProcessId -eq $id}).CommandLine; $t = if($c -match '--type=([a-z-]+)'){$Matches[1]}else{'browser'}; $out += [pscustomobject]@{pid=$id;name=$g.ProcessName;type=$t;peakWS=$g.PeakWorkingSet64;ws=$g.WorkingSet64;peakCommit=$g.PeakPagedMemorySize64;commit=$g.PagedMemorySize64} } }; ConvertTo-Json -InputObject @($out) -Compress`;
  return ps(s);
};
const killTree = (dir) => { try { const key = path.basename(dir); execFileSync('powershell', ['-NoProfile', '-Command', `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*${key}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`]); } catch { } };
const MB = (b) => +(b / 1048576).toFixed(1);
const ev = (page, expr, a) => page.evaluate(new Function('a', `return (${expr})(a)`), a);

try {
  for (const task of tasks) {
    const t0 = Date.now(); console.log(`[${task}] start`);
    if (task === 'memory') {
      const res = {};
      for (const kind of ['idle', 'whole', 'pages']) for (let rep = 0; rep < 2; rep++) {
        const { ctx, page, dir } = await launch();
        try {
          const r = await page.evaluate(([k]) => window.T.memory({ kind: k }), [kind]);
          await new Promise((r2) => setTimeout(r2, 500));
          const procs = procTree(dir); const rend = procs.filter((p) => p.type === 'renderer');
          r.renderers = rend.map((p) => ({ pid: p.pid, peakWS_MB: MB(p.peakWS), peakCommit_MB: MB(p.peakCommit), ws_MB: MB(p.ws) }));
          r.rendererMaxPeakWS_MB = Math.max(...rend.map((p) => MB(p.peakWS))); r.rendererMaxPeakCommit_MB = Math.max(...rend.map((p) => MB(p.peakCommit)));
          (res[kind] ??= []).push(r); console.log(kind, rep, r.rendererMaxPeakWS_MB, 'MB peak WS', r.rendererMaxPeakCommit_MB, 'MB peak commit');
        } finally { await ctx.close(); killTree(dir); }
      }
      save('memory', res); continue;
    }
    const { ctx, page, dir } = await launch(); const out = {};
    try {
      page.setDefaultTimeout(600000);
      const T = (fn, a) => page.evaluate(([f, x]) => window.T[f](x), [fn, a]);
      if (task === 'env') { out.env = await T('env'); const v = ctx.browser()?.version?.(); out.version = v; }
      if (task === 'smoke') {
        const s = await T('setup', { kind: 'pages' }); out.setup = s; console.log(JSON.stringify(s));
        out.run = await page.evaluate(async () => { const T = window.T; const sleep = (ms) => new Promise((r) => setTimeout(r, ms)); T.seek(0); await T.play(); await sleep(4000); const st = await T.stats(); await T.stop(); return { st, state: T.state() }; });
        await T('teardown');
      }
      if (task === 'correctness') {
        for (const sc of arg.scenarios ?? ['plain', 'seeks', 'loop', 'loopBoundary']) { const r = await page.evaluate(([n]) => window.T.scenario(n, {}), [sc]); out[sc] = r;
          console.log(sc, 'pass', r.main.pass, 'rolling', r.main.rolling, 'mismatchQ', r.main.mismatchQuanta, 'posBad', r.main.posBad, 'silentBad', r.main.silentBad, JSON.stringify(r.main.covered), 'buf', r.nodeStats.bufQuanta); }
      }
      if (task === 'controls') {
        for (const sc of ['seeks', 'loop']) { out[sc] = await page.evaluate(([n]) => window.T.scenario(n, { controls: true }), [sc]);
          console.log(sc, 'clean', out[sc].controlsOnClean, (out[sc].controls ?? []).map((c) => `${c.failed ? 'FAIL-as-expected' : 'MISSED'}:${c.name}`).join('\n  ')); }
        for (const bug of ['edge', 'loopGain', 'swap', 'shift']) { out['bug_' + bug] = await page.evaluate(([n, b]) => window.T.scenario(n, { bug: b }), [bug === 'loopGain' ? 'loop' : 'plain', bug]); const r = out['bug_' + bug]; console.log('bug', bug, 'pass', r.main.pass, 'mismatchQ', r.main.mismatchQuanta, 'posBad', r.main.posBad); }
      }
      if (task === 'stretch') {
        for (const [name, tempo, n] of [['plain', 0.75, 1], ['plain', 0.75, 3], ['loop', 0.75, 1], ['loop', 1.5, 3], ['seeks', 0.75, 1], ['seeks', 1.25, 3]]) {
          const r = await page.evaluate(([a, b, c]) => window.T.stretchScenario(a, b, c), [name, tempo, n]); (out.rows ??= []).push(r); console.log(name, tempo, n, 'input pass', r.input?.pass, r.input?.quanta, 'mismatch', r.input?.mismatchQuanta, 'buf', r.nodeStats.bufQuanta, 'errors', r.nodeStats.errors);
        }
      }
      if (task === 'dbg') { out.s = await page.evaluate(async () => { const r = await window.T.seekLatency({ kind: 'whole', mode: 'warm', n: 5 }); return r; }); console.log(JSON.stringify(out.s.rows)); }
      if (task === 'latency') {
        const n = arg.n ?? 200;
        for (const [kind, mode] of [['whole', 'warm'], ['pages', 'warm'], ['pages', 'cold'], ['whole', 'cold']]) {
          const r = await page.evaluate(([k, m, nn]) => window.T.seekLatency({ kind: k, mode: m, n: nn }), [kind, mode, n]); out[`${kind}_${mode}`] = r;
          console.log(kind, mode, 'lat', JSON.stringify(r.latMs), 'wall', JSON.stringify(r.wallMs), 'timeouts', r.timeouts, 'warmFlag', r.warmFlagged);
        }
        for (const tp of [0.75]) { const r = await page.evaluate(([nn, t]) => window.T.seekLatency({ kind: 'pages', mode: 'cold', n: nn, tempo: t }), [Math.min(n, 60), tp]); out[`pages_cold_tempo${tp}`] = r; console.log('pages cold tempo', tp, JSON.stringify(r.latMs), JSON.stringify(r.wallMs)); }
      }
      if (task === 'underrun') {
        const dur = arg.durationMs ?? 20000;
        const defs = [
          { name: 'plain' }, { name: 'tempo075_1st', tempo: 0.75, stretchers: 1 }, { name: 'tempo075_3st', tempo: 0.75, stretchers: 3 }, { name: 'tempo075_3st_aligned', tempo: 0.75, stretchers: 3, stagger: false },
          { name: 'loop4s', loop: true }, { name: 'loop4s_tempo075_3st', loop: true, tempo: 0.75, stretchers: 3 },
          { name: 'seeks500_cold', seekEveryMs: 500, seekMode: 'cold' }, { name: 'seeks500_near', seekEveryMs: 500, seekMode: 'near' },
          { name: 'jank200_1s', jank: true }, { name: 'jank200_1s_tempo075_3st', jank: true, tempo: 0.75, stretchers: 3 }, { name: 'stress1', stress: 1 }, { name: 'stress24', stress: 24 }, { name: 'stress24_tempo075_3st', stress: 24, tempo: 0.75, stretchers: 3 }, { name: 'stress64_tempo075_3st_jank', stress: 64, tempo: 0.75, stretchers: 3, jank: true }, { name: 'stress64_seeks500_near_jank', stress: 64, seekEveryMs: 500, seekMode: 'near', jank: true },
        ];
        for (const d of defs.filter((x) => !arg.only || arg.only.includes(x.name))) {
          const r = await page.evaluate((a) => window.T.underrun(a), { ...d, durationMs: dur }); (out.rows ??= []).push(r);
          console.log(d.name, 'ps delta', JSON.stringify(r.playbackStatsDelta), 'buf q', r.spike.bufQuanta, 'ev', r.spike.bufEvents, 'quanta', r.spike.quanta, 'expected', r.quantaExpected, 'procMean us', +(r.spike.procMs / r.spike.quanta * 1000).toFixed(1), 'heavy', r.spike.procHeavy, 'errors', r.spike.errors);
        }
      }
      if (task === 'cost') { out.cost = await T('messageCost', {}); console.log(JSON.stringify(out.cost)); }
      save(arg.tag ? `${task}-${arg.tag}` : task, { version: ctx.browser()?.version?.(), ...out });
    } finally { await ctx.close(); killTree(dir); }
    console.log(`[${task}] took ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
} finally { server.close(); for (const d of dirs) killTree(d); }
process.exit(0);
