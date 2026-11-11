import fs from 'node:fs'; import path from 'node:path'; import os from 'node:os'; import { execFileSync } from 'node:child_process';
import { chromium, webkit } from 'playwright';
import { startServer } from './server.mjs';

const [engine, taskArg] = process.argv.slice(2); const tasks = taskArg.split(',');
const root = import.meta.dirname; const outDir = path.join(root, 'out'); fs.mkdirSync(outDir, { recursive: true });
const inputs = JSON.parse(fs.readFileSync(path.join(root, 'data/inputs.json'), 'utf8'));
const save = (task, obj) => { const f = path.join(outDir, `${engine}-${task}.json`); fs.writeFileSync(f, JSON.stringify(obj, null, 1)); console.log('saved', f); };

const { server, port } = await startServer(); const base = `http://127.0.0.1:${port}`;
const launch = async () => {
  const opts = { headless: true, args: engine === 'webkit' ? [] : ['--force_high_performance_gpu', '--enable-precise-memory-info', '--js-flags=--expose-gc'] };
  if (engine === 'msedge') opts.channel = 'msedge';
  const type = engine === 'webkit' ? webkit : chromium;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),`aacp-${engine}-`));
  const ctx = await type.launchPersistentContext(dir, opts); const page = await ctx.newPage(); await page.goto(`${base}/web/index.html`); await page.waitForFunction(() => window.ready);
  return { ctx, page, dir };
};
const call = (page, id, cmd, args) => page.evaluate(([i, c, a]) => window.callWorker(i, c, a), [id, cmd, args]);
const ps = (script) => JSON.parse(execFileSync('powershell', ['-NoProfile', '-Command', script], { encoding: 'utf8', maxBuffer: 1 << 26 }) || 'null');
const procTree = (dir) => {
  const key = path.basename(dir);
  const s = `$all = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine; $roots = $all | Where-Object { $_.CommandLine -and $_.CommandLine -like '*${key}*' }; $ids = New-Object System.Collections.Generic.HashSet[int]; foreach($r in $roots){ [void]$ids.Add($r.ProcessId) }; $changed=$true; while($changed){ $changed=$false; foreach($p in $all){ if($ids.Contains([int]$p.ParentProcessId) -and -not $ids.Contains([int]$p.ProcessId)){ [void]$ids.Add([int]$p.ProcessId); $changed=$true } } }; $out = @(); foreach($id in $ids){ $g = Get-Process -Id $id -ErrorAction SilentlyContinue; if($g){ $c = ($all | Where-Object {$_.ProcessId -eq $id}).CommandLine; $t = if($c -match '--type=([a-z-]+)'){$Matches[1]}else{'browser'}; $out += [pscustomobject]@{pid=$id;name=$g.ProcessName;type=$t;peakWS=$g.PeakWorkingSet64;ws=$g.WorkingSet64;peakCommit=$g.PeakPagedMemorySize64;commit=$g.PagedMemorySize64;peakVirt=$g.PeakVirtualMemorySize64} } }; ConvertTo-Json -InputObject @($out) -Compress`;
  return ps(s);
};
const MB = (b) => +(b / 1048576).toFixed(1);
const summarize = (procs) => { const by = {}; for (const p of procs) { const k = p.type; by[k] ??= { n: 0, peakWS_MB: 0, sumPeakWS_MB: 0, peakCommit_MB: 0 }; by[k].n++; by[k].peakWS_MB = Math.max(by[k].peakWS_MB, MB(p.peakWS)); by[k].sumPeakWS_MB += MB(p.peakWS); by[k].peakCommit_MB = Math.max(by[k].peakCommit_MB, MB(p.peakCommit)); }
  const r = procs.filter((p) => p.type === 'renderer'); return { byType: by, rendererMaxPeakWS_MB: Math.max(0, ...r.map((p) => MB(p.peakWS))), rendererMaxPeakCommit_MB: Math.max(0, ...r.map((p) => MB(p.peakCommit))), totalSumPeakWS_MB: +procs.reduce((s, p) => s + MB(p.peakWS), 0).toFixed(1) }; };

const names = ['chord', 'noise', 'clicks', 'sweep', 'mix240'];
try {
  for (const task of tasks) {
    const t0 = Date.now(); console.log(`[${engine}] ${task}`);
    if (task === 'mem') {
      const res = {};
      for (const sc of [['idle', 'memIdle', {}], ['whole3', 'memWhole', { name: 'mix240', stems: 3 }], ['pages3_ring5', 'memPages', { name: 'mix240', stems: 3, ringPages: 5 }], ['whole1', 'memWhole', { name: 'mix240', stems: 1 }], ['pages1_ring5', 'memPages', { name: 'mix240', stems: 1, ringPages: 5 }]]) {
        for (let rep = 0; rep < 2; rep++) { const { ctx, page, dir } = await launch(); try { const r = await call(page, 'm', sc[1], sc[2]); await new Promise((r2) => setTimeout(r2, 500)); r.proc = summarize(procTree(dir)); (res[sc[0]] ??= []).push(r); console.log(sc[0], rep, JSON.stringify(r.proc.rendererMaxPeakWS_MB), 'MB renderer peakWS', r.ms); } finally { await ctx.close(); } }
      }
      save(task, res); continue;
    }
    const { ctx, page } = await launch(); const out = {};
    try {
      page.on('pageerror', (e) => console.log('pageerror', e.message));
      if (task === 'env') { out.env = await call(page, 'w', 'env'); out.version = ctx.browser()?.version?.() ?? null; }
      if (task === 'controls') out.controls = await call(page, 'w', 'controls');
      if (task === 'indexcheck') for (const n of names) out[n] = await call(page, 'w', 'indexcheck', { name: n });
      if (task === 'correctness') for (const n of names.slice(0, 4)) { out[n] = await call(page, 'w', 'correctness', { name: n }); const p = out[n].perP; console.log(n, 'wholeFrames', out[n].wholeFrames, Object.entries(p).map(([k, v]) => `P${k}:${v.pagesOk}/${v.pages} max=${v.maxDiff}`).join(' '), JSON.stringify(out[n].seqP0)); }
      if (task === 'bench') { out.decoder = await call(page, 'w', 'benchDecoder', { name: 'mix240' }); out.seek = await call(page, 'w', 'benchSeek', { name: 'mix240' }); out.seekChord = await call(page, 'w', 'benchSeek', { name: 'chord' }); out.seekMfra = await call(page, 'w', 'benchSeek', { name: 'mix240_mfra' }); }
      if (task === 'parallel') {
        out.oneWorker = await call(page, 'w', 'benchParallel', { names: ['chord', 'sweep', 'clicks'], iterations: 20 });
        const it = 30; const seq = await call(page, 'a', 'loopPages', { name: 'chord', iterations: it * 3 }); const stems = ['chord', 'sweep', 'clicks'];
        for (const s of stems) await call(page, 'w' + s, 'env');
        const startAt = Date.now() + 1500; const t = Date.now(); const par = await Promise.all(stems.map((s) => call(page, 'w' + s, 'loopPages', { name: s, iterations: it, startAt })));
        out.threeWorkers = { pagesPerStem: it, sequential_oneStem_x3_ms: +seq.ms.toFixed(0), parallel_three_workers_each_ms: par.map((p) => +p.ms.toFixed(0)), parallelWallApprox_ms: Math.max(...par.map((p) => p.ms)) };
      }
      if (task === 'flac') for (const n of names.slice(0, 4).concat('mix240')) { out[n] = await call(page, 'w', 'flac', { name: n }); console.log(n, JSON.stringify(out[n].pagesAll ?? out[n].variants)); }
      save(task, { engine, inputs, ...out });
    } finally { await ctx.close(); }
    console.log(`[${engine}] ${task} took ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
} finally { server.close(); }
process.exit(0);
