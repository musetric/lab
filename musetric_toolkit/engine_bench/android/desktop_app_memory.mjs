import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { RawPage, cdpPort, outDir, sleep } from './device.mjs';

const { values } = parseArgs({
  options: {
    pid: { type: 'string' },
    project: { type: 'string', default: '/project/1' },
    cycles: { type: 'string', default: '3' },
    'open-wait': { type: 'string', default: '7' },
    'close-wait': { type: 'string', default: '4' },
  },
});
if (!values.pid) {
  console.error('usage: node desktop_app_memory.mjs --pid <pid of the app> [--project /project/1] [--cycles 3]');
  process.exit(2);
}
const MB = (bytes) => +(bytes / 1048576).toFixed(1);
const processes = () => {
  const script = `$all = Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine
$ids = New-Object System.Collections.Generic.HashSet[int]; [void]$ids.Add(${+values.pid}); $changed = $true
while ($changed) { $changed = $false; foreach ($p in $all) { if ($ids.Contains([int]$p.ParentProcessId) -and $ids.Add([int]$p.ProcessId)) { $changed = $true } } }
$out = foreach ($p in $all) { if ($ids.Contains([int]$p.ProcessId)) { $g = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
  if ($g) { [pscustomobject]@{ pid = $p.ProcessId; name = $p.Name; type = if ($p.CommandLine -match '--type=([\\w-]+)') { $matches[1] } else { 'browser' }; ws = $g.WorkingSet64; peakWS = $g.PeakWorkingSet64; priv = $g.PrivateMemorySize64 } } } }
ConvertTo-Json -Compress @($out)`;
  return JSON.parse(execFileSync('powershell', ['-NoProfile', '-Command', script], { encoding: 'utf8' }));
};
const rows = [];
const record = (state) => {
  const procs = processes().map((p) => ({ ...p, ws: MB(p.ws), peakWS: MB(p.peakWS), priv: MB(p.priv) }));
  const renderers = procs.filter((p) => p.type === 'renderer');
  const row = { state, renderers, sumWsMB: +procs.reduce((sum, p) => sum + p.ws, 0).toFixed(1), procs };
  rows.push(row);
  console.log(state.padEnd(10), `sum ${row.sumWsMB} MB`, renderers.map((p) => `renderer ${p.pid}=${p.ws}`).join(' '));
};

const page = await RawPage.open(cdpPort);
const at = await page.evaluate(() => location.pathname);
if (at !== '/') throw new Error(`the app is not on its home page: ${at}`);
record('base');
for (let i = 1; i <= +values.cycles; i++) {
  await page.evaluate((href) => document.querySelector(`a[href="${href}"]`).click(), values.project);
  await sleep(+values['open-wait'] * 1000);
  if ((await page.evaluate(() => location.pathname)) !== values.project) throw new Error('the project did not open');
  record(`open ${i}`);
  await page.evaluate(() => history.back());
  await sleep(+values['close-wait'] * 1000);
  record(`closed ${i}`);
}
const version = await (await fetch(`http://127.0.0.1:${cdpPort}/json/version`)).json();
fs.writeFileSync(
  outDir('desktop_app_memory.json'),
  JSON.stringify({ project: values.project, version, rows }, null, 1),
);
page.ws.close();
await sleep(200);
process.exit(0);
