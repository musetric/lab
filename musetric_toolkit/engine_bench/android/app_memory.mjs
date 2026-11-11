import fs from 'node:fs';
import { parseArgs } from 'node:util';
import * as device from './device.mjs';

const { values } = parseArgs({
  options: {
    package: { type: 'string' },
    tap: { type: 'string' },
    cycles: { type: 'string', default: '3' },
    'open-wait': { type: 'string', default: '7' },
    'close-wait': { type: 'string', default: '4' },
  },
});
const pkg = values.package ?? device.webviewPackage();
const [x, y] = values.tap ? values.tap.split(',').map(Number) : device.screenCenter();
const cycles = +values.cycles;
const openWaitMs = +values['open-wait'] * 1000;
const closeWaitMs = +values['close-wait'] * 1000;

const listing = `for p in $(ps -A -o PID,NAME | grep -E "${pkg}|webview:sandboxed|webview:privileged" | while read p n; do echo $p; done); do n=$(cat /proc/$p/cmdline | tr "\\0" " "); echo "$p $n"; dumpsys meminfo $p | grep "TOTAL PSS"; done`;

const snapshot = () => {
  const lines = device.sh(listing).replace(/\r/g, '').split('\n');
  const procs = [];
  for (let i = 0; i < lines.length; i++) {
    const head = /^(\d+) (\S+)/.exec(lines[i]);
    const mem = /TOTAL PSS:\s+(\d+)\s+TOTAL RSS:\s+(\d+)/.exec(lines[i + 1] ?? '');
    if (!head || !mem) continue;
    procs.push({
      pid: +head[1],
      name: head[2],
      pssMB: +(+mem[1] / 1024).toFixed(1),
      rssMB: +(+mem[2] / 1024).toFixed(1),
    });
  }
  return { procs, sumPssMB: +procs.reduce((sum, p) => sum + p.pssMB, 0).toFixed(1) };
};

const rows = [];
const record = (state) => {
  const row = { state, ...snapshot() };
  rows.push(row);
  console.log(state.padEnd(10), `sum ${row.sumPssMB} MB`, row.procs.map((p) => `${p.name}=${p.pssMB}`).join(' '));
};

device.wake();
const before = await device.gate();
console.log('tap', x, y, 'package', pkg);
record('base');
for (let i = 1; i <= cycles; i++) {
  device.sh(`input tap ${x} ${y}`);
  await device.sleep(openWaitMs);
  record(`open ${i}`);
  fs.writeFileSync(device.outDir(`app_memory-open${i}.png`), device.adbBinary('exec-out', 'screencap', '-p'));
  device.sh('input keyevent KEYCODE_BACK');
  await device.sleep(closeWaitMs);
  record(`closed ${i}`);
}
fs.writeFileSync(
  device.outDir('app_memory.json'),
  JSON.stringify({ package: pkg, tap: [x, y], device: device.versions(), before, after: device.env(), rows }, null, 1),
);
