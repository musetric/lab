import fs from 'node:fs';
import * as device from './device.mjs';
import { startServer } from './server.mjs';

const [label, roundsArg, secondsArg, idsArg, pacedArg] = process.argv.slice(2);
if (!label || !roundsArg || !secondsArg || !idsArg) {
  console.error('usage: node stepper.mjs <chrome|webview> <rounds> <seconds> <ids> [paced]');
  console.error('ids: S0 S1 S2 S2c S3 S4 S4b S5 S6 S7a S7b, comma separated; env TAG, STEM_S');
  process.exit(2);
}
const rounds = +roundsArg;
const seconds = +secondsArg;
const ids = idsArg.split(',');
const paced = pacedArg === 'paced';
const target = device.targetOf(label);
const tag = `stepper-${process.env.TAG ?? ''}${label}${paced ? '-paced' : ''}`;
const results = [];

const { server, port } = await startServer({ isolated: true });
device.mapPort(port);
let opened;
try {
  device.wake();
  console.log('pre', JSON.stringify(await device.gate()));
  opened = await device.openTarget(target, `http://localhost:${port}/web/stepper.html`);
  const { page } = opened;
  await page.waitForFunction(() => window.ready);
  device.front(target);
  await device.sleep(1500);
  for (let r = 0; r < rounds; r++) {
    for (const id of r % 2 ? [...ids].reverse() : ids) {
      device.assertFront(target);
      const before = device.env();
      if (before.thermalStatus >= 2) {
        console.log('thermal stop', before);
        break;
      }
      const res = await page.evaluate((m) => window.runScenario(m), {
        cmd: 'run',
        id,
        seconds,
        stemSeconds: +(process.env.STEM_S ?? 100),
        paced,
        label: `r${r}`,
      });
      const after = device.env();
      results.push({ ...res, before, after });
      const q = res.quantumMs;
      const us = (v) => (v * 1000).toFixed(0);
      console.log(
        id,
        `r${r}`,
        `wall ${(res.wallMs / 1000).toFixed(1)}s med ${us(q.median)}us p99 ${us(q.p99)} p99.9 ${us(q.p999)} max ${us(q.max)}`,
        `over50 ${q.over50} over100 ${q.over100} | cpu ${before.cpuMaxC}->${after.cpuMaxC} th ${after.thermalStatus}`,
      );
      fs.writeFileSync(
        device.outDir(`${tag}.json`),
        JSON.stringify({ label, device: device.versions(), paced, seconds, results }, null, 1),
      );
    }
  }
} finally {
  await opened?.close();
  device.cleanup();
  server.close();
}
process.exit(0);
