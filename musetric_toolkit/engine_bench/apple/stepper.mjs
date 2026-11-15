import fs from 'node:fs';
import path from 'node:path';
import * as D from './device.mjs';
import { session, stands } from './session.mjs';

const [target, roundsArg, secondsArg, idsArg, pacedArg] = process.argv.slice(2);
if (!idsArg) {
  console.error('usage: node stepper.mjs <target> <rounds> <seconds> <ids> [paced]');
  process.exit(2);
}
const rounds = +roundsArg;
const seconds = +secondsArg;
const ids = idsArg.split(',');
const paced = pacedArg === 'paced';
const tag = `stepper-${process.env.TAG ?? ''}${target}${paced ? '-paced' : ''}`;

await session(
  target,
  async ({ open, meta, env, gate }) => {
    console.log('pre', JSON.stringify(await gate()));
    const page = await open('/web/stepper.html');
    await page.waitForFunction(() => Boolean(window.ready));
    await D.sleep(1500);
    const results = [];
    const base = meta();
    for (let r = 0; r < rounds; r++) {
      for (const id of r % 2 ? [...ids].reverse() : ids) {
        const before = await gate();
        const res = await page.evaluate((m) => window.runScenario(m), {
          cmd: 'run', id, seconds, stemSeconds: +(process.env.STEM_S ?? 100), paced, label: `r${r}`,
        });
        const after = env();
        results.push({ ...res, before, after });
        const q = res.quantumMs;
        const us = (v) => (v * 1000).toFixed(0);
        console.log(id, `r${r}`, `wall ${(res.wallMs / 1000).toFixed(1)}s med ${us(q.median)}us p99 ${us(q.p99)} p99.9 ${us(q.p999)} max ${us(q.max)}`,
          `over50 ${q.over50} over100 ${q.over100} | ${JSON.stringify(after)}`);
        fs.writeFileSync(D.outDir(`${tag}.json`), JSON.stringify({ ...base, paced, seconds, results }, null, 1));
      }
    }
  },
  { roots: [path.join(stands, 'android')], isolated: true },
);
