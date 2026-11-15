import fs from 'node:fs';
import path from 'node:path';
import * as D from './device.mjs';
import { session, stands } from './session.mjs';

const [target] = process.argv.slice(2);
await session(target, async ({ open, meta, handleExtra, call }) => {
  let report;
  handleExtra((req, res, url) => {
    if (req.method === 'POST' && url.pathname === '/report') {
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', () => {
        report = JSON.parse(body);
        res.end('ok');
      });
      return true;
    }
    return false;
  });
  const page = await open('/index.html');
  for (let i = 0; i < 100 && !report; i++) {
    await D.sleep(200);
  }
  const audio = await page.evaluate(async () => {
    const c = new AudioContext();
    const before = c.state;
    await Promise.race([c.resume(), new Promise((r) => setTimeout(r, 3000))]);
    const after = c.state;
    const t0 = c.currentTime;
    await new Promise((r) => setTimeout(r, 500));
    const advanced = c.currentTime - t0;
    await c.close();
    return { before, after, advancedS: advanced };
  });
  const withGesture = await call(page, async () => {
    const c = new AudioContext();
    await Promise.race([c.resume(), new Promise((r) => setTimeout(r, 15000))]);
    const state = c.state;
    await c.close();
    return { state };
  });
  const pool = await open('/web/index.html');
  await pool.waitForFunction(() => Boolean(window.T?.env));
  const poolEnv = await pool.evaluate(() => window.T.env());
  const viewport = await pool.evaluate(() => ({ width: innerWidth, height: innerHeight }));
  const out = { ...meta(), facts: report?.facts ?? null, audioWithoutGesture: audio, audioAfterTap: withGesture, poolEnv, viewport };
  console.log(JSON.stringify(out, null, 1));
  fs.writeFileSync(D.outDir(`facts-${target}.json`), JSON.stringify(out, null, 1));
}, { roots: [path.join(stands, 'page_pool_spike'), path.join(stands, 'ios/www')] });
