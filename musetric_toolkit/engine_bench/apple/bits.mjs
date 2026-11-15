import fs from 'node:fs';
import path from 'node:path';
import * as D from './device.mjs';
import { session, stands } from './session.mjs';

const names = ['chord', 'noise', 'clicks', 'sweep'];
const packets = 400;
const [target] = process.argv.slice(2);

if (target === 'compare') {
  const f32 = (file) => {
    const b = fs.readFileSync(file);
    return new Float32Array(b.buffer, b.byteOffset, b.length / 4);
  };
  const decoded = D.targets.filter((t) => names.every((n) => fs.existsSync(D.outDir(`bits-${t}-${n}.f32`))));
  const out = {};
  for (let i = 0; i < decoded.length; i++) {
    for (let j = i + 1; j < decoded.length; j++) {
      const pair = `${decoded[i]} vs ${decoded[j]}`;
      out[pair] = {};
      for (const name of names) {
        const x = f32(D.outDir(`bits-${decoded[i]}-${name}.f32`));
        const y = f32(D.outDir(`bits-${decoded[j]}-${name}.f32`));
        let differing = 0;
        let maxAbsDiff = 0;
        let sumSq = 0;
        let sigSq = 0;
        for (let k = 0; k < Math.min(x.length, y.length); k++) {
          const d = Math.abs(x[k] - y[k]);
          if (d > 0 || Number.isNaN(d)) {
            differing++;
            maxAbsDiff = Math.max(maxAbsDiff, d);
          }
          sumSq += d * d;
          sigSq += y[k] * y[k];
        }
        out[pair][name] = {
          samples: [x.length, y.length],
          differing,
          maxAbsDiff,
          snrDb: differing ? +(10 * Math.log10(sigSq / sumSq)).toFixed(1) : null,
        };
      }
      console.log(pair, JSON.stringify(Object.fromEntries(names.map((n) => [n, [out[pair][n].differing, out[pair][n].snrDb]]))));
    }
  }
  fs.writeFileSync(D.outDir('bits-compare.json'), JSON.stringify(out, null, 1));
  process.exit(0);
}

const aacPages = path.join(stands, 'aac_pages');
await session(
  target,
  async ({ open, meta }) => {
    const page = await open('/web/decode-bits.html');
    await page.waitForFunction(() => Boolean(window.ready));
    const hashes = {};
    for (const name of names) {
      const base64 = await page.evaluate(([n, count]) => window.decodeN(n, count), [name, packets]);
      const bytes = Buffer.from(base64, 'base64');
      fs.writeFileSync(D.outDir(`bits-${target}-${name}.f32`), bytes);
      hashes[name] = { bytes: bytes.length, sha256: (await import('node:crypto')).createHash('sha256').update(bytes).digest('hex') };
    }
    console.log(target, JSON.stringify(hashes));
    fs.writeFileSync(D.outDir(`bits-${target}.json`), JSON.stringify({ ...meta(), packets, hashes }, null, 1));
  },
  {
    roots: [path.join(stands, 'android'), aacPages],
    routes: { '/mediabunny.mjs': path.join(aacPages, 'node_modules/mediabunny/dist/bundles/mediabunny.min.mjs') },
    isolated: true,
  },
);
