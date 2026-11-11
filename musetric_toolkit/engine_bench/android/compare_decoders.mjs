import fs from 'node:fs';
import path from 'node:path';
import * as device from './device.mjs';
import { here, startServer } from './server.mjs';

const aacPages = path.resolve(here, '../aac_pages');
const channel = process.env.DESKTOP_CHANNEL ?? 'msedge';
const names = ['chord', 'noise', 'clicks', 'sweep'];
const packets = 400;
const results = {};

const { server, port } = await startServer({
  roots: [here, aacPages],
  routes: { '/mediabunny.mjs': path.join(aacPages, 'node_modules/mediabunny/dist/bundles/mediabunny.min.mjs') },
  isolated: true,
});
device.mapPort(port);

const decodeAll = async (page, key) => {
  results[key] = {};
  for (const name of names) {
    const base64 = await page.evaluate(([n, count]) => window.decodeN(n, count), [name, packets]);
    results[key][name] = Buffer.from(base64, 'base64');
  }
};

try {
  device.wake();
  console.log(await device.gate());

  const chrome = await device.openTarget('chrome', `http://localhost:${port}/web/decode-bits.html`);
  await chrome.page.waitForFunction(() => window.ready);
  await decodeAll(chrome.page, 'phoneChrome');
  await chrome.close();

  const webview = await device.openTarget('webview', `http://localhost:${port}/web/decode-bits.html`);
  await webview.page.waitForFunction(() => window.ready);
  await decodeAll(webview.page, 'phoneWebview');
  await webview.close();

  const options = { args: ['--force_high_performance_gpu'] };
  if (channel !== 'chromium') options.channel = channel;
  const desktop = await device.chromium.launch(options);
  const page = await desktop.newPage();
  await page.goto(`http://127.0.0.1:${port}/web/decode-bits.html`);
  await page.waitForFunction(() => window.ready);
  await decodeAll(page, 'desktop');
  await desktop.close();
} finally {
  device.cleanup();
  server.close();
}

const f32 = (b) => new Float32Array(b.buffer, b.byteOffset, b.length / 4);
const compare = (a, b) => {
  const x = f32(a);
  const y = f32(b);
  let differing = 0;
  let maxAbsDiff = 0;
  let sumSq = 0;
  let sigSq = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - y[i]);
    if (d > 0) {
      differing++;
      maxAbsDiff = Math.max(maxAbsDiff, d);
    }
    sumSq += d * d;
    sigSq += y[i] * y[i];
  }
  return {
    frames: x.length / 2,
    samples: x.length,
    differing,
    maxAbsDiff,
    snrDb: differing ? +(10 * Math.log10(sigSq / sumSq)).toFixed(1) : null,
  };
};

const out = {};
for (const name of names) {
  out[name] = {
    chromeVsWebview: compare(results.phoneChrome[name], results.phoneWebview[name]),
    phoneChromeVsDesktop: compare(results.phoneChrome[name], results.desktop[name]),
  };
}
console.log(JSON.stringify(out, null, 1));
fs.writeFileSync(device.outDir('compare_decoders.json'), JSON.stringify(out, null, 1));
process.exit(0);
