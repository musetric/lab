import fs from 'node:fs';
import path from 'node:path';
import * as device from './device.mjs';
import { here, startServer } from './server.mjs';

const [label] = process.argv.slice(2);
if (!['chrome', 'webview'].includes(label)) {
  console.error('usage: node facts.mjs <chrome|webview>');
  process.exit(2);
}
const www = path.resolve(here, '../ios/www');
const readFacts = (page) =>
  page.evaluate(async () => {
    for (let i = 0; i < 100; i++) {
      const text = document.getElementById('out').textContent;
      if (text !== 'checking...') {
        return JSON.parse(text);
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error('facts timeout');
  });
const appFacts = () => ({
  url: location.href,
  secureContext: isSecureContext,
  crossOriginIsolated: self.crossOriginIsolated,
  sharedArrayBuffer: typeof SharedArrayBuffer,
});

const plain = await startServer({ roots: [www] });
const isolated = await startServer({ roots: [www], isolated: true });
device.mapPort(plain.port);
device.mapPort(isolated.port);
const out = { label, device: device.versions() };
let opened;
try {
  device.wake();
  out.before = await device.gate();
  if (label === 'webview') {
    device.front('webview');
    await device.sleep(3000);
    await device.connectWebview();
    const targets = await (await fetch(`http://127.0.0.1:${device.cdpPort}/json`)).json();
    out.appPages = [];
    for (const target of targets.filter((t) => t.type === 'page')) {
      const page = await device.RawPage.open(device.cdpPort, () => target);
      out.appPages.push({ title: target.title, ...(await page.evaluate(appFacts)) });
      await page.close();
    }
  }
  opened = await device.openTarget(label, `http://localhost:${plain.port}/index.html`);
  out.plain = await readFacts(opened.page);
  await opened.page.goto(`http://localhost:${isolated.port}/index.html`);
  out.isolated = await readFacts(opened.page);
  out.after = device.env();
} finally {
  await opened?.close();
  device.cleanup();
  plain.server.close();
  isolated.server.close();
}
console.log(JSON.stringify(out, null, 1));
fs.writeFileSync(device.outDir(`facts-${label}.json`), JSON.stringify(out, null, 1));
process.exit(0);
