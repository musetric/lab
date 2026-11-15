import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { RawPage, cdpPort, sleep } from './device.mjs';

export { outDir, sleep } from './device.mjs';

const executable = process.env.DESKTOP_BROWSER ?? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
let browser;
let current;

export const mapPort = () => {};
export const cleanup = () => {};
export const wake = () => {};
export const env = () => ({ thermalStatus: null, focus: null });
export const gate = async () => env();
export const front = () => {};
export const assertFront = () => {};
export const tapActivation = async () => {};
export const chromeFresh = () => {};

const close = async () => {
  if (!browser) return;
  const { child, profile } = browser;
  browser = null;
  child.kill();
  await new Promise((resolve) => child.once('exit', resolve));
  await sleep(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
};

export const connect = async () => {
  await close();
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'engine-bench-'));
  const child = spawn(executable, [
    `--remote-debugging-port=${cdpPort}`,
    `--user-data-dir=${profile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--force_high_performance_gpu',
    '--autoplay-policy=no-user-gesture-required',
    'about:blank',
  ]);
  browser = { child, profile };
  for (let i = 0; i < 100; i++) {
    try {
      browser.version = await (await fetch(`http://127.0.0.1:${cdpPort}/json/version`)).json();
      break;
    } catch {
      await sleep(100);
    }
  }
  return { close };
};

const json = async (route) =>
  (await fetch(`http://127.0.0.1:${cdpPort}/json/${route}`, { method: 'PUT' })).json().catch(() => null);

export const openPage = async (_, url) => {
  let targetId;
  current = await RawPage.open(cdpPort, (targets) => {
    const pages = targets.filter((t) => t.type === 'page');
    const target = pages.find((t) => t.url === 'about:blank') ?? pages[0];
    targetId = target.id;
    return target;
  });
  current.targetId = targetId;
  current.setDefaultTimeout = () => {};
  current.close = close;
  await current.goto(url);
  await json(`activate/${targetId}`);
  for (let i = 0; (await current.evaluate(() => document.visibilityState)) !== 'visible'; i++) {
    if (i === 50) throw new Error('the page is not visible');
    await sleep(100);
  }
  return current;
};

let other;
export const toBackground = async () => {
  other = await json('new?about:blank');
};
export const toFront = async () => {
  await json(`activate/${current.targetId}`);
  if (other) await json(`close/${other.id}`);
  other = null;
};

export const versions = () => ({
  browser: browser?.version?.Browser ?? null,
  os: `${os.type()} ${os.release()}`,
  cpu: os.cpus()[0].model.trim(),
});
