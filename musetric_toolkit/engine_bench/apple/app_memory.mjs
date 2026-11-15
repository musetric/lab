import fs from 'node:fs';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import * as D from './device.mjs';

const [target, command, ...rest] = process.argv.slice(2);
if (!['mac-app', 'iphone-app'].includes(target) || !['prepare', 'run'].includes(command)) {
  console.error('usage: node app_memory.mjs <mac-app|iphone-app> <prepare <song>|run [cycles] [openWaitS] [closeWaitS]>');
  process.exit(2);
}
const appBundle = process.env.APP_BUNDLE ?? 'com.musetric.client.dev';

class MacApp {
  async start() {
    this.dir = D.outDir('appjs');
    fs.rmSync(this.dir, { recursive: true, force: true });
    fs.mkdirSync(this.dir, { recursive: true });
    const target = new D.Target('mac-app', {});
    this.process = spawn(process.env.APP_BINARY, [], {
      env: { ...process.env, HOME: process.env.APP_HOME, DYLD_INSERT_LIBRARIES: D.navigateLibrary(), BENCH_JS_DIR: this.dir },
      stdio: ['ignore', 'ignore', fs.openSync(D.outDir('mac-app.log'), 'a')],
    });
    this.pid = this.process.pid;
    this.n = 0;
    void target;
    for (let i = 0; i < 60; i++) {
      await D.sleep(1000);
      const origin = await this.eval('location.origin').catch(() => null);
      if (origin && origin.startsWith('http')) {
        const listening = execFileSync('lsof', ['-a', '-p', String(this.pid), '-iTCP', '-sTCP:LISTEN', '-n', '-P', '-Fn'], { encoding: 'utf8' })
          .split('\n')
          .filter((l) => l.startsWith('n'))
          .map((l) => l.slice(1));
        for (const address of listening) {
          const ok = await fetch(`http://${address}/api/processing`).then((r) => r.ok, () => false);
          if (ok) {
            this.base = `http://${address}`;
            return;
          }
        }
      }
    }
    throw new Error('the app page did not answer');
  }

  async eval(code, timeoutMs = 10000) {
    const name = `c${++this.n}`;
    fs.writeFileSync(path.join(this.dir, `${name}.js`), code);
    const out = path.join(this.dir, `${name}.json`);
    const end = Date.now() + timeoutMs;
    while (Date.now() < end) {
      if (fs.existsSync(out)) {
        const r = JSON.parse(fs.readFileSync(out, 'utf8'));
        fs.rmSync(out);
        if (r.error) {
          throw new Error(r.error);
        }
        return r.value;
      }
      await D.sleep(100);
    }
    throw new Error(`no answer to ${code}`);
  }

  async stop() {
    try {
      process.kill(this.pid, 'SIGTERM');
    } catch {}
    await D.sleep(1500);
  }
}

class PhoneApp {
  async start() {
    D.phoneTerminate('MobileSafari.app');
    D.phoneLaunch(appBundle);
    await D.sleep(5000);
    this.pid = D.phoneAppPid(appBundle);
    this.inspector = spawn('pymobiledevice3', ['webinspector', 'cdp', '--udid', D.udid(), '--port', '9444'], { stdio: 'ignore' });
    await D.sleep(4000);
    for (let i = 0; i < 30; i++) {
      const origin = await this.eval('location.origin').catch(() => null);
      if (origin && origin.startsWith('http')) {
        const port = new URL(origin).port;
        this.forward = spawn('pymobiledevice3', ['usbmux', 'forward', '--serial', D.udid(), '--host', '127.0.0.1', '28555', port], { stdio: 'ignore' });
        await D.sleep(1500);
        this.base = 'http://127.0.0.1:28555';
        this.devicePort = port;
        return;
      }
      await D.sleep(1000);
    }
    throw new Error('the app page did not show up in Web Inspector');
  }

  async page() {
    const list = await (await fetch('http://127.0.0.1:9444/json')).json();
    const page = list.find((p) => /^http:\/\/127\.0\.0\.1:\d+/.test(p.url ?? '') && !/executor|ortWarmup/.test(p.url ?? ''));
    if (!page) {
      throw new Error('no app page');
    }
    return page;
  }

  async eval(code) {
    const page = await this.page();
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = reject;
    });
    const answer = new Promise((resolve) => {
      ws.onmessage = (event) => resolve(JSON.parse(event.data));
    });
    ws.send(JSON.stringify({ id: 1, method: 'Runtime.evaluate', params: { expression: code, returnByValue: true } }));
    const r = await Promise.race([answer, D.sleep(10000).then(() => null)]);
    ws.close();
    if (!r) {
      throw new Error('no answer');
    }
    return r.result?.result?.value;
  }

  async stop() {
    this.forward?.kill('SIGINT');
    this.inspector?.kill('SIGINT');
  }
}

const api = async (base, method, pathname, body) => {
  const res = await fetch(base + pathname, {
    method,
    headers: body && !(body instanceof FormData) ? { 'content-type': 'application/json' } : undefined,
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`${method} ${pathname}: ${res.status} ${text.slice(0, 300)}`);
  }
  return text ? JSON.parse(text) : null;
};

const app = target === 'mac-app' ? new MacApp() : new PhoneApp();
await app.start();
console.log('app', target, 'pid', app.pid, 'api', app.base);
try {
  if (command === 'prepare') {
    const song = rest[0];
    const name = path.basename(song).replace(/\.[^.]+$/, '');
    const list = await api(app.base, 'GET', '/api/project/list');
    let project = list.find((p) => p.name === name);
    if (!project) {
      const form = new FormData();
      form.append('song', new Blob([fs.readFileSync(song)], { type: 'audio/wav' }), path.basename(song));
      form.append('name', name);
      project = await api(app.base, 'POST', '/api/project/create', form);
      console.log('created', project.id);
    }
    const started = Date.now();
    for (;;) {
      const item = await api(app.base, 'GET', `/api/project/${project.id}`);
      const steps = Object.entries(item.processing?.steps ?? {}).map(([k, v]) => `${k}:${v.status}`);
      console.log(new Date().toISOString().slice(11, 19), steps.join(' '));
      if (steps.length && Object.values(item.processing.steps).every((v) => v.status === 'done' || v.status === 'skipped')) {
        console.log('processed in', ((Date.now() - started) / 1000).toFixed(0), 's');
        break;
      }
      if (Object.values(item.processing?.steps ?? {}).some((v) => v.status === 'error' || v.status === 'failed')) {
        throw new Error(`processing failed: ${JSON.stringify(item.processing.steps).slice(0, 600)}`);
      }
      if (Date.now() - started > 30 * 60 * 1000) {
        throw new Error('processing takes over 30 minutes');
      }
      await D.sleep(10000);
    }
  }
  if (command === 'run') {
    const cycles = +(rest[0] ?? 3);
    const openWaitMs = +(rest[1] ?? 7) * 1000;
    const closeWaitMs = +(rest[2] ?? 4) * 1000;
    const list = await api(app.base, 'GET', '/api/project/list');
    if (!list.length) {
      throw new Error('the app holds no project: run prepare first');
    }
    const project = list.find((p) => p.name === (rest[3] ?? 'synthetic-52s')) ?? list[0];
    const footprint = new D.Footprint(target, app.pid, 250);
    const rows = [];
    let from = Date.now();
    const record = async (state) => {
      const to = Date.now();
      const peaks = footprint.peaks(from, to);
      const end = footprint.endValues(to);
      const route = await app.eval('location.pathname').catch(() => null);
      rows.push({ state, route, from, to, peaks, end });
      console.log(state.padEnd(9), String(route).padEnd(12), peaks.map((p) => `${p.pid}: peak ${p.polledMaxMB} end ${end[p.pid] ?? '-'}${p.lifetimeMaxMB ? ` life ${p.lifetimeMaxMB}` : ''}`).join(' | '));
      from = Date.now();
    };
    const env = D.env(null);
    await D.sleep(6000);
    await record('base');
    for (let i = 1; i <= cycles; i++) {
      await app.eval(`document.querySelector('a[href="/project/${project.id}"]').click(), location.pathname`);
      await D.sleep(openWaitMs);
      await record(`open ${i}`);
      await app.eval('history.back(), location.pathname');
      await D.sleep(closeWaitMs);
      await record(`closed ${i}`);
    }
    await footprint.stop();
    fs.writeFileSync(
      D.outDir(`app_memory-${target}.json`),
      JSON.stringify({ target, device: D.versions(target), project: { id: project.id, name: project.name }, env, cycles, openWaitMs, closeWaitMs, rows }, null, 1),
    );
  }
} finally {
  await app.stop();
}
process.exit(0);
