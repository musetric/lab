import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { execFileSync, spawn } from 'node:child_process';
import { startPool } from './pool.mjs';

export const here = import.meta.dirname;
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export const targets = ['mac-safari', 'mac-chrome', 'mac-app', 'iphone-safari', 'iphone-app'];
export const isPhone = (target) => target.startsWith('iphone');

const run = (file, args, options = {}) =>
  execFileSync(file, args, { encoding: 'utf8', maxBuffer: 1 << 26, ...options });
const tryRun = (file, args) => {
  try {
    return run(file, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  } catch (error) {
    return String(error.stdout ?? '') + String(error.stderr ?? '');
  }
};

export const outDir = (name) => {
  const dir = path.join(here, 'out');
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, name);
};

const appBundle = () => process.env.APP_BUNDLE ?? 'com.musetric.client.dev';
const chromeBinary = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const safariBundle = 'com.apple.mobilesafari';

export const udid = () => {
  if (process.env.IPHONE_UDID) {
    return process.env.IPHONE_UDID;
  }
  const listed = JSON.parse(run('pymobiledevice3', ['usbmux', 'list']));
  const phone = listed.find((d) => d.DeviceClass === 'iPhone' && d.ConnectionType === 'USB');
  if (!phone) {
    throw new Error('no iPhone on USB');
  }
  return phone.UniqueDeviceID;
};

const devicectl = (...args) => {
  const file = outDir(`devicectl-${process.pid}.json`);
  run('xcrun', ['devicectl', ...args, '--json-output', file, '-q']);
  const out = JSON.parse(fs.readFileSync(file, 'utf8'));
  fs.rmSync(file, { force: true });
  return out.result;
};

const phoneProcesses = () =>
  devicectl('device', 'info', 'processes', '--device', udid()).runningProcesses.map((p) => ({
    pid: p.processIdentifier,
    path: decodeURIComponent(p.executable ?? ''),
  }));

export const phonePid = (fragment) => phoneProcesses().find((p) => p.path.includes(fragment))?.pid;

export const phoneAppPid = (bundle) => {
  const app = devicectl('device', 'info', 'apps', '--device', udid(), '--bundle-id', bundle).apps[0];
  const root = decodeURIComponent(new URL(app.url).pathname);
  return phoneProcesses().find((p) => p.path.replace(/^file:\/\//, '').startsWith(root))?.pid;
};

export const phoneTerminate = (fragment) => {
  const pid = phonePid(fragment);
  if (pid) {
    tryRun('xcrun', ['devicectl', 'device', 'process', 'terminate', '--device', udid(), '--pid', String(pid), '-q']);
  }
};

export const phoneLaunch = (bundle, url) => {
  const args = ['devicectl', 'device', 'process', 'launch', '--device', udid(), '-q'];
  if (url) {
    args.push('--payload-url', url);
  } else {
    args.push('--terminate-existing');
  }
  run('xcrun', [...args, bundle]);
};

const xctestrun = () => {
  const dir = path.join(here, 'phonebridge/build/Build/Products');
  const file = fs.existsSync(dir) && fs.readdirSync(dir).find((f) => f.endsWith('.xctestrun'));
  if (!file) {
    throw new Error('build the phone bridge first: see readme.md');
  }
  return path.join(dir, file);
};

export class Bridge {
  constructor({ listenPort = 18443, tunnelPort = 18444, forwardPort = 28444 } = {}) {
    Object.assign(this, { listenPort, tunnelPort, forwardPort });
    this.thermal = null;
    this.lines = [];
    this.order = '';
  }

  handle(req, res, url) {
    if (url.pathname !== '/__bridge/command') {
      return false;
    }
    res.end(this.order);
    this.order = '';
    return true;
  }

  tap() {
    this.order = 'tap';
  }

  home(seconds) {
    this.order = `home ${seconds}`;
  }

  async start(serverPort, bundle, seconds = 4 * 3600) {
    this.forward = spawn(
      'pymobiledevice3',
      ['usbmux', 'forward', '--serial', udid(), '--host', '127.0.0.1', String(this.forwardPort), String(this.tunnelPort)],
      { stdio: 'ignore' },
    );
    this.runner = spawn(
      'xcodebuild',
      ['test-without-building', '-xctestrun', xctestrun(), '-destination', `platform=iOS,id=${udid()}`],
      {
        env: {
          ...process.env,
          TEST_RUNNER_BRIDGE_SECONDS: String(seconds),
          TEST_RUNNER_BRIDGE_LISTEN: String(this.listenPort),
          TEST_RUNNER_BRIDGE_TUNNEL: String(this.tunnelPort),
          TEST_RUNNER_BRIDGE_BUNDLE: bundle,
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    );
    const log = fs.createWriteStream(outDir('bridge.log'), { flags: 'a' });
    for (const stream of [this.runner.stdout, this.runner.stderr]) {
      readline.createInterface({ input: stream }).on('line', (line) => {
        log.write(line + '\n');
        const m = /^BRIDGE .*thermal (\d)/.exec(line);
        if (m) {
          this.thermal = { state: +m[1], at: Date.now() };
        }
        if (line.startsWith('BRIDGE')) {
          this.lines.push(line);
        }
      });
    }
    for (let i = 0; i < 120 && !this.lines.some((l) => l.includes('listener ready')); i++) {
      await sleep(1000);
    }
    if (!this.lines.some((l) => l.includes('listener ready'))) {
      throw new Error('the phone bridge did not start: see out/bridge.log');
    }
    this.pool = startPool(this.forwardPort, serverPort);
  }

  stop() {
    this.pool?.close();
    this.runner?.kill('SIGINT');
    this.forward?.kill('SIGINT');
  }
}

const macThermalTool = () => {
  const tool = outDir('thermal');
  if (!fs.existsSync(tool)) {
    const source = outDir('thermal.swift');
    fs.writeFileSync(source, 'import Foundation\nprint(ProcessInfo.processInfo.thermalState.rawValue)\n');
    run('xcrun', ['swiftc', '-O', '-o', tool, source]);
  }
  return tool;
};

export const navigateLibrary = () => {
  const library = outDir('navigate.dylib');
  const source = path.join(here, 'navigate.m');
  if (!fs.existsSync(library) || fs.statSync(library).mtimeMs < fs.statSync(source).mtimeMs) {
    run('clang', ['-dynamiclib', '-fobjc-arc', '-framework', 'AppKit', '-framework', 'WebKit', '-o', library, source]);
  }
  return library;
};

const macPids = (pattern) =>
  tryRun('pgrep', ['-f', pattern])
    .split('\n')
    .filter(Boolean)
    .map(Number);

const waitGone = async (pid) => {
  for (let i = 0; i < 100; i++) {
    try {
      process.kill(pid, 0);
    } catch {
      return;
    }
    await sleep(100);
  }
};

export const env = (bridge) => {
  const mac = {
    thermalState: +run(macThermalTool(), []).trim(),
    cpuSpeedLimit: +(/CPU_Speed_Limit\s*=\s*(\d+)/.exec(tryRun('pmset', ['-g', 'therm']))?.[1] ?? 100),
    load1: +run('sysctl', ['-n', 'vm.loadavg']).trim().split(/\s+/)[1],
  };
  if (!bridge) {
    return { mac };
  }
  const fresh = bridge.thermal && Date.now() - bridge.thermal.at < 15000;
  return { mac, phone: { thermalState: fresh ? bridge.thermal.state : null } };
};

export const gate = async (target, bridge) => {
  for (let i = 0; i < 120; i++) {
    const state = env(bridge);
    const level = isPhone(target) ? state.phone?.thermalState : state.mac.thermalState;
    if (level !== null && level !== undefined && level <= 1) {
      return state;
    }
    if (level === null || level === undefined) {
      await sleep(1000);
      continue;
    }
    console.log('thermal wait', JSON.stringify(state));
    await sleep(20000);
  }
  throw new Error('thermal gate');
};

export const versions = (target) => {
  const mac = {
    macos: run('sw_vers', ['-productVersion']).trim(),
    build: run('sw_vers', ['-buildVersion']).trim(),
    chip: run('sysctl', ['-n', 'machdep.cpu.brand_string']).trim(),
    model: run('sysctl', ['-n', 'hw.model']).trim(),
  };
  if (!isPhone(target)) {
    return { mac };
  }
  const info = JSON.parse(run('pymobiledevice3', ['usbmux', 'list'])).find((d) => d.UniqueDeviceID === udid());
  return { mac, phone: { model: info.ProductType, ios: info.ProductVersion, build: info.BuildVersion } };
};

export class Footprint {
  constructor(target, responsiblePid, intervalMs = 250) {
    this.samples = [];
    if (isPhone(target)) {
      this.child = spawn(
        'pymobiledevice3',
        [
          'developer', 'dvt', 'sysmon', 'process', 'monitor', 'threshold', '0', '--udid', udid(),
          '-k', 'pid', '-k', 'name', '-k', 'physFootprint', '-k', 'coalitionID',
        ],
        { stdio: ['ignore', 'pipe', 'ignore'] },
      );
      let coalition = null;
      readline.createInterface({ input: this.child.stdout }).on('line', (line) => {
        try {
          const p = JSON.parse(line);
          if (p.pid === responsiblePid) {
            coalition = p.coalitionID;
          }
          if (p.name === 'com.apple.WebKit.WebContent' && coalition !== null && p.coalitionID === coalition) {
            this.samples.push({ t: Date.now(), procs: [{ pid: p.pid, footprintMB: +(p.physFootprint / 2 ** 20).toFixed(1) }] });
          }
        } catch {}
      });
    } else {
      this.child = spawn('python3', [path.join(here, 'footprint.py'), String(responsiblePid), String(intervalMs)], {
        stdio: ['pipe', 'pipe', 'inherit'],
      });
      readline.createInterface({ input: this.child.stdout }).on('line', (line) => this.samples.push(JSON.parse(line)));
    }
  }

  peaks(from = 0, to = Infinity) {
    const byPid = new Map();
    for (const s of this.samples.filter((x) => x.t >= from && x.t <= to)) {
      for (const p of s.procs) {
        const row = byPid.get(p.pid) ?? { pid: p.pid, name: p.name, polledMaxMB: 0, lifetimeMaxMB: null, samples: 0 };
        row.polledMaxMB = Math.max(row.polledMaxMB, p.footprintMB);
        if (p.lifetimeMaxMB !== undefined) {
          row.lifetimeMaxMB = Math.max(row.lifetimeMaxMB ?? 0, p.lifetimeMaxMB);
        }
        row.samples++;
        byPid.set(p.pid, row);
      }
    }
    return [...byPid.values()].sort((a, b) => b.polledMaxMB - a.polledMaxMB);
  }

  endValues(to = Infinity) {
    const last = new Map();
    for (const s of this.samples.filter((x) => x.t <= to)) {
      for (const p of s.procs) {
        last.set(p.pid, p.footprintMB);
      }
    }
    return Object.fromEntries(last);
  }

  async stop() {
    if (this.child.stdin) {
      this.child.stdin.end();
    } else {
      this.child.kill('SIGINT');
    }
    await sleep(300);
  }
}

const hostFor = (target) => (isPhone(target) ? '127.0.0.1' : 'localhost');

export class Target {
  constructor(target, { remote, serverPort, bridge, home, appBinary } = {}) {
    if (!targets.includes(target)) {
      throw new Error(`target is one of ${targets.join(', ')}`);
    }
    Object.assign(this, { target, remote, serverPort, bridge, home, appBinary });
    this.responsiblePid = null;
  }

  url(pathname) {
    const port = isPhone(this.target) ? this.bridge.listenPort : this.serverPort;
    return `http://${hostFor(this.target)}:${port}${pathname}`;
  }

  async open(pathname) {
    await this.close();
    const since = Date.now();
    const url = this.url(pathname);
    if (this.target === 'mac-safari') {
      await this.webdriverStart();
      this.session = (await this.webdriver('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } })).sessionId;
      await this.webdriver('POST', `/session/${this.session}/url`, { url });
      this.responsiblePid = Number(tryRun('pgrep', ['-x', 'Safari']).split('\n')[0]) || null;
    }
    if (this.target === 'mac-chrome') {
      const profile = outDir('chrome-profile');
      this.process = spawn(
        chromeBinary,
        [
          `--user-data-dir=${profile}`,
          '--no-first-run',
          '--no-default-browser-check',
          '--autoplay-policy=no-user-gesture-required',
          '--new-window',
          url,
        ],
        { stdio: 'ignore' },
      );
      this.responsiblePid = this.process.pid;
    }
    if (this.target === 'mac-app') {
      if (!this.appBinary || !this.home) {
        throw new Error('mac-app needs APP_BINARY and APP_HOME');
      }
      fs.mkdirSync(this.home, { recursive: true });
      this.process = spawn(this.appBinary, [], {
        env: { ...process.env, HOME: this.home, DYLD_INSERT_LIBRARIES: navigateLibrary(), BENCH_URL: url },
        stdio: ['ignore', 'ignore', fs.openSync(outDir('mac-app.log'), 'a')],
      });
      this.responsiblePid = this.process.pid;
    }
    if (this.target === 'iphone-safari') {
      phoneTerminate('MobileSafari.app');
      await sleep(1000);
      phoneLaunch(safariBundle, url);
      await sleep(1500);
      this.responsiblePid = phonePid('MobileSafari.app');
    }
    if (this.target === 'iphone-app') {
      phoneTerminate('MobileSafari.app');
      phoneLaunch(appBundle());
      await sleep(4000);
      this.responsiblePid = phoneAppPid(appBundle());
      await this.inspectorNavigate(url);
    }
    this.page = await this.remote.page(pathname.split('?')[0], { since, timeoutMs: 90000 });
    return this.page;
  }

  async webdriverStart() {
    if (this.driver) {
      return;
    }
    this.driverPort = 4445;
    this.driver = spawn('safaridriver', ['-p', String(this.driverPort)], { stdio: 'ignore' });
    for (let i = 0; i < 50; i++) {
      try {
        await fetch(`http://127.0.0.1:${this.driverPort}/status`);
        return;
      } catch {
        await sleep(200);
      }
    }
    throw new Error('safaridriver did not start');
  }

  async webdriver(method, pathname, body) {
    const res = await fetch(`http://127.0.0.1:${this.driverPort}${pathname}`, {
      method,
      headers: { 'content-type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    const json = await res.json();
    if (json.value?.error) {
      throw new Error(`safaridriver ${pathname}: ${json.value.error} ${json.value.message}`);
    }
    return json.value;
  }

  async activate() {
    if (isPhone(this.target)) {
      this.bridge.tap();
      return true;
    }
    if (this.target === 'mac-safari' && this.session) {
      const body = await this.webdriver('POST', `/session/${this.session}/element`, { using: 'css selector', value: 'body' });
      const id = Object.values(body)[0];
      await this.webdriver('POST', `/session/${this.session}/element/${id}/click`, {});
      return true;
    }
    return false;
  }

  async inspectorNavigate(url) {
    if (!this.inspector) {
      this.inspectorPort = 9444;
      this.inspector = spawn(
        'pymobiledevice3',
        ['webinspector', 'cdp', '--udid', udid(), '--port', String(this.inspectorPort)],
        { stdio: ['ignore', fs.openSync(outDir('inspector.log'), 'a'), fs.openSync(outDir('inspector.log'), 'a')] },
      );
      await sleep(4000);
    }
    for (let attempt = 0; attempt < 30; attempt++) {
      try {
        const list = await (await fetch(`http://127.0.0.1:${this.inspectorPort}/json`)).json();
        const page = list.find((p) => /^http:\/\/127\.0\.0\.1:\d+/.test(p.url ?? '') && !/executor|ortWarmup/.test(p.url ?? ''));
        if (page) {
          const ws = new WebSocket(page.webSocketDebuggerUrl);
          await new Promise((resolve, reject) => {
            ws.onopen = resolve;
            ws.onerror = reject;
          });
          ws.send(JSON.stringify({ id: 1, method: 'Runtime.evaluate', params: { expression: `location.href = ${JSON.stringify(url)}` } }));
          await sleep(500);
          ws.close();
          return;
        }
      } catch {}
      await sleep(1000);
    }
    throw new Error('the app page did not show up in Web Inspector');
  }

  async close() {
    this.page = null;
    if (this.target === 'mac-safari' && this.session) {
      await this.webdriver('DELETE', `/session/${this.session}`).catch(() => {});
      this.session = null;
      for (const pid of tryRun('pgrep', ['-x', 'Safari']).split('\n').filter(Boolean).map(Number)) {
        process.kill(pid, 'SIGTERM');
        await waitGone(pid);
      }
      await sleep(1000);
    }
    if ((this.target === 'mac-chrome' || this.target === 'mac-app') && this.process) {
      const pid = this.process.pid;
      try {
        process.kill(pid, 'SIGTERM');
      } catch {}
      await waitGone(pid);
      this.process = null;
    }
    if (this.target === 'iphone-safari') {
      phoneTerminate('MobileSafari.app');
    }
    if (this.target === 'iphone-app') {
      phoneTerminate(`${appBundle()}`);
    }
  }

  async dispose() {
    await this.close();
    this.inspector?.kill('SIGINT');
    this.driver?.kill('SIGINT');
  }
}
