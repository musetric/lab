import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';

export { chromium };

export const ADB = process.env.ADB ?? 'adb';
export const cdpPort = 9333;
export const chromePackage = 'com.android.chrome';
const webviewProviderPackage = 'com.google.android.webview';

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const adb = (...args) => execFileSync(ADB, args, { encoding: 'utf8', maxBuffer: 1 << 26 });
export const sh = (command) => adb('shell', command);

export const defaultAppPackage = 'com.musetric.client.dev';
export const webviewPackage = () => process.env.APP_PACKAGE ?? defaultAppPackage;

export const env = () => {
  const thermal = sh('dumpsys thermalservice');
  const thermalStatus = +/Thermal Status: (\d+)/.exec(thermal)[1];
  const temps = {};
  for (const m of thermal.matchAll(/mValue=([\d.]+), mType=\d+, mName=(\w+)/g)) {
    temps[m[2]] = +m[1];
  }
  const cpuMaxC = Math.max(
    ...Object.entries(temps)
      .filter(([name]) => name.startsWith('CPU'))
      .map(([, value]) => value),
  );
  const batteryC = +/temperature: (\d+)/.exec(sh('dumpsys battery'))[1] / 10;
  const focus = /mCurrentFocus=.*? (\S+)\}/.exec(sh('dumpsys window | grep mCurrentFocus'))?.[1];
  return { thermalStatus, batteryC, cpuMaxC, focus };
};

export const gate = async () => {
  for (let i = 0; i < 60; i++) {
    const state = env();
    if (state.thermalStatus <= 1) {
      return state;
    }
    console.log('thermal wait', JSON.stringify(state));
    await sleep(20000);
  }
  throw new Error('thermal gate');
};

export const wake = () => {
  sh('input keyevent KEYCODE_WAKEUP');
  sh('wm dismiss-keyguard');
};

export const screenCenter = () => {
  const size = sh('wm size');
  const m = /Override size: (\d+)x(\d+)/.exec(size) ?? /Physical size: (\d+)x(\d+)/.exec(size);
  if (!m) {
    throw new Error(`cannot read the screen size: ${size}`);
  }
  return [Math.round(+m[1] / 2), Math.round(+m[2] / 2)];
};

export const tapActivation = async () => {
  const [x, y] = process.env.ACTIVATION_TAP ? process.env.ACTIVATION_TAP.split(',').map(Number) : screenCenter();
  sh(`input tap ${x} ${y}`);
  await sleep(300);
};

export const mapPort = (port) => {
  adb('reverse', `tcp:${port}`, `tcp:${port}`);
};

export const cleanup = () => {
  try {
    adb('reverse', '--remove-all');
  } catch {}
  try {
    adb('forward', '--remove-all');
  } catch {}
};

export const versions = () => {
  const version = (pkg) => {
    try {
      return /versionName=(\S+)/.exec(sh(`dumpsys package ${pkg}`))?.[1] ?? null;
    } catch {
      return null;
    }
  };
  return {
    chrome: version(chromePackage),
    webview: version(webviewProviderPackage),
    android: sh('getprop ro.build.version.release').trim(),
    device: sh('getprop ro.product.model').trim(),
  };
};

const webviewSocket = (pkg) => {
  const pid = sh(`pidof ${pkg}`).trim().split(/\s+/)[0];
  if (!pid) {
    throw new Error(`${pkg} is not running`);
  }
  return `webview_devtools_remote_${pid}`;
};

const chromeSocketUp = () => sh('cat /proc/net/unix').includes('chrome_devtools_remote');

export const connect = async () => {
  if (!chromeSocketUp()) {
    sh(`am start -a android.intent.action.VIEW -d about:blank ${chromePackage}`);
    for (let i = 0; i < 50 && !chromeSocketUp(); i++) {
      await sleep(200);
    }
  }
  adb('forward', `tcp:${cdpPort}`, 'localabstract:chrome_devtools_remote');
  return chromium.connectOverCDP(`http://127.0.0.1:${cdpPort}`);
};

export const openPage = async (browser, url) => {
  const page = await browser.contexts()[0].newPage();
  await page.goto(url);
  await page.bringToFront();
  return page;
};

export const chromeFresh = () => {
  sh(`am force-stop ${chromePackage}`);
  sh(`am start -a android.intent.action.VIEW -d about:blank ${chromePackage}`);
};

export class RawPage {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.onmessage = (event) => {
      const m = JSON.parse(event.data);
      const call = this.pending.get(m.id);
      if (call) {
        this.pending.delete(m.id);
        if (m.error) {
          call.reject(new Error(JSON.stringify(m.error)));
        } else {
          call.resolve(m.result);
        }
      }
    };
  }

  static async open(port) {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    const target = targets.find((x) => x.type === 'page');
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = reject;
    });
    return new RawPage(ws);
  }

  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async goto(url) {
    await this.send('Page.enable');
    await this.send('Page.navigate', { url });
    for (let i = 0; i < 100; i++) {
      await sleep(200);
      try {
        if ((await this.evaluate(() => document.readyState)) === 'complete') {
          return;
        }
      } catch {}
    }
    throw new Error('goto timeout');
  }

  async evaluate(fn, arg) {
    const expression = `(${fn.toString()})(${arg === undefined ? '' : JSON.stringify(arg)})`;
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) {
      throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    }
    return r.result.value;
  }

  async waitForFunction(fn) {
    for (let i = 0; i < 200; i++) {
      if (await this.evaluate(fn)) {
        return;
      }
      await sleep(100);
    }
    throw new Error('wait timeout');
  }

  on() {}

  async close() {
    try {
      this.ws.close();
    } catch {}
  }
}

export const connectWebview = async () => {
  adb('forward', `tcp:${cdpPort}`, `localabstract:${webviewSocket(webviewPackage())}`);
  return RawPage.open(cdpPort);
};

export const front = (target) => {
  if (target === 'webview') {
    sh(`monkey -p ${webviewPackage()} -c android.intent.category.LAUNCHER 1`);
  } else {
    sh(`am start -n ${chromePackage}/org.chromium.chrome.browser.ChromeTabbedActivity`);
  }
};

export const assertFront = (target) => {
  const focus = env().focus ?? '';
  const pkg = target === 'webview' ? webviewPackage() : chromePackage;
  if (!focus.startsWith(pkg)) {
    throw new Error(`not in front: ${focus}`);
  }
};

export const targetOf = (label) => (label.startsWith('webview') ? 'webview' : 'chrome');

export const openTarget = async (target, url) => {
  if (target === 'webview') {
    front('webview');
    await sleep(3000);
    const page = await connectWebview();
    await page.goto(url);
    return { page, close: () => page.close() };
  }
  const browser = await connect();
  const page = await openPage(browser, url);
  return {
    page,
    close: async () => {
      try {
        await page.close();
        await browser.close();
      } catch {}
    },
  };
};

export const outDir = (name) => {
  const dir = path.join(import.meta.dirname, 'out');
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, name);
};

export const adbBinary = (...args) => execFileSync(ADB, args, { maxBuffer: 1 << 28 });
