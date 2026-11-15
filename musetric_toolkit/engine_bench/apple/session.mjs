import path from 'node:path';
import { startServer } from '../android/server.mjs';
import { Remote } from './remote.mjs';
import * as D from './device.mjs';

export const stands = path.resolve(D.here, '..');

export const session = async (target, body, { roots = [stands], routes = {}, isolated = false } = {}) => {
  if (!D.targets.includes(target)) {
    console.error(`target: ${D.targets.join(' | ')}`);
    process.exit(2);
  }
  const remote = new Remote();
  let extra = null;
  const { server, port } = await startServer({
    roots,
    routes,
    isolated,
    handle: (req, res, url) =>
      Boolean(extra?.(req, res, url)) || Boolean(bridge?.handle(req, res, url)) || remote.handle(req, res, url),
    inject: remote.inject,
  });
  let bridge = null;
  const opened = new D.Target(target, {
    remote,
    serverPort: port,
    home: process.env.APP_HOME,
    appBinary: process.env.APP_BINARY,
  });
  let failed = false;
  try {
    if (D.isPhone(target)) {
      bridge = new D.Bridge();
      await bridge.start(port, target === 'iphone-app' ? process.env.APP_BUNDLE ?? 'com.musetric.client.dev' : 'com.apple.mobilesafari');
      opened.bridge = bridge;
    }
    let warned = false;
    const call = async (page, fn, arg) => {
      let done = false;
      const result = page.evaluate(fn, arg).finally(() => {
        done = true;
      });
      (async () => {
        while (!done) {
          await D.sleep(700);
          if (done) {
            break;
          }
          const waiting = await page.evaluate(() => window.__remoteSuspended?.() ?? 0).catch(() => 0);
          if (waiting > 0 && !(await opened.activate().catch(() => false)) && !warned) {
            warned = true;
            console.log(`${target}: an AudioContext waits for a user gesture`);
          }
        }
      })();
      return result;
    };
    await body({
      call,
      target,
      port,
      remote,
      bridge,
      device: opened,
      open: (pathname) => opened.open(pathname),
      close: () => opened.close(),
      gate: () => D.gate(target, bridge),
      env: () => D.env(bridge),
      meta: () => ({ target, device: D.versions(target), at: new Date().toISOString(), env: D.env(bridge) }),
      handleExtra: (fn) => {
        extra = fn;
      },
    });
  } catch (error) {
    failed = true;
    console.error(error);
  } finally {
    await opened.dispose();
    bridge?.stop();
    remote.close();
    server.close();
  }
  process.exit(failed ? 1 : 0);
};
