import fs from 'node:fs';
import path from 'node:path';

const script = path.join(import.meta.dirname, 'web/remote.js');
const pollMs = 15000;
const goneMs = 150000;

const readBody = (req) =>
  new Promise((resolve) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
  });

export class Remote {
  constructor({ log = (line) => console.log(line) } = {}) {
    this.clients = new Map();
    this.pending = new Map();
    this.logs = [];
    this.log = log;
    this.nextId = 0;
    this.handle = this.handle.bind(this);
    this.inject = this.inject.bind(this);
    this.watchdog = setInterval(() => this.reap(), 5000);
    this.watchdog.unref();
  }

  inject(html) {
    const tag = '<script src="/__remote.js"></script>';
    return /^<!doctype html>/i.test(html) ? html.replace(/^<!doctype html>/i, (m) => m + tag) : tag + html;
  }

  client(id, page, ua) {
    let c = this.clients.get(id);
    if (!c) {
      c = { id, page, ua, first: Date.now(), seen: Date.now(), queue: [], waiter: null };
      this.clients.set(id, c);
    }
    c.seen = Date.now();
    return c;
  }

  handle(req, res, url) {
    if (url.pathname === '/__remote.js') {
      res.writeHead(200, { 'content-type': 'text/javascript', 'cache-control': 'no-store' });
      res.end(fs.readFileSync(script));
      return true;
    }
    if (url.pathname === '/__remote/next') {
      const c = this.client(url.searchParams.get('client'), url.searchParams.get('page'), req.headers['user-agent']);
      res.setHeader('content-type', 'application/json');
      res.setHeader('cache-control', 'no-store');
      if (c.queue.length) {
        res.end(JSON.stringify(c.queue.shift()));
        return true;
      }
      c.waiter = res;
      const timer = setTimeout(() => {
        if (c.waiter === res) {
          c.waiter = null;
          res.end('null');
        }
      }, pollMs);
      res.on('close', () => {
        clearTimeout(timer);
        if (c.waiter === res) {
          c.waiter = null;
        }
      });
      return true;
    }
    if (url.pathname === '/__remote/result') {
      readBody(req).then((text) => {
        res.end('ok');
        const message = JSON.parse(text);
        const call = this.pending.get(message.id);
        if (!call) {
          return;
        }
        this.pending.delete(message.id);
        if ('error' in message) {
          call.reject(new Error(message.error));
        } else {
          call.resolve(message.value);
        }
      });
      return true;
    }
    if (url.pathname === '/__remote/log') {
      readBody(req).then((text) => {
        res.end('ok');
        const entry = { at: Date.now(), ...JSON.parse(text) };
        this.logs.push(entry);
        this.log(`page ${entry.kind}: ${String(entry.text).slice(0, 400)}`);
      });
      return true;
    }
    return false;
  }

  send(clientId, fn, arg) {
    const c = this.clients.get(clientId);
    if (!c) {
      return Promise.reject(new Error(`no client ${clientId}`));
    }
    const id = ++this.nextId;
    const command = { id, fn: fn.toString(), arg };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, clientId });
      if (c.waiter) {
        const res = c.waiter;
        c.waiter = null;
        res.end(JSON.stringify(command));
      } else {
        c.queue.push(command);
      }
    });
  }

  reap() {
    const now = Date.now();
    for (const [id, c] of this.clients) {
      if (now - c.seen > goneMs) {
        this.clients.delete(id);
        for (const [callId, call] of this.pending) {
          if (call.clientId === id) {
            this.pending.delete(callId);
            call.reject(new Error(`page gone: ${c.page}`));
          }
        }
      }
    }
  }

  async page(prefix, { since = 0, timeoutMs = 60000 } = {}) {
    const end = Date.now() + timeoutMs;
    while (Date.now() < end) {
      const found = [...this.clients.values()]
        .filter((c) => c.first >= since && decodeURIComponent(c.page).startsWith(prefix))
        .sort((a, b) => b.first - a.first)[0];
      if (found) {
        return new RemotePage(this, found);
      }
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    throw new Error(`no page ${prefix} within ${timeoutMs} ms`);
  }

  close() {
    clearInterval(this.watchdog);
    for (const c of this.clients.values()) {
      c.waiter?.end('null');
    }
  }
}

export class RemotePage {
  constructor(remote, client) {
    this.remote = remote;
    this.client = client;
  }

  get ua() {
    return this.client.ua;
  }

  evaluate(fn, arg) {
    return this.remote.send(this.client.id, fn, arg);
  }

  async waitForFunction(fn, arg, timeoutMs = 60000) {
    const end = Date.now() + timeoutMs;
    while (Date.now() < end) {
      if (await this.evaluate(fn, arg)) {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    throw new Error('wait timeout');
  }
}
