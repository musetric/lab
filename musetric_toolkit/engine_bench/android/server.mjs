import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

export const here = import.meta.dirname;
const stretchModule = path.join(here, 'node_modules/signalsmith-stretch/SignalsmithStretch.mjs');
const types = {
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.html': 'text/html',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.mp4': 'video/mp4',
  '.flac': 'audio/flac',
};

export const startServer = ({ roots = [here], routes = {}, isolated = false, handle, inject } = {}) =>
  new Promise((resolve) => {
    const stats = { requests: 0, ranged: 0, bytes: 0, sent: 0 };
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, 'http://127.0.0.1');
      if (handle?.(req, res, url)) {
        return;
      }
      if (url.pathname === '/__stats') {
        res.setHeader('content-type', 'application/json');
        res.setHeader('cache-control', 'no-store');
        return res.end(JSON.stringify(stats));
      }
      if (url.pathname === '/__reset') {
        stats.requests = stats.ranged = stats.bytes = stats.sent = 0;
        return res.end('ok');
      }
      const pathname = decodeURIComponent(url.pathname);
      const file =
        pathname === '/vendor/SignalsmithStretch.mjs'
          ? stretchModule
          : (routes[pathname] ??
            roots.map((root) => path.join(root, pathname)).find((f) => fs.existsSync(f) && fs.statSync(f).isFile()));
      if (pathname.includes('..') || !file || !fs.existsSync(file)) {
        res.statusCode = 404;
        return res.end('nf');
      }
      const size = fs.statSync(file).size;
      const headers = {
        'content-type': types[path.extname(file)] ?? 'application/octet-stream',
        'accept-ranges': 'bytes',
        'cache-control': 'no-store',
        ...(isolated && {
          'cross-origin-opener-policy': 'same-origin',
          'cross-origin-embedder-policy': 'require-corp',
          'cross-origin-resource-policy': 'same-origin',
        }),
      };
      stats.requests++;
      const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
      if (inject && !range && path.extname(file) === '.html') {
        const body = inject(fs.readFileSync(file, 'utf8'));
        stats.bytes += Buffer.byteLength(body);
        stats.sent += Buffer.byteLength(body);
        res.writeHead(200, { ...headers, 'content-length': Buffer.byteLength(body) });
        return res.end(body);
      }
      if (range) {
        const start = Number(range[1]);
        const end = range[2] === '' ? size - 1 : Math.min(size - 1, Number(range[2]));
        stats.ranged++;
        stats.bytes += end - start + 1;
        res.writeHead(206, {
          ...headers,
          'content-range': `bytes ${start}-${end}/${size}`,
          'content-length': end - start + 1,
        });
        const stream = fs.createReadStream(file, { start, end });
        stream.on('data', (chunk) => {
          stats.sent += chunk.length;
        });
        stream.pipe(res);
      } else {
        stats.bytes += size;
        res.writeHead(200, { ...headers, 'content-length': size });
        const stream = fs.createReadStream(file);
        stream.on('data', (chunk) => {
          stats.sent += chunk.length;
        });
        stream.pipe(res);
      }
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
