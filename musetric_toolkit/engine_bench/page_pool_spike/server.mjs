import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
const root = path.resolve(import.meta.dirname);
const types = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.mp4': 'video/mp4' };
export const startServer = () => new Promise((resolve) => {
  const stats = { requests: 0, ranged: 0, bytes: 0 };
  const server = http.createServer((req, res) => {
    const u = new URL(req.url, 'http://127.0.0.1');
    if (u.pathname === '/__stats') { res.setHeader('content-type', 'application/json'); return res.end(JSON.stringify(stats)); }
    const f = u.pathname === '/vendor/SignalsmithStretch.mjs' ? path.join(root, 'node_modules/signalsmith-stretch/SignalsmithStretch.mjs') : path.join(root, decodeURIComponent(u.pathname));
    if (!f.startsWith(root) || u.pathname.includes('..') || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.statusCode = 404; return res.end('nf'); }
    const size = fs.statSync(f).size; const h = { 'content-type': types[path.extname(f)] ?? 'application/octet-stream', 'accept-ranges': 'bytes', 'cache-control': 'no-store' };
    stats.requests++;
    const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
    if (m) { const s = Number(m[1]); const e = m[2] === '' ? size - 1 : Math.min(size - 1, Number(m[2])); stats.ranged++; stats.bytes += e - s + 1;
      res.writeHead(206, { ...h, 'content-range': `bytes ${s}-${e}/${size}`, 'content-length': e - s + 1 }); fs.createReadStream(f, { start: s, end: e }).pipe(res); }
    else { stats.bytes += size; res.writeHead(200, { ...h, 'content-length': size }); fs.createReadStream(f).pipe(res); }
  });
  server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
});
