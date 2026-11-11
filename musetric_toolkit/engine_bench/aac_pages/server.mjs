import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
const root = path.resolve(import.meta.dirname);
const types = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.mp4': 'video/mp4', '.flac': 'audio/flac', '.wav': 'audio/wav' };
export const startServer = () => new Promise((resolve) => {
  const stats = { requests: 0, ranged: 0, bytes: 0, sent: 0, log: [] };
  const server = http.createServer((req, res) => {
    const u = new URL(req.url, 'http://127.0.0.1');
    if (u.pathname === '/__stats') { res.setHeader('content-type', 'application/json'); res.setHeader('cache-control', 'no-store'); return res.end(JSON.stringify(stats)); }
    if (u.pathname === '/__reset') { stats.requests = stats.ranged = stats.bytes = stats.sent = 0; stats.log = []; return res.end('ok'); }
    let f;
    if (u.pathname === '/mediabunny.mjs') f = path.join(root, 'node_modules/mediabunny/dist/bundles/mediabunny.min.mjs');
    else f = path.join(root, decodeURIComponent(u.pathname));
    if (!f.startsWith(root) || u.pathname.includes('..') || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.statusCode = 404; return res.end('nf'); }
    const size = fs.statSync(f).size; const h = { 'content-type': types[path.extname(f)] ?? 'application/octet-stream', 'accept-ranges': 'bytes',
      'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'require-corp', 'cross-origin-resource-policy': 'same-origin', 'cache-control': 'no-store' };
    stats.requests++;
    const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '');
    if (m) { let s = m[1] === '' ? size - Number(m[2]) : Number(m[1]); let e = m[1] === '' || m[2] === '' ? size - 1 : Math.min(size - 1, Number(m[2]));
      if (s > e || s >= size) { res.writeHead(416, { 'content-range': `bytes */${size}` }); return res.end(); }
      stats.ranged++; stats.bytes += e - s + 1; if (stats.log.length < 40) stats.log.push([s, e]);
      res.writeHead(206, { ...h, 'content-range': `bytes ${s}-${e}/${size}`, 'content-length': e - s + 1 }); const rs = fs.createReadStream(f, { start: s, end: e }); rs.on('data', (c) => { stats.sent += c.length; }); rs.pipe(res); }
    else { stats.bytes += size; if (stats.log.length < 40) stats.log.push(['full', size]); res.writeHead(200, { ...h, 'content-length': size }); const rs = fs.createReadStream(f); rs.on('data', (c) => { stats.sent += c.length; }); rs.pipe(res); }
  });
  server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
});
