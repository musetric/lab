import { createServer } from 'node:https';
import { readFileSync, appendFileSync, statSync, createReadStream, existsSync } from 'node:fs';
import { join, extname, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(process.argv[2] ?? join(here, 'www'));
const port = Number(process.argv[3] ?? 8443);
const log = join(here, 'reports.jsonl');
const certDir = process.env.CERT_DIR ?? join(here, 'certs');
if (!existsSync(join(certDir, 'key.pem')) || !existsSync(join(certDir, 'cert.pem'))) {
  throw new Error('key.pem and cert.pem are missing in ' + certDir + ': see readme.md');
}
const aliases = {
  '/mediabunny.mjs': 'node_modules/mediabunny/dist/bundles/mediabunny.min.mjs',
  '/vendor/SignalsmithStretch.mjs': 'node_modules/signalsmith-stretch/SignalsmithStretch.mjs',
};
const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.wasm': 'application/wasm', '.mp4': 'audio/mp4',
  '.flac': 'audio/flac', '.bin': 'application/octet-stream', '.css': 'text/css',
};

const server = createServer({
  key: readFileSync(join(certDir, 'key.pem')),
  cert: readFileSync(join(certDir, 'cert.pem')),
}, (req, res) => {
  const url = new URL(req.url, `https://${req.headers.host}`);
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method === 'POST' && url.pathname === '/report') {
    let body = '';
    req.on('data', (c) => { body += c; });
    req.on('end', () => {
      appendFileSync(log, JSON.stringify({ at: new Date().toISOString(), ua: req.headers['user-agent'], body }) + '\n');
      res.end('ok');
    });
    return;
  }
  const path = join(root, aliases[url.pathname] ?? decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
  if (!path.startsWith(root) || !existsSync(path) || statSync(path).isDirectory()) {
    res.statusCode = 404; res.end('not found'); return;
  }
  const size = statSync(path).size;
  res.setHeader('Content-Type', types[extname(path)] ?? 'application/octet-stream');
  res.setHeader('Accept-Ranges', 'bytes');
  const range = req.headers.range && /bytes=(\d+)-(\d*)/.exec(req.headers.range);
  if (range) {
    const start = Number(range[1]);
    const end = range[2] ? Number(range[2]) : size - 1;
    res.statusCode = 206;
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
    res.setHeader('Content-Length', end - start + 1);
    createReadStream(path, { start, end }).pipe(res);
    return;
  }
  res.setHeader('Content-Length', size);
  createReadStream(path).pipe(res);
});
server.listen(port, '0.0.0.0', () => console.log(`ready https://0.0.0.0:${port} root=${root}`));
