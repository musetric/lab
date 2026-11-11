import { readFileSync } from 'node:fs';
import { SourceMap } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const id = process.argv[2];
const top = Number(process.argv[3] ?? 25);
const { quanta, profile } = JSON.parse(readFileSync(path.join(here, 'results', `profile_${id}.json`), 'utf8'));
const maps = new Map();
const mapFor = (url) => {
  if (!maps.has(url)) {
    let m;
    try {
      const file = url.startsWith('file:') ? fileURLToPath(url) : url;
      m = new SourceMap(JSON.parse(readFileSync(`${file}.map`, 'utf8')));
    } catch {
      m = undefined;
    }
    maps.set(url, m);
  }
  return maps.get(url);
};
const sites = new Map();
let total = 0;
const walk = (node, stack) => {
  const cf = node.callFrame;
  const here = { fn: cf.functionName || '(anon)', url: cf.url, line: cf.lineNumber, col: cf.columnNumber };
  const next = [...stack, here];
  if (node.selfSize > 0) {
    total += node.selfSize;
    let label;
    for (let i = next.length - 1; i >= 0; i--) {
      const f = next[i];
      const sm = f.url ? mapFor(f.url) : undefined;
      if (!sm) continue;
      const e = sm.findEntry(f.line, f.col);
      if (e?.originalSource) {
        const src = e.originalSource.replace(/^.*\/(packages|src)\//, '$1/');
        label = `${src}:${e.originalLine + 1}  ${f.fn}`;
        break;
      }
    }
    label ??= `${cf.url || '(native)'}:${cf.lineNumber + 1}  ${here.fn}`;
    const chain = next
      .map((f) => f.fn)
      .filter((n) => n && n !== '(root)' && !n.startsWith('('))
      .slice(-4)
      .join(' < ');
    const key = label;
    const s = sites.get(key) ?? { bytes: 0, chains: new Map() };
    s.bytes += node.selfSize;
    s.chains.set(chain, (s.chains.get(chain) ?? 0) + node.selfSize);
    sites.set(key, s);
  }
  for (const c of node.children) walk(c, next);
};
walk(profile.head, []);
const rows = [...sites].sort((a, b) => b[1].bytes - a[1].bytes).slice(0, top);
console.log(`${id}: ${quanta} quanta, sampled total ${(total / quanta).toFixed(0)} B/quantum`);
for (const [label, s] of rows) {
  const chain = [...s.chains].sort((a, b) => b[1] - a[1])[0][0];
  console.log(`${(s.bytes / quanta).toFixed(1).padStart(10)} B/q  ${label}   [${chain}]`);
}
