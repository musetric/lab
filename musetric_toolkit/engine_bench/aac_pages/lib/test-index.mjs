import fs from 'node:fs';
import { indexFromBuffer, indexFromRanges } from '../web/mp4index.js';
import { runControls } from '../web/compare.js';
for (const k of ['chord','noise','clicks','sweep','mix240']) {
  const b = fs.readFileSync(`data/${k}.mp4`); const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
  const a = indexFromBuffer(ab); const r = await indexFromRanges(async (s, e) => ab.slice(s, e), ab.byteLength);
  const same = a.packets.length === r.packets.length && a.packets.every((p, i) => p.offset === r.packets[i].offset && p.size === r.packets[i].size);
  const sizes = a.packets.map(p=>p.size); const sum=sizes.reduce((x,y)=>x+y,0);
  console.log(k, 'packets', a.packets.length, 'fragments', a.fragments, 'ranged==buffer', same, 'req', r.requests, 'bytes', r.bytes, 'min/mean/max size', Math.min(...sizes), (sum/sizes.length).toFixed(0), Math.max(...sizes), 'covers', sum, 'all dur1024', a.packets.every(p=>p.dur===1024));
}
console.log(runControls().map(c=>`${c.pass?'PASS':'FAIL'} ${c.name} ${c.reasons.join('; ')}`).join('\n'));
