import fs from 'node:fs';
const [src, dst] = process.argv.slice(2); const b = fs.readFileSync(src); const v = new DataView(b.buffer, b.byteOffset, b.length);
let pos = 0; const entries = []; let t = 0;
while (pos + 8 <= b.length) { const size = v.getUint32(pos), type = b.toString('latin1', pos + 4, pos + 8);
  if (type === 'moof') { entries.push({ time: t, offset: pos });
    let p = pos + 8 + 16 + 8; const traf_end = pos + size; while (p < traf_end) { const s = v.getUint32(p), ty = b.toString('latin1', p + 4, p + 8); if (ty === 'trun') { t += v.getUint32(p + 12) * 1024; break; } p += s; } }
  pos += size; }
const tfra = Buffer.alloc(8 + 4 + 4 + 4 + 4 + entries.length * 19); let o = 0;
tfra.writeUInt32BE(tfra.length, o); o += 4; tfra.write('tfra', o); o += 4; tfra.writeUInt32BE(0x01000000, o); o += 4; tfra.writeUInt32BE(1, o); o += 4; tfra.writeUInt32BE(0, o); o += 4; tfra.writeUInt32BE(entries.length, o); o += 4;
for (const e of entries) { tfra.writeBigUInt64BE(BigInt(e.time), o); o += 8; tfra.writeBigUInt64BE(BigInt(e.offset), o); o += 8; tfra[o++] = 1; tfra[o++] = 1; tfra[o++] = 1; }
const mfroSize = 16; const mfraSize = 8 + tfra.length + mfroSize; const mfro = Buffer.alloc(16); mfro.writeUInt32BE(16, 0); mfro.write('mfro', 4); mfro.writeUInt32BE(0, 8); mfro.writeUInt32BE(mfraSize, 12);
const head = Buffer.alloc(8); head.writeUInt32BE(mfraSize, 0); head.write('mfra', 4);
fs.writeFileSync(dst, Buffer.concat([b, head, tfra, mfro])); console.log('entries', entries.length, 'mfra bytes', mfraSize, 'lastTime', entries.at(-1).time);
