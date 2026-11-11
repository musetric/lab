const u32 = (v, o) => v.getUint32(o), u64 = (v, o) => v.getUint32(o) * 2 ** 32 + v.getUint32(o + 4);
export const parseMoof = (v, viewStart, base, moofSize) => {
  const packets = []; let pos = base - viewStart + 8; const end = base - viewStart + moofSize; let baseTime = 0;
  while (pos < end) {
    const size = u32(v, pos), type = u32(v, pos + 4);
    if (type === 0x74726166) {
      let p = pos + 8; const e = pos + size; let dataOffsetBase = base;
      while (p < e) {
        const s = u32(v, p), t = u32(v, p + 4);
        if (t === 0x74666474) { const ver = v.getUint8(p + 8); baseTime = ver ? u64(v, p + 12) : u32(v, p + 12); }
        if (t === 0x7472756e) {
          const flags = u32(v, p + 8) & 0xffffff, count = u32(v, p + 12); let q = p + 16; let dataOffset = 0;
          if (flags & 1) { dataOffset = v.getInt32(q); q += 4; } if (flags & 4) q += 4;
          let off = dataOffsetBase + dataOffset;
          for (let i = 0; i < count; i++) {
            let dur = 0, sz = 0; if (flags & 0x100) { dur = u32(v, q); q += 4; } if (flags & 0x200) { sz = u32(v, q); q += 4; } if (flags & 0x400) q += 4; if (flags & 0x800) q += 4;
            packets.push({ offset: off, size: sz, dur }); off += sz;
          }
        }
        p += s;
      }
    }
    pos += size;
  }
  return { packets, baseTime };
};
export const indexFromBuffer = (buf) => {
  const v = new DataView(buf); const out = []; let pos = 0, fragments = 0;
  while (pos + 8 <= buf.byteLength) {
    const size = u32(v, pos), type = u32(v, pos + 4);
    if (type === 0x6d6f6f66) { out.push(...parseMoof(v, 0, pos, size).packets); fragments++; }
    if (size < 8) break; pos += size;
  }
  return { packets: out, fragments };
};
export const indexFromRanges = async (fetchRange, fileSize, READ = 4096) => {
  let pos = 0, requests = 0, bytes = 0, fragments = 0; const out = [];
  while (pos + 8 <= fileSize) {
    const ab = await fetchRange(pos, Math.min(fileSize, pos + READ)); requests++; bytes += ab.byteLength;
    const v = new DataView(ab); let p = 0, jumped = false, need = 0;
    while (p + 8 <= ab.byteLength) {
      const size = u32(v, p), type = u32(v, p + 4);
      if (type === 0x6d646174) { pos += p + size; jumped = true; break; }
      if (p + size > ab.byteLength) { need = size; break; }
      if (type === 0x6d6f6f66) { out.push(...parseMoof(v, pos, pos + p, size).packets); fragments++; }
      p += size;
    }
    if (!jumped) { pos += p; if (need > READ) READ = need + 16; if (p === 0 && !need) break; }
  }
  return { packets: out, fragments, requests, bytes };
};
