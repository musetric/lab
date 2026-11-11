import fs from 'node:fs'; import crypto from 'node:crypto';
const SR = 48000;
let s = 0x12345678;
const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
const gauss = () => { let a = 0; for (let i = 0; i < 6; i++) a += rnd(); return (a - 3) / 0.7071; };
const writeWav = (name, L, R) => {
  const n = L.length, buf = Buffer.alloc(44 + n * 6);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 6, 4); buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(2, 22);
  buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * 6, 28); buf.writeUInt16LE(6, 32); buf.writeUInt16LE(24, 34);
  buf.write('data', 36); buf.writeUInt32LE(n * 6, 40);
  let o = 44;
  const q = (x) => Math.max(-8388608, Math.min(8388607, Math.round(x * 8388608)));
  for (let i = 0; i < n; i++) for (const ch of [L, R]) { buf.writeIntLE(q(ch[i]), o, 3); o += 3; }
  fs.writeFileSync(name, buf);
  return crypto.createHash('sha256').update(buf).digest('hex');
};
const chord = (secs) => {
  const n = secs * SR, L = new Float32Array(n), R = new Float32Array(n);
  const f = [220, 261.63, 329.63, 392, 110];
  for (let i = 0; i < n; i++) { const t = i / SR; let l = 0, r = 0;
    f.forEach((fr, k) => { l += Math.sin(2 * Math.PI * fr * t + k) / f.length; r += Math.sin(2 * Math.PI * fr * 1.002 * t + 2 * k) / f.length; });
    const env = 0.35 + 0.15 * Math.sin(2 * Math.PI * 0.3 * t); L[i] = l * env; R[i] = r * env; }
  return [L, R];
};
const noise = (secs) => {
  const n = secs * SR, L = new Float32Array(n), R = new Float32Array(n);
  const pink = () => { let b = [0,0,0,0,0,0,0]; return () => { const w = rnd()*2-1;
    b[0]=0.99886*b[0]+w*0.0555179; b[1]=0.99332*b[1]+w*0.0750759; b[2]=0.969*b[2]+w*0.153852; b[3]=0.8665*b[3]+w*0.3104856; b[4]=0.55*b[4]+w*0.5329522; b[5]=-0.7616*b[5]-w*0.016898;
    const p=(b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6]+w*0.5362)*0.11; b[6]=w*0.115926; return p; }; };
  const pl = pink(), pr = pink();
  for (let i = 0; i < n; i++) { const t = i / SR, seg = Math.floor(t * 2); if (seg % 2) continue;
    const kind = Math.floor(seg / 2) % 2; const a = 0.4 * Math.min(1, (t * 2 - seg) * 20);
    L[i] = a * (kind ? pl() : (rnd()*2-1)); R[i] = a * (kind ? pr() : (rnd()*2-1)); }
  return [L, R];
};
const clicks = (secs) => {
  const n = secs * SR, L = new Float32Array(n), R = new Float32Array(n);
  let pos = 1000; while (pos < n - 5000) { const amp = 0.3 + 0.6 * rnd(), len = 1 + Math.floor(rnd() * 200), sign = rnd() < 0.5 ? -1 : 1;
    for (let k = 0; k < len; k++) { const d = Math.exp(-k / (len / 3 + 1)) * amp * sign * (k % 2 ? -1 : 1); L[pos + k] += d; R[pos + k] += d * (rnd() < 0.3 ? -1 : 1); }
    pos += 500 + Math.floor(rnd() * 30000); }
  return [L, R];
};
const sweep = (secs) => {
  const n = secs * SR, L = new Float32Array(n), R = new Float32Array(n), T = 30, f0 = 20, f1 = 20000, k = Math.log(f1 / f0);
  for (let i = 0; i < n; i++) { const t = (i / SR) % T; const ph = 2 * Math.PI * f0 * T / k * (Math.exp(k * t / T) - 1);
    L[i] = 0.5 * Math.sin(ph); const t2 = T - t; R[i] = 0.5 * Math.sin(2 * Math.PI * f0 * T / k * (Math.exp(k * t2 / T) - 1)); }
  return [L, R];
};
const mix = (secs) => { const a = chord(secs), b = noise(secs), c = clicks(secs), d = sweep(secs);
  const n = secs*SR, L = new Float32Array(n), R = new Float32Array(n);
  for (let i = 0; i < n; i++) { L[i] = 0.5*a[0][i] + 0.4*b[0][i] + 0.4*c[0][i] + 0.2*d[0][i]; R[i] = 0.5*a[1][i] + 0.4*b[1][i] + 0.4*c[1][i] + 0.2*d[1][i]; }
  return [L, R]; };
const mixN = (secs, parts) => { const n = secs * SR, L = new Float32Array(n), R = new Float32Array(n);
  for (const [p, g] of parts) for (let i = 0; i < n; i++) { L[i] += g * p[0][i]; R[i] += g * p[1][i]; } return [L, R]; };
fs.mkdirSync('data', { recursive: true });
const defs = { stemA: () => mixN(240, [[chord(240), 0.7], [sweep(240), 0.3]]), stemB: () => mixN(240, [[noise(240), 0.6], [clicks(240), 0.5]]), stemC: () => mix(240) };
const out = {};
for (const [k, fn] of Object.entries(defs)) { s = 0x12345678; const [L, R] = fn(); out[k] = writeWav(`data/${k}.wav`, L, R); console.log(k, out[k]); }
