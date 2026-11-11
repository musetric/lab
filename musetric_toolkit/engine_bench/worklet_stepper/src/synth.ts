export const sampleRate = 48000;
const rng = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const hz = (midi: number) => 440 * 2 ** ((midi - 69) / 12);
type Bq = { b0: number; b1: number; b2: number; a1: number; a2: number; x1: number; x2: number; y1: number; y2: number };
const bandpass = (f: number, q: number): Bq => {
  const w = (2 * Math.PI * f) / sampleRate;
  const al = Math.sin(w) / (2 * q);
  const a0 = 1 + al;
  return { b0: al / a0, b1: 0, b2: -al / a0, a1: (-2 * Math.cos(w)) / a0, a2: (1 - al) / a0, x1: 0, x2: 0, y1: 0, y2: 0 };
};
const lowpass = (f: number, q: number): Bq => {
  const w = (2 * Math.PI * f) / sampleRate;
  const c = Math.cos(w);
  const al = Math.sin(w) / (2 * q);
  const a0 = 1 + al;
  return { b0: (1 - c) / 2 / a0, b1: (1 - c) / a0, b2: (1 - c) / 2 / a0, a1: (-2 * c) / a0, a2: (1 - al) / a0, x1: 0, x2: 0, y1: 0, y2: 0 };
};
const run = (f: Bq, x: number) => {
  const y = f.b0 * x + f.b1 * f.x1 + f.b2 * f.x2 - f.a1 * f.y1 - f.a2 * f.y2;
  f.x2 = f.x1;
  f.x1 = x;
  f.y2 = f.y1;
  f.y1 = y;
  return y;
};
const saw = (p: number) => 2 * (p - Math.floor(p + 0.5));

const beat = 0.5;
const bars = 8;
const tileSeconds = bars * 4 * beat;
const chords = [[57, 60, 64], [53, 57, 60], [60, 64, 67], [55, 59, 62], [57, 60, 64], [53, 57, 60], [52, 56, 59], [52, 56, 59]];

export type Stem = Float32Array[];
const tile = (): Record<'lead' | 'backing' | 'instrumental', Stem> => {
  const n = tileSeconds * sampleRate;
  const mk = (): Stem => [new Float32Array(n), new Float32Array(n)];
  const lead = mk();
  const backing = mk();
  const inst = mk();
  const r = rng(7);
  const scale = [57, 60, 62, 64, 67, 69, 72, 74, 76];
  let idx = 4;
  const f1 = bandpass(700, 4);
  const f2 = bandpass(1300, 5);
  const f3 = bandpass(2600, 6);
  let phase = 0;
  for (let b = 0; b < bars * 4; b++) {
    const step = Math.floor(r() * 3) - 1;
    idx = Math.max(0, Math.min(scale.length - 1, idx + step * (r() < 0.5 ? 1 : 2)));
    const rest = r() < 0.15;
    const f0 = hz(scale[idx]);
    for (let i = 0; i < beat * sampleRate; i++) {
      const t = i / sampleRate;
      const s = b * beat * sampleRate + i;
      const env = rest ? 0 : Math.min(1, t / 0.012) * (t < beat * 0.9 ? 1 : Math.max(0, 1 - (t - beat * 0.9) / 0.05));
      phase += (f0 * (1 + 0.006 * Math.sin(2 * Math.PI * 5.2 * (s / sampleRate)))) / sampleRate;
      const g = saw(phase) + 0.2 * (r() - 0.5);
      const v = (run(f1, g) + 0.7 * run(f2, g) + 0.4 * run(f3, g)) * env * 2.2;
      lead[0][s] = v * 0.9;
      lead[1][s] = v;
    }
  }
  for (let b = 0; b < bars; b++) {
    for (let k = 0; k < 3; k++) {
      const f0 = hz(chords[b][k] + 12);
      for (let ch = 0; ch < 2; ch++) {
        let p = r();
        const det = ch === 0 ? 0.997 : 1.003;
        for (let i = 0; i < 4 * beat * sampleRate; i++) {
          const s = b * 4 * beat * sampleRate + i;
          const t = i / sampleRate;
          p += (f0 * det * (1 + 0.003 * Math.sin(2 * Math.PI * 4.1 * (s / sampleRate)))) / sampleRate;
          const env = Math.min(1, t / 0.4) * Math.min(1, (4 * beat - t) / 0.3);
          backing[ch][s] += (Math.abs(saw(p)) * 2 - 1) * 0.12 * env;
        }
      }
    }
  }
  const lp = lowpass(600, 0.8);
  const padL = lowpass(1800, 0.7);
  const padR = lowpass(1800, 0.7);
  const pp = [0, 0, 0, 0, 0, 0];
  for (let b = 0; b < bars; b++) {
    const root = chords[b][0] - 12;
    for (let e = 0; e < 8; e++) {
      const f0 = hz(root + (e % 4 === 3 ? 7 : 0));
      let p = 0;
      for (let i = 0; i < beat * 0.5 * sampleRate; i++) {
        const s = Math.round((b * 4 + e * 0.5) * beat * sampleRate) + i;
        const t = i / sampleRate;
        p += f0 / sampleRate;
        const v = run(lp, saw(p)) * Math.exp(-t * 7) * 0.5;
        inst[0][s] += v;
        inst[1][s] += v;
      }
    }
    for (let i = 0; i < 4 * beat * sampleRate; i++) {
      const s = b * 4 * beat * sampleRate + i;
      let l = 0;
      let rr = 0;
      for (let k = 0; k < 3; k++) {
        pp[k] += (hz(chords[b][k]) * 0.998) / sampleRate;
        pp[k + 3] += (hz(chords[b][k]) * 1.002) / sampleRate;
        l += saw(pp[k]);
        rr += saw(pp[k + 3]);
      }
      inst[0][s] += run(padL, l) * 0.05;
      inst[1][s] += run(padR, rr) * 0.05;
    }
  }
  const hat = bandpass(9000, 2);
  for (let q = 0; q < bars * 4 * 2; q++) {
    const start = Math.round(q * beat * 0.5 * sampleRate);
    const isBeat = q % 2 === 0;
    const beatNo = (q / 2) % 4;
    for (let i = 0; i < 0.25 * sampleRate; i++) {
      const s = start + i;
      const t = i / sampleRate;
      if (s >= n) break;
      let v = 0;
      if (isBeat && (beatNo === 0 || beatNo === 2)) {
        v += Math.sin(2 * Math.PI * (50 * t + (90 * (1 - Math.exp(-t * 30))) / 30)) * Math.exp(-t * 18) * 0.7;
      }
      if (isBeat && (beatNo === 1 || beatNo === 3)) v += (r() - 0.5) * Math.exp(-t * 22) * 0.5;
      if (i < 0.05 * sampleRate) v += run(hat, r() - 0.5) * Math.exp(-t * 80) * 0.9;
      inst[0][s] += v;
      inst[1][s] += v;
    }
  }
  return { lead, backing, instrumental: inst };
};

export type Stems = Record<'lead' | 'backing' | 'instrumental', Float32Array<ArrayBuffer>[]>;

export const makeStems = (seconds: number): Stems => {
  const base = tile();
  let peak = 0;
  for (let i = 0; i < base.lead[0].length; i++) {
    for (let ch = 0; ch < 2; ch++) {
      peak = Math.max(peak, Math.abs(base.lead[ch][i] + base.backing[ch][i] + base.instrumental[ch][i]));
    }
  }
  const gain = 0.8 / peak;
  const frames = Math.round(seconds * sampleRate);
  const len = base.lead[0].length;
  const out: Record<string, Float32Array<ArrayBuffer>[]> = {};
  for (const [name, stem] of Object.entries(base)) {
    out[name] = stem.map((src) => {
      const dst = new Float32Array(frames);
      for (let o = 0; o < frames; o += len) dst.set(src.subarray(0, Math.min(len, frames - o)), o);
      for (let i = 0; i < frames; i++) dst[i] *= gain;
      return dst;
    });
  }
  return out as Stems;
};

export const makeVoiceTake = (seconds: number, seed: number) => {
  const r = rng(seed);
  const n = Math.round(seconds * sampleRate);
  const out = new Float32Array(n);
  const f1 = bandpass(600, 4);
  const f2 = bandpass(1100, 5);
  let ph = 0;
  let f0 = 220;
  for (let i = 0; i < n; i++) {
    if (i % 12000 === 0) f0 = hz(57 + Math.floor(r() * 10));
    ph += (f0 * (1 + 0.005 * Math.sin((i / sampleRate) * 31))) / sampleRate;
    out[i] = (run(f1, saw(ph)) + 0.6 * run(f2, saw(ph))) * 0.9;
  }
  return out;
};
