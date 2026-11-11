import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHarness, quantumFrames } from './harness.js';
import { makeStems, sampleRate } from './synth.js';

type Stereo = Float32Array[];
const wraps = 8;

const loadMusic = async (file: string, startSec: number, seconds: number): Promise<Stereo> => {
  const { MPEGDecoder } = await import('mpg123-decoder');
  const decoder = new MPEGDecoder();
  await decoder.ready;
  const res = decoder.decode(new Uint8Array(readFileSync(file)));
  const [l, r = l] = res.channelData;
  const srcRate = res.sampleRate;
  const from = Math.floor(startSec * srcRate);
  const n = Math.floor(seconds * srcRate);
  const out = [resample(l.subarray(from, from + n), srcRate), resample(r.subarray(from, from + n), srcRate)];
  decoder.free();
  return out;
};
const resample = (x: Float32Array, srcRate: number): Float32Array<ArrayBuffer> => {
  if (srcRate === sampleRate) return Float32Array.from(x);
  const ratio = srcRate / sampleRate;
  const outLen = Math.floor(x.length / ratio);
  const out = new Float32Array(outLen);
  const zc = 16;
  const cutoff = Math.min(1, sampleRate / srcRate);
  for (let i = 0; i < outLen; i += 1) {
    const pos = i * ratio;
    const c = Math.floor(pos);
    let acc = 0;
    for (let k = -zc + 1; k <= zc; k += 1) {
      const idx = c + k;
      if (idx < 0 || idx >= x.length) continue;
      const d = pos - idx;
      const a = Math.PI * d * cutoff;
      const sinc = d === 0 ? 1 : Math.sin(a) / a;
      const w = 0.5 * (1 + Math.cos((Math.PI * d) / zc));
      acc += x[idx] * sinc * w * cutoff;
    }
    out[i] = acc;
  }
  return out;
};

const estimateBeat = (x: Stereo) => {
  const hop = Math.round(sampleRate * 0.01);
  const frames = Math.floor(x[0].length / hop);
  const env = new Float64Array(frames);
  for (let f = 0; f < frames; f += 1) {
    let s = 0;
    for (let i = f * hop; i < (f + 1) * hop; i += 1) s += x[0][i] ** 2 + x[1][i] ** 2;
    env[f] = Math.sqrt(s / hop);
  }
  const onset = new Float64Array(frames);
  for (let f = 1; f < frames; f += 1) onset[f] = Math.max(0, env[f] - env[f - 1]);
  let best = { lag: 60, score: -1 };
  for (let lag = 30; lag <= 100; lag += 1) {
    let s = 0;
    for (let f = lag; f < frames; f += 1) s += onset[f] * onset[f - lag];
    s /= frames - lag;
    if (s > best.score) best = { lag, score: s };
  }
  let bestFrac = { lag: best.lag, score: -1 };
  for (let lag = best.lag - 1; lag <= best.lag + 1; lag += 0.05) {
    let s = 0;
    let n = 0;
    for (let f = 0; f < frames; f += 1) {
      const g = f - lag;
      if (g < 0) continue;
      const g0 = Math.floor(g);
      const w = g - g0;
      s += onset[f] * (onset[g0] * (1 - w) + onset[g0 + 1] * w);
      n += 1;
    }
    s /= n;
    if (s > bestFrac.score) bestFrac = { lag, score: s };
  }
  const beat = (bestFrac.lag * hop) / sampleRate;
  const phaseScore = new Float64Array(Math.round(beat / 0.01));
  for (let f = 0; f < frames; f += 1) phaseScore[Math.floor(((f * 0.01) % beat) / 0.01)] += onset[f];
  let phase = 0;
  for (let i = 1; i < phaseScore.length; i += 1) if (phaseScore[i] > phaseScore[phase]) phase = i;
  return { beatSec: beat, bpm: 60 / beat, phaseSec: phase * 0.01 };
};

const renderRuntime = async (x: Stereo, fromFrame: number, lenFrames: number, outFrames: number): Promise<Stereo> => {
  const h = await createHarness(sampleRate);
  const frames = x[0].length;
  const empty: Float32Array<ArrayBuffer>[] = [];
  h.mainData.mount({
    frameCount: frames,
    tracks: { lead: empty, backing: empty, instrumental: x.map((c) => c.slice()) as Float32Array<ArrayBuffer>[] },
    recording: [],
  });
  h.main.setOutputClock({ clockOffset: 0 });
  h.main.setRange({ range: { fromFrame, toFrame: fromFrame + lenFrames }, repeat: 'cycle' });
  h.main.seek({ frameIndex: fromFrame, revision: 1 });
  h.main.play({ revision: 2, latencyFrameCount: 0, inputLatencyFrameCount: 0 });
  await h.pump();
  const total = Math.ceil(outFrames / quantumFrames) * quantumFrames;
  const out: Stereo = [new Float32Array(total), new Float32Array(total)];
  for (let q = 0; q < total / quantumFrames; q += 1) {
    h.step(1);
    out[0].set(h.outputs[0], q * quantumFrames);
    out[1].set(h.outputs[1], q * quantumFrames);
  }
  return [out[0].subarray(0, outFrames), out[1].subarray(0, outFrames)];
};

type Variant = 'hard' | 'product256' | 'xfade';
const renderOffline = (x: Stereo, from: number, len: number, passes: number, variant: Variant, fadeFrames: number): Stereo => {
  const out: Stereo = [new Float32Array(len * passes), new Float32Array(len * passes)];
  for (let ch = 0; ch < 2; ch += 1) {
    for (let p = 0; p < passes; p += 1) {
      for (let i = 0; i < len; i += 1) {
        let v = x[ch][from + i];
        if (variant === 'product256') {
          const ramp = (d: number) => (d < 256 ? Math.max(0, d) / 256 : 1);
          v *= Math.min(ramp(i), ramp(len - i));
        } else if (variant === 'xfade' && i >= len - fadeFrames) {
          const j = i - (len - fadeFrames);
          const t = (j + 0.5) / fadeFrames;
          v = v * Math.cos((t * Math.PI) / 2) + x[ch][from - fadeFrames + j] * Math.sin((t * Math.PI) / 2);
        }
        out[ch][p * len + i] = v;
      }
    }
  }
  return out;
};

const rmsDb = (a: Stereo, centre: number, win: number) => {
  let s = 0;
  for (let i = centre - win / 2; i < centre + win / 2; i += 1) s += a[0][i] ** 2 + a[1][i] ** 2;
  return 10 * Math.log10(s / (2 * win) + 1e-20);
};
const dips = (variant: Stereo, hard: Stereo, len: number) => {
  const win = Math.round(0.005 * sampleRate);
  const hop = Math.round(0.001 * sampleRate);
  const span = Math.round(0.1 * sampleRate);
  const perWrap: { minDb: number; maxDb: number; atMs: number; msBelow3: number; maxStepRatio: number }[] = [];
  const steps: number[] = [];
  for (let i = 1; i < hard[0].length; i += 7) steps.push(Math.abs(hard[0][i] - hard[0][i - 1]));
  steps.sort((p, q) => p - q);
  const typical = steps[Math.floor(steps.length * 0.99)];
  for (let k = 1; k <= wraps; k += 1) {
    const w = k * len;
    let minDb = Infinity;
    let maxDb = -Infinity;
    let at = 0;
    let below = 0;
    for (let c = w - span; c <= w + span; c += hop) {
      const d = rmsDb(variant, c, win) - rmsDb(hard, c, win);
      if (d < minDb) {
        minDb = d;
        at = c - w;
      }
      if (d > maxDb) maxDb = d;
      if (d < -3) below += 1;
    }
    let maxStep = 0;
    for (let i = w - 240; i < w + 240; i += 1) maxStep = Math.max(maxStep, Math.abs(variant[0][i] - variant[0][i - 1]));
    perWrap.push({ minDb, maxDb, atMs: (at / sampleRate) * 1000, msBelow3: (below * hop * 1000) / sampleRate, maxStepRatio: maxStep / typical });
  }
  return perWrap;
};

const writeWav = (file: string, a: Stereo, scale: number) => {
  const n = a[0].length;
  const buf = Buffer.alloc(44 + n * 4);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * 4, 4);
  buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(2, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 4, 28);
  buf.writeUInt16LE(4, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i += 1) {
    for (let ch = 0; ch < 2; ch += 1) {
      buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(a[ch][i] * scale * 32767))), 44 + i * 4 + ch * 2);
    }
  }
  writeFileSync(file, buf);
};
const concat = (parts: Stereo[], gapSec: number): Stereo => {
  const gap = Math.round(gapSec * sampleRate);
  const total = parts.reduce((s, p) => s + p[0].length + gap, 0);
  const out: Stereo = [new Float32Array(total), new Float32Array(total)];
  let o = 0;
  for (const p of parts) {
    out[0].set(p[0], o);
    out[1].set(p[1], o);
    o += p[0].length + gap;
  }
  return out;
};

type Result = Record<string, unknown>;
const processSignal = async (name: string, x: Stereo, fromSec: number, lenSec: number, notes: Result) => {
  const from = Math.round(fromSec * sampleRate);
  const len = Math.round(lenSec * sampleRate);
  const passes = wraps + 1;
  const real = await renderRuntime(x, from, len, passes * len);
  const hard = renderOffline(x, from, len, passes, 'hard', 0);
  const prod = renderOffline(x, from, len, passes, 'product256', 0);
  let maxDiff = 0;
  for (let ch = 0; ch < 2; ch += 1) for (let i = 0; i < prod[0].length; i += 1) maxDiff = Math.max(maxDiff, Math.abs(real[ch][i] - prod[ch][i]));
  const xf31 = renderOffline(x, from, len, passes, 'xfade', Math.round(0.031 * sampleRate));
  const xf5 = renderOffline(x, from, len, passes, 'xfade', Math.round(0.005 * sampleRate));
  const variants: Record<string, Stereo> = { a_product256_runtime: real, b_xfade31ms: xf31, c_xfade5ms: xf5, ref_hardcut: hard };
  let peak = 0;
  for (const v of Object.values(variants)) for (const ch of v) for (const s of ch) peak = Math.max(peak, Math.abs(s));
  const scale = 0.9 / peak;
  mkdirSync('wav', { recursive: true });
  const files: Record<string, string> = {};
  for (const [k, v] of Object.entries(variants)) {
    files[k] = `wav/${name}_${k}.wav`;
    writeWav(files[k], v, scale);
  }
  files.ALL = `wav/${name}_ALL_a-b-c_1s_gaps.wav`;
  writeWav(files.ALL, concat([real, xf31, xf5], 1), scale);
  const metrics: Result = {};
  for (const [k, v] of Object.entries(variants)) if (k !== 'ref_hardcut') metrics[k] = dips(v, hard, len);
  metrics.hardcut_maxStep = dips(hard, hard, len).map((d) => d.maxStepRatio);
  notes[name] = { fromSec, lenSec, wraps, runtimeVsOfflineMaxAbsDiff: maxDiff, scale, files, metrics };
};

const notes: Result = {};
const synth = makeStems(24);
const mix: Stereo = [0, 1].map((ch) => {
  const m = new Float32Array(synth.lead[0].length);
  for (let i = 0; i < m.length; i += 1) m[i] = synth.lead[ch][i] + synth.backing[ch][i] + synth.instrumental[ch][i];
  return m;
});
await processSignal('synthetic', mix, 8, 4, notes);

const musicFile = process.env.MUSIC_FILE;
if (musicFile) {
  const music = await loadMusic(musicFile, 20, 50);
  const beat = estimateBeat(music);
  const beats = Math.max(1, Math.round(4 / beat.beatSec));
  const lenSec = beats * beat.beatSec;
  const startSec = beat.phaseSec + Math.ceil((10 - beat.phaseSec) / beat.beatSec) * beat.beatSec;
  (notes as Result).musicBeat = { ...beat, beats, lenSec, startSec };
  await processSignal('music', music, startSec, lenSec, notes);
}
mkdirSync('results', { recursive: true });
writeFileSync('results/loops.json', JSON.stringify(notes, null, 1));
console.log(JSON.stringify(notes, null, 1).slice(0, 6000));
process.exit(0);
