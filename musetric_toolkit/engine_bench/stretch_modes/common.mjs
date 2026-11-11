export const SR = 48000;
export const Q = 128;
export const BUDGET_US = (Q / SR) * 1e6;

const readStretchSourceInNode = async () => {
  const { readFile } = await import('node:fs/promises');
  return readFile(new URL('./node_modules/signalsmith-stretch/SignalsmithStretch.mjs', import.meta.url), 'utf8');
};
const fetchStretchSourceFromServerVendorRoute = async () => (await fetch('/vendor/SignalsmithStretch.mjs')).text();
let source;
export const createModule = async () => {
  source ??= globalThis.process?.versions?.node
    ? await readStretchSourceInNode()
    : await fetchStretchSourceFromServerVendorRoute();
  const i = source.indexOf('function registerWorkletProcessor');
  const factory = new Function(`${source.slice(0, i)}; return SignalsmithStretch;`)();
  return factory();
};

export const rangeFadeFrameCount = 256;
const ramp = (d) => (d < rangeFadeFrameCount ? Math.max(0, d) / rangeFadeFrameCount : 1);
export const createRangeFrames = (range, repeat, stopFrame) => {
  const { fromFrame, toFrame } = range;
  const length = toFrame - fromFrame;
  const endFrame =
    repeat === 'once' && stopFrame !== undefined
      ? Math.min(toFrame, Math.max(fromFrame + 1, stopFrame))
      : toFrame;
  const songFrame =
    repeat === 'cycle'
      ? (frame) => (frame < toFrame ? frame : fromFrame + ((frame - fromFrame) % length))
      : (frame) => frame;
  const gain = (frame) => {
    if (frame >= endFrame) return 0;
    if (frame < fromFrame) return 1;
    return Math.min(ramp(frame - fromFrame), ramp(endFrame - frame));
  };
  return { range, repeat, endFrame, songFrame, gain };
};

export const mixTrackIntoBuffers = (options) => {
  const { inputBuffers, outputFrameIndex, inputFrameOffset, inputFrameCount, channelCount, track, volume, frames } =
    options;
  if (!track) return;
  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const input = inputBuffers[channelIndex];
    const samples = track[channelIndex] ?? track[0];
    if (!samples) continue;
    const frameBase = outputFrameIndex + inputFrameOffset;
    if (!frames) {
      for (let offset = 0; offset < inputFrameCount; offset += 1) {
        const sample = samples[frameBase + offset] ?? 0;
        input[offset] += sample * volume;
      }
      continue;
    }
    for (let offset = 0; offset < inputFrameCount; offset += 1) {
      const songFrame = frames.songFrame(frameBase + offset);
      const sample = samples[songFrame] ?? 0;
      input[offset] += sample * volume * frames.gain(songFrame);
    }
  }
};

const getFrameCounts = (outputFrameCount, tempoRatio, inputFrameFraction) => {
  const raw = outputFrameCount * tempoRatio;
  const inputFrameCount = Math.floor(raw + inputFrameFraction);
  inputFrameFraction += raw - inputFrameCount;
  return { inputFrameCount, inputFrameFraction };
};
const tonalityLimitHz = 8000;

const createState = (m, outputs, sampleRate, outputFrameCount, split, minBuffer) => {
  if (split) m._configure(outputs.length, Math.floor(sampleRate * 0.12), Math.floor(sampleRate * 0.03), true);
  else m._presetDefault(outputs.length, sampleRate);
  m._reset();
  const inputLatency = m._inputLatency();
  const outputLatency = m._outputLatency();
  const bufferLength = Math.max(outputFrameCount, inputLatency + outputLatency, minBuffer);
  const pointer = m._setBuffers(outputs.length, bufferLength);
  const lengthBytes = bufferLength * 4;
  const inputBuffers = [];
  const outputBuffers = [];
  for (let c = 0; c < outputs.length; c += 1) {
    inputBuffers.push(new Float32Array(m.HEAP8.buffer, pointer + lengthBytes * c, bufferLength));
    outputBuffers.push(new Float32Array(m.HEAP8.buffer, pointer + lengthBytes * (c + outputs.length), bufferLength));
  }
  return { bufferLength, channelCount: outputs.length, inputLatency, outputLatency, inputBuffers, outputBuffers };
};

export const createProcessor = (m, sampleRate, { mode = 'const', split = false, probe = null } = {}) => {
  const tonalityLimit = tonalityLimitHz / sampleRate;
  let state;
  let inputFrameFraction = 0;
  let transposeSemitones = 0;
  let tempoRatio = 1;
  let primed = false;
  let headOffset = 0;
  const now = () => performance.now();
  return {
    mode,
    probe,
    get state() {
      return state;
    },
    get headOffset() {
      return headOffset;
    },
    process: (outputs, fillInput) => {
      const outputFrameCount = outputs[0].length;
      if (transposeSemitones === 0 && tempoRatio === 1) {
        primed = false;
        fillInput(outputs, 0, outputFrameCount);
        return outputFrameCount;
      }
      const fc = getFrameCounts(outputFrameCount, tempoRatio, inputFrameFraction);
      inputFrameFraction = fc.inputFrameFraction;
      if (!state || state.channelCount !== outputs.length || state.bufferLength < outputFrameCount) {
        state = createState(m, outputs, sampleRate, outputFrameCount, split, mode === 'stream' ? 7500 : 0);
        primed = false;
      }
      m._setTransposeSemitones(transposeSemitones, tonalityLimit);
      m._setFormantSemitones(0, false);
      m._setFormantBase(0);
      let t0, t1, t2, t3;
      if (probe) t0 = now();
      if (mode === 'const') {
        for (const b of state.inputBuffers) b.fill(0);
        const end = Math.round(state.outputLatency * tempoRatio) + state.inputLatency;
        fillInput(state.inputBuffers, end - state.bufferLength, state.bufferLength);
        if (probe) t1 = now();
        m._seek(state.bufferLength, tempoRatio);
        if (probe) t2 = now();
        m._process(0, outputFrameCount);
      } else {
        if (!primed) {
          headOffset = Math.round(state.outputLatency * tempoRatio) + state.inputLatency;
          for (const b of state.inputBuffers) b.fill(0);
          fillInput(state.inputBuffers, headOffset - state.bufferLength, state.bufferLength);
          m._seek(state.bufferLength, tempoRatio);
          primed = true;
        }
        const n = fc.inputFrameCount;
        for (const b of state.inputBuffers) b.fill(0, 0, n);
        fillInput(state.inputBuffers, headOffset, n);
        if (probe) t1 = now();
        t2 = t1;
        m._process(n, outputFrameCount);
      }
      if (probe) t3 = now();
      for (let c = 0; c < outputs.length; c += 1) {
        outputs[c].set(state.outputBuffers[c].subarray(0, outputFrameCount));
      }
      if (probe) {
        const t4 = now();
        probe.js += t1 - t0;
        probe.seek += t2 - t1;
        probe.proc += t3 - t2;
        probe.copy += t4 - t3;
      }
      return fc.inputFrameCount;
    },
    reset: () => {
      if (mode === 'const') state = undefined;
      else {
        if (state) m._reset();
        primed = false;
      }
      inputFrameFraction = 0;
    },
    setTransposeSemitones: (v) => (transposeSemitones = v),
    setTempoRatio: (v) => (tempoRatio = v),
  };
};

const mulberry32 = (a) => () => {
  a |= 0;
  a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

export const makeStems = (seconds) => {
  const n = Math.round(seconds * SR);
  const rnd = mulberry32(12345);
  const mk = () => [new Float32Array(n), new Float32Array(n)];
  const chord = mk();
  const drums = mk();
  const lead = mk();
  const chords = [
    [220, 261.63, 329.63],
    [174.61, 220, 261.63],
    [196, 246.94, 293.66],
    [164.81, 207.65, 246.94],
  ];
  let lp = [0, 0];
  const hitEnv = new Float32Array(n);
  for (let t = 0; t < n; t += SR / 2) for (let i = 0; i < 6000 && t + i < n; i += 1) hitEnv[t + i | 0] = Math.exp(-i / 700);
  for (let i = 0; i < n; i += 1) {
    const s = i / SR;
    const cs = chords[Math.floor(s / 4) % 4];
    const trem = 0.8 + 0.2 * Math.sin(2 * Math.PI * 0.3 * s);
    const f0 = 330 * Math.pow(2, Math.round(6 * Math.sin(2 * Math.PI * s / 7)) / 12);
    const vib = f0 * (1 + 0.01 * Math.sin(2 * Math.PI * 5.5 * s));
    for (let c = 0; c < 2; c += 1) {
      let v = 0;
      for (const f of cs) for (let h = 1; h <= 4; h += 1) v += (0.12 / h) * Math.sin(2 * Math.PI * f * h * s + c * 0.7 * h);
      chord[c][i] = v * trem;
      const white = rnd() * 2 - 1;
      lp[c] += 0.15 * (white - lp[c]);
      drums[c][i] = 0.04 * lp[c] + 0.35 * hitEnv[i] * white;
      lead[c][i] = 0.25 * Math.sin(2 * Math.PI * vib * s + 0.3 * c) + 0.08 * Math.sin(2 * Math.PI * 2 * vib * s);
    }
  }
  return { chord, drums, lead };
};

export const render = (opts) => {
  const { procs, groups, range, startFrame, nQ, transpose = 0, events = [], timing = 'none', record = false } = opts;
  const startQ = opts.startQ ?? procs.map(() => 0);
  let tempo = typeof opts.tempo === 'number' ? opts.tempo : opts.tempo(0);
  for (const p of procs) {
    p.reset();
    p.setTempoRatio(tempo);
    p.setTransposeSemitones(transpose);
  }
  let renderFrameIndex = startFrame;
  let u = startFrame;
  const out = [new Float32Array(Q), new Float32Array(Q)];
  const tmp = procs.map(() => [new Float32Array(Q), new Float32Array(Q)]);
  const rec = record ? [new Float32Array(nQ * Q), new Float32Array(nQ * Q)] : undefined;
  const hist = [];
  const evLog = [];
  const times = timing !== 'none' ? new Float64Array(nQ) : undefined;
  const fine = timing === 'fine' ? { js: new Float64Array(nQ), seek: new Float64Array(nQ), proc: new Float64Array(nQ), copy: new Float64Array(nQ) } : undefined;
  let seekCostMs = 0;
  const now = () => performance.now();
  for (let k = 0; k < nQ; k += 1) {
    if (opts.pace) opts.pace(k);
    const q0 = timing !== 'none' ? now() : 0;
    for (const ev of events) {
      if (ev.k !== k) continue;
      if (ev.type === 'seek') {
        renderFrameIndex = ev.frame;
        u = ev.frame;
        for (const p of procs) p.reset();
        evLog.push({ k, type: 'seek', frame: ev.frame });
      } else if (ev.type === 'tempo') {
        tempo = ev.value;
        for (const p of procs) p.setTempoRatio(tempo);
        evLog.push({ k, type: 'tempo', value: ev.value });
      }
    }
    if (typeof opts.tempo === 'function' && k > 0) {
      const nt = opts.tempo(k);
      if (nt !== tempo) {
        tempo = nt;
        for (const p of procs) p.setTempoRatio(tempo);
      }
    }
    out[0].fill(0);
    out[1].fill(0);
    const cur = renderFrameIndex;
    let processed = 0;
    for (let pi = 0; pi < procs.length; pi += 1) {
      if (k < startQ[pi]) continue;
      const target = procs.length === 1 ? out : tmp[pi];
      if (procs.length > 1) {
        target[0].fill(0);
        target[1].fill(0);
      }
      const stems = groups[pi];
      const n = procs[pi].process(target, (inputBuffers, inputFrameOffset, inputFrameCount) => {
        for (const track of stems) {
          mixTrackIntoBuffers({
            inputBuffers,
            outputFrameIndex: cur,
            inputFrameOffset,
            inputFrameCount,
            channelCount: 2,
            track,
            volume: 1,
            frames: range,
          });
        }
      });
      if (pi === 0) processed = n;
      if (procs.length > 1) {
        for (let c = 0; c < 2; c += 1) {
          const o = out[c];
          const t = target[c];
          for (let i = 0; i < Q; i += 1) o[i] += t[i];
        }
      }
    }
    const next = cur + processed;
    hist.push({ k, u, p: cur, n: processed, tempo, ho: procs[0].headOffset, bypass: tempo === 1 && transpose === 0 });
    u += processed;
    renderFrameIndex = range ? range.songFrame(next) : next;
    if (rec) {
      rec[0].set(out[0], k * Q);
      rec[1].set(out[1], k * Q);
    }
    if (times) times[k] = (now() - q0) * 1000;
    if (fine) for (const p of procs) { const pr = p.probe; fine.js[k] += pr.js * 1000; fine.seek[k] += pr.seek * 1000; fine.proc[k] += pr.proc * 1000; fine.copy[k] += pr.copy * 1000; pr.js = pr.seek = pr.proc = pr.copy = 0; }
  }
  return { rec, hist, evLog, times, fine, procs };
};
