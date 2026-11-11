import {
  createHarness,
  createRecorder,
  quantumFrames,
  type Harness,
  type Msg,
} from '../../worklet_stepper/src/harness.js';
import { makeStems, makeVoiceTake, sampleRate, type Stems } from '../../worklet_stepper/src/synth.js';
const dataTypes = new Set(['mount', 'unmount', 'setRecordingPieces', 'beginLiveTake', 'appendLiveTake']);
const budgetMs = (quantumFrames / sampleRate) * 1000;

type Ev = { q: number; msg: Msg };
type Scenario = {
  id: string;
  title: string;
  quanta: (seconds: number) => number;
  setup: (h: Harness, stems: Stems, seconds: number) => Promise<Ev[]>;
};

const sec = (s: number) => Math.round(s * sampleRate);

type Base = {
  tempo?: number;
  transpose?: number;
  octave?: number;
  range?: [number, number];
  repeat?: 'cycle' | 'once';
  startSec?: number;
  recording?: { songStartSec: number; seconds: number; seed: number }[];
};

let revision = 0;
const base = async (h: Harness, stems: Stems, o: Base) => {
  const copy = (arr: Float32Array<ArrayBuffer>[]) => arr.map((c) => c.slice());
  const tracks = { lead: copy(stems.lead), backing: copy(stems.backing), instrumental: copy(stems.instrumental) };
  const recording = (o.recording ?? []).map((r) => ({
    songStartFrame: sec(r.songStartSec),
    tempo: 1,
    transpose: 0,
    octave: 0,
    samples: makeVoiceTake(r.seconds, r.seed) as Float32Array<ArrayBuffer>,
  }));
  h.mainData.mount({ frameCount: tracks.lead[0].length, tracks, recording });
  h.main.setOutputClock({ clockOffset: 0 });
  h.main.setTempoRatio({ tempoRatio: o.tempo ?? 1 });
  h.main.setTransposeSemitones({ transposeSemitones: o.transpose ?? 0 });
  h.main.setVoiceOctave({ voiceOctave: o.octave ?? 0, shifted: (o.octave ?? 0) !== 0 });
  if (o.range) {
    h.main.setRange({ range: { fromFrame: sec(o.range[0]), toFrame: sec(o.range[1]) }, repeat: o.repeat ?? 'cycle' });
  }
  revision += 1;
  h.main.seek({ frameIndex: sec(o.startSec ?? 0), revision });
  revision += 1;
  h.main.play({ revision, latencyFrameCount: 4096, inputLatencyFrameCount: 0 });
  await h.pump();
  h.step(300);
};

const quanta60 = (s: number) => Math.round((s * sampleRate) / quantumFrames);

const scenarios: Scenario[] = [
  {
    id: 'S0',
    title: 'idle (stopped; harness baseline)',
    quanta: quanta60,
    setup: async (h, stems) => {
      await base(h, stems, {});
      h.main.stop({ revision: ++revision, frameIndex: 0 });
      await h.pump();
      return [];
    },
  },
  { id: 'S1', title: 'tempo 1, no range', quanta: quanta60, setup: async (h, s) => (await base(h, s, {}), []) },
  {
    id: 'S2',
    title: 'tempo 0.75, +3 st',
    quanta: quanta60,
    setup: async (h, s) => (await base(h, s, { tempo: 0.75, transpose: 3 }), []),
  },
  {
    id: 'S2c',
    title: 'S2 but playback starts at 30 s (no read before song start)',
    quanta: quanta60,
    setup: async (h, s) => (await base(h, s, { tempo: 0.75, transpose: 3, startSec: 30 }), []),
  },
  {
    id: 'S3',
    title: 'S2 + cycle range 4 s',
    quanta: quanta60,
    setup: async (h, s) => (await base(h, s, { tempo: 0.75, transpose: 3, range: [10, 14], startSec: 10 }), []),
  },
  {
    id: 'S4',
    title: 'tempo 1, lead +1 octave (leadVoice on)',
    quanta: quanta60,
    setup: async (h, s) => (await base(h, s, { octave: 1 }), []),
  },
  {
    id: 'S4b',
    title: 'S2 + lead +1 octave',
    quanta: quanta60,
    setup: async (h, s) => (await base(h, s, { tempo: 0.75, transpose: 3, octave: 1 }), []),
  },
  {
    id: 'S5',
    title: 'tempo 0.75 + recording, 2 pieces recorded at tempo 1',
    quanta: quanta60,
    setup: async (h, s) => (
      await base(h, s, {
        tempo: 0.75,
        recording: [
          { songStartSec: 3, seconds: 20, seed: 1 },
          { songStartSec: 26, seconds: 20, seed: 2 },
        ],
      }),
      []
    ),
  },
  {
    id: 'S6',
    title: 'live take appended in 256-frame chunks (tempo 1), 4 min',
    quanta: () => quanta60(240) - 200,
    setup: async (h, s) => {
      await base(h, s, {});
      const rec = createRecorder();
      rec.data.beginLiveTake({ takeId: 'take', tempo: 1, transpose: 0, octave: 0 });
      for (const m of rec.take()) h.dataRx.queue.push(m);
      await h.pump();
      const total = makeVoiceTake(240, 9);
      const chunks = Math.floor(total.length / 256);
      const evs: Ev[] = [];
      for (let i = 0; i < chunks; i += 1) {
        rec.data.appendLiveTake({
          startFrame: 0,
          offset: i * 256,
          samples: total.subarray(i * 256, i * 256 + 256) as Float32Array<ArrayBuffer>,
        });
        evs.push({ q: i * 2, msg: rec.take()[0] });
      }
      return evs;
    },
  },
  {
    id: 'S7a',
    title: 'seek every 500 ms, tempo 1',
    quanta: quanta60,
    setup: async (h, s, seconds) => (await base(h, s, {}), seekEvents(seconds)),
  },
  {
    id: 'S7b',
    title: 'seek every 500 ms, tempo 0.75 +3 st',
    quanta: quanta60,
    setup: async (h, s, seconds) => (await base(h, s, { tempo: 0.75, transpose: 3 }), seekEvents(seconds)),
  },
];

const seekEvents = (seconds: number): Ev[] => {
  const rec = createRecorder();
  const evs: Ev[] = [];
  const n = Math.floor(seconds / 0.5);
  let seed = 12345;
  for (let k = 1; k < n; k += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    rec.ctl.seek({ frameIndex: sec(5 + (seed / 2 ** 32) * 150), revision: 1000 + k });
    evs.push({ q: Math.round((k * 0.5 * sampleRate) / quantumFrames), msg: rec.take()[0] });
  }
  return evs;
};

const pct = (sorted: Float64Array, p: number) =>
  sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
const stats = (values: Float64Array) => {
  const sorted = Float64Array.from(values).sort();
  let sum = 0,
    o50 = 0,
    o100 = 0;
  for (const v of values) {
    sum += v;
    if (v > budgetMs / 2) o50 += 1;
    if (v > budgetMs) o100 += 1;
  }
  return {
    n: values.length,
    mean: sum / values.length,
    median: pct(sorted, 50),
    p99: pct(sorted, 99),
    p999: pct(sorted, 99.9),
    max: sorted[sorted.length - 1],
    over50: o50,
    over100: o100,
  };
};
let sleepSab: Int32Array | undefined;
let stemCache: { seconds: number; stems: Stems } | undefined;
type Req =
  | { cmd: 'run'; id: string; seconds: number; stemSeconds: number; paced: boolean; label: string; waitGo?: boolean }
  | { cmd: 'go' };
let goResolve: (() => void) | undefined;
self.onmessage = async (e: MessageEvent<Req>) => {
  if (e.data.cmd === 'go') {
    goResolve?.();
    return;
  }
  const { id, seconds, stemSeconds, paced, label, waitGo } = e.data;
  try {
    if (!stemCache || stemCache.seconds !== stemSeconds)
      stemCache = { seconds: stemSeconds, stems: makeStems(stemSeconds) };
    const scn = scenarios.find((s) => s.id === id)!;
    const h = await createHarness(sampleRate);
    const events = await scn.setup(h, stemCache.stems, seconds);
    const quanta = scn.quanta(seconds);
    const times = new Float64Array(quanta);
    const { runtime, inputs, outputs, state } = h;
    let ev = 0;
    if (waitGo) {
      postMessage({ ready: true });
      await new Promise<void>((r) => (goResolve = r));
    }
    const wall0 = performance.now();
    const periodMs = budgetMs;
    for (let q = 0; q < quanta; q += 1) {
      if (ev < events.length && events[ev].q <= q) {
        while (ev < events.length && events[ev].q <= q) {
          (dataTypes.has(events[ev].msg.type) ? h.dataRx : h.ctlRx).queue.push(events[ev].msg);
          ev += 1;
        }
        await h.pump();
      }
      if (paced) {
        const target = wall0 + q * periodMs;
        const wait = target - performance.now();
        if (wait > 0.05) {
          sleepSab ??= new Int32Array(new SharedArrayBuffer(4));
          Atomics.wait(sleepSab, 0, 0, wait);
        }
      }
      const t0 = performance.now();
      runtime.process(inputs, outputs);
      times[q] = performance.now() - t0;
      state.frame += quantumFrames;
    }
    const wallMs = performance.now() - wall0;
    h.mainData.unmount({});
    await h.pump();
    const s = stats(times);
    postMessage({
      result: {
        id,
        label,
        paced,
        quanta,
        wallMs,
        quantumMs: s,
        budgetMs,
        sent: { runtimePosts: h.runtimePort.counts },
      },
    });
  } catch (err) {
    postMessage({ error: String((err as Error).stack ?? err) });
  }
};
postMessage({ loaded: true });
