import { mkdirSync, writeFileSync } from 'node:fs';
import inspector from 'node:inspector';
import os from 'node:os';
import { PerformanceObserver, constants as perfConstants } from 'node:perf_hooks';
import { createHarness, createRecorder, quantumFrames, type Harness, type Msg, type HandlerLog } from './harness.js';
import { makeStems, makeVoiceTake, sampleRate, type Stems } from './synth.js';

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
  { id: 'S2', title: 'tempo 0.75, +3 st', quanta: quanta60, setup: async (h, s) => (await base(h, s, { tempo: 0.75, transpose: 3 }), []) },
  { id: 'S2c', title: 'S2 but playback starts at 30 s (no read before song start)', quanta: quanta60, setup: async (h, s) => (await base(h, s, { tempo: 0.75, transpose: 3, startSec: 30 }), []) },
  {
    id: 'S3',
    title: 'S2 + cycle range 4 s',
    quanta: quanta60,
    setup: async (h, s) => (await base(h, s, { tempo: 0.75, transpose: 3, range: [10, 14], startSec: 10 }), []),
  },
  { id: 'S4', title: 'tempo 1, lead +1 octave (leadVoice on)', quanta: quanta60, setup: async (h, s) => (await base(h, s, { octave: 1 }), []) },
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
    setup: async (h, s) =>
      (await base(h, s, {
        tempo: 0.75,
        recording: [
          { songStartSec: 3, seconds: 20, seed: 1 },
          { songStartSec: 26, seconds: 20, seed: 2 },
        ],
      }), []),
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
        rec.data.appendLiveTake({ startFrame: 0, offset: i * 256, samples: total.subarray(i * 256, i * 256 + 256) as Float32Array<ArrayBuffer> });
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

const pct = (sorted: Float64Array | number[], p: number) => sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
const stats = (values: ArrayLike<number>) => {
  const sorted = Float64Array.from(values as ArrayLike<number>).sort();
  let sum = 0;
  for (const v of sorted) sum += v;
  return {
    n: sorted.length,
    mean: sum / sorted.length,
    median: pct(sorted, 50),
    p99: pct(sorted, 99),
    p999: pct(sorted, 99.9),
    max: sorted[sorted.length - 1],
  };
};

const cpuSnapshot = () => os.cpus().map((c) => ({ ...c.times }));
const busyPct = (a: ReturnType<typeof cpuSnapshot>, b: ReturnType<typeof cpuSnapshot>) => {
  let busy = 0;
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    const idle = b[i].idle - a[i].idle;
    const all = (b[i].user - a[i].user) + (b[i].sys - a[i].sys) + (b[i].nice - a[i].nice) + (b[i].irq - a[i].irq) + idle;
    busy += all - idle;
    total += all;
  }
  return (100 * busy) / total;
};

const mem = () => {
  const m = process.memoryUsage();
  return m.heapUsed + m.arrayBuffers;
};

const gcEvery = Number(process.env.GC_EVERY ?? 500);
const gcFn = (globalThis as { gc?: () => void }).gc;

const makeHandlerLog = (store: Map<string, { sync: number[]; total: number[]; bytes: number[]; q: number[] }>, qRef: { q: number }): HandlerLog =>
  (type, syncMs, totalMs, bytes) => {
    let e = store.get(type);
    if (!e) store.set(type, (e = { sync: [], total: [], bytes: [], q: [] }));
    e.sync.push(syncMs);
    e.total.push(totalMs);
    e.bytes.push(bytes);
    e.q.push(qRef.q);
  };

const summarizeHandlers = (store: Map<string, { sync: number[]; total: number[]; bytes: number[]; q: number[] }>) => {
  const out: Record<string, unknown> = {};
  for (const [type, e] of store) {
    const top = e.total.map((t, i) => [t, i] as const).sort((x, y) => y[0] - x[0]).slice(0, 5).map(([t, i]) => ({ ms: t, atQuantum: e.q[i], bytes: e.bytes[i] }));
    out[type] = { count: e.total.length, syncMs: stats(e.sync), totalMs: stats(e.total), bytes: stats(e.bytes), slowest: top };
  }
  return out;
};

type Mode = 'alloc' | 'time' | 'profile' | 'smoke';

const runOne = async (scn: Scenario, stems: Stems, mode: Mode, seconds: number, label: string) => {
  const h = await createHarness(sampleRate);
  const events = await scn.setup(h, stems, seconds);
  const quanta = Math.min(scn.quanta(seconds), mode === 'profile' ? Number(process.env.PQ ?? 20000) : Infinity);
  const store = new Map<string, { sync: number[]; total: number[]; bytes: number[]; q: number[] }>();
  const qRef = { q: 0 };
  const log = makeHandlerLog(store, qRef);
  const times = new Float64Array(quanta);
  const procBytes = new Float64Array(quanta);
  let ev = 0;
  let baseline = 0;
  if (mode === 'alloc') {
    const cal = new Float64Array(2000);
    for (let i = 0; i < cal.length; i += 1) {
      const a = mem();
      cal[i] = mem() - a;
    }
    baseline = stats(cal).median;
    gcFn?.();
  }
  let session: inspector.Session | undefined;
  if (mode === 'profile') {
    session = new inspector.Session();
    session.connect();
    session.post('HeapProfiler.enable');
    session.post('HeapProfiler.startSampling', { samplingInterval: Number(process.env.INTERVAL ?? 128), includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true } as never);
  }
  const gcEntries: PerformanceEntry[] = [];
  const obs = new PerformanceObserver((list) => gcEntries.push(...list.getEntries()));
  if (mode === 'time') obs.observe({ entryTypes: ['gc'] });
  const cpu0 = cpuSnapshot();
  const wall0 = performance.now();
  const { runtime, inputs, outputs, state } = h;
  const useLog = mode === 'alloc' || mode === 'time';
  for (let q = 0; q < quanta; q += 1) {
    qRef.q = q;
    if (ev < events.length && events[ev].q <= q) {
      while (ev < events.length && events[ev].q <= q) {
        (dataTypes.has(events[ev].msg.type) ? h.dataRx : h.ctlRx).queue.push(events[ev].msg);
        ev += 1;
      }
      await h.pump(useLog ? log : undefined);
    }
    if (mode === 'alloc') {
      if (q % gcEvery === 0) gcFn?.();
      const a = mem();
      runtime.process(inputs, outputs);
      procBytes[q] = mem() - a - baseline;
    } else if (mode === 'time') {
      const t0 = performance.now();
      runtime.process(inputs, outputs);
      times[q] = performance.now() - t0;
    } else {
      runtime.process(inputs, outputs);
    }
    state.frame += quantumFrames;
  }
  const wallMs = performance.now() - wall0;
  const busy = busyPct(cpu0, cpuSnapshot());
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setTimeout(resolve, 20));
  obs.disconnect();
  const result: Record<string, unknown> = { scenario: scn.id, title: scn.title, mode, label, quanta, wallMs, realtimeFactor: (quanta * quantumFrames) / sampleRate / (wallMs / 1000), sysBusyPct: busy };
  if (mode === 'alloc') {
    const clean = Array.from(procBytes).filter((b) => b > -1e6);
    let neg = 0;
    for (const b of procBytes) if (b < 0) neg += 1;
    const hb = [...store.values()].reduce((s, e) => s + e.bytes.reduce((a, b) => a + b, 0), 0);
    if (process.env.KEEP_TIMES) writeFileSync('results/bytes_' + scn.id + '.bin', Buffer.from(procBytes.buffer));
    result.processBytes = stats(clean.map((b) => Math.max(0, b)));
    result.processBytesNegative = neg;
    result.processBytesPerQuantumMean = clean.reduce((a, b) => a + Math.max(0, b), 0) / quanta;
    result.handlerBytesTotal = hb;
    result.handlers = summarizeHandlers(store);
    result.totalBytesPerQuantumInclHandlers = result.processBytesPerQuantumMean as number + hb / quanta;
    result.baselinePairBytes = baseline;
  } else if (mode === 'time') {
    const gcByKind: Record<string, { count: number; totalMs: number; maxMs: number }> = {};
    for (const e of gcEntries) {
      const kind = ({ [perfConstants.NODE_PERFORMANCE_GC_MINOR]: 'minor', [perfConstants.NODE_PERFORMANCE_GC_MAJOR]: 'major', [perfConstants.NODE_PERFORMANCE_GC_INCREMENTAL]: 'incremental', [perfConstants.NODE_PERFORMANCE_GC_WEAKCB]: 'weakcb' } as Record<number, string>)[(e as unknown as { detail: { kind: number } }).detail.kind] ?? 'other';
      const g = (gcByKind[kind] ??= { count: 0, totalMs: 0, maxMs: 0 });
      g.count += 1;
      g.totalMs += e.duration;
      g.maxMs = Math.max(g.maxMs, e.duration);
    }
    result.gc = gcByKind;
    result.quantumMs = stats(times);
    let over = 0;
    let half = 0;
    for (const t of times) {
      if (t > budgetMs) over += 1;
      if (t > budgetMs / 2) half += 1;
    }
    result.overBudget = over;
    result.overHalfBudget = half;
    result.budgetMs = budgetMs;
    result.handlers = summarizeHandlers(store);
    if (process.env.KEEP_TIMES) writeFileSync(`results/times_${scn.id}_${label}.bin`, Buffer.from(times.buffer));
  } else if (mode === 'profile') {
    const { profile } = await new Promise<{ profile: unknown }>((resolve, reject) =>
      session!.post('HeapProfiler.stopSampling', (err, res) => (err ? reject(err) : resolve(res as never))),
    );
    session!.disconnect();
    writeFileSync(`results/profile_${scn.id}.json`, JSON.stringify({ quanta, profile }));
  }
  result.lastPeak = Math.max(...Array.from(h.outputs[0]).map(Math.abs)); result.frameEnd = h.state.frame; result.sent = { runtimePosts: h.runtimePort.counts, playheadPosts: h.playheadPort.counts };
  h.mainData.unmount({});
  await h.pump();
  return result;
};

mkdirSync('results', { recursive: true });
const args = process.argv.slice(2);
const mode = (args[0] ?? 'smoke') as Mode;
const pick = (args[1] ?? 'all') === 'all' ? scenarios.map((s) => s.id) : (args[1] ?? '').split(',');
const flag = (name: string, dflt: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : dflt;
};
const rounds = flag('rounds', 1);
const seconds = flag('seconds', 60);

const stemSeconds = 241;
console.log(`node ${process.version}, mode=${mode}, scenarios=${pick.join(',')}, rounds=${rounds}, seconds=${seconds}`);
const stems = makeStems(stemSeconds);
const all: unknown[] = [];

for (let round = 0; round < rounds; round += 1) {
  for (const id of pick) {
    const scn = scenarios.find((s) => s.id === id)!;
    gcFn?.();
    const r = await runOne(scn, stems, mode, seconds, `r${round}`);
    all.push(r);
    if (mode === 'smoke') console.log('   ', JSON.stringify({ peak: r.lastPeak, sent: r.sent }));
    const q = r.quantumMs as { median: number; p99: number; max: number } | undefined;
    console.log(
      `${id} r${round} ${mode} quanta=${r.quanta} wall=${(r.wallMs as number).toFixed(0)}ms x${(r.realtimeFactor as number).toFixed(1)} busy=${(r.sysBusyPct as number).toFixed(1)}%` +
        (q ? ` median=${(q.median * 1000).toFixed(1)}us p99=${(q.p99 * 1000).toFixed(1)}us max=${q.max.toFixed(3)}ms` : '') +
        (r.totalBytesPerQuantumInclHandlers !== undefined ? ` proc=${(r.processBytesPerQuantumMean as number).toFixed(0)}B/q total=${(r.totalBytesPerQuantumInclHandlers as number).toFixed(0)}B/q` : ''),
    );
  }
}
if (mode !== 'smoke' && mode !== 'profile') {
  writeFileSync(`results/${mode}_${process.env.TAG ?? 'run'}.json`, JSON.stringify(all, null, 1));
}
process.exit(0);
