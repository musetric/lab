import { stemTypes } from 'musetric-root/packages/audio/src/common/stemType.es.js';
import {
  playerChannel,
  playerDataChannel,
  type PlayerTracks,
} from 'musetric-root/packages/engine/src/player/protocol.cross.js';
import {
  createPlayerRuntime,
  type PlayerRuntime,
} from 'musetric-root/packages/engine/src/player/runtime.worklet.js';

export const quantumFrames = 128;
export const channelCount = 2;

export type Msg = { type: string; payload: unknown };

export class FakePort {
  peer: FakePort | undefined = undefined;
  onmessage: ((event: { data: Msg }) => unknown) | null = null;
  queue: (Msg | undefined)[] = [];
  head = 0;
  sink = false;
  counts: Record<string, number> = {};
  captured: Msg[] | undefined = undefined;
  postMessage(message: Msg, _transfer?: unknown[]) {
    if (this.sink) {
      this.counts[message.type] = (this.counts[message.type] ?? 0) + 1;
      this.captured?.push(message);
      return;
    }
    this.peer!.queue.push(message);
  }
}

export type HandlerLog = (type: string, syncMs: number, totalMs: number, bytes: number) => void;

export type Harness = {
  runtime: PlayerRuntime;
  main: ReturnType<typeof playerChannel.outbound<FakePort>>['methods'];
  mainData: ReturnType<typeof playerDataChannel.outbound<FakePort>>['methods'];
  ctlRx: FakePort;
  dataRx: FakePort;
  runtimePort: FakePort;
  playheadPort: FakePort;
  state: { frame: number };
  outputs: Float32Array[];
  inputs: Float32Array[][];
  pump: (log?: HandlerLog) => Promise<void>;
  step: (quanta: number) => void;
};

const trackBytes = process.env.TRACK_BYTES === '1';
const memNow = () => {
  const m = process.memoryUsage();
  return m.heapUsed + m.arrayBuffers;
};

export const createHarness = async (sampleRate: number): Promise<Harness> => {
  const state = { frame: 0 };
  const g = globalThis as Record<string, unknown>;
  g.sampleRate = sampleRate;
  g.currentTime = 0;
  g.currentFrame = 0;

  const ctlRx = new FakePort();
  const dataRx = new FakePort();
  const mainCtl = new FakePort();
  const mainData = new FakePort();
  mainCtl.peer = ctlRx;
  mainData.peer = dataRx;

  const runtimePort = new FakePort();
  runtimePort.sink = true;
  const runtimeDataPort = new FakePort();
  runtimeDataPort.sink = true;
  const playheadPort = new FakePort();
  playheadPort.sink = true;

  const runtime = await createPlayerRuntime({
    port: playerChannel.inbound(runtimePort as never) as never,
    dataPort: playerDataChannel.inbound(runtimeDataPort as never) as never,
    playheadPorts: [playheadPort as never],
    sampleRate,
    getCurrentTime: () => state.frame / sampleRate,
    getCurrentFrame: () => state.frame,
  });
  const outputs = [new Float32Array(quantumFrames), new Float32Array(quantumFrames)];
  const inputs: Float32Array[][] = [[]];

  const deliver = async (rx: FakePort, target: FakePort, log?: HandlerLog) => {
    while (rx.head < rx.queue.length) {
      const m = rx.queue[rx.head]!;
      rx.queue[rx.head++] = undefined;
      const b0 = trackBytes ? memNow() : 0;
      const t0 = performance.now();
      const p = target.onmessage!({ data: m });
      const t1 = performance.now();
      await p;
      const t2 = performance.now();
      log?.(m.type, t1 - t0, t2 - t0, trackBytes ? memNow() - b0 : 0);
    }
    rx.queue.length = 0;
    rx.head = 0;
  };

  return {
    runtime,
    main: playerChannel.outbound(mainCtl as never).methods as never,
    mainData: playerDataChannel.outbound(mainData as never).methods as never,
    ctlRx,
    dataRx,
    runtimePort,
    playheadPort,
    state,
    outputs,
    inputs,
    pump: async (log) => {
      await deliver(dataRx, runtimeDataPort, log);
      await deliver(ctlRx, runtimePort, log);
    },
    step: (quanta) => {
      for (let i = 0; i < quanta; i += 1) {
        runtime.process(inputs, outputs);
        state.frame += quantumFrames;
      }
    },
  };
};

export const createRecorder = () => {
  const port = new FakePort();
  port.sink = true;
  port.captured = [];
  return {
    ctl: playerChannel.outbound(port as never).methods,
    data: playerDataChannel.outbound(port as never).methods,
    take: () => port.captured!.splice(0),
  };
};

export const emptyTracks = (): PlayerTracks =>
  Object.fromEntries(stemTypes.map((type) => [type, []])) as unknown as PlayerTracks;
