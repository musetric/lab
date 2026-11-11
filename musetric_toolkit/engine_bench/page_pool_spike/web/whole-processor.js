import { Q, rangeGain, wrapFrame } from './common.js';
class WholeProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.stems = []; this.cur = 0; this.motion = 0; this.loop = null; this.seek = null; this.q = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.t === 'stems') { this.stems = m.stems; this.port.postMessage({ t: 'ready' }); }
      else if (m.t === 'seek') { this.cur = m.frame; this.seek = { seq: m.seq, handlerFrame: currentFrame, awaiting: true }; this.port.postMessage({ t: 'ack', seq: m.seq }); }
      else if (m.t === 'play') { this.motion = 1; this.port.postMessage({ t: 'ack', seq: m.seq }); }
      else if (m.t === 'stop') { this.motion = 0; this.port.postMessage({ t: 'ack', seq: m.seq }); }
      else if (m.t === 'loop') { this.loop = m.on ? { from: m.from, to: m.to } : null; this.port.postMessage({ t: 'ack', seq: m.seq }); }
      else if (m.t === 'stats') this.port.postMessage({ t: 'stats', st: { q: this.q, cur: this.cur, motion: this.motion, n: this.stems.length, len: this.stems[0] && this.stems[0][0].length } }); else if (m.t === 'tempo') this.port.postMessage({ t: 'ack', seq: m.seq });
    };
  }
  process(inputs, outputs) { try { return this.proc(outputs); } catch (e) { if (!this.err) { this.err = 1; this.port.postMessage({ t: 'error', message: String(e.stack) }); } return true; } }
  proc(outputs) { this.q++;
    const L = outputs[0][0], R = outputs[0][1];
    if (this.motion === 0) return true;
    const loop = this.loop; const cur = this.cur;
    for (const s of this.stems) for (let i = 0; i < Q; i++) {
      const uf = cur + i; let f = uf, g = 1;
      if (loop) { f = wrapFrame(loop.from, loop.to, uf); g = rangeGain(loop.from, loop.to, f); }
      L[i] += (s[0][f] ?? 0) * 1 * g; R[i] += (s[1][f] ?? 0) * 1 * g;
    }
    if (this.seek && this.seek.awaiting) {
      let loud = false; for (let i = 0; i < Q; i++) if (L[i] !== 0 || R[i] !== 0) { loud = true; break; }
      if (loud) { this.seek.awaiting = false; this.port.postMessage({ t: 'sound', seq: this.seek.seq, handlerFrame: this.seek.handlerFrame, soundFrame: currentFrame, warm: true }); }
    }
    const next = cur + Q; this.cur = loop ? wrapFrame(loop.from, loop.to, next) : next;
    return true;
  }
}
registerProcessor('whole-processor', WholeProcessor);
