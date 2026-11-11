import { Q, PAGE, FADE, rampGain, rangeGain, wrapFrame } from './common.js';

const SEG_MAX = 16;
const TONALITY = 8000 / 48000;

class PageProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions;
    this.nStems = o.nStems;
    this.nFrames = o.nFrames;
    this.nPages = o.nFrames.map((n) => Math.ceil(n / PAGE));
    this.ch0 = []; this.ch1 = []; this.pbuf = []; this.pgen = [];
    for (let s = 0; s < this.nStems; s++) {
      this.ch0.push(new Array(this.nPages[s]).fill(null));
      this.ch1.push(new Array(this.nPages[s]).fill(null));
      this.pbuf.push(new Array(this.nPages[s]).fill(null));
      this.pgen.push(new Int32Array(this.nPages[s]));
    }
    this.segF = new Int32Array(SEG_MAX); this.segLen = new Int32Array(SEG_MAX); this.segOff = new Int32Array(SEG_MAX);
    this.motion = 0; this.cur = 0; this.tempo = 1; this.gen = 0;
    this.loopOn = false; this.loopFrom = 0; this.loopTo = 0;
    this.fadePos = -1; this.seekFlag = 0; this.frac = 0;
    this.sinceAnchor = 0; this.dp = null; this.q = 0;
    this.bug = '';
    this.nStretch = 1; this.stagger = true; this.stretch = []; this.mods = null;
    this.cap = null;
    this.seek = null;
    this.resetStats();
    this.port.onmessage = (e) => this.onMain(e.data);
  }

  resetStats() {
    this.st = { quanta: 0, bufQuanta: 0, bufEvents: 0, pages: 0, stale: 0, dup: 0, evicts: 0, refused: 0, rets: 0, anchors: 0, procMs: 0, procHeavy: 0, hMs: 0, hCount: 0, hHeavy: 0,
      errors: 0, heapMoved: 0, benchMsgs: 0, msgBytes: 0, firstQ: 0 };
  }

  onMain(m) {
    const t0 = Date.now();
    switch (m.t) {
      case 'init': this.init(m); break;
      case 'seek':
        this.gen = m.gen; this.cur = m.frame; this.seekFlag = 1; this.fadePos = -1; this.frac = 0;
        this.invalidateStretchers();
        this.seek = { seq: m.seq, handlerFrame: currentFrame, warm: this.segsReadyAt(this.cur), awaiting: true };
        this.ack(m.seq);
        break;
      case 'play': if (this.motion === 0) this.motion = 1; this.ack(m.seq); this.anchor(); break;
      case 'stop': this.motion = 0; this.invalidateStretchers(); this.ack(m.seq); this.anchor(); break;
      case 'tempo': if ((m.value === 1) !== (this.tempo === 1)) this.invalidateStretchers(); this.tempo = m.value; this.ack(m.seq); break;
      case 'loop': this.loopOn = m.on; this.loopFrom = m.from; this.loopTo = m.to; this.gen = m.gen; this.ack(m.seq); break;
      case 'stretchers': this.nStretch = m.count; this.stagger = m.stagger; this.invalidateStretchers(); this.ack(m.seq); break;
      case 'capture': this.armCapture(m); break;
      case 'dump': this.dump(); break;
      case 'stats': this.port.postMessage({ t: 'stats', st: this.st, resident: this.residentCount(), q: this.q, cur: this.cur, motion: this.motion, gen: this.gen, nStretch: this.nStretch }); break;
      case 'resetStats': this.resetStats(); break;
      case 'bug': this.bug = m.name; break;
      case 'ping': this.port.postMessage({ t: 'pong', seq: m.seq, frame: currentFrame, hasPerformance: typeof performance !== 'undefined', hasDate: typeof Date !== 'undefined' }); break;
      default: break;
    }
    this.st.hMs += Date.now() - t0;
  }

  ack(seq) { if (seq !== undefined) this.port.postMessage({ t: 'ack', seq, contextFrame: currentFrame, song: this.cur }); }

  init(m) {
    this.dp = m.port;
    this.dp.onmessage = (e) => this.onPage(e.data);
    if (m.wasmSource) this.initWasm(m);
    else this.port.postMessage({ t: 'ready', wasm: false });
  }

  initWasm(m) {
    try {
      const src = m.wasmSource.replace('instantiateAsync(wasmBinary,wasmBinaryFile,info,receiveInstantiationResult)',
        'WebAssembly.instantiate(Module.__wasmModule,info).then(function(i){receiveInstantiationResult({instance:i})})');
      const factory = new Function(`${src}; return SignalsmithStretch;`)();
      const mkOne = async () => {
        const mod = await factory({ __wasmModule: m.wasmModule });
        mod._main();
        const channels = 2;
        mod._presetDefault(channels, sampleRate);
        mod._reset();
        const inLat = mod._inputLatency(), outLat = mod._outputLatency();
        const bufLen = Math.max(Q * 2, inLat + outLat + 4800, 7500);
        const ptr = mod._setBuffers(channels, bufLen);
        const heap = mod.HEAP8.buffer;
        const inp = [], outp = [];
        for (let c = 0; c < channels; c++) {
          inp.push(new Float32Array(heap, ptr + bufLen * 4 * c, bufLen));
          outp.push(new Float32Array(heap, ptr + bufLen * 4 * (c + channels), bufLen));
        }
        return { mod, heap, inLat, outLat, bufLen, inp, outp, primed: false, ho: 0, wait: 0 };
      };
      Promise.all([mkOne(), mkOne(), mkOne()]).then((a) => { this.stretch = a; this.port.postMessage({ t: 'ready', wasm: true, inLat: a[0].inLat, outLat: a[0].outLat, bufLen: a[0].bufLen }); })
        .catch((e) => this.port.postMessage({ t: 'ready', wasm: false, error: String(e && e.stack || e) }));
    } catch (e) {
      this.port.postMessage({ t: 'ready', wasm: false, error: String(e && e.stack || e) });
    }
  }

  invalidateStretchers() { for (let i = 0; i < this.stretch.length; i++) { this.stretch[i].primed = false; this.stretch[i].wait = this.stagger ? i * 4 : 0; } }

  residentCount() { let n = 0; for (let s = 0; s < this.nStems; s++) for (let k = 0; k < this.nPages[s]; k++) if (this.ch0[s][k]) n++; return n; }

  onPage(m) {
    const t0 = Date.now();
    this.st.hCount++;
    if (m.t === 'page') {
      this.st.pages++;
      const s = m.stem, k = m.page;
      if (m.gen < this.gen) { this.st.stale++; this.retOne(s, k, m.gen, m.buf, 'stale'); }
      else if (this.ch0[s][k]) { this.st.dup++; this.retOne(s, k, m.gen, m.buf, 'dup'); }
      else {
        this.ch0[s][k] = new Float32Array(m.buf, 0, PAGE);
        this.ch1[s][k] = new Float32Array(m.buf, PAGE * 4, PAGE);
        this.pbuf[s][k] = m.buf; this.pgen[s][k] = m.gen;
      }
    } else if (m.t === 'evict') {
      const items = []; const bufs = []; const refused = [];
      const lo = Math.floor((this.cur - 16384) / PAGE), hi = Math.floor((this.cur + 16384 + 8 * Q + (this.stretch.length ? 8192 : 0)) / PAGE);
      for (let i = 0; i < m.items.length; i += 2) {
        const s = m.items[i], k = m.items[i + 1];
        if (!this.ch0[s][k]) { refused.push(s, k); continue; }
        if (k >= lo && k <= hi) { this.st.refused++; refused.push(s, k); continue; }
        items.push({ stem: s, page: k, gen: this.pgen[s][k], buf: this.pbuf[s][k] });
        bufs.push(this.pbuf[s][k]);
        this.ch0[s][k] = null; this.ch1[s][k] = null; this.pbuf[s][k] = null;
      }
      this.st.evicts += items.length;
      if (items.length || refused.length) { this.st.rets++; this.dp.postMessage({ t: 'ret', items, refused, reason: 'evict' }, bufs); }
    } else if (m.t === 'bench') {
      this.st.benchMsgs++;
      if (m.i === 0) this.benchT0 = Date.now();
      (this.benchBufs ??= []).push(m.buf);
      if (this.benchBufs.length === m.n) {
        this.dp.postMessage({ t: 'benchDone', first: this.benchT0, last: Date.now(), count: m.n, bufs: this.benchBufs }, this.benchBufs);
        this.benchBufs = [];
      }
    }
    const d = Date.now() - t0;
    this.st.hMs += d; if (d >= 1) this.st.hHeavy++;
  }

  retOne(stem, page, gen, buf, reason) {
    this.st.rets++;
    this.dp.postMessage({ t: 'ret', items: [{ stem, page, gen, buf }], refused: [], reason }, [buf]);
  }

  anchor() {
    if (!this.dp) return;
    const msg = { t: 'anchor', gen: this.gen, song: this.cur, tempo: this.tempo, motion: this.motion, contextFrame: currentFrame, loop: this.loopOn ? [this.loopFrom, this.loopTo] : null };
    this.st.anchors++;
    this.dp.postMessage(msg);
    this.port.postMessage(msg);
  }

  buildSegs(u0, count) {
    let i = 0, k = 0;
    const loopOn = this.loopOn, from = this.loopFrom, to = this.loopTo;
    while (i < count && k < SEG_MAX) {
      const uf = u0 + i;
      let f, run = count - i;
      if (uf < 0) { f = -1; if (-uf < run) run = -uf; }
      else {
        f = loopOn ? wrapFrame(from, to, uf) : uf;
        const pageEnd = ((f >>> 15) + 1) * PAGE;
        if (pageEnd - f < run) run = pageEnd - f;
        if (loopOn && f < to && to - f < run) run = to - f;
      }
      this.segF[k] = f; this.segLen[k] = run; this.segOff[k] = i; k++; i += run;
    }
    return k;
  }

  segsReady(k, mask) {
    for (let j = 0; j < k; j++) {
      const f = this.segF[j];
      if (f < 0) continue;
      const page = f >>> 15;
      for (let s = 0; s < this.nStems; s++) {
        if (!(mask & (1 << s))) continue;
        if (page < this.nPages[s] && this.ch0[s][page] === null) return false;
      }
    }
    return true;
  }

  segsReadyAt(frame) { return this.segsReady(this.buildSegs(frame, Q), 7); }

  mixSegs(k, mask, d0, d1) {
    const loopOn = this.loopOn, from = this.loopFrom, to = this.loopTo;
    const bug = this.bug;
    for (let s = 0; s < this.nStems; s++) {
      if (!(mask & (1 << s))) continue;
      let ss = s;
      for (let j = 0; j < k; j++) {
        let f = this.segF[j]; const run = this.segLen[j], o = this.segOff[j];
        if (f < 0) continue;
        if (bug === 'shift') f += 1;
        const page = f >>> 15;
        if (page >= this.nPages[ss]) continue;
        if (bug === 'swap' && page >= 10 && s === 1) ss = 0; else ss = s;
        const p0 = this.ch0[ss][page], p1 = this.ch1[ss][page];
        const off = f & (PAGE - 1);
        const gainActive = loopOn && bug !== 'loopGain' && ((f < from + FADE && f + run > from) || (f + run > to - FADE));
        if (gainActive) {
          for (let i = 0; i < run; i++) {
            const g = rangeGain(from, to, f + i);
            d0[o + i] += p0[off + i] * 1 * g; d1[o + i] += p1[off + i] * 1 * g;
          }
        } else {
          for (let i = 0; i < run; i++) { d0[o + i] += p0[off + i]; d1[o + i] += p1[off + i]; }
        }
        if (bug === 'edge' && off === 0 && page > 0) { d0[o] -= p0[0]; d1[o] -= p1[0]; }
      }
    }
  }

  armCapture(m) {
    const nq = m.nq;
    this.cap = { nq, i: 0, l: new Float32Array(nq * Q), r: new Float32Array(nq * Q), pos: new Int32Array(nq), state: new Uint8Array(nq), fade: new Int32Array(nq).fill(-1),
      loopFrom: new Int32Array(nq).fill(-1), loopTo: new Int32Array(nq).fill(-1), seekAt: new Uint8Array(nq), frame: new Float64Array(nq),
      inCap: m.input ? new Float32Array(nq * 6 * 256) : null, inN: m.input ? new Int32Array(nq) : null, inHo: m.input ? new Int32Array(nq * 3) : null, inOn: m.input ? new Uint8Array(nq * 3) : null };
    this.port.postMessage({ t: 'armed' });
  }

  dump() {
    const c = this.cap; this.cap = null;
    if (!c) { this.port.postMessage({ t: 'dump', none: true }); return; }
    const n = c.i;
    const out = { t: 'dump', n, l: c.l.slice(0, n * Q), r: c.r.slice(0, n * Q), pos: c.pos.slice(0, n), state: c.state.slice(0, n), fade: c.fade.slice(0, n), loopFrom: c.loopFrom.slice(0, n),
      loopTo: c.loopTo.slice(0, n), seekAt: c.seekAt.slice(0, n), frame: c.frame.slice(0, n) };
    const tr = [out.l.buffer, out.r.buffer, out.pos.buffer, out.state.buffer, out.fade.buffer, out.loopFrom.buffer, out.loopTo.buffer, out.seekAt.buffer, out.frame.buffer];
    if (c.inCap) { out.inCap = c.inCap.slice(0, n * 6 * 256); out.inN = c.inN.slice(0, n); out.inHo = c.inHo.slice(0, n * 3); out.inOn = c.inOn.slice(0, n * 3); tr.push(out.inCap.buffer, out.inN.buffer, out.inHo.buffer, out.inOn.buffer); }
    this.port.postMessage(out, tr);
  }

  record(state, L, R, fadeStart) {
    const c = this.cap;
    if (!c || c.i >= c.nq) return;
    const i = c.i++;
    c.l.set(L, i * Q); c.r.set(R, i * Q);
    c.pos[i] = this.cur; c.state[i] = state; c.fade[i] = fadeStart;
    c.seekAt[i] = this.seekFlagLogged; c.frame[i] = currentFrame;
    if (this.loopOn) { c.loopFrom[i] = this.loopFrom; c.loopTo[i] = this.loopTo; }
  }

  process(inputs, outputs) {
    const t0 = Date.now();
    const out = outputs[0];
    const L = out[0], R = out[1] ?? out[0];
    try {
      this.st.quanta++; this.q++;
      this.seekFlagLogged = this.seekFlag; this.seekFlag = 0;
      if (this.motion === 0) { this.record(0, L, R, -1); this.pulse(); this.finish(t0); return true; }
      const stretching = this.tempo !== 1 && this.stretch.length > 0;
      let ok;
      let nIn = Q;
      if (stretching) nIn = this.stretchInputCount();
      ok = stretching ? this.stretchReady(nIn) : this.segsReady(this.buildSegs(this.cur, Q), 7);
      if (!ok) {
        if (this.motion === 1) { this.motion = 2; this.st.bufEvents++; this.port.postMessage({ t: 'state', motion: 2, contextFrame: currentFrame, song: this.cur }); this.anchor(); this.invalidateStretchers(); }
        this.st.bufQuanta++;
        this.record(2, L, R, -1);
        this.pulse(); this.finish(t0); return true;
      }
      let fadeStart = -1;
      if (this.motion === 2) {
        this.motion = 1; this.fadePos = 0; this.invalidateStretchers(); this.frac = 0;
        this.port.postMessage({ t: 'state', motion: 1, contextFrame: currentFrame, song: this.cur }); this.anchor();
        if (stretching) { nIn = this.stretchInputCount(); }
      }
      if (this.fadePos >= 0) fadeStart = this.fadePos;
      if (stretching) this.renderStretched(L, R, nIn);
      else { const k = this.buildSegs(this.cur, Q); this.mixSegs(k, 7, L, R); }
      if (this.fadePos >= 0) {
        const fp = this.fadePos;
        for (let i = 0; i < Q; i++) { const g = rampGain(fp + i); L[i] *= g; R[i] *= g; }
        this.fadePos = fp + Q >= FADE ? -1 : fp + Q;
      }
      this.record(1, L, R, fadeStart);
      if (this.seek && this.seek.awaiting) {
        let loud = false;
        for (let i = 0; i < Q; i++) if (L[i] !== 0 || R[i] !== 0) { loud = true; break; }
        if (loud) { this.seek.awaiting = false; this.port.postMessage({ t: 'sound', seq: this.seek.seq, handlerFrame: this.seek.handlerFrame, soundFrame: currentFrame, warm: this.seek.warm }); }
      }
      const next = this.cur + nIn;
      this.cur = this.loopOn ? wrapFrame(this.loopFrom, this.loopTo, next) : next;
      this.pulse();
    } catch (e) {
      this.st.errors++; this.lastError = String(e && e.stack || e);
      L.fill(0); R.fill(0);
      if (this.st.errors === 1) this.port.postMessage({ t: 'error', message: this.lastError });
    }
    this.finish(t0);
    return true;
  }

  finish(t0) { const d = Date.now() - t0; this.st.procMs += d; if (d >= 1) this.st.procHeavy++; }

  pulse() {
    this.sinceAnchor += Q;
    if (this.sinceAnchor >= 2400) { this.sinceAnchor = 0; this.anchor(); }
  }

  stretchInputCount() {
    const raw = Q * this.tempo;
    const n = Math.floor(raw + this.frac);
    this.frac += raw - n;
    return n;
  }

  stretchMask(i) { return this.nStretch === 1 ? 7 : (1 << i); }

  stretchReady(nIn) {
    let ok = true;
    for (let i = 0; i < this.nStretch && ok; i++) {
      const s = this.stretch[i];
      const ho = s.primed ? s.ho : Math.round(s.outLat * this.tempo) + s.inLat;
      const start = s.primed ? this.cur + ho : this.cur + ho - s.bufLen;
      const len = s.primed ? nIn : s.bufLen + nIn;
      if (!s.primed && s.wait > 0) continue;
      ok = this.segsReady(this.buildSegs(start, len), this.stretchMask(i));
    }
    if (!ok) this.frac = 0;
    return ok;
  }

  renderStretched(L, R, nIn) {
    const cap = this.cap;
    for (let i = 0; i < this.nStretch; i++) {
      const s = this.stretch[i];
      const m = s.mod;
      if (!s.primed) {
        if (s.wait > 0) { s.wait--; continue; }
        s.ho = Math.round(s.outLat * this.tempo) + s.inLat;
        m._reset();
        s.inp[0].fill(0); s.inp[1].fill(0);
        const k = this.buildSegs(this.cur + s.ho - s.bufLen, s.bufLen);
        this.mixSegs(k, this.stretchMask(i), s.inp[0], s.inp[1]);
        m._setTransposeSemitones(0, TONALITY); m._setFormantSemitones(0, false); m._setFormantBase(0);
        m._seek(s.bufLen, this.tempo);
        s.primed = true;
      }
      if (s.heap !== m.HEAP8.buffer) this.st.heapMoved++;
      m._setTransposeSemitones(0, TONALITY); m._setFormantSemitones(0, false); m._setFormantBase(0);
      s.inp[0].fill(0, 0, nIn); s.inp[1].fill(0, 0, nIn);
      const k = this.buildSegs(this.cur + s.ho, nIn);
      this.mixSegs(k, this.stretchMask(i), s.inp[0], s.inp[1]);
      if (cap && cap.inCap && cap.i < cap.nq) {
        const base = (cap.i * 3 + i) * 2 * 256;
        cap.inCap.set(s.inp[0].subarray(0, nIn), base); cap.inCap.set(s.inp[1].subarray(0, nIn), base + 256);
        cap.inN[cap.i] = nIn; cap.inHo[cap.i * 3 + i] = s.ho; cap.inOn[cap.i * 3 + i] = 1;
      }
      m._process(nIn, Q);
      const o0 = s.outp[0], o1 = s.outp[1];
      for (let j = 0; j < Q; j++) { L[j] += o0[j]; R[j] += o1[j]; }
    }
  }
}

registerProcessor('page-processor', PageProcessor);
