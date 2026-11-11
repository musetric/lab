class Probe extends AudioWorkletProcessor {
  constructor() {
    super();
    this.n = 0;
    this.cap = new Float64Array(3 * 4e4);
    this.rec = false;
    this.stallMs = 0;
    this.total = 0;
    this.len = 0;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.t === 'start') {
        this.n = 0;
        this.rec = true;
      } else if (m.t === 'stop') {
        this.rec = false;
      } else if (m.t === 'dump') {
        this.port.postMessage({
          t: 'dump',
          n: this.n,
          cap: this.cap.slice(0, 3 * this.n),
          total: this.total,
          len: this.len,
          sr: sampleRate,
        });
      } else if (m.t === 'stall') {
        this.stallMs = m.ms;
      } else if (m.t === 'count')
        this.port.postMessage({
          t: 'count',
          total: this.total,
          now: Date.now(),
          frame: currentFrame,
          time: currentTime,
        });
    };
  }
  process(inputs, outputs) {
    this.total++;
    this.len = outputs[0][0].length;
    if (this.rec && this.n < 4e4) {
      const o = 3 * this.n++;
      this.cap[o] = Date.now();
      this.cap[o + 1] = currentFrame;
      this.cap[o + 2] = outputs[0][0].length;
    }
    if (this.stallMs) {
      const e = Date.now() + this.stallMs;
      this.stallMs = 0;
      while (Date.now() < e);
    }
    return true;
  }
}
registerProcessor('probe', Probe);
