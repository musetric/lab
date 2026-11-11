export const compare = (ref, test) => {
  const r = { ok: false, reasons: [], channels: [], maxDiff: 0, diffCount: 0, bitDiffCount: 0, nonFinite: 0, firstDiff: -1 };
  if (!Array.isArray(ref) || !Array.isArray(test)) { r.reasons.push('not-arrays'); return r; }
  if (ref.length !== test.length) r.reasons.push(`channel-count ${ref.length} vs ${test.length}`);
  for (let c = 0; c < Math.min(ref.length, test.length); c++) {
    const a = ref[c], b = test[c];
    const ch = { lenRef: a.length, lenTest: b.length };
    if (a.length !== b.length) r.reasons.push(`length ch${c} ${a.length} vs ${b.length}`);
    const n = Math.min(a.length, b.length);
    const ua = new Uint32Array(a.buffer, a.byteOffset, a.length), ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
    for (let i = 0; i < n; i++) {
      const x = a[i], y = b[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) { r.nonFinite++; if (r.firstDiff < 0) r.firstDiff = i; r.maxDiff = Infinity; }
      if (ua[i] !== ub[i]) {
        r.bitDiffCount++;
        if (x !== y || Number.isNaN(x) || Number.isNaN(y)) {
          r.diffCount++; if (r.firstDiff < 0) r.firstDiff = i;
          const d = Math.abs(x - y); if (!(d <= r.maxDiff)) r.maxDiff = Number.isNaN(d) ? Infinity : d;
        }
      }
    }
    r.channels.push(ch);
  }
  if (r.nonFinite) r.reasons.push(`non-finite ${r.nonFinite}`);
  if (r.bitDiffCount) r.reasons.push(`bit-diff ${r.bitDiffCount} (numeric ${r.diffCount})`);
  r.ok = r.reasons.length === 0;
  return r;
};

export const firstNonZero = (chs) => { let m = Infinity; for (const a of chs) { for (let i = 0; i < a.length && i < m; i++) if (a[i] !== 0) { m = i; break; } } return m === Infinity ? -1 : m; };

export const runControls = () => {
  let s = 1; const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296 - 0.5; };
  const N = 100000; const mk = () => [new Float32Array(N).map(rnd), new Float32Array(N).map(rnd)];
  const ref = mk(); const copy = () => ref.map((a) => a.slice());
  const out = [];
  const add = (name, test, expectOk) => { const r = compare(ref, test); out.push({ name, expectOk, ok: r.ok, pass: r.ok === expectOk, reasons: r.reasons }); };
  add('identical copy (positive)', copy(), true);
  { const t = copy(); t[0][777] = NaN; add('one NaN', t, false); }
  { const t = copy(); t[1][5] = Infinity; add('one +Infinity', t, false); }
  { const t = copy(); t[1][5] = -Infinity; add('one -Infinity', t, false); }
  { const a2 = ref.map((a) => a.slice()); a2[0][9] = NaN; const t = a2.map((a) => a.slice()); const r = compare(a2, t); out.push({ name: 'NaN in both ref and test (bitwise equal)', expectOk: false, ok: r.ok, pass: !r.ok, reasons: r.reasons }); }
  { const t = copy().map((a) => { const b = new Float32Array(N + 1); b.set(a); return b; }); add('longer by 1 frame', t, false); }
  { const t = copy().map((a) => a.slice(0, N - 1000)); add('shortened by 1000', t, false); }
  { const t = copy(); t[0].subarray(100, 1100).reverse(); add('permutation (segment reversed)', t, false); }
  { const t = copy(); const [x, y] = [t[0].slice(2000, 3000), t[0].slice(5000, 6000)]; t[0].set(y, 2000); t[0].set(x, 5000); add('permutation (two blocks swapped)', t, false); }
  { const t = copy(); t.reverse(); add('channels swapped', t, false); }
  { const t = copy(); t.pop(); add('channel dropped', t, false); }
  { const t = copy(); const u = new Uint32Array(t[1].buffer); u[N - 1]++; add('one element +1 ulp (last)', t, false); }
  { const t = copy(); t[0][0] = t[0][0] + 1e-6; add('one element +1e-6', t, false); }
  { const t = copy().map((a) => new Float32Array(a.length)); add('all zeros, right length', t, false); }
  { const t = copy(); t[0][1] = 0; ref[0][1] = -0; const r = compare(ref, t); out.push({ name: '-0 vs +0 (bitwise differs)', expectOk: false, ok: r.ok, pass: !r.ok, reasons: r.reasons }); ref[0][1] = t[0][1]; }
  return out;
};
