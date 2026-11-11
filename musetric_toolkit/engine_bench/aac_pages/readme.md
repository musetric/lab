# aac_pages

Decodes the host's fMP4/AAC and FLAC in a browser worker one page at a time (a
page is 32768 frames, 32 AAC packets) and compares every sample with the whole
decode of the same file, for several pre-roll lengths P. Also measures page
cost, seek cost (an own packet index against mediabunny), parallel decoders and
renderer memory.

```sh
npm ci
export MEDIA_BIN=<built media example>   # or MUSETRIC_ROOT=<musetric checkout>
npm run inputs
node run.mjs msedge env,controls,indexcheck,correctness,bench,parallel,mem,flac
```

- `npm run inputs` generates the WAVs, then encodes fMP4 and FLAC with the host
  encoder and appends an `mfra` to a copy of `mix240.mp4`.
- Engines: `msedge`, `chromium`, `webkit` (`npx playwright install` fetches the
  last two). The task `mem` needs Windows.
- Inputs (`gen.mjs`, deterministic): chord, noise, clicks and sweep of 90 s and
  `mix240` of 240 s, stereo 48 kHz 24-bit. `npm run check:index` checks the
  packet index and the comparator controls without a browser.
- Prints one line per task and saves `out/<engine>-<task>.json`.
- `webkitprobe.mjs` prints which WebCodecs classes a WebKit build has.
- Optional PNS switch: in `packages/server/crates/media/src/aac.rs` of the
  checkout add `pns: std::env::var("EXP_PNS").is_ok(),` to the
  `AacEncoderConfig`, then encode with `EXP_PNS=1`. It is a local experiment
  and is not committed.
