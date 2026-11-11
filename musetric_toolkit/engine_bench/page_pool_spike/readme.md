# page_pool_spike

A browser spike of the page design: a media worker range-fetches and decodes
AAC pages (WebCodecs) into a pool of transferable buffers that an AudioWorklet
mixes, with loops, seeks and 1 or 3 streaming stretchers. A whole-buffer
baseline and a reference decode check every capture. No SharedArrayBuffer.

```sh
npm ci
export MEDIA_BIN=<built media example>   # or MUSETRIC_ROOT=<musetric checkout>
npm run inputs
node run.mjs env,smoke
node run.mjs correctness,controls,stretch,latency,underrun,cost,memory
node run.mjs correctness '{"scenarios":["plain"]}'
```

- `npm run inputs` generates three distinct synthetic 240 s stems as WAV and
  encodes them to fMP4 with the host encoder.
- Microsoft Edge (`channel: msedge`), headed, on a loopback server; `memory`
  needs Windows. Options go as a JSON second argument.
- Prints one line per scenario and saves `out/<task>.json`.
- `window.T` in `web/main.js` is the scenario API of the page.
