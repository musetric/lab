# android

The engine-bench stands run on an Android phone: Node on the computer serves
the pages, a script drives Chrome on the phone (or the WebView of an installed
app) over adb and the Chrome DevTools protocol, and the page reports what the
phone did. Inputs and page logic are those of the sibling stands, reused by
relative path and not copied: [aac_pages](../aac_pages/readme.md),
[page_pool_spike](../page_pool_spike/readme.md),
[stretch_modes](../stretch_modes/readme.md) and
[worklet_stepper](../worklet_stepper/readme.md). Every script writes its
results to `out/` (ignored). No number here is a claim about another phone;
the stands print the device, the Chrome and the WebView versions into every
result.

## Setup

1. Computer: Node 24 or newer, adb (`ADB=<path to adb>` or `adb` on `PATH`),
   and `npm ci` in this folder. Scripts that reuse a sibling stand need
   `npm ci` there too: `decode_pages.mjs` and `compare_decoders.mjs` use
   [aac_pages](../aac_pages/readme.md) (and its `npm run inputs`),
   `page_pool.mjs` uses [page_pool_spike](../page_pool_spike/readme.md) (and
   its `npm run inputs`).
2. Phone: Developer options on, USB debugging on, the computer authorized, the
   screen on and unlocked. With two phones connected set
   `ANDROID_SERIAL=<serial from adb devices>`; with one it is optional. adb
   reads it itself, so every script follows it.
3. Pages are opened as `http://localhost:<port>` on the phone, which is a
   secure context (WebCodecs and AudioWorklet need one). The script starts a
   server on a free port of the computer and maps it with
   `adb reverse tcp:<port> tcp:<port>`; at the end it removes every
   `adb reverse` and `adb forward` of the device. Every request goes over USB,
   so page fetches are slower than on a loopback.
4. Chrome: `device.mjs` forwards the DevTools socket
   `chrome_devtools_remote` to a local port and attaches Playwright with
   `connectOverCDP`; Chrome is started if it is not running, and
   `page_pool.mjs` and `platform_probes.mjs` restart it with
   `am force-stop` before every configuration so that each one gets a fresh
   browser process. A tap on the screen center (from `adb shell wm size`; set
   `ACTIVATION_TAP=x,y` to change it) gives the page its user activation for
   `AudioContext`.
5. WebView: the app must be installed on the phone as a debug build, so that
   its WebView is debuggable. The script brings the app to the front, forwards
   `webview_devtools_remote_<pid of the app>` and drives the page with a small
   raw CDP client, because Playwright cannot attach to a WebView. The page of
   the app is replaced in memory by the stand's page; nothing is installed or
   written on the phone. The app package is `APP_PACKAGE`
   (default `com.musetric.client.dev`).
6. Thermal gate: before every series and every run the scripts read
   `dumpsys thermalservice` and wait while the thermal status is above 1 (up
   to 20 minutes); a series stops when a run starts at status 2 or more. The
   result files keep the status, the battery and the hottest CPU zone before
   and after each run. Run the A and B of a comparison alternately, as the
   scripts do: short bursts on a phone depend on the frequency governor.

Which scripts need what:

| Script | Needs |
| --- | --- |
| `decode_pages.mjs chrome`, `stepper.mjs chrome`, `stretch_timing.mjs chrome`, `page_pool.mjs`, `platform_probes.mjs` | Chrome only |
| `decode_pages.mjs webview`, `stepper.mjs webview`, `stretch_timing.mjs webview`, `compare_decoders.mjs`, `app_memory.mjs` | the debug build of the app (`compare_decoders.mjs` also needs a desktop browser) |
| `report_stepper.mjs`, `report_stretch.mjs` | nothing: they read result files |

## Scripts

Environment variables: `ADB`, `ANDROID_SERIAL`, `APP_PACKAGE`,
`ACTIVATION_TAP`, `DESKTOP_CHANNEL` (`compare_decoders.mjs`), `MUSETRIC_ROOT`
(`build.mjs`), `TAG` and `STEM_S` (`stepper.mjs`).

- `decode_pages.mjs <chrome|webview> <task[,task]>`: the page comparator of
  aac_pages on the phone. Tasks: `env controls indexcheck correctness bench
  parallel flac`. It shows whether the phone's decoder gives the same bits for
  a page as for the whole decode at pre-roll P of 0 to 3, the cost of a page
  and of a seek, parallel decoders and FLAC pages. Generate the inputs first
  with `npm run inputs` in `../aac_pages`. Output:
  `out/decode_pages-<label>-<task>.json`.
- `compare_decoders.mjs`: decodes the first 400 packets of four stems in phone
  Chrome, in the WebView and in a desktop browser and counts the differing
  float samples and the signal to difference ratio. The desktop browser is
  Playwright's channel `msedge`; `DESKTOP_CHANNEL=chrome` or
  `DESKTOP_CHANNEL=chromium` (Playwright's own Chromium) select another one.
  Output: `out/compare_decoders.json`.
- `build.mjs`: bundles the real Musetric worklet runtime for a browser Worker
  (`src/stepper-worker.ts`, which uses the harness and the synthetic stems of
  [worklet_stepper](../worklet_stepper/readme.md)) into
  `dist/stepper-worker.js` (ignored). It needs `MUSETRIC_ROOT=<musetric
  checkout>`, as `../worklet_stepper/build.mjs` does, and must be run before
  `stepper.mjs`.
- `stepper.mjs <chrome|webview> <rounds> <seconds> <ids> [paced]`: steps that
  runtime quantum by quantum in a dedicated Worker and times every quantum with
  `performance.now()`. Scenarios `S0` to `S7b` are those of worklet_stepper
  (for example `S2` is tempo 0.75 with +3 semitones, `S2c` the same starting
  at 30 s). `paced` waits for the real-time period before each quantum.
  Output: `out/stepper-<TAG><label>[-paced].json`; `report_stepper.mjs` prints
  a table of it.
- `stretch_timing.mjs <chrome|webview> <main|tempo125|paced|webview|quick>`:
  the variants of stretch_modes (constant seek, streaming, one or three
  stretchers, staggered phases, split computation) in a Worker, A-B-A-B over
  two repetitions. The Worker runs
  [stretch_modes/common.mjs](../stretch_modes/common.mjs) itself, served next
  to this stand's copy of the stretcher. Output:
  `out/stretch_timing-<label>-<block>.json`; `report_stretch.mjs` prints a
  table of it.
- `page_pool.mjs <task> [json]`: the page-pool spike in a real AudioWorklet in
  phone Chrome. Tasks: `env correctness controls stretch latency underrun
  memory`; the JSON selects scenarios, repetitions, duration and so on (see the
  source). It serves the spike's own `web/` and `data/` folders, so run
  `npm run inputs` in `../page_pool_spike` first. `underrun` reads
  `AudioContext.playbackStats` and takes a `kind` of `pages` or `whole`;
  `memory` polls `dumpsys meminfo` of every Chrome process about four times a
  second. Output: `out/page_pool-<task>.json`.
- `platform_probes.mjs <task> [json]`: `renderSize` (`renderSizeHint` and the
  quantum the worklet sees), `callbacks` (how many quanta come per callback and
  the gaps), `bug` (a Chromium issue: suspend and resume after a worklet stall,
  checked by the rate of render quanta), `clock` (`outputLatency`,
  `baseLatency` and `getOutputTimestamp()` before and after suspend and
  resume) and `stallCheck` (the longest gap and the underruns of one stall).
  Sound goes to the phone's output; nothing needs to be paired. Output:
  `out/platform_probes-<task>.json`.
- `app_memory.mjs [--package com.musetric.client.dev] [--tap x,y] [--cycles 3]`:
  reads `dumpsys meminfo` (PSS and RSS) of the app and of the WebView
  processes at the start, then taps a project card, waits, reads, presses Back,
  waits and reads again, for each cycle, and saves a screenshot per open. The
  tap position is the card of the project to open on the screen the app starts
  on; it defaults to the screen center, which is only right by chance. The app
  must already hold a project; the script creates and deletes nothing. Output:
  `out/app_memory.json` and `out/app_memory-open<N>.png`.
- `device.mjs` and `server.mjs` are the shared pieces: adb, the thermal gate,
  the CDP connections and the static server (Range requests, cross-origin
  isolation headers where a script asks for them).

## Hypotheses these numbers fed

- [musetric/musetric#1274](https://github.com/musetric/musetric/issues/1274),
  H2 (a pool of pages over the whole decoded stems): `page_pool.mjs`
  (correctness, seek-to-sound latency, underruns, the memory of the browser
  processes) and `app_memory.mjs` (what an installed app holds when a project
  is open).
- The same issue, H3 (PCM pages from FLAC) and H4 (AAC pages with an index):
  `decode_pages.mjs` and `compare_decoders.mjs`.
- [musetric/musetric#1275](https://github.com/musetric/musetric/issues/1275),
  baseline and H1 (the bounds check where the mixer reads a stem): `stepper.mjs`
  and `report_stepper.mjs`, which compare `S2` with `S2c`.
- The same issue, H2 to H4 (streaming the stretcher, staggering the three
  stretchers, splitting the computation): `stretch_timing.mjs` and
  `report_stretch.mjs` in a Worker, and the `stretch` and `underrun` tasks of
  `page_pool.mjs` in a real worklet.
- [musetric/musetric#1278](https://github.com/musetric/musetric/issues/1278),
  baseline: `platform_probes.mjs`.
