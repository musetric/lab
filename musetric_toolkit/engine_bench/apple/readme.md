# apple

The engine-bench stands on Apple devices, driven from a Mac: Safari, Chrome
and the Musetric app's WKWebView on the Mac, Safari and the app's WKWebView on
an iPhone attached over USB. The tasks, scenarios and pages are those of
[android](../android/readme.md) and the sibling stands, reused by path. Every
script writes its results to `out/` (ignored).

## How a page is driven

- **Remote console.** WebKit has no Chrome DevTools protocol, so
  [../android/server.mjs](../android/server.mjs) injects
  [web/remote.js](web/remote.js) into every page it serves, and the page runs
  the functions the driver sends it ([remote.mjs](remote.mjs)), as
  `page.evaluate` does on Android. The same script resumes a waiting
  `AudioContext` on a user gesture.
- **Targets** ([device.mjs](device.mjs)), each page in a fresh process:
  - `mac-safari`: a WebDriver session of `safaridriver`; its click is the user
    gesture Safari asks before an `AudioContext` starts. Needs
    `sudo safaridriver --enable` once.
  - `mac-chrome`: Chrome with a profile of its own in `out/`, with
    `--autoplay-policy=no-user-gesture-required` in place of the tap.
  - `mac-app`: the debug build of the app, `APP_BINARY`, with its data in
    `APP_HOME`; [navigate.m](navigate.m), loaded with `DYLD_INSERT_LIBRARIES`,
    points its main WKWebView at the stand.
  - `iphone-safari`, `iphone-app`: the phone opens the stand as
    `http://127.0.0.1:18443`, a secure context, through the phone bridge
    below; the app (`APP_BUNDLE`, default `com.musetric.client.dev`, a debug
    build) is pointed at it once over `pymobiledevice3 webinspector cdp`.
- **Phone bridge** ([phonebridge](phonebridge), [pool.mjs](pool.mjs)): an
  XCUITest runner listens on 127.0.0.1 of the phone and pairs every page
  connection with one the Mac opened over usbmux, like `adb reverse`. It keeps
  the target app in front, taps the screen when a page waits for a user
  gesture, sends the app to the background on request and prints the phone's
  thermal state. Build it once per phone:
  ```sh
  cd phonebridge && DEVELOPMENT_TEAM=<your team> xcodegen generate
  xcodebuild build-for-testing -project PhoneBridge.xcodeproj -scheme PhoneBridgeUITests \
    -destination "platform=iOS,id=<udid>" -derivedDataPath build -allowProvisioningUpdates
  ```
  iOS asks for the passcode to enable UI automation on the first run after a
  restart.
- **Thermal gate**: a run starts at `ProcessInfo.thermalState` 0 or 1 of the
  device under test (nominal, fair) and waits otherwise.
- **Memory** ([footprint.py](footprint.py), sysmon on the iPhone): the
  physical footprint of the target's page processes, polled every 250 ms on
  the Mac (with the kernel's lifetime peak; Chromium's renderers by parent,
  WebKit's WebContent by responsible process) and about every second on the
  iPhone (WebContent by the coalition of the app or Safari).
- **Underruns**: WebKit has no `AudioContext.playbackStats`, and neither
  WebKit nor the audio server logs a missed cycle on the iPhone, so the
  worklets of the page pool count the pauses between their `process()`
  calls (`gap6` to `gap41`, `gapMaxMs`) against the platform's callback
  period; [overloads.mjs](overloads.mjs) adds the audio IO overloads that
  coreaudiod logs on the Mac.

## Setup

Node 24 or newer, Xcode, xcodegen, `pymobiledevice3` on `PATH`, and `npm ci`
plus `npm run inputs` in the stands a script reuses, as for Android; build
`../android/dist/stepper-worker.js` with `../android/build.mjs`. The iPhone
must be unlocked, in Developer Mode, with the screen on.

## Scripts

`<target>` is one of the five above.

| Script | What |
| --- | --- |
| `facts.mjs <target>` | what the engine has: `SharedArrayBuffer`, `crossOriginIsolated`, AudioWorklet, AudioDecoder with AAC and FLAC, `playbackStats`, `renderSizeHint`, and whether an `AudioContext` starts without a gesture |
| `decode_pages.mjs <target> <tasks>` | the tasks of `../android/decode_pages.mjs` |
| `bits.mjs <target>`, `bits.mjs compare` | the whole decode of `../android/compare_decoders.mjs` per target, then every pair compared |
| `page_pool.mjs <target> <task> [json]` | the tasks of `../android/page_pool.mjs` |
| `stepper.mjs <target> <rounds> <seconds> <ids> [paced]` | `../android/stepper.mjs` |
| `stretch_timing.mjs <target> <block>` | `../android/stretch_timing.mjs` |
| `platform_probes.mjs <target> <task> [json]` | the probes of `../android/platform_probes.mjs`, plus `background` on the iPhone: the context's state, time and `resume()` after the app returns from the background |
| `app_memory.mjs <mac-app\|iphone-app> prepare <song>`, `... run` | creates a processed project through the app's API, then opens and closes it in the app's own page three times and reports the footprint of the app's WebContent processes |
