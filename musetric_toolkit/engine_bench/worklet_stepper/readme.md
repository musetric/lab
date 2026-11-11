# worklet_stepper

Steps the real Musetric player runtime (`runtime.worklet.ts` and its protocol)
quantum by quantum in Node with fake message ports, and measures allocation,
time per quantum and the loop-wrap audio. The runtime is bundled by esbuild
from a musetric checkout, untouched.

```sh
npm ci
export MUSETRIC_ROOT=<musetric checkout>
node build.mjs
node orchestrate.mjs alloc 3
node orchestrate.mjs time 3
node report.mjs
INTERVAL=256 node --expose-gc dist/run.mjs profile S1 --seconds 60 && node analyze_profile.mjs S1
node dist/loops.mjs
```

- `orchestrate.mjs` runs each scenario in a fresh process, in rounds; `SECONDS`
  sets the length. Scenarios `S0` to `S7b` are in `src/run.ts`.
- Inputs: deterministic synthetic stems and a voice take (`src/synth.ts`).
- `dist/loops.mjs` renders the product's loop wrap against two crossfades into
  `wav/` and `results/loops.json`. An openly licensed track can be added with
  `MUSIC_FILE=<mp3>`, for example Kevin MacLeod, "Sneaky Snitch", CC BY 4.0
  (https://incompetech.com/); it is not part of the repository.
- The imports follow the paths of the checkout of the time; adjust
  `src/harness.ts` if they moved.
