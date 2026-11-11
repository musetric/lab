# stretch_modes

Compares the product's way of driving signalsmith-stretch (a seek on every
quantum, "const") with a streaming stretcher ("stream"), offline in Node: time
per 128-frame quantum with 1 and 3 stretchers, loops and seeks, and the offset
of a click train against the expected position after wraps, seeks and tempo
changes. `common.mjs` holds the synthetic stems, a copy of the product's range
logic and the render loop.

```sh
npm ci
node smoke.mjs
node timing.mjs --quick && node timing_report.mjs
node accuracy_run.mjs && node accuracy_report.mjs
node tempo_sweep.mjs
node disc_run.mjs
node seekcost.mjs
```

- Inputs: synthetic stems and click trains, generated in memory.
- Prints tables to the terminal and writes `results/*.json` (ignored).
- Timings depend on the machine: compare modes within one run.
