# Listening rounds

`musetric-listening` runs a blind listening round of two separation systems:
it searches the corpus for clips, builds the round, serves the stand a listener
labels it on, and reports the answers. The flags of a subcommand are in its
`--help`; every path is an argument.

A round is a folder outside the repository with `selection.json`, `clips/`,
`manifest.json` for the page, `key.json` with the system behind A and B and why
a clip was chosen, and the listener's `labels.json` and `labels.log.jsonl`. It
has one of two forms: without `--ask` the listener compares A and B per stem and
may outline sounds on the spectrogram; with `--ask present` the listener only
ticks what sounds in the song itself.

## Rules

- A system is an argument: `name=release folder` of a `LeadBackingRoformer`, or `name=vocals teacher folder,KARA 2 folder`. Its name is written only to `key.json`.
- `select` never picks a track of `excluded_tracks.csv` of `musetric_toolkit/distill`.
- `/key.json` and `/selection.json` are not served, and `check-round` fails on a manifest that gives away a system, a group or a track.
- Round data, clips and labels never enter the repository; a listener's verdicts stay in the round.
- The page speaks Russian to its listener: its words are in `stand/messages.ru.json`, and `build` copies the titles of stems and classes from there into the manifest.
- A browser check writes labels, so it only ever runs on a copy from `copy-round`.

## Commands

The order of a round of ticks, with paths shortened:

```bash
musetric-listening scan --labels labels --corpus corpus --voices voices-fma.json voices-fma-more.json --earlier-round rounds/round1 --out levels.jsonl
musetric-listening words --labels labels --levels levels.jsonl --out words.jsonl
musetric-listening teacher --labels labels --teacher teachers/duality --levels levels.jsonl --words words.jsonl --out windows.jsonl
musetric-listening select --labels labels --levels levels.jsonl --windows windows.jsonl --rule-features rounds/round1/features.json --repeats-from rounds/round1 --out rounds/round2/selection.json
musetric-listening build --selection rounds/round2/selection.json --labels labels --corpus corpus --system student=student --system teacher=teachers/duality,teachers/kara2 --training-data student/training-data.csv --ask present --default-stem mix --id-width 3 --out rounds/round2
musetric-listening check-round --round rounds/round2 --corpus corpus --voices voices-fma.json voices-fma-more.json --earlier-round rounds/round1
musetric-listening serve --round rounds/round2 --open
musetric-listening features --round rounds/round2 --corpus corpus --echonest echonest.csv
musetric-listening report --round rounds/round2 --earlier-round rounds/round1 --rule-features rounds/round1/features.json --windows windows.jsonl --out rounds/round2/report.txt
```

A blind comparison of two students leaves out `--ask`, names both releases, and
needs no `features`; `report` then counts the votes per system:

```bash
musetric-listening select --labels labels --levels levels.jsonl --detector-quiet-clips 0 --detector-loud-clips 0 --reverse-clips 0 --reverse-teacher-clips 0 --random-clips 40 --own-repeats random=4 --repeat-distance 15 --out rounds/round3/selection.json
musetric-listening build --selection rounds/round3/selection.json --labels labels --corpus corpus --system old=releases/old --system new=releases/new --out rounds/round3
```

`scan`, `words` and `teacher` go on where a stopped run ended. A `selection.json`
entry with `stems` maps each system to a folder of stems separated beforehand,
and with `train: false` and a `note` it is a clip that is not for training.

The browser checks are in `stand/checks`, with Playwright from any folder whose
`node_modules` has it:

```bash
musetric-listening copy-round --round rounds/round2 --out tmp/check
musetric-listening serve --round tmp/check --port 8770
node stand/checks/smoke_present.mjs http://127.0.0.1:8770 tmp/shots playwright-home
```

`smoke.mjs` is the same check for a round that compares A and B.
