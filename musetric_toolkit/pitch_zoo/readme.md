# Pitch zoo: checking the pitch reference

`musetric-pitch-zoo` checks the reference of `musetric-pitch` that the pitch
bench of Musetric (musetric/musetric#969) scores the Notes view against: it
runs openly licensed pitch models, converts openly licensed ground truth,
scores the models, the reference and the app's trackers with the bench's
metrics, and draws the windows worth looking at. The plan and its results are
in musetric/musetric#978, the flags of a subcommand in its `--help`.

## Rules

- Every model and data set is used under its license, with its package version, checkpoint revision or archive checksum pinned; data, weights and outputs stay outside git.
- `score` is a port of the bench's metrics, and `parity` checks it against a report of the bench after a change to either side, before its numbers are trusted.
- A reference or a model is better only if it scores better on ground truth.
- `align` times every model on synthetic vibrato and glides, and `lag` every truth set against the aligned models; PENN and CREPE run without a lag correction.
- A model is judged only on the resynthesis of another model's curve, a tracker of the app on every set.
- vocadito is scored with `--no-voicing`, as truth for pitch only.
- Runs are compared by their scores, not their bytes.

The sets are in `truth.py`, the models in `registry.py` and the reference in
`pitch_audio/ensemble.py`. A set is `<data>/<set>/audio/<track>.<ext>` and
`<data>/<set>/tracks/<track>/`, which holds `truth.csv`, `<model>.csv` of `run`,
`reference.csv` of `musetric-pitch` and `<tag>.csv` of the bench side by side.

## Procedure

```sh
uv sync --group pitch-research
musetric-pitch-zoo align
musetric-pitch-zoo truth --data-dir $D
musetric-pitch-zoo run --audio-path $LEADS --out-dir $D/leads --models rmvpe fcpe
musetric-pitch-zoo resynth --audio-path $LEADS --f0-dir $D/leads --f0-name rmvpe --data-dir $D --name resynth
musetric-pitch-zoo run --audio-path $D/<set>/audio --out-dir $D/<set>/tracks
musetric-pitch-zoo lag --tracks-dir $D/<set>/tracks --models rmvpe crepe swiftf0 fcpe pesto
musetric-pitch --audio-path $D/<set>/audio --out-dir $D/<set>/tracks
musetric-pitch-zoo score --tracks-dir $D/<set>/tracks --models rmvpe crepe swiftf0 fcpe reference <tag>
musetric-pitch-zoo disputes --tracks-dir $D/<set>/tracks --models rmvpe crepe swiftf0 fcpe
musetric-pitch-zoo plot --tracks-dir $D/<set>/tracks --audio-dir $D/<set>/audio --reference truth --compare <tag> --out-dir $D/plots
musetric-pitch-zoo parity --tracks-dir <bench out> --bench-report <bench out>/<tag>.corpus.json --tag <tag>
```

`$LEADS` holds the lead stems of the #969 corpus, and `<tag>.csv` comes from
`measure:pitch extract --tag <tag>` of the bench. An overlay of `plot` draws one
`--compare` against one `--reference`; the numbers of several go into a table.

## Changing it

- **The reference**: new weights or a new composition in `pitch_audio/ensemble.py`, chosen with `tune`; the old and the new commit write `musetric-pitch --name reference-<commit>` side by side, compared with `score` and `plot --select differ` on every set and the #969 corpus; #969 records the move with its start re-measured.
- **A model**: a class whose `estimate(audio)` returns a `PitchEstimate`, an entry in `registry.py`, a pin in `pitch-research`, a section in `thirdPartyNotices.md`, and a lag within 2.5 ms in `align` or corrected in its module.
- **A data set**: an open license without NonCommercial or NoDerivatives terms, an f0 that no model under test produced, a `fetch_*` function in `truth.py` and a `lag` check.
