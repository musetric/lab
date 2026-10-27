# Rhythm bench: the beats the metronome clicks

`musetric-rhythm-bench` measures the beats the metronome of Musetric clicks on
a set of tracks, for the hypotheses of musetric/musetric#1108; it is not a
processing step. It writes the beats of Beat This! in torch for any input,
window and postprocessing, counts the spurious beats, gaps and tempo jumps of
each variant, finds the passages where two variants differ and renders them
with clicks to listen to. The flags of a subcommand are in its `--help`.

## Rules

- A variant is `<track>/<variant>.json` of a results directory, with `beats` and `downbeats` in seconds; any such JSON counts, such as the app's rhythm result saved as `<track>/app.json`.
- `reference` writes `<label>-w<window>-minimal` and `<label>-w<window>-dbn` as the metronome would click them, and their logits as `<label>-w<window>-beat.npy` and `-downbeat.npy`.
- The rhythm step of the app is the variant `instrumental-w1500-dbn`; that the app's tracker gives the beats of the `dbn` postprocessing is the rhythm case of `musetric-parity`.
- `score` counts as #1108 defines it, in `beat_counts.py`.
- Where the counts disagree with what `listen` plays, the ear decides.

## Commands

```bash
musetric-rhythm-bench reference --audio-path corpus --file-name instrumental.flac --label instrumental --results-dir results
musetric-rhythm-bench reference --audio-path corpus --file-name mix.flac --label mix --results-dir results
musetric-rhythm-bench mean --results-dir results --logits instrumental-w1500 mix-w1500 --label mean-w1500
musetric-rhythm-bench score --results-dir results --variants instrumental-w1500-dbn mix-w1500-dbn mean-w1500-dbn --reference instrumental-w1500-dbn
musetric-rhythm-bench windows --results-dir results --variants instrumental-w1500-dbn mix-w1500-dbn
musetric-rhythm-bench listen --audio-path corpus --file-name mix.flac --results-dir results --windows results/windows-instrumental-w1500-dbn-mix-w1500-dbn.json
musetric-rhythm-bench stems --instrumental corpus/song/instrumental.flac --vocals corpus/song/vocals.flac --mix corpus/song/mix.flac --start 60 --end 120
```

`reference` runs on the CPU by default and skips tracks whose variants exist
unless `--overwrite` is given. `score` writes `score.md` and `score.json` into
the results directory, `windows` writes `windows-<first>-<second>.json`, and
`listen` writes a WAV per window into `<results-dir>/listen`, the audio with
the clicks of the player's metronome on top.
