# Rhythm bench: the beats the metronome clicks

The metronome of Musetric clicks on the beats of the rhythm step, which runs
Beat This! on the instrumental stem. Where the instrumental has no drums the
model loses the beat, and the clicks bunch up or leave gaps
(musetric/musetric#1108). This package measures that on a set of tracks: it
writes the beats of Beat This! in torch for any input, window and
postprocessing, counts the spurious beats, gaps and tempo jumps of each
variant, finds the passages where two variants differ and renders them with
clicks to listen to. It is the measuring side of the hypotheses of #1108, not
a processing step.

## Variants

A results directory holds one folder per track and one JSON per variant:
`<track>/<variant>.json` with `beats` and `downbeats` in seconds.

- `reference` writes `<label>-w<window>-minimal` and `<label>-w<window>-dbn`:
  Beat This! "final0" in torch, in windows of `--window` frames with a border
  of 6 where the first window's frames are kept, then the `minimal` (peak
  picking) or the `dbn` (madmom) postprocessing of Beat This!, then the rhythm
  summary of the app, which folds a double-time grid. The variant is what the
  metronome would click. Its logits are kept next to it as
  `<label>-w<window>-beat.npy` and `-downbeat.npy`.
- `mean` writes the same two variants for the mean of the logits of two
  inputs, such as the instrumental and the full mix.
- Any other JSON with `beats` and `downbeats` is a variant too, such as the
  rhythm result the app serves for a project, saved as `<track>/app.json`.

The rhythm step of the app at #1031 is the variant `instrumental-w1500-dbn`.
That the app's own tracker gives the same beats as the `dbn` postprocessing
on the same logits is the rhythm case of `musetric-parity`, run by the app's
`measure:parity`.

## Counts

`score` counts per track and in total, as #1108 defines them. An interval is
judged against the local beat, the median of the 13 intervals around it:

- **spurious beats**: intervals shorter than 0.8 of the local beat;
- **gap seconds**: the seconds of intervals longer than 1.25 of it;
- **tempo jumps**: the local beat moving by more than 15 % over 8 beats, a
  switch to double or half time that spurious beats and gaps miss;
- with `--reference`, the **F-measure** of the beats against another variant,
  each beat matched once within 70 ms.

The counts decide only together with the ear: where they disagree with what
`listen` plays, the ear decides (#1108).

## Commands

```bash
# Beat This! on the instrumental of every song of a corpus <song>/instrumental.flac
musetric-rhythm-bench reference --audio-path corpus --file-name instrumental.flac \
  --label instrumental --window 1500 --results-dir results

# the same on the full mix, and the mean of both
musetric-rhythm-bench reference --audio-path corpus --file-name mix.flac \
  --label mix --window 1500 --results-dir results
musetric-rhythm-bench mean --results-dir results \
  --logits instrumental-w1500 mix-w1500 --label mean-w1500

# counts, with the agreement against the app's step
musetric-rhythm-bench score --results-dir results \
  --variants instrumental-w1500-dbn mix-w1500-dbn mean-w1500-dbn \
  --reference instrumental-w1500-dbn

# the passages where two variants differ, and renders of them to listen to
musetric-rhythm-bench windows --results-dir results \
  --variants instrumental-w1500-dbn mix-w1500-dbn
musetric-rhythm-bench listen --audio-path corpus --file-name mix.flac \
  --results-dir results \
  --windows results/windows-instrumental-w1500-dbn-mix-w1500-dbn.json

# why one passage loses the beat: levels and onset periods per 4 s
musetric-rhythm-bench stems --instrumental corpus/song/instrumental.flac \
  --vocals corpus/song/vocals.flac --mix corpus/song/mix.flac --start 60 --end 120
```

`score` writes `score.md` and `score.json` into the results directory,
`windows` writes `windows-<first>-<second>.json`, and `listen` writes WAV files
into `<results-dir>/listen`: each window with 3 s around it, the audio at 0.7
and the clicks of the player's metronome on top, higher on a downbeat.

`reference` runs on the CPU by default, where the counts of #1108 are taken,
and skips tracks whose variants exist unless `--overwrite` is given.
