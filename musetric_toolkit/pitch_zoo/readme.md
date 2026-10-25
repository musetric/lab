# Pitch zoo: checking the pitch reference

The pitch bench of Musetric (musetric/musetric#969) scores the line of the
Notes view against the reference of `musetric-pitch`, a combination of pitch
models described below. Where the reference is wrong, a tracker that is right
there scores worse, and without ground truth nobody knows how often that
happens. This package checks the reference: it
runs openly licensed pitch models side by side, converts openly licensed
ground truth, scores the models, the reference and the trackers of the app
with the metrics of the bench, and draws the windows worth looking at. Every
step is repeatable, so the check can be run again when the reference, the
models or the data change. The plan and its history are in
musetric/musetric#978.

## Principles

1. **Open licenses only.** Every model and data set is used under its
   license and nothing is redistributed: data, weights and outputs are
   downloaded or generated outside git.
2. **Pinned and scripted.** Each step is a command; package versions,
   checkpoint revisions and archive checksums are pinned.
3. **One set of metrics.** `score` is a port of the metrics of the bench:
   the accuracy of #966 and the line classes and events of #977. Before its
   numbers are trusted, `parity` checks it against a report of the bench.
4. **Ground truth decides.** A reference or a model is better only if it
   scores better on ground truth. Agreement between models is evidence, not
   proof.
5. **Time alignment is measured, not assumed.** Every model is checked on
   synthetic vibrato and glides with `align`; every truth set is checked
   against the aligned models with `lag`.

## Ground truth

| Set | What | Truth | License |
| --- | --- | --- | --- |
| `resynth` | the 21 lead stems of #969, resynthesized with WORLD along the RMVPE curve | exact by construction | as the corpus |
| `resynth-fcpe` | the same stems, resynthesized along the FCPE curve | exact by construction | as the corpus |
| `vocadito` | 40 solo excerpts in 7 languages | pYIN in Tony, corrected by a trained musician | CC BY 4.0 |
| `dcs` | Dagstuhl ChoirSet, 8 quartet voices, dynamic close microphone (hears the other singers) | annotated by hand on the larynx microphone | CC BY 4.0 |
| `ptdb` | PTDB-TUG speech, 20 speakers: `sa1`, `sa2` and the first three `si` prompts each | RAPT on the laryngograph | ODbL 1.0, DbCL 1.0 |

Checked properties of the sets:

- **vocadito** marks glides between notes, and some quiet but clearly
  harmonic phrases, as unvoiced; the spectrum shows voice there and every
  model tracks it. Its voiced frames are truth for pitch, its unvoiced frames
  are not truth for voicing: score it with `--no-voicing`. `lag` finds it
  about 2 ms early, under half a grid step.
- **PTDB-TUG** reference frame `i` describes the audio at 20.5 ms plus
  `i` times 10 ms, not at `i` times 10 ms: every aligned model finds the
  same shift with `lag`, and `truth` applies it.
- **A resynthesis favours the model whose curve it follows.** The voice
  inherits the voicing and the fine pitch movement of its source: along the
  RMVPE curve RMVPE has the cleanest line (80.4 % of the singing against
  78.3 % for FCPE), along the FCPE curve FCPE has (81.8 % against 71.8 %).
  Build one resynthesis per strong model and judge a model only on the sets
  built from the curves of the others. A tracker of the app built no curve
  and can be judged on all of them.

`resynth` follows the method of MDB-stem-synth: CheapTrick and D4C analyse
the original vocal along the curve, and the WORLD synthesizer rebuilds it with
that exact f0, keeping the timbre, the consonants and the breath. Whatever
the curve gets wrong, the resynthesized voice sings the curve, so the truth
is exact.

Not ground truth, and not to be added as such:

- the CREPE and pYIN trajectories of Dagstuhl ChoirSet: produced by models
  under test;
- Choral Singing Dataset: f0 extracted automatically (SAC);
- Cantoria: pYIN and CREPE;
- Annotated-VocalSet: pYIN smoothed with a median filter;
- MDB-stem-synth, M4Singer, MedleyDB: NonCommercial licenses;
- MIR-1K: no open license stated.

Layout of a set: `<data>/<set>/audio/<track>.<ext>` and
`<data>/<set>/tracks/<track>/`, which holds `truth.csv` (the ground truth
on a 5 ms grid, `trusted` where the annotation covers the frame),
`<model>.csv` from `run`, `reference.csv` from `musetric-pitch` and
`<tag>.csv` from the bench of #969, the same names the bench uses.

## Models

| Name | Package | License | Decode and voicing |
| --- | --- | --- | --- |
| `rmvpe` | vendored (`pitch_audio`) | Apache-2.0, MIT | global Viterbi over the salience, energy gate; the model of `musetric-pitch` |
| `crepe` | `torchcrepe` | MIT | `full` salience through the same global Viterbi; periodicity at least 0.21 above -60 dB |
| `swiftf0` | `swift-f0` | MIT | confidence at least 0.5, as calibrated by its authors |
| `fcpe` | `torchfcpe` | MIT | local argmax, latent peak above 0.006 |
| `penn` | `penn` (FCNF0++) | MIT | Viterbi, periodicity at least 0.065 |
| `pesto` | `pesto-pitch` (`mir-1k_g7`) | LGPL-3.0 | confidence at least 0.5 |
| `pyin` | `librosa` | ISC | 64 ms frames every 10 ms |

Every model with a range runs on 50 to 1100 Hz. Output: `time_s`, `f0_hz`
(0 when unvoiced) and `confidence` per frame, on the grid of the bench.

Known properties:

- The torchcrepe decoders add up to one bin (20 cents) of random dither and
  decode each batch of frames apart, so `crepe` uses the Viterbi of `rmvpe`.
- PENN lags the pitch by 0 to 15 ms depending on the pitch; there is no
  constant correction, so it is used as it is.
- CREPE is 4.5 ms early on a 110 Hz vibrato and within 1.5 ms above it.
- The models are not bit-reproducible run to run, on the CPU as on the GPU:
  two runs of RMVPE on the same audio differ by 0.001 Hz on a few percent of
  frames. Compare runs by their scores, not their bytes.

## The reference of `musetric-pitch`

`musetric-pitch` combines RMVPE, CREPE, SwiftF0 and FCPE, run on the same
audio and put on the same grid:

1. **Voicing**: a frame is voiced when the models that voice it hold at least
   0.6 of the voicing weights, RMVPE 3 and the others 1 each, so RMVPE and at
   least one more model must voice it. On the choir voices of Dagstuhl
   ChoirSet CREPE, SwiftF0 and FCPE voice the neighbouring singers and RMVPE
   does so least. RMVPE drops frames more than 18 dB below the tenth
   percentile of its confident frames, the floor held between -55 and
   -48 dB, which keeps the quiet ends of sung notes.
2. **Pitch**: within each voiced run, every voiced model adds a Gaussian of
   30 cents around its pitch on a 10 cent grid, scaled by its pitch weight,
   RMVPE and FCPE 2, CREPE and SwiftF0 1; a Viterbi over that salience picks
   the path, moving at most 100 cents per frame or jumping to the best bin of
   the previous frame at a fixed cost, so that the path follows the majority
   when a second source, such as a held backing note, sits in the stem. The
   pitch is the weighted mean, in cents, of the models within 50 cents of the
   path.
3. **Trust**: a voiced frame is trusted when at least three models agree on
   it within 50 cents plus the change of the pitch over one frame, so that
   models a few milliseconds apart on a fast glide still agree, or when RMVPE
   and CREPE agree and every other voiced model is at least 1000 cents below
   them. On sung glides and high notes SwiftF0 and FCPE fall to a
   subharmonic, often lagging on it; on the singing of vocadito and Dagstuhl
   ChoirSet the upper pair is right in those frames, on the speech of
   PTDB-TUG mostly the lower one. The other voiced frames stay in the output,
   and the bench leaves them out; `disputes` lists them with the pitch of
   every model.

The weights and parameters live in `pitch_audio/ensemble.py` and were chosen
with `tune` on the truth sets. `confidence` in the output is the share of the
pitch weight that agrees with the path.

## Procedure

Every step is a command; the data lives outside git, in a directory of your
choice.

```sh
uv sync --group pitch-research
D=<data directory>
L=<directory with the lead stems of the #969 corpus>

# 1. models: timing on synthetic vibrato and glides; every lag within 2.5 ms
musetric-pitch-zoo align

# 2. ground truth, and a resynthesis of the corpus along two curves
musetric-pitch-zoo truth --data-dir $D
musetric-pitch-zoo run --audio-path $L --out-dir $D/leads --models rmvpe fcpe
musetric-pitch-zoo resynth --audio-path $L --f0-dir $D/leads --f0-name rmvpe --data-dir $D --name resynth
musetric-pitch-zoo resynth --audio-path $L --f0-dir $D/leads --f0-name fcpe --data-dir $D --name resynth-fcpe

# 3. every model on every set, then the timing of each truth against them:
#    under 2.5 ms the shift is recorded, above it truth.py corrects it
musetric-pitch-zoo run --audio-path $D/<set>/audio --out-dir $D/<set>/tracks
musetric-pitch-zoo lag --tracks-dir $D/<set>/tracks --models rmvpe crepe swiftf0 fcpe pesto

# 4. the reference of the bench, and the trackers of the app, on every set
musetric-pitch --audio-path $D/<set>/audio --out-dir $D/<set>/tracks
yarn workspace @musetric/spectrogram measure:pitch extract --audio $D/<set>/audio --out $D/<set>/tracks --tag <tag>

# 5. scores
musetric-pitch-zoo score --tracks-dir $D/<set>/tracks --models rmvpe crepe swiftf0 fcpe reference <tag>
musetric-pitch-zoo score --tracks-dir $D/vocadito/tracks --models rmvpe fcpe reference <tag> --no-voicing

# 6. looking at it: the spectrum and the lines of the worst or disputed windows
musetric-pitch-zoo disputes --tracks-dir $D/<set>/tracks --models rmvpe crepe swiftf0 fcpe
musetric-pitch-zoo plot --tracks-dir $D/<set>/tracks --audio-dir $D/<set>/audio \
  --reference truth --compare <tag> --select worst --out-dir $D/plots/<set>

# once per change of either implementation of the metrics
musetric-pitch-zoo parity --tracks-dir <bench out> --bench-report <bench out>/<tag>.corpus.json --tag <tag>
```

`plot` writes `<track>/<from>-<to>s/spectrum.png`, the spectrum with the
trusted line of `--reference`, and `overlay.png`, the lines of `--compare`
and `--reference` and the confidence of the reference, one window per 5 s
(`--dpi` sets the resolution). `--select worst` takes the worst windows of
`--compare` against `--reference`, `disputes` the longest voiced but
untrusted runs of `--reference`, `differ` the windows where frames trusted
by `--compare` are more than 50 cents from `--reference`;
`--window <track>:<from>-<to>` draws a given window. On the corpus of #969,
`--reference reference` draws the reference of the bench; on a truth set,
`--reference truth` draws the ground truth.

To change the reference, `tune` scores a grid of weights and parameters of
the ensemble on the truth sets, next to every model alone, in four tables:
the clean share, the trusted frames within 50 cents of the truth, the share
of the voiced truth trusted and the false alarm. The grid is a JSON file;
`anchors` are indices into `--models`:

```json
{
  "voicing_weights": [[3, 1, 1, 1], [1, 1, 1, 1]],
  "pitch_weights": [[2, 1, 1, 2], [1, 0, 0, 1]],
  "anchors": [[0, 1], []],
  "params": {"voicing_share": [0.5, 0.6], "jump_penalty": [4, 8]}
}
```

```sh
musetric-pitch-zoo tune --tracks-dirs $D/resynth-fcpe/tracks $D/vocadito/tracks $D/dcs/tracks $D/ptdb/tracks \
  --models rmvpe crepe swiftf0 fcpe --grid grid.json --out $D/tune
```

A model with a pitch weight of 0 still votes on voicing and counts for trust
but does not move the pitch.

### Comparing commits

Every command writes named CSVs next to each other, so two versions of the
reference, or of a tracker, are compared by running each commit in turn into
the same track directories under its own name, then scoring and drawing the
two names:

```sh
git checkout <commit A>
uv run musetric-pitch --audio-path $D/<set>/audio --out-dir $D/<set>/tracks --name reference-<A>
git checkout <commit B>
uv run musetric-pitch --audio-path $D/<set>/audio --out-dir $D/<set>/tracks --name reference-<B>

musetric-pitch-zoo score --tracks-dir $D/<set>/tracks --models reference-<A> reference-<B>
musetric-pitch-zoo plot --tracks-dir $D/<set>/tracks --audio-dir $D/<set>/audio \
  --reference reference-<B> --compare reference-<A> --select differ --out-dir $D/plots/<A>-<B>
```

The trackers of the app take the same path with `measure:pitch extract --tag
<tag>` of the bench, run from a checkout of each commit. A commit of
`musetric-pitch` older than `--name` writes `reference.csv`; rename it after
the run.

## Reading the scores

The tables are the tables of the bench, one row per scored CSV, as means over
the tracks of a set. For the reference of the bench the last table matters
most:

- **trusted rpa**: the share of trusted frames within 50 cents of the truth.
  The bench counts every trusted frame against the tracker, so every wrong
  trusted frame is a false penalty.
- **trusted share**: the share of the voiced truth the reference trusts; the
  rest is not scored at all.
- **trusted false voicing**: trusted frames where the truth is unvoiced.

A model or a tracker has no trusted mask and trusts every voiced frame.

## Known limits

- **Other singers.** A microphone that hears other singers (the rest of the
  quartet in Dagstuhl ChoirSet, backing vocals left in a lead stem) gives
  every model real voice to track. Where the target singer is silent, the
  models voice the others; Dagstuhl ChoirSet measures it as false alarm, and
  the reference leaves voicing to RMVPE, which does so least.
- **Speech.** PTDB-TUG is speech: it shows how the models handle a voice
  that glides and creaks, but the reference is judged first on singing.

## Extending

- **A model**: a module with a class whose `estimate(audio)` returns a
  `PitchEstimate`, an entry in `registry.py`, a pin in the `pitch-research`
  group and a section in `thirdPartyNotices.md`. `align` must show every lag
  within 2.5 ms, or the lag is corrected in the module. Then run it on every
  set, score it, and give it to `tune`.
- **A change of the reference**: a new composition or new weights in
  `pitch_audio/ensemble.py`, chosen with `tune`, then the old and the new
  commit compared as above on every set and on the corpus of #969. The move
  of the bench is recorded in #969 with the start re-measured.
- **A data set**: an open license without NonCommercial or NoDerivatives
  terms, an f0 that no model under test produced, a `fetch_*` function in
  `truth.py`, and a `lag` check against the aligned models.
