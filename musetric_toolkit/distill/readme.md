# Distillation of the vocals student

`musetric-distill` is the recipe that distilled
`musetric/mel-band-roformer-duality-student` from Aname's Mel-Band RoFormer
Duality (musetric/musetric#1081): the open corpus and its curation, the
teacher's labels, the least-squares fits that shape a pooled student out of the
teacher's weights, the training with every option the work used, weight
averaging, the release that `PooledMelBandRoformer` of `separate_audio` reads,
and the evaluation on MUSDB18-HQ. The flags of a subcommand are in its `--help`.

## Rules

- A student is named by its shape: `pool-s` and six per-pair time factors, then `+` modifiers: `w` local time attention, `m` merged high bands, `f` narrowed FFNs, `a` pruned attention heads, `n`, `hr`, `hs` and `hp` the mask estimator, `d` detail paths, `x` replaced sub-layers.
- The fits that the modifiers read are named in one JSON given as `--fits`, with the keys of `Fits` in `student.py`; the teacher is a folder with its `config.yaml` and `model.ckpt`; every path is an argument.
- Training uses openly licensed audio only, and the released student lists it (`release-credits`); MUSDB18-HQ is for evaluation only.
- A listener's verdicts and hand-made decisions are input files of `data-curate`, never code.
- `release-convert` keeps writing the published weights byte for byte from the training checkpoint of `long-gate`.

## Commands

The order the published student was made in, with paths shortened:

```bash
musetric-distill data-open --song-corpus songs --lyrics-corpus lyrics --out open
musetric-distill data-fma-fetch --meta fma_metadata --corpus open --dataset dataset.csv --select
musetric-distill data-ccmixter --corpus open --listing ccmixter --exclude-artists held-out-artists.txt
musetric-distill label --teacher teacher --corpus open --labels labels
musetric-distill data-analyze --labels labels --out analysis
musetric-distill data-curate --corpus open --labels labels --analysis analysis --meta fma_metadata --dataset dataset.csv --verdicts verdicts --decisions decisions.csv
musetric-distill data-variant --dataset dataset.csv --added added.txt --out variants
musetric-distill fit-ffn-importance --teacher teacher --labels labels --dataset variants/dataset-fma.csv --out ffn-importance.json
musetric-distill fit-ffn-refit --teacher teacher --labels labels --dataset variants/dataset-fma.csv --keep 50 --pairs 01 --ranking ffn-importance.json --out ffn-refit-50.pt
musetric-distill train --student "pool-s114422+w32@01+m44@012345+f50r@01+a4@01+hr128t+hp2+d+xb0:lin/b1:lin" --teacher teacher --labels labels --dataset variants/dataset-fma-pells.csv --fits fits.json --out run
musetric-distill soup --out soup/student.pt run-a/student.pt run-b/student.pt run-c/student.pt
musetric-distill release-convert --checkpoint run/student.pt --teacher teacher --out release
musetric-distill release-credits --dataset variants/dataset-fma-pells.csv --release release
musetric-distill musdb-score --release release --musdb musdb --teacher-vocals teacher-vocals --out scores
```

`train` saves `student.pt` every `--save-minutes`, and fp16 snapshots of the
learnt weights every `--snapshot-steps`; `evaluate` scores a checkpoint on the
held-out tracks of the corpus, and `musdb-separate`, `musdb-compare` and
`musdb-bleed` score and compare checkpoints on MUSDB18-HQ.
