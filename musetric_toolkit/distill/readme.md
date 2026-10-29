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

## Lead and backing

The split of musetric/musetric#1084 gives the released student two more
outputs in the same pass: `LeadBackingRoformer` of `separate_audio` keeps the
student's vocals mask and adds a mask that takes the backing out of the
vocals; the lead is the vocals minus the backing, so the two sum to the
vocals. Its labels are the student's own vocals and KARA 2's backing of them,
taken at the volume compensation of KARA 2's model data, which
`label-voices` writes to `teacher.json` of the labels: the part of the
vocals that today's cascade takes out.

```bash
musetric-distill label-voices --student student --kara kara2 --corpus open --manifest voices.json --labels voices
musetric-distill train-voices --student student --labels voices --out split
musetric-distill data-ccmixter-voices --crawl ccmixter-crawl --out ccmixter-voices
musetric-distill score-voices --student student --kara kara2 --cascade --checkpoint split/split.pt --golden ccmixter-voices/mix --out scores.json
musetric-distill release-voices --student student --checkpoint split/split.pt --labels voices --corpus open --out release
```

`release-voices` writes the whole model's weights and config, and
`training-data.csv` with every track it learned from: the student's and those
of the split.

`kara2` holds UVR's `UVR_MDXNET_KARA_2.onnx` and `model_data.json`; KARA 2
runs on onnxruntime's CUDA provider. `score-voices` scores the split and the
cascade of the student and KARA 2 against true lead and backing stems.
`data-ccmixter-voices` builds such a set from ccMixter a cappellas whose lead
and backing parts are separate uploads (CC BY and CC0), as `acappella/` and,
over the upload's own instrument stems, `mix/`; it is evaluation only, and its
uploaders are kept out of training.
